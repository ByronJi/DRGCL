import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from aug import TUDataset_aug as TUDataset
from torch_geometric.loader import DataLoader

from loss import BarlowTwinsLoss, CCALoss
import sys
import copy
import argparse
import random
import numpy as np
import os.path as osp

import argparse
from mask_generator import FeatureMask
from torch.autograd import Variable
from gsimclr import MaskSimclr
from evaluate_embedding import evaluate_embedding

parser = argparse.ArgumentParser(description='Meta Mask Training for Simclr on TU datasets')
parser.add_argument('--DS', dest='DS', help='Dataset', default='PROTEINS')
parser.add_argument('--lr', dest='lr', type=float,
                    help='Learning rate.', default=0.01)
parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=3,
                    help='Number of graph convolution layers before each pooling')
parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=32,
                    help='')
parser.add_argument('--aug', type=str, default='random4')
parser.add_argument('--max-epochs', default=20, type=int)
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--disable-meta', action="store_true",
                    help='whether using meta learning when training')
parser.add_argument('--disable-sigmoid',
                    action="store_true",
                    help='whether using meta learning when training')
parser.add_argument('--no-second-order', action="store_true")
parser.add_argument('--weight-bl', default=1, type=float)
parser.add_argument('--weight-cl', default=10, type=float,
                    help='weight of simclr loss')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--log-interval', type=int, default=5)
parser.add_argument('--criterion-type', type=str, default='CCA', choices=['BarlowTwins', 'CCA'])


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class MetaMaskBarlowTwins(nn.Module):
    def __init__(self, dataset_num_features, hidden_dim, num_gc_layers, device, enable_meta, enable_sigmoid, weight_cl,
                 second_order, criterion_type):
        super(MetaMaskBarlowTwins, self).__init__()
        self.mask_simclr = MaskSimclr(dataset_num_features, hidden_dim, num_gc_layers, device)
        self.mask_simclr_ = copy.deepcopy(self.mask_simclr)
        if criterion_type == 'BarlowTwins':
            self.criterion = BarlowTwinsLoss(device)
        elif criterion_type == 'CCA':
            self.criterion = CCALoss(device)
        if enable_meta:
            self.auto_mask = FeatureMask(hidden_dim * num_gc_layers, enable_sigmoid)
        self.enable_meta = enable_meta
        self.weight_cl = weight_cl
        self.second_order = second_order

    def unrolled_backward(self, data, data_aug, model_optim, mask_optim, eta):
        """
        Compute un-rolled loss and backward its gradients
        """
        #  compute unrolled multi-task network theta_1^+ (virtual step)
        masks = self.auto_mask()
        x, x_cl = self.mask_simclr(data.x, data.edge_index, data.batch, masks)
        x_aug, x_aug_cl = self.mask_simclr(data_aug.x, data_aug.edge_index, data_aug.batch, masks)
        loss = MaskSimclr.simclr_loss(x_cl, x_aug_cl)

        model_optim.zero_grad()
        mask_optim.zero_grad()
        # calculate a trial step
        loss.backward()
        # copy the gradients
        gradients = copy.deepcopy(
            [v.grad.data if v.grad is not None else None for v in self.mask_simclr.parameters()])

        model_optim.zero_grad()
        mask_optim.zero_grad()

        with torch.no_grad():
            for weight, weight_, d_p in zip(self.mask_simclr.parameters(),
                                            self.mask_simclr_.parameters(),
                                            gradients):
                if d_p is None:
                    weight_.copy_(weight)
                    continue

                d_p = -d_p
                g = model_optim.param_groups[0]
                state = model_optim.state[weight]
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                step_t = state['step']
                step_t += 1

                if g['weight_decay'] != 0:
                    d_p = d_p.add(weight, alpha=g['weight_decay'])
                beta1, beta2 = g['betas']
                beta2 = g['betas'][1]
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(g['betas'][0]).add_(d_p, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(d_p, d_p.conj(), value=1 - beta2)

                step = step_t

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                step_size = g['lr'] / bias_correction1

                bias_correction2_sqrt = math.sqrt(bias_correction2)
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(g['eps'])

                weight.addcdiv_(exp_avg, denom, value=-step_size)
                weight_ = copy.deepcopy(weight)
                weight_.grad = None

        masks = self.auto_mask()
        x, x_cl = self.mask_simclr_(data.x, data.edge_index, data.batch, masks)
        x_aug, x_aug_cl = self.mask_simclr_(data_aug.x, data_aug.edge_index, data_aug.batch, masks)
        loss = MaskSimclr.simclr_loss(x_cl, x_aug_cl)

        mask_optim.zero_grad()
        loss.backward()

        dalpha = [v.grad for v in self.auto_mask.parameters()]
        if self.second_order:
            vector = [v.grad.data if v.grad is not None else None for v in self.mask_simclr_.parameters()]

            implicit_grads = self._hessian_vector_product(vector, data, data_aug)

            for g, ig in zip(dalpha, implicit_grads):
                g.data.sub_(ig.data, alpha=eta)

        for v, g in zip(self.auto_mask.parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _hessian_vector_product(self, gradients, data, data_aug, r=1e-2):
        with torch.no_grad():
            for weight, weight_ in zip(self.mask_simclr.parameters(), self.mask_simclr_.parameters()):
                weight_.copy_(weight)
                weight_.grad = None

        valid_grad = []
        for grad in gradients:
            if grad is not None:
                valid_grad.append(grad)
        R = r / _concat(valid_grad).norm()
        for p, v in zip(self.mask_simclr_.parameters(), gradients):
            if v is not None:
                p.data.add_(v, alpha=R)

        masks = self.auto_mask()
        x, x_cl = self.mask_simclr_(data.x, data.edge_index, data.batch, masks)
        x_aug, x_aug_cl = self.mask_simclr_(data_aug.x, data_aug.edge_index, data_aug.batch, masks)
        loss = MaskSimclr.simclr_loss(x_cl, x_aug_cl)

        grads_p = torch.autograd.grad(loss, self.auto_mask.parameters())

        for p, v in zip(self.mask_simclr_.parameters(), gradients):
            if v is not None:
                p.data.sub_(v, alpha=2 * R)

        masks = self.auto_mask()
        x, x_cl = self.mask_simclr_(data.x, data.edge_index, data.batch, masks)
        x_aug, x_aug_cl = self.mask_simclr_(data_aug.x, data_aug.edge_index, data_aug.batch, masks)
        loss = MaskSimclr.simclr_loss(x_cl, x_aug_cl)

        grads_n = torch.autograd.grad(loss, self.auto_mask.parameters())

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]

    def per_optimizer_step(self,
                           optimizer_a=None,
                           optimizer_b=None,
                           loss=None):

        # update params
        if loss is not None:
            optimizer_a.zero_grad()
            if optimizer_b is not None:
                optimizer_b.zero_grad()
            loss.backward()

        optimizer_a.step()
        optimizer_a.zero_grad()
        if optimizer_b is not None:
            optimizer_b.step()
            optimizer_b.zero_grad()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    args = parser.parse_args()
    setup_seed(args.seed)

    accuracies = {'val': [], 'test': []}
    batch_size = args.batch_size
    lr = args.lr
    DS = args.DS
    path = osp.join(osp.expanduser('~'), 'datasets', DS)
    print(path)
    enable_meta = not args.disable_meta
    enable_sigmoid = not args.disable_sigmoid
    second_order = not args.no_second_order
    # enable_meta = False
    # enable_sigmoid = False
    # second_order = False


    dataset = TUDataset(path, name=DS, aug=args.aug).shuffle()
    dataset_eval = TUDataset(path, name=DS, aug='none').shuffle()
    print(len(dataset))
    print(dataset.get_num_feature())
    try:
        dataset_num_features = dataset.get_num_feature()
    except:
        dataset_num_features = 1

    dataloader = DataLoader(dataset, batch_size=batch_size)
    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MetaMaskBarlowTwins(dataset_num_features, args.hidden_dim, args.num_gc_layers, device, enable_meta,
                                enable_sigmoid,
                                args.weight_cl,
                                second_order, args.criterion_type)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.mask_simclr.parameters(), lr=0.01)

    if enable_meta:
        # mask_optim = torch.optim.Adam(model.auto_mask.parameters(), lr=0.01)
        mask_optim = torch.optim.SGD(model.auto_mask.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
        mask_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(mask_optim, args.max_epochs)

    for epoch in range(1, args.max_epochs + 1):
        loss_all = 0
        loss_bl_all = 0
        loss_cl_all = 0
        model.train()
        for data in dataloader:
            data, data_aug = data
            optimizer.zero_grad()
            node_num, _ = data.x.size()
            data = data.to(device)

            masks = torch.ones((args.hidden_dim * args.num_gc_layers)).to(device)
            print(masks.shape)
            if enable_meta:
                masks = model.auto_mask()
            x, x_cl = model.mask_simclr(data.x, data.edge_index, data.batch, masks)

            if args.aug == 'dnodes' or args.aug == 'subgraph' or args.aug == 'random2' or args.aug == 'random3' or args.aug == 'random4':
                edge_idx = data_aug.edge_index.numpy()
                _, edge_num = edge_idx.shape
                idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]

                node_num_aug = len(idx_not_missing)
                data_aug.x = data_aug.x[idx_not_missing]

                data_aug.batch = data.batch[idx_not_missing]
                idx_dict = {idx_not_missing[n]: n for n in range(node_num_aug)}
                edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if
                            not edge_idx[0, n] == edge_idx[1, n]]
                data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)

            data_aug = data_aug.to(device)
            x_aug, x_aug_cl = model.mask_simclr(data_aug.x, data_aug.edge_index, data_aug.batch, masks)
            loss_cl = MaskSimclr.simclr_loss(x_cl, x_aug_cl)
            print("loss_cl: ", loss_cl)
            loss_bl = model.criterion(x, x_aug)
            print("loss_bl: ", loss_bl)
            loss = args.weight_bl * loss_bl + args.weight_cl * loss_cl
            if enable_meta:
                model.per_optimizer_step(optimizer, None, loss)
                model.unrolled_backward(data, data_aug, optimizer, mask_optim, optimizer.param_groups[0]['lr'])
                model.per_optimizer_step(mask_optim)
            else:
                loss.backward()
                optimizer.step()

            loss_bl_all += loss_bl.item() * data.num_graphs
            loss_cl_all += loss_cl.item() * data.num_graphs
            loss_all += loss.item() * data.num_graphs
            if enable_meta:
                mask_scheduler.step()

        print('Epoch {}, Loss {}, Loss_bl {}, Loss_cl {}'.format(epoch, loss_all / len(dataloader),
                                                                 loss_bl_all / len(dataloader),
                                                                 loss_cl_all / len(dataloader)))

        if epoch % args.log_interval == 0:
            model.eval()
            emb, y = model.mask_simclr.encoder.get_embeddings(dataloader_eval)
            if enable_meta:
                masks = model.auto_mask().detach().cpu().numpy()
                print(masks)
                emb = emb * masks[None, :]
            # if epoch == 20:
            #     np.save("simclr_emb.npy", emb)
            #     np.save("simclr_y.npy", y)
            acc_val, acc = evaluate_embedding(emb, y)
            acc_val = round(acc_val, 8)
            acc = round(acc, 8)
            accuracies['val'].append(acc_val)
            accuracies['test'].append(acc)
            select_val_index = accuracies['val'].index(max(accuracies['val']))
            select_test_value = accuracies['test'][select_val_index]
            print('Eval | Epochs {}: val {}, test {}, cur max test {}'.format(epoch, accuracies['val'][-1], accuracies['test'][-1], select_test_value))

    log_path = 'log_' + args.criterion_type + '_' + args.aug + '_' + str(enable_meta) + '_' + str(
        args.weight_cl) + '_' + str(args.weight_bl) + '_' + str(args.batch_size)
    if not osp.isdir(log_path):
        os.makedirs(log_path)
    with open(log_path + '/' + args.DS, 'a+') as f:
        s = json.dumps(accuracies)
        f.write('DS:{},seed:{},layer_num:{},epochs:{},lr:{}, select test value:{}\n'.format(args.DS, args.seed, args.num_gc_layers,
                                                                      args.max_epochs, args.lr, select_test_value))
        f.write(s)
        f.write('\n')
