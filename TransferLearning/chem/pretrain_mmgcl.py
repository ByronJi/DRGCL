import argparse
import copy
import math

from loader import MoleculeDataset_aug
from torch_geometric.data import DataLoader
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import global_mean_pool

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN
from sklearn.metrics import roc_auc_score

import pandas as pd

from copy import deepcopy

from loss import BarlowTwinsLoss, CCALoss
from mask_generator import FeatureMask
from torch.autograd import Variable


def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr


class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)

    def forward(self, x, summary):
        h = torch.matmul(summary, self.weight)
        return torch.sum(x * h, dim=1)


class graphcl(nn.Module):

    def __init__(self, gnn):
        super(graphcl, self).__init__()
        self.gnn = gnn
        self.pool = global_mean_pool
        self.projection_head = nn.Sequential(nn.Linear(300, 300), nn.ReLU(inplace=True), nn.Linear(300, 300))

    def forward_cl(self, x, edge_index, edge_attr, batch):
        x = self.gnn(x, edge_index, edge_attr)
        x = self.pool(x, batch)
        x = self.projection_head(x)
        return x

    def loss_cl(self, x1, x2):
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss


class mask_graphcl(nn.Module):

    def __init__(self, gnn):
        super(mask_graphcl, self).__init__()
        self.gnn = gnn
        self.pool = global_mean_pool
        self.projection_head = nn.Sequential(nn.Linear(300, 300), nn.ReLU(inplace=True), nn.Linear(300, 300))
        self.projection_head2 = nn.Sequential(nn.Linear(300, 300), nn.ReLU(inplace=True), nn.Linear(300, 300))

    def forward_cl(self, x, edge_index, edge_attr, batch, masks=None):
        x = self.gnn(x, edge_index, edge_attr)
        x = self.pool(x, batch)
        if masks is not None:
            x = x * masks[None, :]
        x_cl = self.projection_head(x)
        x_bl = self.projection_head2(x)
        return x_bl, x_cl

    @staticmethod
    def simclr_loss(x1, x2):
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class MetaMaskBarlowTwins(nn.Module):
    def __init__(self, gnn, device, enable_meta, enable_sigmoid, weight_cl,
                 second_order, criterion_type):
        super(MetaMaskBarlowTwins, self).__init__()
        self.mask_simclr = mask_graphcl(gnn)
        self.mask_simclr_ = copy.deepcopy(self.mask_simclr)
        if criterion_type == 'BarlowTwins':
            self.criterion = BarlowTwinsLoss(device)
        elif criterion_type == 'CCA':
            self.criterion = CCALoss(device)
        if enable_meta:
            self.auto_mask = FeatureMask(300, enable_sigmoid)
        self.enable_meta = enable_meta
        self.weight_cl = weight_cl
        self.second_order = second_order

    def unrolled_backward(self, data, data_aug, model_optim, mask_optim, eta):
        """
        Compute un-rolled loss and backward its gradients
        """
        #  compute unrolled multi-task network theta_1^+ (virtual step)
        masks = self.auto_mask()
        # x1, x1_cl = model.mask_simclr.forward_cl(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch, masks)
        x, x_cl = self.mask_simclr.forward_cl(data.x, data.edge_index, data.edge_attr, data.batch, masks)
        x_aug, x_aug_cl = self.mask_simclr.forward_cl(data_aug.x, data_aug.edge_index, data_aug.edge_attr,
                                                      data_aug.batch, masks)
        loss = mask_graphcl.simclr_loss(x_cl, x_aug_cl)

        model_optim.zero_grad()
        mask_optim.zero_grad()
        # 计算一步梯度
        loss.backward()
        # copy梯度
        gradients = copy.deepcopy(
            [v.grad.data if v.grad is not None else None for v in self.mask_simclr.parameters()])

        model_optim.zero_grad()
        mask_optim.zero_grad()
        # do virtual step: theta_1^+ = theta_1 - alpha * (primary loss + auxiliary loss)
        # optimizer.param_groups：是长度为2的list，其中的元素是2个字典；
        # optimizer.param_groups[0]：长度为6的字典，
        # 包括[‘amsgrad’, ‘params’, ‘lr’, ‘betas’, ‘weight_decay’, ‘eps’]这6个参数
        # optimizer.param_groups[1]：好像是表示优化器的状态的一个字典
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
        x, x_cl = self.mask_simclr_.forward_cl(data.x, data.edge_index, data.edge_attr, data.batch, masks)
        x_aug, x_aug_cl = self.mask_simclr_.forward_cl(data_aug.x, data_aug.edge_index, data_aug.edge_attr,
                                                       data_aug.batch, masks)
        loss = mask_graphcl.simclr_loss(x_cl, x_aug_cl)

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
        x, x_cl = self.mask_simclr_.forward_cl(data.x, data.edge_index, data.edge_attr, data.batch, masks)
        x_aug, x_aug_cl = self.mask_simclr_.forward_cl(data_aug.x, data_aug.edge_index, data_aug.edge_attr,
                                                       data_aug.batch, masks)
        loss = mask_graphcl.simclr_loss(x_cl, x_aug_cl)

        grads_p = torch.autograd.grad(loss, self.auto_mask.parameters())

        for p, v in zip(self.mask_simclr_.parameters(), gradients):
            if v is not None:
                p.data.sub_(v, alpha=2 * R)

        masks = self.auto_mask()
        x, x_cl = self.mask_simclr_.forward_cl(data.x, data.edge_index, data.edge_attr, data.batch, masks)
        x_aug, x_aug_cl = self.mask_simclr_.forward_cl(data_aug.x, data_aug.edge_index, data_aug.edge_attr,
                                                       data_aug.batch, masks)
        loss = mask_graphcl.simclr_loss(x_cl, x_aug_cl)

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


def train(args, model, device, dataset, enable_meta, optimizer, mask_optim, mask_scheduler):
    dataset.aug = "none"
    dataset1 = dataset.shuffle()
    dataset2 = deepcopy(dataset1)
    dataset1.aug, dataset1.aug_ratio = args.aug1, args.aug_ratio1
    dataset2.aug, dataset2.aug_ratio = args.aug2, args.aug_ratio2

    loader1 = DataLoader(dataset1, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    model.train()

    train_acc_accum = 0
    train_loss_accum = 0

    for step, batch in enumerate(tqdm(zip(loader1, loader2), desc="Iteration")):
        batch1, batch2 = batch
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)

        optimizer.zero_grad()

        masks = torch.ones(300).to(device)
        if enable_meta:
            masks = model.auto_mask()

        x1, x1_cl = model.mask_simclr.forward_cl(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch, masks)
        x2, x2_cl = model.mask_simclr.forward_cl(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch, masks)
        loss_cl = model.mask_simclr.simclr_loss(x1_cl, x2_cl)
        # print("loss_cl: ", loss_cl)
        loss_bl = model.criterion(x1, x2)
        # print("loss_bl: ",loss_bl)
        loss = args.weight_bl * loss_bl + args.weight_cl * loss_cl
        if enable_meta:
            model.per_optimizer_step(optimizer, None, loss)
            model.unrolled_backward(batch1, batch2, optimizer, mask_optim, optimizer.param_groups[0]['lr'])
            model.per_optimizer_step(mask_optim)
        else:
            loss.backward()
            optimizer.step()

        if enable_meta:
            mask_scheduler.step()

        train_loss_accum += float(loss.detach().cpu().item())
        acc = torch.tensor(0)
        train_acc_accum += float(acc.detach().cpu().item())

    return train_acc_accum / (step + 1), train_loss_accum / (step + 1)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=3,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default='zinc_standard_agent',
                        help='root directory of dataset. For now, only classification.')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--model_file', type=str, default='', help='filename to output the pre-trained model')
    parser.add_argument('--seed', type=int, default=0, help="Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataset loading')
    parser.add_argument('--aug1', type=str, default='random')
    parser.add_argument('--aug_ratio1', type=float, default=0.2)
    parser.add_argument('--aug2', type=str, default='random')
    parser.add_argument('--aug_ratio2', type=float, default=0.2)

    parser.add_argument('--disable-meta', action="store_true",
                        help='whether using meta learning when training')
    parser.add_argument('--disable-sigmoid',
                        action="store_true",
                        help='whether using meta learning when training')
    parser.add_argument('--no-second-order', action="store_true")
    parser.add_argument('--weight-bl', default=1, type=float)
    parser.add_argument('--weight-cl', default=10, type=float,
                        help='weight of simclr loss')
    parser.add_argument('--criterion-type', type=str, default='CCA', choices=['BarlowTwins', 'CCA'])

    args = parser.parse_args()

    enable_meta = not args.disable_meta
    enable_sigmoid = not args.disable_sigmoid
    second_order = not args.no_second_order

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # set up dataset
    # dataset = MoleculeDataset_aug("dataset/" + args.dataset, dataset=args.dataset)
    dataset = MoleculeDataset_aug("/pub/data/hujie/zdata/data/CHEM/dataset/" + args.dataset, dataset=args.dataset)
    print(dataset)

    # set up model
    gnn = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type)

    # model = graphcl(gnn)
    model = MetaMaskBarlowTwins(gnn, device, enable_meta, enable_sigmoid, args.weight_cl,
                                second_order, args.criterion_type)
    model.to(device)

    # set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    if enable_meta:
        # mask_optim = torch.optim.Adam(model.auto_mask.parameters(), lr=0.001)
        mask_optim = torch.optim.SGD(model.auto_mask.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
        mask_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(mask_optim, 100)

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))

        # train(args, model, device, dataset, enable_meta, optimizer, mask_optim, mask_scheduler)
        train_acc, train_loss = train(args, model, device, dataset, enable_meta, optimizer, mask_optim, mask_scheduler)

        print(train_acc)
        print(train_loss)

        if epoch % 20 == 0:
            torch.save(model.state_dict(), "./models_mmgcl/mmgcl_" + str(epoch) + ".pth")


if __name__ == "__main__":
    main()
