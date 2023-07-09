import copy
import os.path as osp
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
# from core.encoders import *
import datetime

# from torch_geometric.datasets import TUDataset
from aug import TUDataset_aug as TUDataset
from torch_geometric.loader import DataLoader
import sys
import json
from torch import optim

from gin import Encoder
from evaluate_embedding import evaluate_embedding

from arguments import arg_parse


class MaskSimclr(nn.Module):
    def __init__(self, dataset_num_features, hidden_dim, num_gc_layers, device):
        super(MaskSimclr, self).__init__()

        self.embedding_dim = hidden_dim * num_gc_layers
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers, device)

        # self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim),
        #                                nn.ReLU(inplace=True),
        #                                nn.Linear(self.embedding_dim, self.embedding_dim))
        # self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, 512, bias=False),
        #                                nn.ReLU(inplace=True),
        #                                nn.BatchNorm1d(512),
        #                                nn.Linear(512, 512, bias=False))
        # self.proj_head2 = nn.Sequential(nn.Linear(self.embedding_dim, 512, bias=False),
        #                                nn.ReLU(inplace=True),
        #                                nn.BatchNorm1d(512),
        #                                nn.Linear(512, 512, bias=False))
        # # projector
        sizes = [self.embedding_dim] + list(map(int, '512-512-512'.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        layers2 = copy.deepcopy(layers)
        self.proj_head = nn.Sequential(*layers)
        self.proj_head2 = nn.Sequential(*layers2)
        self.init_emb()
        self.device = device

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch, masks=None):

        # batch_size = data.num_graphs
        if x is None:
            x = torch.ones(batch.shape[0]).to(self.device)

        y, M = self.encoder(x, edge_index, batch)
        if masks is not None:
            y = y * masks[None, :]
        y_bl = self.proj_head2(y)
        y_cl = self.proj_head(y)

        return y_bl, y_cl

    @staticmethod
    def simclr_loss(x, x_aug):

        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        return loss


