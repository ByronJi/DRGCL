import numpy as np
import torch
import torch.nn as nn


class BarlowTwinsLoss(torch.nn.Module):

    def __init__(self, device, lambda_param=5e-3):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param
        self.device = device

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor):
        # normalize repr. along the batch dimension
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0)  # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0)  # NxD

        N = z_a.size(0)
        D = z_a.size(1)

        # cross-correlation matrix
        c = torch.mm(z_a_norm.T, z_b_norm) / N  # DxD
        # loss
        c_diff = (c - torch.eye(D, device=self.device)).pow(2)  # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_param
        loss = c_diff.sum()

        return loss


class CCALoss(torch.nn.Module):

    def __init__(self, device, lambda_param=1e-3):
        super(CCALoss, self).__init__()
        self.lambda_param = lambda_param
        self.device = device

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        z1_norm = (z1 - z1.mean(0)) / z1.std(0)  # NxD
        z2_norm = (z2 - z2.mean(0)) / z2.std(0)  # NxD
        D = z1.shape[1]
        # c = torch.mm(z1_norm.T, z2_norm) / D
        c1 = torch.mm(z1_norm.T, z1_norm) / D
        c2 = torch.mm(z2_norm.T, z2_norm) / D

        loss_inv = (z1_norm - z2_norm).pow(2).sum() / D
        iden = torch.tensor(np.eye(D)).to(self.device)
        loss_dec1 = (iden - c1).pow(2).sum()
        loss_dec2 = (iden - c2).pow(2).sum()
        loss = loss_inv + self.lambda_param * (loss_dec1 + loss_dec2)

        return loss
