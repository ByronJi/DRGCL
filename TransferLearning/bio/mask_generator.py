import torch
import torch.nn as nn


class FeatureMask(nn.Module):
    def __init__(self, projection_out_dim, enable_sigmoid):
        super(FeatureMask, self).__init__()
        init_value = 1.0
        self.enable_sigmoid = enable_sigmoid
        if self.enable_sigmoid:
            init_value = 0.0
        self.mask = nn.Parameter(torch.ones(int(projection_out_dim)) * init_value)

    def forward(self):
        if self.enable_sigmoid:
            return torch.sigmoid(torch.ones_like(self.mask) * self.mask)
        return torch.ones_like(self.mask) * self.mask
