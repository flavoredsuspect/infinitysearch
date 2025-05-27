# Models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.BatchNorm1d(dim),
        )

    def forward(self, x):
        return x + self.net(x)


class EmbNet(nn.Module):
    def __init__(self, input_dim=784, output_dim=300):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            ResidualBlock(256),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim),
        )
        self.alpha = nn.Parameter(torch.zeros(1))
        self.max_scale = 2

    def forward(self, x):
        emb = F.normalize(self.net(x), p=2, dim=-1)
        scale = 1.0 + (self.max_scale - 1.0) * torch.sigmoid(self.alpha)
        return emb * scale
