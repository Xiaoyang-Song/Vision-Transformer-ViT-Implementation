import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
from torchvision.datasets import CIFAR10
import torch.nn as nn
import einops
from einops import rearrange
import scipy
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
import pickle
import math
import os


class Encoder(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass


class PatchEmbedding(nn.Module):
    def __init__(self, hidden_vec_size, H, W, patch_size=16, in_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_vec_size
        self.in_proj = in_channels * (patch_size ** 2)
        self.lin_proj = nn.Linear(in_features=self.in_proj, out_features=self.hidden_size)
        self.n_w = W // self.patch_size
        self.n_h = H // self.patch_size
        # Note that all embedded sequences share the same positional embedding
        # TODO: Change the initialization of these parameters later
        self.pos_emb = nn.Parameter(torch.randn((self.n_w * self.n_h + 1, self.hidden_size)))
        self.cls = nn.Parameter(torch.randn(1, 1, self.hidden_size))

    def initialize(self):
        pass

    def forward(self, X):
        B, C, H, W = X.shape
        X = X.reshape((B, self.n_w * self.n_h, self.in_proj))
        # projected_X: (B, n_w * n_h, hidden_size)
        projected_X = self.lin_proj(X)
        # batch_cls_token: (B, 1, hidden_size)
        batch_cls_token = torch.repeat_interleave(self.cls, B, dim=0)
        # patch_embedded_X: (B, self.n_w * self.n_h + 1, hidden_size)
        patch_embedded_X = torch.cat([batch_cls_token, projected_X], dim=1)
        # take advantages of broadcasting
        pos_embedded_X = patch_embedded_X + self.pos_emb
        return pos_embedded_X


class MultiHeadAttention(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass


class Residual(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass
