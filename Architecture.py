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
        # TODO: in ViT paper implementation, a CNN architecture is used for performance gain
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


class Attention(nn.Module):
    def __init__(self, D, D_h):
        super().__init__()
        self.hidden_size = D
        self.D_h = D_h
        self.lin_Q = nn.Linear(in_features=D, out_features=D_h)
        self.lin_K = nn.Linear(in_features=D, out_features=D_h)
        self.lin_V = nn.Linear(in_features=D, out_features=D_h)

    def forward(self, X):
        # Q, K, V: (B, self.n_w * self.n_h + 1, D_h)
        Q, K, V = self.lin_Q(X), self.lin_K(X), self.lin_V(X)
        # Attention: (self.n_w * self.n_h + 1, self.n_w * self.n_h + 1)
        A = torch.softmax(torch.bmm(Q, torch.transpose(K, 1, 2)) / math.sqrt(self.D_h), dim=-1)
        return torch.bmm(A, V)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, D):
        super().__init__()
        self.SA_Blocks = []
        self.SA_Outputs = torch.empty(0)
        self.D = D
        self.D_h = D // num_heads
        self.num_heads = num_heads
        self.lin_proj = nn.Linear(in_features=self.D, out_features=self.D)
        self.initialize()

    def initialize(self):
        for i in range(self.num_heads):
            self.SA_Blocks.append(Attention(self.D, self.D_h))

    def forward(self, X):
        B, _, _ = X.shape
        for i in range(self.num_heads):
            self.SA_Outputs = torch.cat([self.SA_Outputs, self.SA_Blocks[i](X)], dim=-1)
        msa_cat = self.SA_Outputs.reshape((B, -1, self.D))
        out = self.lin_proj(msa_cat)
        return out


class Residual(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass
