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
    def __init__(self, hidden_vec_size, patch_size=16, in_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_vec_size
        self.in_projection = in_channels * (patch_size ** 2)
        self.projection = nn.Linear(in_features=self.in_projection, out_features=self.hidden_size)

    def forward(self, X):
        B, C, H, W = X.shape
        n_w = W // self.patch_size
        n_h = H // self.patch_size
        X = X.reshape((B, n_w * n_h, self.in_projection))
        projected_X = self.projection(X)
        return projected_X


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
