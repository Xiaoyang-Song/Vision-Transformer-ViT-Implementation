import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
from torchvision.datasets import CIFAR10
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
import pickle
import math
import os
from Utility import *

check_version()
check_device()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# a = torch.ones((1,1,5))
# print(a.shape)
# print(torch.repeat_interleave(a, 5, dim=0).shape)
# a = torch.ones((2,1)).T
# b = torch.ones((1,2)).T
# print((a@b).shape)
# a, b, c, _  = 1,2,3,4
# print(a)
# print(b)
# print(c)
# a = []
# a.append(torch.Tensor([1,2]))
# a.append([1,3])
# a.append([1,3])
# a = torch.Tensor(a)
# a = a.reshape(-1, 6)
# print(a.shape)
# a = torch.empty(0)
# a = torch.cat([a, torch.ones((1,2,3))],dim=0)
# a = torch.cat([a, torch.ones((1,2,3))],dim=0)
# print(a)
# print(a.shape)