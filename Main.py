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

a = torch.ones((1,1,5))
print(a.shape)
print(torch.repeat_interleave(a, 5, dim=0).shape)
