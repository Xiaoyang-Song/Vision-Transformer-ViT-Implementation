import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
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

width = 32
height = 32
batch_size = 32
patch_size = 4
# Get training / testing data and data loader
train_data, test_data = get_cifar_10_dataset(width=width, height=height)
train_dataLoader = load_data(dataset=train_data, batch_size=batch_size, shuffle=True)
test_dataLoader = load_data(dataset=test_data, batch_size=batch_size, shuffle=False)

# Declare Model
model = ViT(hidden_size=32, H=height, W=width, num_msa_heads=4,
            patch_size=4, mlp_expansion=2, num_encoders=6,
            in_channels=3, mlp_p_out=0.5, num_classes=10)

learning_rate = 0.05
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
criterion = nn.CrossEntropyLoss()

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    print(f"Epoch: {epoch}")
    for index, (images, labels) in enumerate(train_dataLoader):
        images, labels = images.to(device), labels.to(device)
        # print(images.shape)
        predicted = model(images)
        loss = criterion(predicted, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    scheduler.step()

    model.eval()
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for _, (images, labels) in enumerate(test_dataLoader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
        acc = 100.0 * n_correct / n_samples
        print(f'-- ViT Accuracy: {acc}%')

# a = torch.ones((4,3,5))
# print(a.shape)
# print(a[:,0:1,:].shape)
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
