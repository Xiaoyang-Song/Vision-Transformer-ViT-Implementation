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
import pickle
import math
import os
import sys

from tqdm import tqdm

from Utility import *

from device import default_device


def main():
    check_version()
    check_device()

    width = 32
    height = 32
    batch_size = 256 if torch.cuda.is_available() else 1
    patch_size = 4

    # Get training / testing data and data loader
    train_data, test_data = get_dataset(width=width, height=height, dataset="CIFAR-10")
    train_dataLoader = load_data(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_dataLoader = load_data(dataset=test_data, batch_size=batch_size, shuffle=False)

    # Declare Model
    model = ViT(hidden_size=128, H=height, W=width, num_msa_heads=4,
                patch_size=patch_size, mlp_expansion=2, num_encoders=6,
                in_channels=3, mlp_p_out=0.1, num_classes=10, device=default_device)

    # Declare loss function & Optimizer
    learning_rate = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    # Declare training & testing information
    num_epochs = 100
    acc_epochs = []

    for epoch in range(num_epochs):
        model.train()
        print(f"Epoch: {epoch}")
        print('train')
        for index, (images, labels) in enumerate(tqdm(train_dataLoader)):
            images, labels = images.to(default_device), labels.to(default_device)
            outputs = model(images).to(default_device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # scheduler.step()

        model.eval()
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            print('test')
            for _, (images, labels) in enumerate(tqdm(test_dataLoader)):
                images, labels = images.to(default_device), labels.to(default_device)
                outputs = model(images).to(default_device)
                _, predicted = torch.max(outputs, dim=1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()
            acc = 100.0 * n_correct / n_samples
            acc_epochs.append(acc)
            print(f'-- ViT Accuracy: {acc}%')


if __name__ == "__main__":
    main()
