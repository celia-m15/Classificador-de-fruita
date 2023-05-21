# -*- coding: utf-8 -*-
"""
Created on Mon May  8 22:20:12 2023

@author: alexc
"""

import torch
from torch.utils import data
import torch.nn.functional as F
import torch.nn as nn

# create a custom dataset from the preprocessed images and labels
class ImageDataset(data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        # convert the image and label to PyTorch tensors
        image = torch.from_numpy(image.transpose((2, 0, 1))).float()
        label = torch.tensor(label).long()
        return image, label

    def __len__(self):
        return len(self.images)

    
class CNNModel(nn.Module):
    def __init__(self, num_labels, image_size):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * (image_size // 8)  * (image_size // 8) , 256)
        self.fc2 = nn.Linear(256, num_labels)

    def forward(self, x, image_size):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 64 * (image_size // 8) * (image_size // 8))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    