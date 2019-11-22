from __future__ import print_function, division
import os
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim

class SongDataset(Dataset):
    def __init__(self, X_train, y_train, transform = None, type_ = 'mlp'):
        assert(type_ == 'mlp' or type_ == 'cnn')
        self.X = X_train
        self.y = y_train
        self.type_ = type_
        if transform:
            self.X = transform(self.X)
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        x = self.X[idx]
        if len(self.y) == 0:
            y = -1
        else:
            y = self.y[idx][0]
        if self.type_ == 'mlp':
            sample = {'x': torch.from_numpy(x).type(torch.float32),
                      'y': torch.tensor(np.array([float(y)]))}
        else:
            sample = {'x': torch.from_numpy(x).type(torch.float32).reshape(1, 90, 1),
                      'y': torch.tensor(np.array([float(y)]))}
        return sample


class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.fc_1 = nn.Linear(90, 128)
        self.norm1 = nn.BatchNorm1d(128)
        self.fc_2 = nn.Linear(128, 64)
        self.norm2 = nn.BatchNorm1d(64)
        self.fc_3 = nn.Linear(64, 32)
        self.norm3 = nn.BatchNorm1d(32)
        self.fc_4 = nn.Linear(32, 16)
        self.norm4 = nn.BatchNorm1d(16)
        self.fc_5 = nn.Linear(16, 1)
        
    def forward(self, x):
        x = self.norm1(self.fc_1(x))
        x = F.relu(x)
        x = self.norm2(self.fc_2(x))
        x = F.relu(x)
        x = self.norm3(self.fc_3(x))
        x = F.relu(x)
        x = self.norm4(self.fc_4(x))
        x = F.relu(x)
        x = self.fc_5(x)

        return x

class ResDense(nn.Module):    
    def __init__(self):
        super(ResDense, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 1)
        self.cnn = models.resnet18(pretrained=False)
        self.fc_1 = nn.Linear(1000, 256)
        self.norm1 = nn.BatchNorm1d(256)
        self.fc_2 = nn.Linear(256, 64)
        self.norm2 = nn.BatchNorm1d(64)
        self.fc_3 = nn.Linear(64, 16)
        self.norm3 = nn.BatchNorm1d(16)
        self.fc_4 = nn.Linear(16, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.cnn(x)
        x = self.norm1(self.fc_1(x))
        x = F.relu(x)
        x = self.norm2(self.fc_2(x))
        x = F.relu(x)
        x = self.norm3(self.fc_3(x))
        x = F.relu(x)
        x = self.fc_4(x)
        return x