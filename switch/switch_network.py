import math
import sys
import numpy
import torch
import torch.nn as nn
from torch.nn import parameter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.append('..')
from train_params import *

METRICS_BATCH = 20 if 'CIFAR' in DATASET else 24
class Conv2Linear(nn.Module):
    def __init__(self):
        super(Conv2Linear, self).__init__()

    def forward(self, x):
        return x.view(-1)

class PredictionNet(nn.Module):
    def __init__(self):
        super(PredictionNet, self).__init__()
        self.fc = nn.Sequential(
            Conv2Linear(),
            nn.Linear(2 * METRICS_BATCH, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    
    def predict(self,x):
        out = self.forward(x)
        raw_pre = torch.argmax(out)
        if 'CIFAR' in DATASET:
            pre = '1' if raw_pre == 0 else '2' 
        else:
            pre = '1' if raw_pre <= 1 else '2' 
        return pre,raw_pre