from random import SystemRandom

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class ResidueBlock(nn.Module) : 
    def __init__(self, in_dim, out_dim) : 
        super(ResidueBlock, self).__init__()

        self.linear1 = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.linear2 = nn.Linear(in_features=out_dim, out_features=out_dim)
        self.shortcut = nn.Sequential()

    def forward(self, x) : 
        out = F.relu(self.linear1(x))
        out = F.relu(self.linear2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out

class CatanNetwork(nn.Module):
    def __init__(self, neurons) :
        super(CatanNetwork, self).__init__()

        self.layer1 = nn.Linear(1782, neurons)
        self.layer2 = ResidueBlock(neurons, neurons)
        self.layer3 = ResidueBlock(neurons, neurons)
        self.layer4 = ResidueBlock(neurons, neurons)
        self.layer5 = ResidueBlock(neurons, neurons)
        self.linear = nn.Linear(in_features=neurons, out_features=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.linear(out)
        return out