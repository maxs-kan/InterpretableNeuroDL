import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as torch_data
import torch.nn.functional as F

hidden = lambda c_in, c_out: nn.Sequential(
    nn.Conv3d(c_in, c_out, (3,3,3)),
    nn.BatchNorm3d(c_out),
    nn.ReLU(),
    nn.MaxPool3d(2)
)

class MriNet(nn.Module):
    def __init__(self, c):
        super(MriNet, self).__init__()
        self.hidden1 = hidden(1, c)
        self.hidden2 = hidden(c, 2*c)
        self.hidden3 = hidden(2*c, 4*c)
        self.linear = nn.Linear(128*5*7*5, 2)
        self.flatten = nn.Flatten()
        self.drop_layer = nn.Dropout(p=.5) #dropout for preventing overfitting

    def forward(self, x):
        x = self.hidden1(x)
        #x = self.drop_layer(x)
        x = self.hidden2(x)
        #x = self.drop_layer(x)
        x = self.hidden3(x)
        #x = self.drop_layer(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = F.log_softmax(x, dim=1)
        x = self.drop_layer(x)
        return x

class MriData(torch.utils.data.Dataset):
    def __init__(self, X, y):
        super(MriData, self).__init__()
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
