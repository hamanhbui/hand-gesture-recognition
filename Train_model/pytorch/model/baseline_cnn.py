import numpy as np
import torch
import torch.nn as nn

class Baseline_CNN(nn.Module):

    def __init__(self):
        super(Baseline_CNN, self).__init__()
        self.relu=nn.ReLU()
        self.conv1=nn.Conv2d(in_channels=1, out_channels=64, kernel_size=[5,1],stride=1,padding=0)
        self.conv2=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[5,1],stride=1,padding=0)
        self.conv3=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[5,1],stride=1,padding=0)
        self.conv4=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[5,1],stride=1,padding=0)
        self.fc1=nn.Linear(in_features=22272, out_features=128)
        self.dropout=nn.Dropout(p=0.5)  
        self.fc2=nn.Linear(in_features=128, out_features=128)
        self.dropout=nn.Dropout(p=0.5) 
        self.fc3=nn.Linear(in_features=128, out_features=22)

    def forward(self, x):
        x=self.conv1(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.relu(x)
        x=self.conv3(x)
        x=self.relu(x)
        x=self.conv4(x)
        x=self.relu(x)
        x=x.view(-1,22272)
        x=self.fc1(x)
        x=self.relu(x)
        x=self.dropout(x)
        x=self.fc2(x)
        x=self.relu(x)
        x=self.dropout(x)
        x=self.fc3(x)
        return x