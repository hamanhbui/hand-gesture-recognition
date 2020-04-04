import torch
import torch.nn as nn
from radam import RAdam
class CNN_LSTM(nn.Module):

    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.relu=nn.ReLU()
        self.conv1=nn.Conv2d(in_channels=1, out_channels=64, kernel_size=[5,1],stride=1,padding=0)
        self.conv2=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[5,1],stride=1,padding=0)
        self.conv3=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[5,1],stride=1,padding=0)
        self.conv4=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[5,1],stride=1,padding=0)
        self.rnn = nn.LSTM(input_size=384, hidden_size=128, num_layers=2,dropout=0.5)
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
        x=x.permute(2,0,1,3).contiguous()
        x = x.view(58,-1,384)
        # h0 = torch.randn(2*1, 100, 128)
        # c0 = torch.randn(2*1, 100, 128)
        # x,_ = self.rnn(x, (h0, c0))
        x,_ = self.rnn(x)
        # x=x.contiguous()
        # x = x.view(-1,128)
        # x=self.fc3(x)
        # x = x.view(58,-1,22)
        # x=x[57]

        x=x[-1]
        x=self.fc3(x)
        return x