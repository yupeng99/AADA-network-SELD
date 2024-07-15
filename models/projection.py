import torch
import torch.nn as nn
from config import OPS_NAMES
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x

class Projection(nn.Module):
    def __init__(self, in_features, n_layers, n_hidden=128):
        super(Projection, self).__init__()
        self.n_layers = n_layers
        # self.convblock = nn.Sequential(ConvBlock(7,64),
        #                                nn.MaxPool2d((1,4)),
        #                                nn.Dropout2d(0.05),
        #                                ConvBlock(64,64),
        #                                nn.MaxPool2d((5, 4)),
        #                                nn.Dropout2d(0.05),)
        #
        # in_features = 64*100*4
        # self.gru = nn.GRU(input_size=in_features, hidden_size=128,
        #                         num_layers=2, batch_first=True,
        #                         dropout=0.05, bidirectional=True)
        if self.n_layers > 0:
            layers = [nn.Linear(in_features, n_hidden), nn.ReLU()]
            for _ in range(self.n_layers-1):
                layers.append(nn.Linear(n_hidden, n_hidden))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(n_hidden, 2*len(OPS_NAMES)))
        else:
            layers = [nn.Linear(in_features, 2*len(OPS_NAMES))]
        self.projection = nn.Sequential(*layers)

    def forward(self, x):


        x = x.reshape(x.size(0),-1)
        # (x,_) = self.gru(x)
        x = self.projection(x)  # (32,20)
        # x = x.mean(dim=(2))   #这样些可能有问题
        # print(f"x.shape:{x.shape}")
        return x
    

