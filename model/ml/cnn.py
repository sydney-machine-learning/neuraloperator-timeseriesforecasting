from __future__ import absolute_import

from typing import Tuple
import torch
import torch.nn as nn


class ConvNet(nn.Module):

    def __init__(self, n_features: int, n_channels: int=1,
                 out_dim: int=1, n_filters: Tuple[int, int, int]=(8, 8, 8), 
                 dropout_p: float=0.2) -> None:
        
        # Initialise module class
        super(ConvNet, self).__init__()

        # Define Layers
        self.conv_1 = nn.Conv2d(in_channels=n_channels,
                                out_channels=1,
                                kernel_size=(2, 1),
                                bias=True)
        
        self.conv_2 = nn.Conv2d(in_channels=1,
                                 out_channels=1,
                                 kernel_size=(3, 3),
                                 bias=True)

        self.max_pool = nn.MaxPool2d(kernel_size=(2, 1))

        self.conv_3 = nn.Conv2d(in_channels=1,
                                 out_channels=1,
                                 kernel_size=(2, 2),
                                 bias=True)
        
        self.flatten = nn.Flatten()

        self.dropout = nn.Dropout(p=dropout_p)

        self.linear = nn.Linear(in_features=2,
                                out_features=out_dim)
        
    
    def forward(self, x):
        out = self.conv_1(x).relu()
        # print("1:", out.shape)
        out = self.conv_2(out).relu()
        # print("2:", out.shape)
        out = self.max_pool(out)
        # print("3:", out.shape)
        out = self.conv_3(out).relu()
        # print("4:", out.shape)
        out = self.dropout(self.flatten(out))
        # print("5:", out.shape)
        out = self.linear(out)
        # print("6:", out.shape)
        return out


