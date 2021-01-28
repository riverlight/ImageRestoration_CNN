# -*- coding: utf-8 -*-

from torch import nn
import torch as t


class IRCNN(nn.Module):
    def __init__(self, num_channels=3):
        super(IRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2, bias=True)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2, bias=True)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


if __name__ == "__main__":
    ir = IRCNN(num_channels=3).to('cuda')
    for name, params in ir.named_parameters():
        print(name, params.shape)

    inputs = t.rand(1, 3, 32, 32).to('cuda')
    print(inputs.shape)
    outputs = ir(inputs)
    # print(outputs)
