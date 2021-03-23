# -*- coding: utf-8 -*-

from torch import nn
import torch as t
import torchvision
import torch.nn.functional as F
from torchsummary import summary


class BlockDW(nn.Module):
    def __init__(self, chl_in, chl_out, kernel):
        super(BlockDW, self).__init__()
        self.convDW = nn.Conv2d(chl_in, chl_in, kernel_size=kernel, stride=1, padding=kernel//2, groups=chl_in, bias=False)
        self.convPW = nn.Conv2d(chl_in, chl_out, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn = nn.BatchNorm2d(chl_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.convDW(x))
        out = self.relu(self.bn(self.convPW(out)))
        return out

class BlockDW2(nn.Module):
    def __init__(self, chl_in, chl_out, kernel):
        super(BlockDW2, self).__init__()
        self.convDW1 = nn.Conv2d(chl_in, chl_out, kernel_size=5, stride=1, padding=5//2, groups=1, bias=False)
        self.convPW1 = nn.Conv2d(chl_out, chl_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.convDW2 = nn.Conv2d(chl_out, chl_out, kernel_size=5, stride=1, padding=5 // 2, groups=chl_out, bias=False)
        self.convPW2 = nn.Conv2d(chl_out, chl_out, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn = nn.BatchNorm2d(chl_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.convDW1(x))
        out = self.relu(self.convPW1(out))
        out = self.relu(self.convDW2(out))
        out = self.relu(self.bn(self.convPW2(out)))
        return out


class BlockDN(nn.Module):
    def __init__(self, chl_in, chl_out, kernel, expansion):
        super(BlockDN, self).__init__()
        self.convPW1 = nn.Conv2d(chl_in, chl_in//expansion, kernel_size=1, stride=1)
        self.convDW = nn.Conv2d(chl_in//expansion, chl_in//expansion, kernel, stride=1, padding=kernel//2, groups=chl_in//expansion)
        self.convPW2 = nn.Conv2d(chl_in//expansion, chl_out, 1, stride=1, bias=True)
        self.bn = nn.BatchNorm2d(chl_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.convPW1(x))
        out = self.relu(self.convDW(out))
        out = self.relu(self.bn(self.convPW2(out)))
        return out

class BlockOut(nn.Module):
    def __init__(self, chl_in, chl_out, kernel, expansion):
        super(BlockOut, self).__init__()
        self.convPW1 = nn.Conv2d(chl_in, chl_in//expansion, kernel_size=1, stride=1)
        self.convDW = nn.Conv2d(chl_in//expansion, chl_in//expansion, kernel_size=kernel, stride=1, padding=kernel//2, groups=chl_in//expansion)
        self.convPW2 = nn.Conv2d(chl_in//expansion, chl_out, 1, 1)
        self.relu = nn.ReLU()
        self.relu2 = nn.Tanh()

    def forward(self, x):
        out = self.relu(self.convPW1(x))
        out = self.relu(self.convDW(out))
        out = self.relu2(self.convPW2(out))
        return out

class NewIRNet3(nn.Module):
    def __init__(self):
        super(NewIRNet3, self).__init__()
        self.chl_mid = 32
        self.convDW9x9 = BlockDW2(3, self.chl_mid, 9)
        self.dn2 = BlockDN(self.chl_mid, self.chl_mid, 3, 4)
        self.dn3 = BlockDN(self.chl_mid*2, self.chl_mid, 3, 4)
        self.blockOut4 = BlockOut(self.chl_mid*3, 3, 9, 2)

    def forward(self, x):
        dw_out = self.convDW9x9(x)
        dn2_out = self.dn2(dw_out)
        out = t.cat([dw_out, dn2_out], 1)
        dn3_out = self.dn3(out)
        out = t.cat([dn3_out, dn2_out, dw_out], 1)
        out = self.blockOut4(out)
        out = out + x
        return out


def test():
    device = 'cuda'
    inputs = t.rand(2, 3, 96, 64).to(device)
    net = NewIRNet3().to(device)
    outputs = net(inputs)
    print(outputs.shape)
    t.save(net, 'd:/nir2_test.pth')
    summary(net, input_size=(3, 96, 64), device=device)

if __name__=="__main__":
    test()

