# -*- coding: utf-8 -*-

import torch.optim as optim
from torch import nn
import torch as t
import torchvision
import torch.nn.functional as F
from torchsummary import summary


class BlockDW2(nn.Module):
    def __init__(self, chl_in, chl_out, kernel, cfg):
        super(BlockDW2, self).__init__()
        self.convDW1 = nn.Conv2d(cfg[0][0], cfg[0][1], kernel_size=5, stride=1, padding=5//2, groups=1, bias=False)
        self.convDW2 = nn.Conv2d(cfg[1][0], cfg[1][1], kernel_size=5, stride=1, padding=5 // 2, groups=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[1][1])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.convDW1(x))
        out = self.relu(self.bn2(self.convDW2(out)))
        return out

class BlockOut(nn.Module):
    def __init__(self, chl_in, chl_out, kernel, expansion, cfg):
        super(BlockOut, self).__init__()
        self.convDW = nn.Conv2d(cfg[0][0], cfg[0][1], kernel_size=kernel, stride=1, padding=kernel // 2, groups=1)
        self.convPW2 = nn.Conv2d(cfg[1][0], cfg[1][1], kernel_size=5, stride=1, padding=5//2)
        self.relu = nn.ReLU(inplace=True)
        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.leaky(self.convDW(x))
        out = self.tanh(self.convPW2(out))
        return out

class NewIRNet9(nn.Module):
    def __init__(self, cfg=None):
        super(NewIRNet9, self).__init__()
        self.chl_mid = 32
        self.lst_bn_layer_id = [2]
        self.lst_bn_next_layer_id = [3]
        self.lst_bn_next_cat = [[2]]
        self.cfg = cfg
        if self.cfg is None:
            self.cfg = [(3, self.chl_mid), (self.chl_mid, self.chl_mid),
                        (self.chl_mid, self.chl_mid), (self.chl_mid, 3)]

        self.convDW9x9 = BlockDW2(3, self.chl_mid, 9, self.cfg)
        self.blockOut4 = BlockOut(self.chl_mid, 3, 5, 2, self.cfg[2:])

    def forward(self, x):
        dw_out = self.convDW9x9(x)
        out = self.blockOut4(dw_out)
        out = out + x
        return out

#***********************稀疏训练（对BN层γ进行约束）**************************
def updateBN(net, s):
    for m in net.modules():
        #  isinstance() 函数来判断一个对象是否是一个已知的类型
        # print(m)
        if isinstance(m, nn.BatchNorm2d):
            #  hasattr() 函数用于判断对象是否包含对应的属性
            if hasattr(m.weight, 'data'):
                m.weight.grad.data.add_(s*t.sign(m.weight.data)) #L1正则
                m.bias.grad.data.add_(s*t.sign(m.bias.data))


def test():
    device = 'cuda'
    inputs = t.rand(2, 3, 96, 64).to(device)
    targets = t.rand(2, 3, 96,64).to(device)
    net = NewIRNet9().to(device)
    net.train()
    optimizer = optim.Adam(params=net.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()

    outputs = net(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()

    print(outputs.shape)
    t.save(net, 'd:/nir9_test.pth')
    summary(net, input_size=(3, 960, 64), device=device)
    # updateBN(net, 0.0001)

if __name__=="__main__":
    test()

