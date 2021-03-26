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
        self.convDW1 = nn.Conv2d(cfg[0][1], cfg[0][0], kernel_size=5, stride=1, padding=5//2, groups=1, bias=False)
        self.convDW2 = nn.Conv2d(cfg[0][0], cfg[1][0], kernel_size=5, stride=1, padding=5 // 2, groups=cfg[0][0], bias=False)
        self.convPW2 = nn.Conv2d(cfg[2][1], cfg[2][0], kernel_size=1, stride=1, padding=0, bias=True)
        self.bn = nn.BatchNorm2d(cfg[3])
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.convDW1(x))
        out = self.relu(self.convDW2(out))
        out = self.relu(self.bn(self.convPW2(out)))
        return out


class BlockDN(nn.Module):
    def __init__(self, chl_in, chl_out, kernel, expansion, cfg):
        super(BlockDN, self).__init__()
        self.convPW1 = nn.Conv2d(cfg[0][1], cfg[0][0], kernel_size=1, stride=1)
        self.convDW = nn.Conv2d(cfg[0][0], cfg[1][0], kernel, stride=1, padding=kernel//2, groups=cfg[0][0])
        self.convPW2 = nn.Conv2d(cfg[2][1], cfg[2][0], 1, stride=1, bias=True)
        self.bn2 = nn.BatchNorm2d(cfg[3])
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.convPW1(x))
        out = self.relu(self.convDW(out))
        out = self.relu(self.bn2(self.convPW2(out)))
        return out

class BlockOut(nn.Module):
    def __init__(self, chl_in, chl_out, kernel, expansion, cfg):
        super(BlockOut, self).__init__()
        self.convPW1 = nn.Conv2d(cfg[0][1], cfg[0][0], kernel_size=1, stride=1)
        self.convDW = nn.Conv2d(cfg[0][0], cfg[1][0], kernel_size=kernel, stride=1, padding=kernel//2, groups=cfg[0][0])
        self.convPW2 = nn.Conv2d(cfg[2][1], cfg[2][0], 1, 1, padding=0)
        self.relu = nn.ReLU()
        self.relu2 = nn.Tanh()

    def forward(self, x):
        out = self.relu(self.convPW1(x))
        out = self.relu(self.convDW(out))
        out = self.relu2(self.convPW2(out))
        return out

class NewIRNet6(nn.Module):
    def __init__(self, cfg=None):
        super(NewIRNet6, self).__init__()
        self.chl_mid = 32
        self.lst_bn_layer_id = [3, 7, 11]
        self.lst_bn_next_layer_id = [4, 8, 12]
        self.lst_bn_next_cat = [[3], [3, 7], [3, 7, 11]]
        self.cfg = cfg
        if self.cfg is None:
            self.cfg = [(32, 3), (32, 1), (32, 32), 32,
                        (8, 32), (8, 1), (32, 8), 32,
                        (16, 64), (16, 1), (32, 16), 32,
                        (48, 96), (48, 1), (3, 48)]

        self.convDW9x9 = BlockDW2(3, self.chl_mid, 9, self.cfg)
        self.dn2 = BlockDN(self.chl_mid, self.chl_mid, 3, 4, self.cfg[4:])
        self.dn3 = BlockDN(self.chl_mid*2, self.chl_mid, 3, 4, self.cfg[8:])
        self.blockOut4 = BlockOut(self.chl_mid*3, 3, 9, 2, self.cfg[12:])

    def forward(self, x):
        dw_out = self.convDW9x9(x)
        dn2_out = self.dn2(dw_out)
        out = t.cat([dw_out, dn2_out], 1)
        dn3_out = self.dn3(out)
        out = t.cat([dw_out, dn2_out, dn3_out], 1)
        out = self.blockOut4(out)
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


def test():
    device = 'cuda'
    inputs = t.rand(2, 3, 96, 64).to(device)
    targets = t.rand(2, 3, 96,64).to(device)
    net = NewIRNet6().to(device)
    net.train()
    optimizer = optim.Adam(params=net.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()

    outputs = net(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()

    print(outputs.shape)
    t.save(net, 'd:/nir6_test.pth')
    # summary(net, input_size=(3, 96, 64), device=device)
    updateBN(net, 0.0001)

if __name__=="__main__":
    test()

