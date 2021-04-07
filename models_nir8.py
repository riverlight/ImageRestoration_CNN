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
        self.convDW2 = nn.Conv2d(cfg[0][1], cfg[1][1], kernel_size=5, stride=1, padding=5 // 2, groups=cfg[0][1], bias=False)
        self.convPW2 = nn.Conv2d(cfg[2][0], cfg[2][1], kernel_size=1, stride=1, padding=0, bias=True)
        self.bn2 = nn.BatchNorm2d(cfg[3])
        self.prelu = nn.PReLU()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.prelu(self.convDW1(x))
        # out = self.relu(self.convDW1a(out))
        out = self.prelu(self.convDW2(out))
        out = self.prelu(self.bn2(self.convPW2(out)))
        return out

class BlockBottleNet(nn.Module):
    def __init__(self, chl_in, chl_out, kernel, expansion, cfg):
        super(BlockBottleNet, self).__init__()
        self.convPW1 = nn.Conv2d(cfg[0][0], cfg[0][1], kernel_size=1, stride=1)
        self.convDW = nn.Conv2d(cfg[0][1], cfg[1][1], kernel, stride=1, padding=kernel // 2, groups=cfg[0][1])
        self.convPW2 = nn.Conv2d(cfg[2][0], cfg[2][1], 1, stride=1, bias=True)
        self.bn2 = nn.BatchNorm2d(cfg[3])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.convPW1(x))
        out = self.relu(self.convDW(out))
        out = self.bn2(self.convPW2(out))
        # out += x
        out = self.relu(out)
        return out


class BlockOut(nn.Module):
    def __init__(self, chl_in, chl_out, kernel, expansion, cfg):
        super(BlockOut, self).__init__()
        self.convPW1 = nn.Conv2d(cfg[0][0], cfg[0][1], kernel_size=1, stride=1, bias=True)
        self.bn1 = nn.BatchNorm2d(cfg[1])
        self.convDW = nn.Conv2d(cfg[1], cfg[2][1], kernel_size=kernel, stride=1, padding=kernel//2, groups=cfg[1])
        # self.bn2 = nn.BatchNorm2d(cfg[2][1])
        self.convPW2 = nn.Conv2d(cfg[3][0], cfg[3][1], 1, 1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.relu2 = nn.Tanh()

    def forward(self, x):
        out = self.relu(self.bn1(self.convPW1(x)))
        out = self.relu(self.convDW(out))
        out = self.relu2(self.convPW2(out))
        return out

class NewIRNet8(nn.Module):
    def __init__(self, cfg=None):
        super(NewIRNet8, self).__init__()
        self.chl_mid = 64
        self.lst_bn_layer_id = [3, 7, 11, 13]
        self.lst_bn_next_layer_id = [4, 8, 12, 14]
        self.lst_bn_next_cat = [[3], [7], [11], [13]]
        self.cfg = cfg
        if self.cfg is None:
            self.cfg = [(3, 64), (1, 64), (64, 64), 64,
                        (64, 16), (1, 16), (16, 64), 64,
                        (64, 16), (1, 16), (16, 64), 64,
                        (64, 32), 32, (1, 32), (32, 3)]

        self.convDW9x9 = BlockDW2(3, self.chl_mid, 9, self.cfg)
        self.boN2 = BlockBottleNet(self.chl_mid, self.chl_mid, 3, 4, self.cfg[4:])
        self.boN3 = BlockBottleNet(self.chl_mid, self.chl_mid, 3, 4, self.cfg[8:])
        self.blockOut4 = BlockOut(self.chl_mid, 3, 9, 2, self.cfg[12:])

    def forward(self, x):
        dw_out = self.convDW9x9(x)
        bo2_out = self.boN2(dw_out)
        bo3_out = self.boN3(bo2_out)
        out = self.blockOut4(bo3_out)
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
    net = NewIRNet8().to(device)
    net.train()
    optimizer = optim.Adam(params=net.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()

    outputs = net(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()

    print(outputs.shape)
    t.save(net, 'd:/nir8_test.pth')
    summary(net, input_size=(3, 960, 64), device=device)
    # updateBN(net, 0.0001)

if __name__=="__main__":
    test()

