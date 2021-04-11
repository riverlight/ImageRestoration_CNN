# -*- coding: utf-8 -*-

import torch.optim as optim
from torch import nn
import torch as t
import torchvision
import torch.nn.functional as F
from torchsummary import summary


class NewIRNet10(nn.Module):
    def __init__(self):
        super(NewIRNet10, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=1, padding=5 // 2, groups=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 24, kernel_size=5, stride=1, padding=5 // 2, groups=1, bias=False)
        self.bn2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, kernel_size=5, stride=1, padding=5 // 2, groups=1, bias=True)
        self.bn3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 3, kernel_size=5, stride=1, padding=5 // 2, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.prelu = nn.PReLU()
        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out)
        out = self.bn3(self.conv3(out))
        out = self.leaky(out)
        out = self.tanh(self.conv4(out))
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
    net = NewIRNet10().to(device)
    net.train()
    optimizer = optim.Adam(params=net.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()

    outputs = net(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()

    print(outputs.shape)
    t.save(net, 'd:/nir10_test.pth')
    summary(net, input_size=(3, 960, 64), device=device)
    # updateBN(net, 0.0001)

if __name__=="__main__":
    test()

