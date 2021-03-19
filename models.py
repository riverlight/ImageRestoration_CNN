# -*- coding: utf-8 -*-

from torch import nn
import torch as t
import torchvision
import torch.nn.functional as F


class IRCNN(nn.Module):
    def __init__(self, num_channels=3):
        super(IRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2, bias=True)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2, bias=True)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2, bias=True)
        self.relu = nn.PReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class ConvolutionalBlock(nn.Module):
    """
    卷积模块,由卷积层, BN归一化层, 激活层构成.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=False, activation=None):
        """
        :参数 in_channels: 输入通道数
        :参数 out_channels: 输出通道数
        :参数 kernel_size: 核大小
        :参数 stride: 步长
        :参数 batch_norm: 是否包含BN层
        :参数 activation: 激活层类型; 如果没有则为None
        """
        super(ConvolutionalBlock, self).__init__()

        if activation is not None:
            activation = activation.lower()
            assert activation in {'prelu', 'leakyrelu', 'tanh'}

        # 层列表
        layers = list()

        # 1个卷积层
        layers.append(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2))

        # 1个BN归一化层
        if batch_norm is True:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        # 1个激活层
        if activation == 'prelu':
            layers.append(nn.PReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.2))
        elif activation == 'tanh':
            layers.append(nn.Tanh())

        # 合并层
        self.conv_block = nn.Sequential(*layers)

    def forward(self, input):
        """
        前向传播

        :参数 input: 输入图像集，张量表示，大小为 (N, in_channels, w, h)
        :返回: 输出图像集，张量表示，大小为(N, out_channels, w, h)
        """
        output = self.conv_block(input)

        return output


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        x = self.activation(x)
        return x


class Conv9x9(nn.Module):
    def __init__(self):
        super(Conv9x9, self).__init__()
        self.conv1 = ConvolutionalBlock(3, 16, 5, batch_norm=False, activation="leakyrelu")
        self.dwConv2 = SeparableConv2d(16, 32, 3, 1, 1, bias=False)
        self.dwConv3 = SeparableConv2d(32, 64, 3, 1, 1, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dwConv2(x)
        x = self.dwConv3(x)
        return x

class Bottleneck(nn.Module):
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, skip=False):
        super(Bottleneck, self).__init__()
        self.skip = False
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.skip is False:
            out += self.shortcut(x)
        out = F.relu(out)
        return out


class NewIRNet(nn.Module):
    def __init__(self):
        super(NewIRNet, self).__init__()
        self.conv1 = Conv9x9()
        self.bn2 = Bottleneck(64, 16)
        self.bn3 = Bottleneck(64, 16)
        self.bn4 = Bottleneck(64, 16, skip=True)
        # 最后一个卷积模块
        self.conv5 = ConvolutionalBlock(in_channels=64, out_channels=3, kernel_size=3,
                                              batch_norm=False, activation='Tanh')

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn2(out)
        out = self.bn3(out)
        out = self.bn4(out)
        out = self.conv5(out)
        # out = out + x
        return out


class ResidualBlock(nn.Module):
    """
    残差模块, 包含两个卷积模块和一个跳连.
    """

    def __init__(self, kernel_size=3, n_channels=64):
        """
        :参数 kernel_size: 核大小
        :参数 n_channels: 输入和输出通道数（由于是ResNet网络，需要做跳连，因此输入和输出通道数是一致的）
        """
        super(ResidualBlock, self).__init__()

        # 第一个卷积块
        self.conv_block1 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation='PReLu')

        # 第二个卷积块
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation=None)

    def forward(self, input):
        """
        前向传播.

        :参数 input: 输入图像集，张量表示，大小为 (N, n_channels, w, h)
        :返回: 输出图像集，张量表示，大小为 (N, n_channels, w, h)
        """
        residual = input  # (N, n_channels, w, h)
        output = self.conv_block1(input)  # (N, n_channels, w, h)
        output = self.conv_block2(output)  # (N, n_channels, w, h)
        output = output + residual  # (N, n_channels, w, h)

        return output


class IRTestNet(nn.Module):
    # test net
    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16):
        super(IRTestNet, self).__init__()
        # 第一个卷积块
        kernel_size = 3
        stride = 1
        self.conv_block1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2)
        # 一系列残差模块, 每个残差模块包含一个跳连接
        # self.residual_blocks = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=3,
        #                                       batch_norm=True, activation='PReLu')

    def forward(self, lr_imgs):
        output = self.conv_block1(lr_imgs)  # (16, 3, 24, 24)
        residual = output  # (16, 64, 24, 24)
        # output = self.residual_blocks(output)  # (16, 64, 24, 24)
        # output = output + residual
        return output



class IRResNet(nn.Module):
    """
    SRResNet模型
    """
    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16):
        """
        :参数 large_kernel_size: 第一层卷积和最后一层卷积核大小
        :参数 small_kernel_size: 中间层卷积核大小
        :参数 n_channels: 中间层通道数
        :参数 n_blocks: 残差模块数
        :参数 scaling_factor: 放大比例
        """
        super(IRResNet, self).__init__()

        # 第一个卷积块
        self.conv_block1 = ConvolutionalBlock(in_channels=3, out_channels=n_channels, kernel_size=large_kernel_size,
                                              batch_norm=False, activation='PReLu')

        # 一系列残差模块, 每个残差模块包含一个跳连接
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(kernel_size=small_kernel_size, n_channels=n_channels) for i in range(n_blocks)])

        # 第二个卷积块
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels,
                                              kernel_size=small_kernel_size,
                                              batch_norm=True, activation=None)


        # 最后一个卷积模块
        self.conv_block3 = ConvolutionalBlock(in_channels=n_channels, out_channels=3, kernel_size=large_kernel_size,
                                              batch_norm=False, activation='Tanh')

    def forward(self, lr_imgs):
        """
        前向传播.

        :参数 lr_imgs: 低分辨率输入图像集, 张量表示，大小为 (N, 3, w, h)
        :返回: 高分辨率输出图像集, 张量表示， 大小为 (N, 3, w * scaling factor, h * scaling factor)
        """
        output = self.conv_block1(lr_imgs)  # (16, 3, 24, 24)
        residual = output  # (16, 64, 24, 24)
        output = self.residual_blocks(output)  # (16, 64, 24, 24)
        output = self.conv_block2(output)  # (16, 64, 24, 24)
        output = output + residual  # (16, 64, 24, 24)
        sr_imgs = self.conv_block3(output)  # (16, 3, 24 * 4, 24 * 4)

        return sr_imgs


class Generator(nn.Module):
    """生成器：直接用 IR-Resnet """
    def __init__(self):
        super(Generator, self).__init__()
        self.net = IRResNet(n_blocks=3)

    def forward(self, inputs):
        outputs = self.net(inputs)

        return outputs


class TruncatedVGG19(nn.Module):
    """
    truncated VGG19网络，用于计算VGG特征空间的MSE损失
    """

    def __init__(self, i, j):
        """
        :参数 i: 第 i 个池化层
        :参数 j: 第 j 个卷积层
        """
        super(TruncatedVGG19, self).__init__()

        # 加载预训练的VGG模型
        vgg19 = torchvision.models.vgg19(pretrained=True)  # C:\Users\Administrator/.cache\torch\checkpoints\vgg19-dcbb9e9d.pth

        maxpool_counter = 0
        conv_counter = 0
        truncate_at = 0
        # 迭代搜索
        for layer in vgg19.features.children():
            truncate_at += 1

            # 统计
            if isinstance(layer, nn.Conv2d):
                conv_counter += 1
            if isinstance(layer, nn.MaxPool2d):
                maxpool_counter += 1
                conv_counter = 0

            # 截断位置在第(i-1)个池化层之后（第 i 个池化层之前）的第 j 个卷积层
            if maxpool_counter == i - 1 and conv_counter == j:
                break

        # 检查是否满足条件
        assert maxpool_counter == i - 1 and conv_counter == j, "当前 i=%d 、 j=%d 不满足 VGG19 模型结构" % (
            i, j)

        # 截取网络
        self.truncated_vgg19 = nn.Sequential(*list(vgg19.features.children())[:truncate_at + 1])

    def forward(self, input):
        """
        前向传播
        参数 input: 高清原始图或超分重建图，张量表示，大小为 (N, 3, w * scaling factor, h * scaling factor)
        返回: VGG19特征图，张量表示，大小为 (N, feature_map_channels, feature_map_w, feature_map_h)
        """
        output = self.truncated_vgg19(input)  # (N, feature_map_channels, feature_map_w, feature_map_h)

        return output


class Discriminator(nn.Module):
    """ 判别器 """

    def __init__(self, kernel_size=3, n_channels=64, n_blocks=4, fc_size=128):
        super(Discriminator, self).__init__()
        in_channels = 3
        conv_blocks = list()
        for i in range(n_blocks):
            out_channels = (n_channels if i is 0 else in_channels * 2) if i % 2 is 0 else in_channels
            conv_blocks.append(
                ConvolutionalBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=1 if i % 2 is 0 else 2, batch_norm=i is not 0, activation='LeakyReLu'))
            in_channels = out_channels
        self.conv_blocks = nn.Sequential(*conv_blocks)
        # 固定输出大小
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(out_channels * 6 * 6, fc_size)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(fc_size, 1)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        outputs = self.conv_blocks(inputs)
        outputs = self.adaptive_pool(outputs)
        # print("o1", outputs.shape)
        outputs = outputs.view(batch_size, -1)
        # print("o2", outputs.shape)
        outputs = self.fc1(outputs)
        outputs = self.leaky_relu(outputs)
        outputs = self.fc2(outputs)
        return outputs


def test_discriminator():
    d = Discriminator()
    d_p_len = 0
    for name, params in d.named_parameters():
        print(name, params.shape)
        size = 1
        for item in params.shape:
            size *= item
        d_p_len += size
        print('g', type(params), len(params))
    print('d', d_p_len)
    inputs = t.rand(1, 3, 48, 48).to('cpu')
    outputs = d(inputs)
    print('out shape: ', outputs.shape)


def test_generator():
    g = Generator()
    g_p_len = 0
    for name, params in g.named_parameters():
        print(name, params.shape)
        size = 1
        for item in params.shape:
            size *= item
        g_p_len += size
        print('g', type(params), len(params))
    print('g', g_p_len)

    inputs = t.rand(1, 3, 160, 160).to('cpu')
    outputs0 = g(inputs)
    print(outputs0.shape)
    t.save(g.net.state_dict(), 'd:/g.pth')

    ir = IRResNet(n_blocks=3)
    ir.load_state_dict(t.load('d:/g.pth'))
    t.save(ir.state_dict(), 'd:/g1.pth')
    g_p_len = 0
    for name, params in ir.named_parameters():
        print(name, params.shape)
        size = 1
        for item in params.shape:
            size *= item
        g_p_len += size
        print('ir', type(params), len(params))
    print('ir', g_p_len)
    outputs1 = ir(inputs)
    print(outputs1.shape)
    diff = outputs0 - outputs1
    print(diff)


def test_truncated_vgg19():
    tv = TruncatedVGG19(1, 2)
    vgg = torchvision.models.vgg19()

    tv_p_len = 0
    for name, params in tv.named_parameters():
        print(name, params.shape)
        size = 1
        for item in params.shape:
            size *= item
        tv_p_len += size
        print(type(params), len(params))

    vgg_p_len = 0
    for name, params in vgg.named_parameters():
        # print(name, params.shape)
        size = 1
        for item in params.shape:
            size *= item
        vgg_p_len += size

    print('len : ', tv_p_len, vgg_p_len)
    # t.save(tv.state_dict(), 'd:/aa.pth')
    # t.save(vgg.state_dict(), 'd:/bb.pth')
    inputs = t.rand(1, 3, 160, 160).to('cpu')
    outputs = tv(inputs)
    print(outputs.shape)


def test_ircnn():
    ir = IRCNN(num_channels=3).to('cuda')
    for name, params in ir.named_parameters():
        print(name, params.shape)

    inputs = t.rand(1, 3, 32, 32).to('cuda')
    print(inputs.shape)
    outputs = ir(inputs)
    # print(outputs)


def test_irrestnet():
    ir = IRResNet(n_blocks=3).to('cuda')
    for name, params in ir.named_parameters():
        print(name, params.shape)

    inputs = t.rand(1, 3, 32, 32).to('cuda')
    print(inputs.shape)
    outputs = ir(inputs)
    # print(outputs)


def test_conv():
    c = nn.Conv2d(3, 32, 5, 1, 3)
    inputs = t.rand(1, 3, 12, 12).to('cpu')
    print(c.padding, c.stride, c.kernel_size)
    outputs = c(inputs)
    print(outputs.shape)

    seq = nn.Sequential(
        # 输入 3 x 96 x 96
        nn.Conv2d(3, 1, 13, 7, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True))
    outputs = seq(inputs).view(-1)
    print(type(outputs))
    print(outputs.shape)

if __name__ == "__main__":
    # test_irrestnet()
    # test_truncated_vgg19()
    # test_conv()
    # test_generator()
    test_discriminator()
