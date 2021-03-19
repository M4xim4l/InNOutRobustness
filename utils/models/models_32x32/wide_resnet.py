##https://github.com/xternalz/WideResNet-pytorch
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    """
    Swish out-performs Relu for deep NN (more than 40 layers). Although, the performance of activation and swish model
    degrades with increasing batch out_size, swish performs better than activation.
    https://jmlb.github.io/ml/2017/12/31/swish_activation_function/ (December 31th 2017)
    """

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def get_activation(activation):
    if activation == 'relu':
        return nn.ReLU(inplace=False)
    elif activation == 'swish':
        return Swish()
    else:
        raise NotImplementedError()

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, activation='relu', dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.activation = get_activation(activation)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.activation(self.bn1(x))
        else:
            out = self.activation(self.bn1(x))
        out = self.activation(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, activation, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, activation, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, activation, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, activation, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, activation='relu', dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, activation, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, activation, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, activation, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.activation = get_activation(activation)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.activation(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

def WideResNet28x10(num_classes=10, activation='relu', dropRate=0.0):
     return WideResNet(28, num_classes, 10, activation=activation, dropRate=dropRate)

def WideResNet28x2(num_classes=10, activation='relu', dropRate=0.0):
     return WideResNet(28, num_classes, 2, activation=activation, dropRate=dropRate)

def WideResNet28x20(num_classes=10, activation='relu', dropRate=0.0):
    return WideResNet(28, num_classes, 20, activation=activation,dropRate=dropRate)

def WideResNet34x20(num_classes=10, activation='relu', dropRate=0.0):
    return WideResNet(34, num_classes, 20, activation=activation,dropRate=dropRate)

def WideResNet40x10(num_classes=10, activation='relu', dropRate=0.0):
    return WideResNet(40, num_classes, 10, activation=activation, dropRate=dropRate)

def WideResNet70x16(num_classes=10, activation='relu', dropRate=0.0):
    return WideResNet(70, num_classes, 16, activation=activation, dropRate=dropRate)
