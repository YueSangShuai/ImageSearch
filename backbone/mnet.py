# -*- coding: utf-8 -*-
"""
Created on 18-5-21 下午5:26

@author: ronghuaiyang
"""
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                    padding=1, bias=False)

class MaxMin(nn.Module):
    def __init__(self, *args):
        super(MaxMin, self).__init__()

    def forward(self, input):
        c = input.size(1)//2
        i1 = input[:, :c]
        i2 = input[:, c:c*2]
        x1 = torch.max(i1, i2)
        x2 = torch.min(i1, i2)
        if c*2 < input.size(1):
            x = torch.cat([x1, x2, F.relu(input[:, -1:])], dim=1)
        else:
            x = torch.cat([x1, x2], dim=1)
        return x

    def __repr__(self):
        return self.__class__.__name__+"()"


class FReLU(nn.Module):
    r""" FReLU formulation. The funnel condition has a window size of kxk. (k=3 by default)
    """

    def __init__(self, in_channels):
        super().__init__()
        self.conv_frelu = nn.Conv2d(
            in_channels, in_channels, 3, 1, 1, groups=in_channels)

    def forward(self, x):
        x1 = self.conv_frelu(x)
        x = torch.max(x, x1)
        return x


Act = FReLU

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DepthWisePointWiseWBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(DepthWisePointWiseWBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, 1)
        self.prelu1 = Act(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, 3, stride=stride, padding=1, groups=8)
        self.prelu2 = Act(planes)
        self.conv3 = nn.Conv2d(planes, planes, 1)

        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)

    def forward(self, x):
        residual = x
        out = x

        out = self.conv1(out)
        out = self.prelu1(out)
        out = self.conv2(out)
        out = self.prelu2(out)
        out = self.conv3(out)

        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        #out = self.prelu2(out)

        return out


class IRBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(IRBlock, self).__init__()

        self.conv = nn.Sequential(
            Act(inplanes),
            conv3x3(inplanes, inplanes),
            Act(inplanes),
            conv3x3(inplanes, planes, stride),
        )

        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)

    def forward(self, x):
        residual = x
        out = self.conv(x)

        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                              padding=1, bias=False)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.PReLU(channel // reduction),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResNetFace(nn.Module):
    def __init__(self, block, layers, out_features=512, use_se=True):
        s = 2
        self.inplanes = int(32*s)
        self.out_features = out_features
        self.use_se = use_se
        super(ResNetFace, self).__init__()
        #self.down2x = nn.AvgPool2d(2, stride=2)
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False)
        self.prelu = Act(self.inplanes)
        #self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, int(32*s), layers[0], stride=2)
        self.layer2 = self._make_layer(block, int(64*s), layers[1], stride=2)
        self.layer3 = self._make_layer(block, int(128*s), layers[2], stride=2)
        self.layer4 = self._make_layer(block, int(256*s), layers[3], stride=2)
        self.dropout = nn.Dropout(p=0.1)

        self.final_layer = nn.Sequential(
            nn.Conv2d(int(256*s), int(256*s), (3, 3), padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(int(256*s), int(256*s), (3, 3), padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(int(256*s), out_features, (3, 3), padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride,
                            downsample, use_se=self.use_se))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        #x = self.down2x(x)
        x = self.conv1(x)
        x = self.prelu(x)
        #x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.dropout(x)

        x = self.final_layer(x)
        x = x.view(x.size(0), -1)
        return x


class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        # self.conv1 = RIConv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1,
                              bias=False)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(8, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc5 = nn.Linear(512 * 8 * 8, 512)
        self.bn = nn.BatchNorm1d(512)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        x = self.bn(x)
        return x


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet_face18(use_se=True, **kwargs):
    model = ResNetFace(DepthWisePointWiseWBlock, [
                       3, 6, 12, 3], use_se=use_se, **kwargs)
    return model

class RotNet(ResNetFace):
    def __init__(self, **kwargs):
        use_se= kwargs.pop("use_se", True)
        super().__init__(DepthWisePointWiseWBlock, [
                       3, 6, 12, 3], use_se=use_se, **kwargs)
    def forward(self, x):
        y1 = super().forward(x)
        y2 = super().forward(torch.rot90(x, 1, [2, 3]))
        y3 = super().forward(torch.rot90(x, 2, [2, 3]))
        y4 = super().forward(torch.rot90(x, 3, [2, 3]))
        y = y1+y2+y3+y4
        return y

def resnet10(**kwargs):
    model = ResNetFace(DepthWisePointWiseWBlock, [
                       2, 2, 6, 2], use_se=False, **kwargs)
    return model
