# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from .utils import FReLU, CSymConv2d, RIConv2d, RICSymConv2d, ActIdentity, FeatureInfo
import random

Conv2d = nn.Conv2d

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                    padding=1, bias=False)

Act = FReLU

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DepthWisePointWiseWBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(DepthWisePointWiseWBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)

        self.conv1 = Conv2d(inplanes, planes, 1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.prelu1 = Act(planes)
        self.conv2 = Conv2d(
            planes, planes, 3, stride=stride, padding=1, groups=8)
        self.bn2 = nn.BatchNorm2d(planes)
        self.prelu2 = Act(planes)
        self.conv3 = Conv2d(planes, planes, 1)
        self.bn3 = nn.BatchNorm2d(planes)
        #self.prelu3 = Act(planes)

        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)

    def forward(self, x):
        x = self.bn0(x)
        residual = x
        out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.prelu2(out)
        out = self.conv3(out)
        out = self.bn3(out)

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
            nn.BatchNorm2d(inplanes),
            Act(inplanes),
            conv3x3(inplanes, inplanes),
            nn.BatchNorm2d(inplanes),
            Act(inplanes),
            conv3x3(inplanes, planes, stride),
        )

        self.bn_out = nn.BatchNorm2d(planes)

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
        out = self.bn_out(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

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

def random_drop_input_block(x_, ratio=0.5):
    x = x_.clone()
    N,C,H,W = x.shape
    for n in range(N):
        if random.random()<ratio:
            i = random.randint(0,3)
            if i==0:
                x[n,:,:,:W//2]=0
            elif i==1:
                x[n,:,:,W//2:]=0
            elif i==2:
                x[n,:,:H//2]=0
            else:
                x[n,:,H//2:]=0
    return x

class ResNetFace(nn.Module):
    def __init__(self, block, layers, out_features=512, use_se=True, scale=2, steam_conv=None, steam_kernel=None, 
        in_channel=3, drop_input_block=0, reverse_input=False):
        super().__init__()
        self.drop_input_block = drop_input_block
        s = scale
        self.inplanes = int(32*s)
        self.out_features = out_features
        self.use_se = use_se
        self.reverse_input = reverse_input
        if steam_conv is None:
            self.conv1 = nn.Conv2d(
                in_channel, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False)
        else:
            self.conv1 = steam_conv(
                in_channel, self.inplanes, kernel_size=steam_kernel, stride=2, padding=steam_kernel//2, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.prelu = Act(self.inplanes)
        self.layer1 = self._make_layer(block, int(32*s), layers[0], stride=2)
        self.layer2 = self._make_layer(block, int(64*s), layers[1], stride=2)
        self.layer3 = self._make_layer(block, int(128*s), layers[2], stride=2)
        self.layer4 = self._make_layer(block, int(256*s), layers[3], stride=2)
        self.bn_out = nn.BatchNorm2d(int(256*s))
        self.dropout = nn.Dropout(p=0.1)
        self.final_layer = nn.Sequential(
            # Conv2d(int(256*s), int(256*s), (3, 3), padding=1),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv2d(int(256*s), int(256*s), (3, 3), padding=1),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2d(int(256*s), out_features, (3, 3), padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(out_features),
            nn.AdaptiveAvgPool2d(1),
        ) if out_features>0 else nn.Identity()
        self.feature_info = FeatureInfo(
            [32*s, 64*s, 128*s, 256*s, out_features],[4,8,16,32,1024]
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2d(self.inplanes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride,
                            downsample, use_se=self.use_se))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se))

        return nn.Sequential(*layers)

    def forward(self, x, all_features=False):
        if self.reverse_input: x = 1-x
        if self.drop_input_block>0 and self.training:
            x = random_drop_input_block(x, self.drop_input_block)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.bn_out(x4)
        x = self.dropout(x)
        x = self.final_layer(x)
        x = x.view(x.size(0), -1)
        if all_features:
            return [x1,x2,x3,x4,x]
        else: 
            return x

class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = Conv2d(1, 64, kernel_size=3, stride=1, padding=1,
                              bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.fc5 = nn.Linear(512 * 8 * 8, 512)

        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2d(self.inplanes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc5(x)

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

def resnet_face18(use_se=True, **kwargs):
    model = ResNetFace(DepthWisePointWiseWBlock, [
                       3, 6, 12, 3], use_se=use_se, **kwargs)
    return model

def resnet_fp18(use_se=True, **kwargs):
    model = ResNetFace(DepthWisePointWiseWBlock, [
                       3, 6, 12, 3], use_se=use_se, reverse_input=True, **kwargs)
    return model

def face18(use_se=True, **kwargs):
    model = ResNetFace(DepthWisePointWiseWBlock, [
                       3, 6, 12, 3], use_se=use_se, scale=0.5, **kwargs)
    return model

def resnet10(**kwargs):
    use_se= kwargs.pop("use_se", True)
    scale = kwargs.pop("scale", 0.5)
    model = ResNetFace(DepthWisePointWiseWBlock, [
                       2, 4, 8, 2], use_se=use_se, scale=scale, **kwargs)
    return model

def resnet_face18csy(use_se=True, **kwargs):
    global Conv2d
    Conv2d = CSymConv2d
    model = ResNetFace(DepthWisePointWiseWBlock, [
                       3, 6, 12, 3], use_se=use_se, **kwargs)
    return model

def face18csy(use_se=True, **kwargs):
    global Conv2d
    Conv2d = CSymConv2d
    model = ResNetFace(DepthWisePointWiseWBlock, [
                       3, 6, 12, 3], use_se=use_se, scale=0.5, **kwargs)
    return model

def face18csyd(use_se=True, **kwargs):
    kwargs['drop_input_block']=0.5
    return face18csy(use_se=use_se, **kwargs)

def face18csyr(use_se=True, **kwargs):
    global Conv2d
    Conv2d = RICSymConv2d
    model = ResNetFace(DepthWisePointWiseWBlock, [
                       3, 6, 12, 3], use_se=use_se, scale=0.5, **kwargs)
    return model

def face18csyri(use_se=True, **kwargs):
    global Conv2d
    Conv2d = RICSymConv2d
    global Act
    Act = ActIdentity
    model = ResNetFace(DepthWisePointWiseWBlock, [
                       3, 6, 12, 3], use_se=use_se, scale=0.5, **kwargs)
    return model

def face18csyris5(steam_kernel=5, out_features=512):
    return face18csyri(True, out_features=out_features, steam_conv=RICSymConv2d, steam_kernel=steam_kernel)

def face18csyris9(out_features=512):
    return face18csyri(True, out_features=out_features, steam_conv=RICSymConv2d, steam_kernel=9)

def resnet10csy(**kwargs):
    global Conv2d
    Conv2d = CSymConv2d
    use_se= kwargs.pop("use_se", True)
    scale = kwargs.pop("scale", 0.5)
    model = ResNetFace(DepthWisePointWiseWBlock, [
                       2, 4, 8, 2], use_se=use_se, scale=scale, **kwargs)
    return model

if __name__ == '__main__':
    x= torch.rand(1,3,224,224)
    net = resnet_face18(out_features=512)
    net.eval()
    y = net(x, True)
    print(net)
    torch.save(net.state_dict(), "resnet_face18.pth")
    print(net.feature_info.channels(), net.feature_info.reduction())
    for x in y: print(list(x.shape))

