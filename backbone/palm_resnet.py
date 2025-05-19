import torch
import torch.nn as nn
import torch.nn.functional as F
import torch

class FeatureInfo(object):
    def __init__(self, channels, reductions) -> None:
        super().__init__()
        self.channels_ = channels
        self.reductions_ = reductions
    def channels(self):
        return self.channels_
    def reduction(self):
        return self.reductions_

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class FReLU(nn.Module):
    r""" FReLU formulation. The funnel condition has a window size of kxk. (k=3 by default)
    """

    def __init__(self, in_channels):
        super().__init__()
        self.conv_frelu = nn.Conv2d(
            in_channels, in_channels, 3, 1, 1, groups=in_channels)
        self.bn_frelu = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x1 = self.conv_frelu(x)
        x1 = self.bn_frelu(x1)
        x = torch.max(x, x1)
        return x


Act = FReLU

class DepthWisePointWiseWBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(DepthWisePointWiseWBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)

        self.conv1 = nn.Conv2d(inplanes, planes, 1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.prelu1 = Act(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, 3, stride=stride, padding=1, groups=8)
        self.bn2 = nn.BatchNorm2d(planes)
        self.prelu2 = Act(planes)
        self.conv3 = nn.Conv2d(planes, planes, 1)
        self.bn3 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)

    def forward(self, x):
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
        return out

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1),
            Act(channel // reduction),
            nn.Conv2d(channel // reduction, channel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.fc(self.pool(x))
        return x * y


class ResNetFace(nn.Module):
    def __init__(self, out_features, block, layers, use_se=True, s=2, channel=3):
        super().__init__()
        self.s = s
        self.inplanes = int(32*s)
        self.out_features = out_features
        self.use_se = use_se
        self.conv1 = nn.Conv2d(
            channel, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.prelu = Act(self.inplanes)
        self.layer1 = self._make_layer(block, int(32*s), layers[0], stride=2)
        self.layer2 = self._make_layer(block, int(64*s), layers[1], stride=2)
        self.layer3 = self._make_layer(block, int(128*s), layers[2], stride=2)
        self.layer4 = self._make_layer(block, int(256*s), layers[3], stride=2)
        self.bn_out = nn.BatchNorm2d(int(256*s))
        self.dropout = nn.Dropout(p=0.1)

        self.final_layer = nn.Sequential(
            nn.Conv2d(int(256*s), int(256*s), (3, 3), padding=0,groups=8),
            nn.Conv2d(int(256*s), int(256*s), (3, 3), padding=0,groups=8),
            nn.BatchNorm2d(int(256*s)),
            nn.Flatten(),
        )
        self.fc = nn.Linear(int(256*s)*3*2, out_features)

        self.feature_info = FeatureInfo([s*32, s*64, s*128, s*256, out_features], [4,8,16,32,1024])
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
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

    def forward(self, x, return_featuremaps=False):
        layer_outs = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)

        x = self.layer1(x)
        layer_outs.append(x)
        x = self.layer2(x)
        layer_outs.append(x)
        x = self.layer3(x)
        layer_outs.append(x)
        x = self.layer4(x)
        layer_outs.append(x)

        x = self.bn_out(x)
        x = self.dropout(x)
        if (x.shape[1]!=7) or (x.shape[2]!=6):
            x=F.adaptive_avg_pool2d(x, (7,6))
        x = self.final_layer(x)
        features = self.fc(x)
        layer_outs.append(features)
        if return_featuremaps:
            return layer_outs
        else:
            return features


class PalmNet(ResNetFace):
    # 27M
    def __init__(self, out_features=256, channel=3):
        super().__init__(out_features, DepthWisePointWiseWBlock, [
                       3, 6, 12, 3], use_se=True, channel=channel)

class PalmNet50(ResNetFace):
    def __init__(self, out_features=256):
        super().__init__(out_features, DepthWisePointWiseWBlock, [
                       4, 12, 36, 32], use_se=True)

class PalmNet15(ResNetFace):
    # 16M
    def __init__(self, out_features=256):
        super().__init__(out_features, DepthWisePointWiseWBlock, [
                       3, 6, 12, 3], use_se=True, s=1.5)

if __name__ == '__main__':
    from . import utils
    net = utils.test_model(PalmNet15, (3,224,192))
    torch.save(net.state_dict(), f"{net.__class__.__name__}.bin")
