import torch.nn as nn
import math
import torch.nn.functional as F
import torch
from . import utils
import time

class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, theta=0.7):

        super(Conv2d_cd, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff
            
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


Act = utils.FReLU

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=1):
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

class _PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


class _ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(_ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out



class DepthWisePointWiseWBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(DepthWisePointWiseWBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)

        self.conv1 = nn.Conv2d(inplanes, planes, 3, padding=1, groups=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.prelu1 = Act(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, 3, stride=stride, padding=1, groups=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.prelu2 = Act(planes)
        self.conv3 = nn.Conv2d(planes, planes, 1, padding=0, groups=1)
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

class ResNetFace(nn.Module):
    def __init__(self, out_features=256, block=DepthWisePointWiseWBlock, layers=[3,3,3,3], use_se=True, s=2):
        self.sizes = (3, 224, 224)
        self.s = s
        self.inplanes = int(32*s)
        self.out_features = out_features
        self.use_se = use_se

        super(ResNetFace, self).__init__()
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.prelu = Act(self.inplanes)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, int(32*s), layers[0], stride=1)
        self.layer2 = self._make_layer(block, int(64*s), layers[1], stride=1)
        self.layer3 = self._make_layer(block, int(128*s), layers[2], stride=1)
        self.layer4 = self._make_layer(block, int(256*s), layers[3], stride=1)
        
        self.pam = nn.Sequential(
            nn.Conv2d(int(256*s),int(256*s), 1),
            _PositionAttentionModule(int(256*s)),
            nn.Conv2d(int(256*s),int(256*s), 1),
        )
        self.cam = nn.Sequential(
            nn.Conv2d(int(256*s),int(256*s), 1),
            _ChannelAttentionModule(),
            nn.Conv2d(int(256*s),int(256*s), 1),
        )

        self.bn_out = nn.BatchNorm2d(int(256*s))
        self.dropout = nn.Dropout(p=0.1)

        self.cdc_conv = nn.Sequential(
            Conv2d_cd(int(256*s), int(256*s), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(256*s)),
        )

        self.fc = nn.Sequential(
            nn.Conv2d(int(256*s), int(256*s), (3, 3), padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(int(256*s), out_features, (3, 3), padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(out_features),
            nn.Flatten(),
        )
        self.print_feature_size = True
        self.feature_info = utils.FeatureInfo([32*s, 64*s, 128*s, 256*s, 256*s, out_features],[2,4,8,16,32,1024])
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

    def forward(self, x, return_intermediate=False):
        layer_outs = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        layer_outs.append(x)
        x = self.maxpool(x)

        x = self.layer2(x)
        layer_outs.append(x)
        x = self.maxpool(x)
        
        x = self.layer3(x)
        layer_outs.append(x)
        x = self.maxpool(x)

        x = self.layer4(x)
        layer_outs.append(x)
        x = self.maxpool(x)

        x = self.pam(x) + self.cam(x)
        x = self.bn_out(x)
        x = self.cdc_conv(x)
        
        layer_outs.append(x)
        features1 = x
        feature = self.fc(features1)
        layer_outs.append(feature)
        if return_intermediate:
            return layer_outs
        else:
            return feature


class Net1(ResNetFace):
    def __init__(self, out_features=16):
        super().__init__(out_features=out_features, block=DepthWisePointWiseWBlock, layers=[3,3,3,3], use_se=True, s=2)


# def Net1(out_features=16, use_se=True, **kwargs):
#     model = ResNetFace(out_features, DepthWisePointWiseWBlock, [
#                        3, 3, 3, 3], use_se=use_se, **kwargs)
#     return model

class DepthWisePointWiseWBlockSmall(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)

        self.conv1 = nn.Conv2d(inplanes, planes, 3, padding=1, groups=24)
        self.bn1 = nn.BatchNorm2d(planes)
        self.prelu1 = Act(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, 3, stride=stride, padding=1, groups=24)
        self.bn2 = nn.BatchNorm2d(planes)
        self.prelu2 = Act(planes)
        self.conv3 = nn.Conv2d(planes, planes, 1, padding=0, groups=1)
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

class ResNetFaceSmall(nn.Module):
    def __init__(self, out_features=256, block=DepthWisePointWiseWBlockSmall, layers=[1,1,1,2], use_se=False):
        super().__init__()
        self.sizes = (3, 224, 224)
        s = 3
        self.s = s
        self.inplanes = int(32*s)
        self.out_features = out_features
        self.use_se = use_se
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=5, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.prelu = Act(self.inplanes)
        self.layer1 = self._make_layer(block, int(32*s), layers[0], stride=2)
        self.layer2 = self._make_layer(block, int(64*s), layers[1], stride=2)
        self.layer3 = self._make_layer(block, int(128*s), layers[2], stride=2)
        self.layer4 = self._make_layer(block, int(256*s), layers[3], stride=2)
        self.bn_out = nn.BatchNorm2d(int(256*s))
        self.dropout = nn.Dropout(p=0.1)

        self.final_layer = nn.Sequential(
            nn.Conv2d(int(256*s), 1, 1),
            nn.BatchNorm2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(int(256*s), int(256*s), (3, 3), padding=1),
            nn.BatchNorm2d(int(256*s)),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(int(256*s), out_features, (3, 3), padding=1),
            nn.BatchNorm2d(out_features),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )
        self.print_feature_size = True
        self.feature_info = utils.FeatureInfo([32*s, 64*s, 128*s, 256*s, out_features],[4, 8, 16, 32, 1024])

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

    def forward(self, x, return_intermediate=False, return_feature=False):
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
        x = self.bn_out(x)
        b = int(x.size(0))
        xx = x
        x = self.dropout(x)
        features1 = x
        layer_outs.append(x)
        if self.print_feature_size:
            print(features1.size())
            self.print_feature_size = False
        feature = self.classifier(features1)
        layer_outs.append(feature)
        if return_intermediate:
            return layer_outs
        else:
            return feature

class NetSmall(ResNetFaceSmall):
    def __init__(self, out_features=256, block=DepthWisePointWiseWBlockSmall, layers=[1, 1, 1, 2], use_se=False):
        super().__init__(out_features, block, layers, use_se)