# -*- coding: utf-8 -*-
"""
Created on 18-5-21 下午5:26

@author: ronghuaiyang
"""
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.utils.weight_norm as weight_norm
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
    return RIConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
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
        self.conv_frelu = RIConv2d(
            in_channels, in_channels, 3, 1, 1, groups=in_channels)
        self.bn_frelu = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x1 = self.conv_frelu(x)
        x1 = self.bn_frelu(x1)
        x = torch.max(x, x1)
        return x


# from PLA import PLA
Act = FReLU


class RIConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(RIConv2d, self).__init__(*args, **kwargs)

    def forward(self, input):
        if self.kernel_size == (1, 1):
            # print(self.kernel_size)
            return F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

        weight = torch.stack([
            torch.rot90(self.weight, 0, [2, 3]),
            torch.rot90(self.weight, 1, [2, 3]),
            torch.rot90(self.weight, 2, [2, 3]),
            torch.rot90(self.weight, 3, [2, 3]),
        ], dim=1)
        weight = weight.view(weight.size(0)*weight.size(1),
                             weight.size(2), weight.size(3), weight.size(4))
        bias = self.bias
        if bias is not None:
            bias = torch.stack([
                bias, bias, bias, bias
            ], dim=1)
            bias = bias.view(bias.size(0)*bias.size(1))

        out = F.conv2d(input, weight, bias, self.stride,
                       self.padding, self.dilation, self.groups)
        out, _ = out.view(out.size(0), out.size(
            1)//4, 4, out.size(2), out.size(3)).max(dim=2)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
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

        self.conv1 = RIConv2d(inplanes, planes, 1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.prelu1 = Act(planes)
        self.conv2 = RIConv2d(
            planes, planes, 3, stride=stride, padding=1, groups=8)
        self.bn2 = nn.BatchNorm2d(planes)
        self.prelu2 = Act(planes)
        self.conv3 = RIConv2d(planes, planes, 1)
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
        self.conv1 = RIConv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = RIConv2d(planes, planes, kernel_size=3, stride=stride,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = RIConv2d(
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


class ResNetFace(nn.Module):
    def __init__(self, block, layers, out_features=512, use_se=True):
        s = 2
        self.inplanes = int(32*s)
        self.out_features = out_features
        self.use_se = use_se
        super(ResNetFace, self).__init__()
        #self.down2x = nn.AvgPool2d(2, stride=2)
        self.conv1 = RIConv2d(
            3, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.prelu = Act(self.inplanes)
        #self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, int(32*s), layers[0], stride=2)
        self.layer2 = self._make_layer(block, int(64*s), layers[1], stride=2)
        self.layer3 = self._make_layer(block, int(128*s), layers[2], stride=2)
        self.layer4 = self._make_layer(block, int(256*s), layers[3], stride=2)
        self.bn_out = nn.BatchNorm2d(int(256*s))
        self.dropout = nn.Dropout(p=0.1)

        self.final_layer = nn.Sequential(
            RIConv2d(int(256*s), int(256*s), (3, 3), padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            RIConv2d(int(256*s), int(256*s), (3, 3), padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            RIConv2d(int(256*s), out_features, (3, 3), padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(out_features),
        )

        for m in self.modules():
            if isinstance(m, RIConv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        self.feature_info = FeatureInfo(
            [32*s, 64*s, 128*s, 256*s, out_features],[4,8,16,32,1024]
        )
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                RIConv2d(self.inplanes, planes * block.expansion,
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

    def forward(self, x, return_features=False):
        #x = self.down2x(x)
        y = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        #x = self.maxpool(x)

        x = self.layer1(x)
        y.append(x)
        x = self.layer2(x)
        y.append(x)
        x = self.layer3(x)
        y.append(x)
        x = self.layer4(x)
        x = self.bn_out(x)
        y.append(x)
        x = self.dropout(x)

        x = self.final_layer(x)
        x = x.view(x.size(0), -1)
        y.append(x)
        if return_features: return y
        return x


class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        # self.conv1 = RIConv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        self.conv1 = RIConv2d(1, 64, kernel_size=3, stride=1, padding=1,
                              bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(8, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc5 = nn.Linear(512 * 8 * 8, 512)

        for m in self.modules():
            if isinstance(m, RIConv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        s = 2
        self.feature_info = FeatureInfo(
            [32*s, 64*s, 128*s, 256*s, 512],[4,8,16,32,1024]
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                RIConv2d(self.inplanes, planes * block.expansion,
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
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = nn.AvgPool2d(kernel_size=x.size()[2:])(x)
        # x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc5(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


def resnet_face18(use_se=True, **kwargs):
    model = ResNetFace(DepthWisePointWiseWBlock, [
                       3, 6, 12, 3], use_se=use_se, **kwargs)
    return model


def visualization_onnx():
    import torch

    model = resnet_face18().cuda().eval()
    print(model)
    dummy_input = torch.randn(1, 3, 256, 256).cuda()

    import sys
    if len(sys.argv) > 1:
        from thop import profile
        flops, params = profile(model, inputs=(dummy_input, ))
        from thop import clever_format
        flops, params = clever_format([flops, params], "%.3f")
        print('flops, params:\n', flops, params)
        exit()

    outs = model(dummy_input)
    for out in outs:
        print(out.size())

    path = "resnet_face18.onnx"
    torch.onnx.export(model, dummy_input, path, verbose=True,
        export_params=False, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    import onnx
    from onnx import helper, shape_inference
    onnx.save(shape_inference.infer_shapes(onnx.load(path)), path)

    import netron
    netron.start(path, port=12405)


def test_RIConv():
    x = torch.randn(1, 2, 3, 3)
    x[x > 0] = 1
    x[x < 0] = 0

    print('x:', x)
    conv = RIConv2d(2, 2, 3, padding=0, groups=2)
    conv.weight.data[conv.weight > 0] = 1
    conv.weight.data[conv.weight < 0] = 0
    conv.bias.data[conv.bias > 0] = 1
    conv.bias.data[conv.bias < 0] = 0
    print('conv.weight:', conv.weight)
    print('conv.bias:', conv.bias)
    print('conv(x):', conv(x))


if __name__ == '__main__':
    x= torch.rand(1,3,256,256)
    net = resnet_face18(out_features=512)
    net.eval()
    y = net(x, return_features=True)
    print(net)
    torch.save(net.state_dict(), "resnet_face18.pth")
    print(net.feature_info.channels(), net.feature_info.reduction())
    for x in y: print(list(x.shape))

