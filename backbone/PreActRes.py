import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils

class FReLU(nn.Module):
   def __init__(self, in_channels):
       super().__init__()
       self.conv_frelu = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
       self.bn_frelu = nn.BatchNorm2d(in_channels)
    
   def forward(self, x):
       x1 = self.conv_frelu(x)
       x1 = self.bn_frelu(x1)
       x = torch.max(x, x1)
       return x

class GELU(nn.GELU):
    def __init__(self, channel=1):
        super().__init__()

class SiLU(nn.SiLU):
    def __init__(self, channel=1):
        super().__init__()

class ReLU(nn.ReLU):
    def __init__(self, channel=1):
        super().__init__()

# 注意力模块
class AttentionModual(nn.Module):
    def __init__(self, in_channels, order=1):
        super(AttentionModual, self).__init__()
        self.order = order
        self.inter_channels = in_channels // 2
        for j in range(self.order):
            for i in range(j+1):
                name = 'order' + str(self.order) + '_' + str(j+1) + '_' + str(i+1)
                setattr(self, name, nn.Sequential(nn.Conv2d(in_channels, self.inter_channels, 1, padding=0, bias=False)))
        for i in range(self.order):
            name = 'convb' + str(self.order) + '_' + str(i+1)
            setattr(self, name, nn.Sequential(nn.Conv2d(self.inter_channels, in_channels, 
                    1, padding=0, bias=False), nn.Sigmoid()))
    
    def forward(self, x):
        y = []
        for j in range(self.order):
            for i in range(j+1):
                name = 'order' + str(self.order) + '_' + str(j+1) + '_' + str(i+1)
                layer = getattr(self, name)
                y.append(layer(x))
        y_ = []
        cnt=0
        for j in range(self.order):
            y_temp = 1
            for i in range(j+1):
                y_temp = y_temp * y[cnt]
                cnt += 1
            y_.append(F.relu(y_temp))
        y__ = 0
        for i in range(self.order):
            name = 'convb' + str(self.order) + '_' + str(i+1)
            layer = getattr(self, name)
            y__ += layer(y_[i])
        out = x * y__ / self.order
        return out

############################################################
class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2, scale=64, act_fun=FReLU):
        super(PreActResNet, self).__init__()
        self.in_planes = scale
        self.parts = num_blocks[3]
        self.conv1 = nn.Conv2d(3, scale, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, scale, num_blocks[0], stride=1, act_fun=act_fun)
        self.layer2 = self._make_layer(block, scale*2, num_blocks[1], stride=2, act_fun=act_fun)
        self.layer3 = self._make_layer(block, scale*4, num_blocks[2], stride=2, act_fun=act_fun)
        
        for i in range(self.parts):
            name = 'level_' + str(i)
            setattr(self, name, AttentionModual(scale*4, i+1))
        
        self.linear = nn.Linear(196*scale, num_classes)
        self.feature_info = utils.FeatureInfo([scale, scale*2, scale*4, scale*4, num_classes], [1, 2, 4, 4, 1000])
    def _make_layer(self, block, planes, num_blocks, stride, act_fun):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, act_fun))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        y = []
        out = self.conv1(x)
        out = self.layer1(out)
        y.append(out)
        out = self.layer2(out)
        y.append(out)
        out = self.layer3(out)
        y.append(out)
        for i in range(self.parts):
            name = 'level_' + str(i)
            layer = getattr(self, name)
            out = layer(out)
        y.append(out)
        _,__,H,W = out.shape
        if not (H==28 and W==28):
            out = F.interpolate(out, (28,28))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        y.append(out)
        if return_features:
            return y
        return out

############################################################
class PreActBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, act_fun=FReLU):
        super(PreActBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.frelu1 = act_fun(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.frelu2 = act_fun(planes)
        self.relu1 = self.frelu1
        self.relu2 = self.frelu2
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = self.conv1(x)
        out = self.frelu1(self.bn1(out))
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out = self.frelu2(self.bn2(self.conv2(out)))
        out += shortcut
        return out

class Net2233s64frelu(PreActResNet):
    # 30M
    def __init__(self, out_features=2):
        super().__init__(PreActBlock, [2,2,3,3], num_classes=out_features, scale=64, act_fun=FReLU)

class Net3443s96frelu(PreActResNet):
    # 74M
    def __init__(self, out_features=2):
        super().__init__(PreActBlock, [3,4,4,3], num_classes=out_features, scale=96, act_fun=FReLU)

class Net2233s64silu(PreActResNet):
    # 30M
    def __init__(self, out_features=2):
        super().__init__(PreActBlock, [2,2,3,3], num_classes=out_features, scale=64, act_fun=SiLU)

class Net2233s32frelu(PreActResNet):
    # 11M
    def __init__(self, out_features=2):
        super().__init__(PreActBlock, [2,2,3,3], num_classes=out_features, scale=32, act_fun=FReLU)

class Net2233s32relu(PreActResNet):
    # 11M
    def __init__(self, out_features=2):
        super().__init__(PreActBlock, [2,2,3,3], num_classes=out_features, scale=32, act_fun=ReLU)

class Net2442s32relu(PreActResNet):
    # 12M
    def __init__(self, out_features=2):
        super().__init__(PreActBlock, [2,4,4,2], num_classes=out_features, scale=32, act_fun=ReLU)
