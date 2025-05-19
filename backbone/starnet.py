"""
Implementation of Prof-of-Concept Network: StarNet.

We make StarNet as simple as possible [to show the key contribution of element-wise multiplication]:
    - like NO layer-scale in network design,
    - and NO EMA during training,
    - which would improve the performance further.

Created by: Xu Ma (Email: ma.xu1@northeastern.edu)
Modified Date: Mar/29/2024
"""
import torch
import torch.nn as nn
from .utils import DropPath, trunc_normal_, FeatureInfo

model_urls = {
    "starnet_s1": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s1.pth.tar",
    "starnet_s2": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s2.pth.tar",
    "starnet_s3": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s3.pth.tar",
    "starnet_s4": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s4.pth.tar",
}


class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


class StarNet(nn.Module):
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, num_classes=1000, in_channel=3, start_channel=32, steam_stride=2):
        super().__init__()
        self.num_classes = num_classes
        self.in_channel = start_channel
        # stem layer
        kernel_size=max(3, steam_stride)
        self.stem = nn.Sequential(ConvBN(in_channel, self.in_channel, kernel_size=kernel_size, stride=steam_stride, 
                                         padding=1 if kernel_size>steam_stride else 0), nn.ReLU6())
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] # stochastic depth
        # build stages
        self.stages = nn.ModuleList()
        cur = 0
        ch=[]
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            ch.append(self.in_channel)
            blocks = [Block(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))
        # head
        self.norm = nn.BatchNorm2d(self.in_channel)
        if num_classes > 0:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Linear(self.in_channel, num_classes)
            ch.append(num_classes)
        else:
            self.avgpool, self.head = None, None
        self.apply(self._init_weights)
        self.feature_info = FeatureInfo(ch, [steam_stride*2, steam_stride*4, steam_stride*8, steam_stride*16]+[1024] if num_classes>0 else [])
    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, return_featuremaps=False):
        x = self.stem(x)
        y = []
        for stage in self.stages:
            x = stage(x)
            y.append(x)
        if self.head is None:
            return y
        x = torch.flatten(self.avgpool(self.norm(x)), 1)
        x = self.head(x)
        if not return_featuremaps:
            return x
        y.append(x)
        return y


def s1(out_features=256, pretrained=False, **kwargs):
    model = StarNet(24, [2, 2, 8, 3], num_classes=out_features, **kwargs)
    if pretrained:
        url = model_urls['starnet_s1']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    return model


def s2(out_features=256, pretrained=False, **kwargs):
    model = StarNet(32, [1, 2, 6, 2], num_classes=out_features, **kwargs)
    if pretrained:
        url = model_urls['starnet_s2']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    return model


def s3(out_features=256, pretrained=False, **kwargs):
    model = StarNet(32, [2, 2, 8, 4], num_classes=out_features, **kwargs)
    if pretrained:
        url = model_urls['starnet_s3']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    return model


def s4(out_features=256, pretrained=False, **kwargs):
    model = StarNet(32, [3, 3, 12, 5], num_classes=out_features, **kwargs)
    if pretrained:
        url = model_urls['starnet_s4']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    return model


# very small networks #
def s050(out_features=256, pretrained=False, **kwargs):
    return StarNet(16, [1, 1, 3, 1], 3, num_classes=out_features, **kwargs)


def s100(out_features=256, pretrained=False, **kwargs):
    return StarNet(20, [1, 2, 4, 1], 4, num_classes=out_features, **kwargs)

def s150(out_features=256, pretrained=False, **kwargs):
    return StarNet(24, [1, 2, 4, 2], 3, num_classes=out_features, **kwargs)