
from argparse import Action
from requests import delete
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from .utils import FReLU, try_load_state, FeatureInfo, SquaredReLU, StarReLU
        
class GELU(nn.Module):
    def __init__(self, channel=None) -> None:
        super().__init__()
    def forward(self, x):
        return x * torch.sigmoid(1.702*x)
#        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6,
                 kernel_size=7, activation=GELU, expansion=4):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim) # depthwise conv
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, expansion * dim, 1) # pointwise/1x1 convs, implemented with linear layers
        self.act = activation(expansion * dim)
        self.pwconv2 = nn.Conv2d(expansion * dim, dim, 1)
        if layer_scale_init_value > 0:
            self.gamma = nn.BatchNorm2d(dim)
            self.gamma.weight.data.fill_(layer_scale_init_value)
            self.gamma.bias.data.fill_(0)
        else:
            self.gamma = None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma(x)
        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        channel (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, channel=3, num_classes=0, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1., steam_drop_rate=0.,
                 steam_stride=4, steam_padding=0, gap='mean', head_drop_rate=0.,
                 kernel_size=7, activation=GELU, expansion=4,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(channel, dims[0], kernel_size=4, stride=steam_stride, padding=steam_padding),
            nn.BatchNorm2d(dims[0]),
            nn.Dropout(steam_drop_rate, inplace=True),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    nn.BatchNorm2d(dims[i]),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                    layer_scale_init_value=layer_scale_init_value, 
                    kernel_size=kernel_size, activation=activation, expansion=expansion) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        self.gap = nn.AdaptiveAvgPool2d(1) if gap == 'mean' else nn.AdaptiveMaxPool2d(1)
        self.apply(self._init_weights)
        self.num_classes = num_classes
        self.norm = nn.BatchNorm2d(dims[-1]) # final norm layer
        if num_classes>0:
            self.head_dropout = nn.Dropout(head_drop_rate, inplace=True)
            self.head = nn.Linear(dims[-1], num_classes)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)
            self.feature_info = FeatureInfo([*dims,num_classes],[steam_stride,steam_stride*2,steam_stride*4,steam_stride*8,1024])
        else:
            self.feature_info = FeatureInfo(dims,[steam_stride,steam_stride*2,steam_stride*4,steam_stride*8])

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        f=[]
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            f.append(x)
        return f

    def forward(self, x, return_features=False):
        x = self.forward_features(x)
        y = self.norm(x[-1]) 
        y = self.gap(y)
        if self.num_classes>0:
            y = self.head_dropout(y)
            y = self.head(nn.Flatten()(y))
            x.append(y)
        if return_features:
            return x
        return y

@register_model
def convnextf_micro(pretrained=False,in_22k=False, **kwargs):
    # 27M
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[48, 96, 192, 384], num_classes=kwargs["out_features"])
    return model

@register_model
def convnextf_micromax(**kwargs):
    return convnextf_micro(gap='max', **kwargs)


@register_model
class convnextf_micro224(ConvNeXt):
    # 27M
    def __init__(self, **kwargs):
        kwargs['num_classes'] = kwargs["out_features"]
        del kwargs["out_features"]
        super().__init__(depths=[3, 3, 9, 3], dims=[48, 96, 192, 384], **kwargs)
    def forward(self, x, return_features=False):
        x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
        return super().forward(x, return_features)

@register_model
class convnextf_micro224d(convnextf_micro224):
    def __init__(self, **kwargs):
        super().__init__(steam_drop_rate=0.5, **kwargs)

@register_model
class convnextf_micro224dd(convnextf_micro224):
    # 27M
    def __init__(self, **kwargs):
        super().__init__(steam_drop_rate=0.5, head_drop_rate=0.5, **kwargs)

@register_model
def convnextf_lite(pretrained=False,in_22k=False, **kwargs):
    # 40M
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 256, 384], num_classes=kwargs["out_features"])
    return model

@register_model
def convnextf_nano(pretrained=True, in_22k=True, **kwargs):
    # 50M
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[48, 96, 192, 384], num_classes=kwargs["out_features"])
    return model

@register_model
def convnextf_mini(pretrained=True, in_22k=True, **kwargs):
    # 16M
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[24, 48, 96, 256], num_classes=kwargs["out_features"])
    return model

@register_model
def convnextf_mini2(pretrained=True, in_22k=True, **kwargs):
    # 12M
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[16, 32, 64, 256], num_classes=kwargs["out_features"])
    return model


@register_model
def convnextf_tiny(pretrained=True, in_22k=False, **kwargs):
    # 107M
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], num_classes=kwargs["out_features"])
    return model

class convnextf_small(ConvNeXt):
    # 190M
    def __init__(self, out_features=512, **kwargs):
        super().__init__(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], num_classes=out_features)

class convnextf3_small(ConvNeXt):
    # 189M
    def __init__(self, out_features=512, **kwargs):
        super().__init__(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], num_classes=out_features, kernel_size=3, **kwargs)

class convnextf3_s1(ConvNeXt):
    # 66M
    def __init__(self, out_features=512, **kwargs):
        super().__init__(depths=[16, 2, 1, 1], dims=[128, 256, 512, 1024], num_classes=out_features, kernel_size=3, **kwargs)

class convnextf3_s1f(convnextf3_s1):
    # 66M
    def __init__(self, out_features=512):
        super().__init__(out_features=out_features, activation=FReLU)

class convnextf3_s1s(convnextf3_s1):
    # 66M
    def __init__(self, out_features=512):
        super().__init__(out_features=out_features, activation=StarReLU)

class convnextf3_t1(ConvNeXt):
    # 39M
    def __init__(self, out_features=512, **kwargs):
        super().__init__(depths=[16, 2, 1, 1], dims=[128, 256, 512, 1024], num_classes=out_features, kernel_size=3, expansion=2, **kwargs)

class convnextf3_t1f(convnextf3_t1):
    # 40M
    def __init__(self, out_features=512):
        super().__init__(out_features=out_features, activation=FReLU)

class convnextf3_t1s(convnextf3_t1):
    # 40M
    def __init__(self, out_features=512):
        super().__init__(out_features=out_features, activation=StarReLU)

class convnextf_base(ConvNeXt):
    def __init__(self, out_features=512, **kwargs):
        super().__init__(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], num_classes=out_features, **kwargs)

class convnextf_large(ConvNeXt):
    def __init__(self, out_features=512, **kwargs):
        super().__init__(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], num_classes=out_features, **kwargs)

class convnextf_largeS(convnextf_large):
    def __init__(self, out_features=512, **kwargs):
        super().__init__(out_features=out_features, activation=StarReLU, **kwargs)

class convnextf_xlarge(ConvNeXt):
    def __init__(self, out_features=512, **kwargs):
        super().__init__(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], num_classes=out_features, **kwargs)


if __name__=="__main__":
    x= torch.rand(1,3,112,112)
    net = convnextf3_t1s(out_features=512)
    net.eval()
    y = net(x, return_features=True)
    print(net)
    torch.save(net.state_dict(), f"{net.__class__.__name__}.pth")
    print(net.feature_info.channels(), net.feature_info.reduction())
    for x in y: print(list(x.shape))
