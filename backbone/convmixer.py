import torch.nn as nn
import torch
from torch.nn import functional as F
from .utils import FeatureInfo, FReLU, GELU, ReLU, test_model

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class ConvMixer(nn.Module):
    def __init__(self, dim=128, depth=10, kernel_size=9, patch_size=7, pooling_size=0, activation=GELU, conv2d=nn.Conv2d):
        super().__init__()
        self.feature_info = FeatureInfo([dim, dim],[patch_size, patch_size if pooling_size<=0 else patch_size*pooling_size])
        self.steam = nn.Sequential(         
            conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            activation(dim),
            nn.BatchNorm2d(dim),
        )
        net = [
            *[nn.Sequential(
                    Residual(nn.Sequential(
                        conv2d(dim, dim, kernel_size, groups=dim, padding=kernel_size//2),
                        activation(dim),
                        nn.BatchNorm2d(dim)
                    )),
                    conv2d(dim, dim, kernel_size=1),
                    activation(dim),
                    nn.BatchNorm2d(dim)
            ) for i in range(depth)],
        ]
        if pooling_size>0:
            net.append(nn.MaxPool2d(pooling_size))
        self.net=nn.Sequential(*net)
    def forward(self, x, return_features=False):
        x = self.steam(x)
        y = self.net(x)
        if return_features:
            return [x,y]
        return y

class ConvMixer2(nn.Module):
    def __init__(self, dim=128, depth=10, kernel_size=9, patch_size=7, pooling_size=0, activation=FReLU, conv2d=nn.Conv2d, drop_path=0.1):
        super().__init__()
        self.feature_info = FeatureInfo([dim, dim],[patch_size, patch_size if pooling_size<=0 else patch_size*pooling_size])
        self.steam = nn.Sequential(         
            conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(dim),
            activation(dim),
        )
        self.drop_path = drop_path
        net = [
            *[nn.Sequential(
                    Residual(nn.Sequential(
                        conv2d(dim, dim, kernel_size, groups=dim, padding=kernel_size//2),
                        nn.BatchNorm2d(dim),
                        activation(dim),
                    )),
                    conv2d(dim, dim, kernel_size=1),
                    nn.BatchNorm2d(dim),
                    activation(dim),
            ) for i in range(depth)],
        ]
        self.pool = nn.AvgPool2d(pooling_size) if pooling_size>0 else nn.Identity()
        self.net=nn.ModuleList(net)
    def forward(self, x, return_features=False):
        x = self.steam(x)
        for n in self.net:
            #if not self.training or torch.rand(1)>self.drop_path:
            x = n(x)
        y = self.pool(x)
        if return_features:
            return [x,y]
        return y

class NP8K9D10(ConvMixer):
    def __init__(self, out_features=512, depth=10, kernel_size=9, patch_size=8, activation=FReLU, max_pooling=False):
        super().__init__(dim=out_features, depth=depth, kernel_size=kernel_size, patch_size=patch_size, activation=activation)
        if max_pooling:
            self.feature_info.reduction()[-1]=1024
            self.pool = nn.Sequential(nn.AdaptiveMaxPool2d(1), nn.Flatten())
        elif max_pooling == False:
            self.feature_info.reduction()[-1]=1024
            self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        else: # max_pooling == None
            self.pool = nn.Identity()
    def forward(self, x, return_features=False):
        x = super().forward(x, return_features)
        if return_features:
            x[-1]=self.pool(x[-1])
            return x
        return self.pool(x)

class AP8K9D10(ConvMixer2):
    def __init__(self, out_features=512, max_pooling=False, **kwargs):
        super().__init__(dim=out_features, **kwargs)
        if max_pooling:
            self.feature_info.reduction()[-1]=1024
            self.pool = nn.Sequential(nn.AdaptiveMaxPool2d(1), nn.Flatten())
        elif max_pooling == False:
            self.feature_info.reduction()[-1]=1024
            self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        else: # max_pooling == None
            self.pool = nn.Flatten()
    def forward(self, x, return_features=False):
        x = super().forward(x, return_features)
        return x

class NGP8K9D10(NP8K9D10):
    def __init__(self, out_features=512, depth=10, kernel_size=9, patch_size=8, max_pooling=False):
        super().__init__(out_features=out_features, depth=depth, kernel_size=kernel_size, patch_size=patch_size, activation=GELU, max_pooling=max_pooling)

class NP8K5D10(NP8K9D10):
    def __init__(self, out_features=512, depth=10, kernel_size=5, patch_size=8, max_pooling=False):
        super().__init__(out_features=out_features, depth=depth, kernel_size=kernel_size, patch_size=patch_size, activation=FReLU, max_pooling=max_pooling)

class NGP8K9D20(NP8K9D10):
    def __init__(self, out_features=512, depth=20, kernel_size=9, patch_size=8, max_pooling=False):
        super().__init__(out_features=out_features, depth=depth, kernel_size=kernel_size, patch_size=patch_size, activation=GELU, max_pooling=max_pooling)

class AGP8K9D20(AP8K9D10):
    def __init__(self, out_features=512, depth=20, kernel_size=9, patch_size=8):
        super().__init__(out_features=out_features, depth=depth, kernel_size=kernel_size, patch_size=patch_size,
                         drop_path=0.1)

class AGP8K19D20(AP8K9D10):
    def __init__(self, out_features=512, depth=20, kernel_size=19, patch_size=8):
        super().__init__(out_features=out_features, depth=depth, kernel_size=kernel_size, patch_size=patch_size,
                         drop_path=0.1)

class AGP8K9D16(AP8K9D10):
    def __init__(self, out_features=512, depth=16, kernel_size=9, patch_size=8):
        super().__init__(out_features=out_features, depth=depth, kernel_size=kernel_size, patch_size=patch_size,
                         drop_path=0.1)

class AGP16K9D20(AP8K9D10):
    def __init__(self, out_features=512, depth=20, kernel_size=9, patch_size=16):
        super().__init__(out_features=out_features, depth=depth, kernel_size=kernel_size, patch_size=patch_size)

class NGP8K9D20F512(NGP8K9D20):
    def __init__(self, out_features=512, dim=512, depth=20, kernel_size=9, patch_size=8):
        super().__init__(out_features=dim, depth=depth, kernel_size=kernel_size, patch_size=patch_size, max_pooling=None)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(dim*4*4, out_features))
        self.feature_info.reduction().append(1024)
        self.feature_info.channels().append(out_features)
    def forward(self, x, return_features=False):
        y = super().forward(x, return_features=True)
        x = y[-1]
        x = F.adaptive_max_pool2d(x, 4)
        x = self.fc(x)
        if return_features:
            return y+[x]
        return x

class NGP6K9D10(NP8K9D10):
    def __init__(self, out_features=512, depth=10, kernel_size=9, patch_size=6, max_pooling=True):
        super().__init__(out_features=out_features, depth=depth, kernel_size=kernel_size, patch_size=patch_size, activation=GELU, max_pooling=max_pooling)

class NGP16K9D10(NP8K9D10):
    def __init__(self, out_features=512, depth=10, kernel_size=9, patch_size=16, max_pooling=True):
        super().__init__(out_features=out_features, depth=depth, kernel_size=kernel_size, patch_size=patch_size, activation=GELU, max_pooling=max_pooling)

class NGP8K9D20X512(NGP8K9D20):
    def __init__(self, out_features=512, dim=512, depth=20, kernel_size=9, patch_size=8):
        super().__init__(out_features=dim, depth=depth, kernel_size=kernel_size, patch_size=patch_size, max_pooling=None)
        self.fc = nn.Conv2d(dim, out_features, kernel_size=1)
    def forward(self, x, return_features=False):
        y = super().forward(x, return_features=True)
        x = self.fc(y[-1])
        x = F.adaptive_avg_pool2d(x,1)[:,:,0,0]
        if not return_features: return x
        y[-1]=x
        return y

class NGP8K9D20X1536(NGP8K9D20X512):
    def __init__(self, out_features=512):
        super().__init__(out_features=out_features, dim=1536)

class NGP8K9D20X256(NGP8K9D20X512):
    def __init__(self, out_features=512):
        super().__init__(out_features, dim=256)


if __name__=="__main__":
    net = test_model(NGP16K9D10, (3,224,224), 256)
    torch.save(net.state_dict(), "convmixer.NGP8K9D20X512.bin")
