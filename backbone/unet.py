import torch
from torch import clamp_min
import torch.nn as nn
import torch.nn.functional as F
from . import utils

def add_conv_stage(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, useBN=True):
  return nn.Sequential(
    nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
    nn.BatchNorm2d(dim_out),
    nn.LeakyReLU(0.1),
  )

def upsample(ch_coarse, ch_fine):
  return nn.Sequential(
    nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
    nn.ReLU()
  )

class UNet_(nn.Module):
  def __init__(self, inplane=1, outplane=2, scale=2, deep=4, connections=[1,1,1,1,1,1,1,1,1]):
    super().__init__()
    self.deep = deep
    oplane = scale*2**(deep)
    self.outplane=oplane
    self.connections = connections+[0]*deep
    left = []
    right = []
    left_in = scale
    left.append(add_conv_stage(inplane, left_in))
    for i in range(deep-1):
        left.append(add_conv_stage(left_in, left_in*2))
        right.insert(0, upsample(left_in*2, left_in))
        right.insert(1, add_conv_stage(left_in+(left_in if connections[i]==1 else 0), 
            left_in if i>0 else outplane))
        left_in *= 2
    for i, k in enumerate(left): 
        setattr(self, 'left_'+str(i), k)
    for i, k in enumerate(right): 
        setattr(self, 'right_'+str(i), k)
    self.left=left
    self.right = right
  def forward(self, x):
    y0=[]
    for d in range(self.deep-1):
        x = self.left[d](x)
        y0.append(x)
        x = F.max_pool2d(x,2)
    x = self.left[self.deep-1](x)
    y0.append(x)
    y1=[x]
    for d in range(self.deep-1):
        x = self.right[d*2](x)
        if self.connections[self.deep-2-d]==1:
            x0 = y0[self.deep-2-d]
            x = self.right[d*2+1](torch.cat([x0,x], dim=1))
        else:
            x = self.right[d*2+1](x)
        y1.append(x)
    return y1

class UNetClamp(nn.Module):
    def __init__(self, out_features, scale=4, deep=6, connections=[1,1,1,1,0,0,0], clamp=False, max_pool=True, inplane=3):
        super().__init__()
        self.net = UNet_(inplane=inplane, outplane=out_features, scale=scale, deep=deep, connections=connections)
        self.clamp = clamp
        self.max_pool = nn.Sequential(nn.AdaptiveMaxPool2d(1), nn.Flatten()) if max_pool else \
            nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten()) if max_pool==False else nn.Identity()
    def forward(self, x, return_features=False):
        y = self.net(x)
        x = torch.sigmoid(y[-1]) if self.clamp==False else y[-1].clamp(0,1) if self.clamp else y[-1].clamp_min(0) 
        x = self.max_pool(x)
        if return_features:
            y.append(x)
            return y
        return x

class UNetS4(UNetClamp):
    def __init__(self, out_features):
        super().__init__(out_features=out_features, scale=4, deep=6, connections=[0,1,0,1,0,1,0])

class UNetS8(UNetClamp):
    def __init__(self, out_features):
        super().__init__(out_features=out_features, scale=8, deep=6, connections=[0,1,0,1,0,1,0])

class UNetS8m(UNetClamp):
    def __init__(self, out_features):
        super().__init__(out_features=out_features, scale=8, deep=6, connections=[0,1,0,1,0,1,0], clamp=None)

class UNetS8a(UNetClamp):
    def __init__(self, out_features):
        super().__init__(out_features=out_features, scale=8, deep=6, connections=[0,1,0,1,0,1,0], clamp=None, max_pool=False)

class UNetS8n(UNetClamp):
    def __init__(self, out_features):
        super().__init__(out_features=out_features, scale=8, deep=6, connections=[0,1,0,1,0,1,0], clamp=None, max_pool=None)

class UNetS32m(UNetClamp):
    def __init__(self, out_features):
        super().__init__(out_features=out_features, scale=32, deep=6, connections=[0,1,0,1,0,1,0], clamp=None)

class UNetS64m(UNetClamp):
    def __init__(self, out_features):
        super().__init__(out_features=out_features, scale=64, deep=6, connections=[0,1,0,1,0,1,0], clamp=None)


class UClamp(nn.Module):
    def __init__(self, out_features, scale=4, deep=6, connections=[1,1,1,1,0,0,0], clamp=False, max_pool=True, inplane=3):
        super().__init__()
        self.net = UNet_(inplane=inplane, outplane=scale, scale=scale, deep=deep, connections=connections)
        self.clamp = clamp
        self.max_pool = nn.Identity() if (max_pool==None or out_features==0) else nn.Sequential(
            nn.Conv2d(scale, out_features, 4, 4, 0, bias=False), nn.BatchNorm2d(out_features), nn.ReLU(),
            nn.AdaptiveMaxPool2d(1) if max_pool else \
                nn.AdaptiveAvgPool2d(1), nn.Flatten())
        s = scale
        self.feature_info = utils.FeatureInfo(
            [s, 2*s, 4*s, 8*s, 16*s, 32*s, 64*s, 128*s, 256*s][:len(connections)-1]+[out_features],
            [1, 2, 4, 8, 16, 32, 64, 128, 256][:len(connections)-1]+ [1024]
        )
    def forward(self, x, return_features=False):
        y = self.net(x)[::-1]
        x = self.max_pool(y[0])
        if return_features:
            y.append(x)
            return y
        return x

class UClamp2(UClamp):
    def __init__(self, out_features):
        super().__init__(out_features, scale=4, deep=4, connections=[1,1,1,1,0,0,0], clamp=False, max_pool=True, inplane=3)


if __name__ == '__main__':
    utils.test_model(UNetS4)
