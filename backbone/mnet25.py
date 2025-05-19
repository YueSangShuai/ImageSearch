import torch
import torch.nn as nn
import os
from .utils import FeatureInfo
from torch.nn import functional as F

class UpsamplingBilinear2d2DC(nn.ConvTranspose2d):
    def __init__(self, channel, scale_factor=2):
        assert(scale_factor==2)
        super().__init__(channel, channel, 4, 2, 1, bias=False, groups=channel)
        w=torch.FloatTensor([[-0.0033,  0.1873,  0.1872, -0.0035],
          [ 0.1874,  0.6293,  0.6292,  0.1872],
          [ 0.1873,  0.6293,  0.6291,  0.1872],
          [-0.0034,  0.1873,  0.1871, -0.0035]])
        self.weight.data[:,:]=w
        self.weight.requires_grad = False

class UpsamplingBilinear2d2DCLarge(nn.ConvTranspose2d):
    def __init__(self, channel, scale_factor=2):
        assert(scale_factor==2)
        super().__init__(channel, channel, 4, 2, 1, bias=False, groups=1)
        self.weight.data[:]=0
        w=torch.FloatTensor([[-0.0033,  0.1873,  0.1872, -0.0035],
          [ 0.1874,  0.6293,  0.6292,  0.1872],
          [ 0.1873,  0.6293,  0.6291,  0.1872],
          [-0.0034,  0.1873,  0.1871, -0.0035]])
        for i in range(channel):
            self.weight.data[i,i]=w
        self.weight.requires_grad = False

def create_upmodule(mode, in_channels):
    # 用反卷积取代双线性上取样插值，解决有些框架不支持的问题 ---
    if mode == "UCBAC":
        return UpsamplingBilinear2d2DC(in_channels, scale_factor=2)
    if mode == "UCBACLarge":
        return UpsamplingBilinear2d2DCLarge(in_channels, scale_factor=2)
    if mode == "UCBACLargeM":
        up = UpsamplingBilinear2d2DCLarge(in_channels, scale_factor=2)
        up.weight.requires_grad = True
        return up
    return nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)


class Net(nn.Module):
    def __init__(self, out_features=512, upmode='UCBA', channel=3, pretrained=True):
        super().__init__()
        self.x2 = nn.Conv2d(channel, 8, kernel_size=(3, 3), stride=(2, 2), bias=False, padding=(1, 1))
        self.x7 = nn.BatchNorm2d(8, momentum=0.9, eps=1e-05)
        self.x8 = nn.ReLU()
        self.x10 = nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), bias=False, groups=8, padding=(1, 1))
        self.x15 = nn.BatchNorm2d(8, momentum=0.9, eps=1e-05)
        self.x16 = nn.ReLU()
        self.x18 = nn.Conv2d(8, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.x23 = nn.BatchNorm2d(16, momentum=0.9, eps=1e-05)
        self.x24 = nn.ReLU()
        self.x26 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(2, 2), bias=False, groups=16, padding=(1, 1))
        self.x31 = nn.BatchNorm2d(16, momentum=0.9, eps=1e-05)
        self.x32 = nn.ReLU()
        self.x34 = nn.Conv2d(16, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.x39 = nn.BatchNorm2d(32, momentum=0.9, eps=1e-05)
        self.x40 = nn.ReLU()
        self.x42 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), bias=False, groups=32, padding=(1, 1))
        self.x47 = nn.BatchNorm2d(32, momentum=0.9, eps=1e-05)
        self.x48 = nn.ReLU()
        self.x50 = nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.x55 = nn.BatchNorm2d(32, momentum=0.9, eps=1e-05)
        self.x56 = nn.ReLU()
        self.x58 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), bias=False, groups=32, padding=(1, 1))
        self.x63 = nn.BatchNorm2d(32, momentum=0.9, eps=1e-05)
        self.x64 = nn.ReLU()
        self.x66 = nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.x71 = nn.BatchNorm2d(64, momentum=0.9, eps=1e-05)
        self.x72 = nn.ReLU()
        self.x74 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), bias=False, groups=64, padding=(1, 1))
        self.x79 = nn.BatchNorm2d(64, momentum=0.9, eps=1e-05)
        self.x80 = nn.ReLU()
        self.x82 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.x87 = nn.BatchNorm2d(64, momentum=0.9, eps=1e-05)
        self.x88 = nn.ReLU()
        self.x90 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), bias=False, groups=64, padding=(1, 1))
        self.x95 = nn.BatchNorm2d(64, momentum=0.9, eps=1e-05)
        self.x96 = nn.ReLU()
        self.x98 = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.x103 = nn.BatchNorm2d(128, momentum=0.9, eps=1e-05)
        self.x104 = nn.ReLU()
        self.x106 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), bias=False, groups=128, padding=(1, 1))
        self.x111 = nn.BatchNorm2d(128, momentum=0.9, eps=1e-05)
        self.x112 = nn.ReLU()
        self.x114 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.x119 = nn.BatchNorm2d(128, momentum=0.9, eps=1e-05)
        self.x120 = nn.ReLU()
        self.x122 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), bias=False, groups=128, padding=(1, 1))
        self.x127 = nn.BatchNorm2d(128, momentum=0.9, eps=1e-05)
        self.x128 = nn.ReLU()
        self.x130 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.x135 = nn.BatchNorm2d(128, momentum=0.9, eps=1e-05)
        self.x136 = nn.ReLU()
        self.x138 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), bias=False, groups=128, padding=(1, 1))
        self.x143 = nn.BatchNorm2d(128, momentum=0.9, eps=1e-05)
        self.x144 = nn.ReLU()
        self.x146 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.x151 = nn.BatchNorm2d(128, momentum=0.9, eps=1e-05)
        self.x152 = nn.ReLU()
        self.x154 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), bias=False, groups=128, padding=(1, 1))
        self.x159 = nn.BatchNorm2d(128, momentum=0.9, eps=1e-05)
        self.x160 = nn.ReLU()
        self.x162 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.x167 = nn.BatchNorm2d(128, momentum=0.9, eps=1e-05)
        self.x168 = nn.ReLU()
        self.x170 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), bias=False, groups=128, padding=(1, 1))
        self.x175 = nn.BatchNorm2d(128, momentum=0.9, eps=1e-05)
        self.x176 = nn.ReLU()
        self.x178 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.x183 = nn.BatchNorm2d(128, momentum=0.9, eps=1e-05)
        self.x184 = nn.ReLU()
        self.x186 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), bias=False, groups=128, padding=(1, 1))
        self.x191 = nn.BatchNorm2d(128, momentum=0.9, eps=1e-05)
        self.x192 = nn.ReLU()
        self.x194 = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.x199 = nn.BatchNorm2d(256, momentum=0.9, eps=1e-05)
        self.x200 = nn.ReLU()
        self.x202 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), bias=False, groups=256, padding=(1, 1))
        self.x207 = nn.BatchNorm2d(256, momentum=0.9, eps=1e-05)
        self.x208 = nn.ReLU()
        self.x210 = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.x215 = nn.BatchNorm2d(256, momentum=0.9, eps=1e-05)
        self.x216 = nn.ReLU()
        self.x219 = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
        self.x224 = nn.BatchNorm2d(64, momentum=0.9, eps=2e-05)
        self.x225 = nn.ReLU()
        self.x228 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.x233 = nn.BatchNorm2d(32, momentum=0.9, eps=2e-05)
        self.x236 = nn.Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.x241 = nn.BatchNorm2d(16, momentum=0.9, eps=2e-05)
        self.x242 = nn.ReLU()
        self.x245 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.x250 = nn.BatchNorm2d(16, momentum=0.9, eps=2e-05)
        self.x253 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.x258 = nn.BatchNorm2d(16, momentum=0.9, eps=2e-05)
        self.x259 = nn.ReLU()
        self.x262 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.x267 = nn.BatchNorm2d(16, momentum=0.9, eps=2e-05)
        self.x269 = nn.ReLU()
        self.x284 = nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
        self.x289 = nn.BatchNorm2d(64, momentum=0.9, eps=2e-05)
        self.x290 = nn.ReLU()
        self.x296 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.x301 = nn.BatchNorm2d(64, momentum=0.9, eps=2e-05)
        self.x302 = nn.ReLU()
        self.x305 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.x310 = nn.BatchNorm2d(32, momentum=0.9, eps=2e-05)
        self.x313 = nn.Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.x318 = nn.BatchNorm2d(16, momentum=0.9, eps=2e-05)
        self.x319 = nn.ReLU()
        self.x322 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.x327 = nn.BatchNorm2d(16, momentum=0.9, eps=2e-05)
        self.x330 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.x335 = nn.BatchNorm2d(16, momentum=0.9, eps=2e-05)
        self.x336 = nn.ReLU()
        self.x339 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.x344 = nn.BatchNorm2d(16, momentum=0.9, eps=2e-05)
        self.x346 = nn.ReLU()
        self.x361 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        self.x366 = nn.BatchNorm2d(64, momentum=0.9, eps=2e-05)
        self.x367 = nn.ReLU()
        self.x373 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.x378 = nn.BatchNorm2d(64, momentum=0.9, eps=2e-05)
        self.x379 = nn.ReLU()
        self.x382 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.x387 = nn.BatchNorm2d(32, momentum=0.9, eps=2e-05)
        self.x390 = nn.Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.x395 = nn.BatchNorm2d(16, momentum=0.9, eps=2e-05)
        self.x396 = nn.ReLU()
        self.x399 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.x404 = nn.BatchNorm2d(16, momentum=0.9, eps=2e-05)
        self.x407 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.x412 = nn.BatchNorm2d(16, momentum=0.9, eps=2e-05)
        self.x413 = nn.ReLU()
        self.x416 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.x421 = nn.BatchNorm2d(16, momentum=0.9, eps=2e-05)
        self.x423 = nn.ReLU()
        self.up1 = create_upmodule(upmode, 64)
        self.up2 = create_upmodule(upmode, 64)
        self.feature_info = FeatureInfo([32,64,64,64,out_features],[4,8,16,32,1024])
        self.load_pretrained()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(64*16, out_features)) if out_features>0 else nn.Identity()
    def forward_stage1(self, x0):
        x2 = self.x2(x0)
        x7 = self.x7(x2)
        x8 = self.x8(x7)
        x10 = self.x10(x8)
        x15 = self.x15(x10)
        x16 = self.x16(x15)
        x18 = self.x18(x16)
        x23 = self.x23(x18)
        x24 = self.x24(x23)
        x26 = self.x26(x24)
        x31 = self.x31(x26)
        x32 = self.x32(x31)
        x34 = self.x34(x32)
        x39 = self.x39(x34)
        x40 = self.x40(x39)
        x42 = self.x42(x40)
        x47 = self.x47(x42)
        x48 = self.x48(x47)
        x50 = self.x50(x48)
        x55 = self.x55(x50)
        x56 = self.x56(x55)
        return x56
    def forward_stage2(self, x56):
        x58 = self.x58(x56)
        x63 = self.x63(x58)
        x64 = self.x64(x63)
        x66 = self.x66(x64)
        x71 = self.x71(x66)
        x72 = self.x72(x71)
        x74 = self.x74(x72)
        x79 = self.x79(x74)
        x80 = self.x80(x79)
        x82 = self.x82(x80)
        x87 = self.x87(x82)
        x88 = self.x88(x87)
        return x88 
    def forward_stage3(self, x88):
        x90 = self.x90(x88)
        x95 = self.x95(x90)
        x96 = self.x96(x95)
        x98 = self.x98(x96)
        x103 = self.x103(x98)
        x104 = self.x104(x103)
        x106 = self.x106(x104)
        x111 = self.x111(x106)
        x112 = self.x112(x111)
        x114 = self.x114(x112)
        x119 = self.x119(x114)
        x120 = self.x120(x119)
        x122 = self.x122(x120)
        x127 = self.x127(x122)
        x128 = self.x128(x127)
        x130 = self.x130(x128)
        x135 = self.x135(x130)
        x136 = self.x136(x135)
        x138 = self.x138(x136)
        x143 = self.x143(x138)
        x144 = self.x144(x143)
        x146 = self.x146(x144)
        x151 = self.x151(x146)
        x152 = self.x152(x151)
        x154 = self.x154(x152)
        x159 = self.x159(x154)
        x160 = self.x160(x159)
        x162 = self.x162(x160)
        x167 = self.x167(x162)
        x168 = self.x168(x167)
        x170 = self.x170(x168)
        x175 = self.x175(x170)
        x176 = self.x176(x175)
        x178 = self.x178(x176)
        x183 = self.x183(x178)
        x184 = self.x184(x183)
        return x184 
    def forward_stage4(self, x184):
        x186 = self.x186(x184)
        x191 = self.x191(x186)
        x192 = self.x192(x191)
        x194 = self.x194(x192)
        x199 = self.x199(x194)
        x200 = self.x200(x199)
        x202 = self.x202(x200)
        x207 = self.x207(x202)
        x208 = self.x208(x207)
        x210 = self.x210(x208)
        x215 = self.x215(x210)
        x216 = self.x216(x215)
        x219 = self.x219(x216)
        x224 = self.x224(x219)
        x225 = self.x225(x224)
        return x225

    def forward_stages(self, x0):
        x56 = self.forward_stage1(x0)
        x88 = self.forward_stage2(x56)
        x184 = self.forward_stage3(x88)
        x225 = self.forward_stage4(x184)

        x284 = self.x284(x184)
        x289 = self.x289(x284)
        x290 = self.x290(x289)
        x291 = self.up1(x225)
        x293 = x290 + x291
        x296 = self.x296(x293)
        x301 = self.x301(x296)
        x302 = self.x302(x301)

        x361 = self.x361(x88)
        x366 = self.x366(x361)
        x367 = self.x367(x366)
        x368 = self.up2(x302)
        x370 = x367 + x368
        x373 = self.x373(x370)
        x378 = self.x378(x373)
        x379 = self.x379(x378)

        return x56, x379, x302, x225

    def forward(self, x, return_features=False):
        x56, x379, x302, x225 = self.forward_stages(x)
        x382 = self.x382(x379)
        x387 = self.x387(x382)
        x390 = self.x390(x379)
        x395 = self.x395(x390)
        x396 = self.x396(x395)
        x399 = self.x399(x396)
        x404 = self.x404(x399)
        x407 = self.x407(x396)
        x412 = self.x412(x407)
        x413 = self.x413(x412)
        x416 = self.x416(x413)
        x421 = self.x421(x416)
        x422 = torch.cat([x387,x404,x421], dim=1)
        x423 = self.x423(x422)

        x305 = self.x305(x302)
        x310 = self.x310(x305)
        x313 = self.x313(x302)
        x318 = self.x318(x313)
        x319 = self.x319(x318)
        x322 = self.x322(x319)
        x327 = self.x327(x322)
        x330 = self.x330(x319)
        x335 = self.x335(x330)
        x336 = self.x336(x335)
        x339 = self.x339(x336)
        x344 = self.x344(x339)
        x345 = torch.cat([x310,x327,x344], dim=1)
        x346 = self.x346(x345)

        x228 = self.x228(x225)
        x233 = self.x233(x228)
        x236 = self.x236(x225)
        x241 = self.x241(x236)
        x242 = self.x242(x241)
        x245 = self.x245(x242)
        x250 = self.x250(x245)
        x253 = self.x253(x242)
        x258 = self.x258(x253)
        x259 = self.x259(x258)
        x262 = self.x262(x259)
        x267 = self.x267(x262)
        x268 = torch.cat([x233,x250,x267], dim=1)
        x269 = self.x269(x268)
        y = self.fc(x269)
        if return_features:
            return [x56, x423, x346, x269, y]
        return y
        
    def load_pretrained(self):
        state = torch.load(os.path.join(os.path.split(__file__)[0], 'mnet25.pt'), map_location=lambda storage, loc: storage)
        my_state = self.state_dict()
        for k,v in state.items():
            if k in my_state: 
                try:
                    my_state[k][:]=v[:]
                except Exception as e:
                    print(k, e)
        self.load_state_dict(my_state)

class Net2(Net):
    def __init__(self, out_features=512, downsample=4):
        super().__init__(out_features=out_features, upmode='UCBA', pretrained=True)
        self.down1 = nn.Sequential(nn.Conv2d(3,3,downsample,downsample), nn.BatchNorm2d(3), nn.PReLU())
    def forward(self, x, return_features=False):
        # convert RGB to gray
        x = self.down1(x)
        h, w =  x.shape[2:]
        if h!=64 or w!=64:
            x = F.interpolate(x, (64,64))
        return super().forward(x, return_features=return_features)

class GrayNet2A(Net):
    def __init__(self, out_features=512, downsample=4):
        super().__init__(out_features=out_features, upmode='UCBA', pretrained=True)
        self.down1 = nn.Sequential(nn.Conv2d(1,3,downsample,downsample), nn.BatchNorm2d(3), nn.PReLU())
    def forward(self, x, return_features=False):
        x = self.down1(x.mean(1, keepdim=True))
        return super().forward(x, return_features=return_features)
    
class Net3(Net2):
    def __init__(self, out_features=512):
        super().__init__(out_features=out_features, downsample=2)