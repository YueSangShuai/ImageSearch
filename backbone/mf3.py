import torch
from torch import nn
from . import utils
import torch.nn.functional as F
import math

def make_activation(activation, channel):
    return activation(channel)

class GELU(nn.GELU):
    def __init__(self, channel=512):
        super().__init__()

class ReLU(nn.ReLU):
    def __init__(self, channel=512):
        super().__init__()

class LeakyReLU(nn.LeakyReLU):
    def __init__(self, channel=512, negative_slope=0.01, inplace=False):
        super().__init__(negative_slope, inplace)

class PCone(nn.Module):
    # Parabolic-Cone
    def __init__(self, channel=512) -> None:
        super().__init__()
    def forward(self, x):
        return x*(2-x)

class SELayer(nn.Module):
    """Implementation of the Squeeze-Excitaion layer from https://arxiv.org/abs/1709.01507"""
    def __init__(self, inplanes, squeeze_ratio=8, activation=nn.PReLU, size=None):
        super(SELayer, self).__init__()
        assert squeeze_ratio >= 1
        assert inplanes > 0
        if size is not None:
            self.global_avgpool = nn.AvgPool2d(size)
        else:
            self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(inplanes, int(inplanes / squeeze_ratio), kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(int(inplanes / squeeze_ratio), inplanes, kernel_size=1, stride=1)
        self.relu = make_activation(activation, inplanes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.sigmoid(out)
        return x * out

class InvertedResidual(nn.Module):
    """Implementation of the modified Inverted residual block"""
    def __init__(self, in_channels, out_channels, stride, expand_ratio, outp_size=None, activation=nn.PReLU, se=True):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        self.inv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(in_channels * expand_ratio),
            make_activation(activation, in_channels * expand_ratio),

            nn.Conv2d(in_channels * expand_ratio, in_channels * expand_ratio, 3, stride, 1,
                      groups=in_channels * expand_ratio, bias=False),
            nn.BatchNorm2d(in_channels * expand_ratio),
            make_activation(activation, in_channels * expand_ratio),

            nn.Conv2d(in_channels * expand_ratio, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            SELayer(out_channels, 8, activation) if se else nn.Identity()
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.inv_block(x)

        return self.inv_block(x)

def init_block(in_channels, out_channels, stride, activation=nn.PReLU):
    """Builds the first block of the MobileFaceNet"""
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        make_activation(activation, out_channels)
    )


class MobileFaceNet(nn.Module):
    def __init__(self, out_features=512, width_multiplier=1., channel=3, size=[112,112], inverted_residual_setting=None, 
        activation=utils.FReLU, se=False, extra_conv=True, gap="mean", first_channel_num=64, stride=1):
        super(MobileFaceNet, self).__init__()
        self.size = size
        self.channel = channel

        # Set up of inverted residual blocks
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [2, 64, 5, 2],
                [4, 128, 1, 2],
                [2, 128, 6, 1],
                [4, 128, 1, 2],
                [2, 128, 2, 1]
            ]

        last_channel_num = 512 if extra_conv else out_features
        steam = [init_block(channel, first_channel_num, 2, activation)]

        steam.append(nn.Conv2d(first_channel_num, first_channel_num, 3, stride, 1,
                                       groups=first_channel_num, bias=False))
        steam.append(nn.BatchNorm2d(first_channel_num))
        steam.append(make_activation(activation, first_channel_num))
        self.steam = nn.Sequential(*steam)

        # Inverted Residual Blocks
        in_channel_num = first_channel_num
        stage=[]
        self.stages=[]
        stage_index=1
        strides, channels = [], []
        stride *= 2 
        for ind_, (t, c, n, s) in enumerate(inverted_residual_setting):
            output_channel = int(c * width_multiplier)
            for i in range(n):
                if i == 0:
                    stage.append(InvertedResidual(in_channel_num, output_channel,
                                                          s, t, activation=activation, se=se))
                else:
                    stage.append(InvertedResidual(in_channel_num, output_channel,
                                                          1, t, activation=activation, se=se))
                in_channel_num = output_channel
            stride *= s
            if s>1 and ind_<len(inverted_residual_setting)-1:
                stage = nn.Sequential(*stage)
                setattr(self, 'stage'+str(stage_index), stage)
                stage_index += 1
                strides.append(stride)
                channels.append(output_channel)
                self.stages.append(stage)
                stage = []
        # 1x1 expand block
        stage.append(nn.Sequential(nn.Conv2d(in_channel_num, last_channel_num, 1, 1, 0, bias=False),
                                           nn.BatchNorm2d(last_channel_num),
                                           make_activation(activation, last_channel_num)))
        stage = nn.Sequential(*stage)
        setattr(self, 'stage'+str(stage_index), stage)
        self.stages.append(stage)
        strides.append(stride)
        channels.append(last_channel_num)
        strides.append(1024)
        channels.append(out_features)
        self.feature_info = utils.FeatureInfo(channels, strides)
        # Depth-wise pooling
        self.dw_pool = nn.AdaptiveAvgPool2d(1) if gap == "mean" else \
            nn.AdaptiveMaxPool2d(1)  if gap == "max" else nn.Identity()
        self.conv1_extra = nn.Linear(last_channel_num, out_features) if extra_conv else None
        if not extra_conv: print("last_channel_num:", last_channel_num)
        self.init_weights()
    def forward(self, x, return_features=False):
        x = self.steam(x)
        y = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            y.append(x)
        x = self.dw_pool(x)
        x = x.squeeze(3).squeeze(2)
        if self.conv1_extra is not None:
            x = self.conv1_extra(x)
        y.append(x)
        return y if return_features else x

    def init_weights(self):
        """Initializes weights of the model before training"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class Net1(MobileFaceNet):
    # input size: torch.Size([1, 3, 112, 112])
    # param: 1.029M flops: 227.932M
    # time(ms): 3.8525819778442383
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, width_multiplier=1.0, **wargs)

class Net2(MobileFaceNet):
    # input size: torch.Size([1, 3, 112, 112])
    # param: 3.599M flops: 760.523M
    # time(ms): 3.9347410202026367
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, width_multiplier=2.0, **wargs)

class ANet2(MobileFaceNet):
    # input size: torch.Size([1, 3, 112, 112])
    # param: 6.107M flops: 1.267G, 25181582 Bytes
    # time(ms): 10.521578788757324
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, width_multiplier=2.0, inverted_residual_setting = [
                # t, c, n, s
                [2, 64, 8, 2],
                [4, 128, 1, 2],
                [2, 128, 12, 1],
                [4, 128, 1, 2],
                [2, 128, 4, 1]
            ], **wargs)


class ANet2C(ANet2):
    def __init__(self, out_features=512):
        super().__init__(out_features, activation=PCone)

class ANet2P(MobileFaceNet):
    '''
    input size: torch.Size([1, 3, 112, 112])
    param: 5.747M flops: 1.265G
    time(ms): 15.228128433227539
    '''
    def __init__(self, out_features, **wargs):
        width_multiplier = wargs.get('width_multiplier', 2.0)
        if "width_multiplier" in wargs: del wargs['width_multiplier']
        super().__init__(out_features, width_multiplier=width_multiplier, inverted_residual_setting = [
                # t, c, n, s
                [2, 64, 8, 2],
                [4, 128, 1, 2],
                [2, 128, 12, 1],
                [4, 128, 1, 2],
                [2, 128, 4, 1]
            ], activation=nn.PReLU, **wargs)

class ANet2L(MobileFaceNet):
    '''
    input size: torch.Size([1, 3, 112, 112])
    param: 5.747M flops: 1.265G
    time(ms): 15.228128433227539
    '''
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, width_multiplier=2.0, inverted_residual_setting = [
                # t, c, n, s
                [2, 64, 8, 2],
                [4, 128, 1, 2],
                [2, 128, 12, 1],
                [4, 128, 1, 2],
                [2, 128, 4, 1]
            ], activation=LeakyReLU, **wargs)
        
class ANet2R(MobileFaceNet):
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, width_multiplier=2.0, inverted_residual_setting = [
                # t, c, n, s
                [2, 64, 8, 2],
                [4, 128, 1, 2],
                [2, 128, 12, 1],
                [4, 128, 1, 2],
                [2, 128, 4, 1]
            ], activation=ReLU, **wargs)

class ANet2G(MobileFaceNet):
    """
    input size: torch.Size([1, 3, 112, 112])
    param: 5.747M flops: 1.265G
    time(ms): 31.79990291595459
    """
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, width_multiplier=2.0, inverted_residual_setting = [
                # t, c, n, s
                [2, 64, 8, 2],
                [4, 128, 1, 2],
                [2, 128, 12, 1],
                [4, 128, 1, 2],
                [2, 128, 4, 1]
            ], activation=GELU, **wargs)

class ANet2H(ANet2):
    def forward(self, x):
        n,c,h,w=x.shape
        noise = torch.rand(n//2,c,h//2,w)
        x[:n//2,:,:h//2,:]=noise
        return super().forward(x)


class NNet2(MobileFaceNet):
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, width_multiplier=2.0, stride=2, inverted_residual_setting = [
                # t, c, n, s
                [2, 64, 8, 2],
                [4, 128, 1, 2],
                [2, 128, 12, 1],
                [4, 128, 1, 2],
                [2, 128, 4, 1]
            ], **wargs)

class NNet2F768(MobileFaceNet):
    def __init__(self, out_features, **wargs):
        super().__init__(out_features=768, width_multiplier=2.0, stride=2, inverted_residual_setting = [
                # t, c, n, s
                [2, 64, 8, 2],
                [4, 128, 1, 2],
                [2, 128, 12, 1],
                [4, 128, 1, 2],
                [2, 128, 4, 1]
            ], extra_conv=False, **wargs)
        self.out = nn.Linear(768, out_features)
        self.feature_info.channels().append(out_features)
        self.feature_info.reduction().append(1024)
    def forward(self, x, return_features=False):
        y = super().forward(x, return_features)
        y.append(self.out(y[-1]))
        return y

class NNet2P(MobileFaceNet):
    '''
    input size: torch.Size([1, 3, 112, 112])
    param: 5.747M flops: 1.265G
    time(ms): 15.228128433227539
    '''
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, width_multiplier=2.0, stride=2, inverted_residual_setting = [
                # t, c, n, s
                [2, 64, 8, 2],
                [4, 128, 1, 2],
                [2, 128, 12, 1],
                [4, 128, 1, 2],
                [2, 128, 4, 1]
            ], activation=nn.PReLU, **wargs)

class NNet2L(MobileFaceNet):
    '''
    input size: torch.Size([1, 3, 112, 112])
    param: 5.747M flops: 1.265G
    time(ms): 15.228128433227539
    '''
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, width_multiplier=2.0, stride=2, inverted_residual_setting = [
                # t, c, n, s
                [2, 64, 8, 2],
                [4, 128, 1, 2],
                [2, 128, 12, 1],
                [4, 128, 1, 2],
                [2, 128, 4, 1]
            ], activation=LeakyReLU, **wargs)
        
class ANet1(MobileFaceNet):
    # input size: torch.Size([1, 3, 112, 112])
    # param: 1.303M flops: 247.580M
    # time(ms): 5.994439125061035
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, width_multiplier=1.0, inverted_residual_setting = [
                # t, c, n, s
                [2, 64,  4, 2],
                [4, 128, 1, 2],
                [2, 128, 8, 1],
                [4, 128, 1, 2],
                [2, 128, 4, 1]
            ], **wargs)

class ANet3(MobileFaceNet):
# input size: torch.Size([1, 3, 112, 112])
# param: 17.543M flops: 3.530G
# time(ms): 13.867926597595215
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, width_multiplier=3.0, inverted_residual_setting = [
                # t, c, n, s
                [2, 64, 10, 2],
                [4, 128, 1, 2],
                [2, 128, 16, 1],
                [4, 128, 1, 2],
                [2, 128, 6, 1]
            ], **wargs)

class ANet3P(ANet3):
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, activation=nn.PReLU, **wargs)

class ANetA(MobileFaceNet):
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, width_multiplier=1.0, inverted_residual_setting = [
                # t, c, n, s
                [2, 64,  4, 2],
                [4, 128, 1, 2],
                [2, 128, 4, 1],
                [4, 128, 1, 2],
                [2, 128, 6, 1]
            ], **wargs)
            
class ANet1P(ANet1):
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, activation=nn.PReLU, **wargs)

class ANet1L(ANet1):
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, activation=LeakyReLU, **wargs)

class ANet3P(ANet3):
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, activation=nn.PReLU, **wargs)

class ANet0(MobileFaceNet):
    # input size: torch.Size([1, 3, 112, 112])
    # param: 462.167K flops: 99.280M
    # time(ms): 4.000091552734375
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, width_multiplier=0.6, inverted_residual_setting = [
                # t, c, n, s
                [2, 64,  3, 2],
                [4, 128, 1, 2],
                [2, 128, 6, 1],
                [4, 128, 1, 2],
                [2, 128, 3, 1]
            ], **wargs)

class ANet0P(ANet0):
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, activation=nn.PReLU, **wargs)

class ANet0L(ANet0):
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, activation=LeakyReLU, **wargs)

class Tiny(MobileFaceNet):
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, first_channel_num=8, extra_conv=False, inverted_residual_setting = [
                # t, c, n, s
                [2, 12, 3, 2],
                [4, 24, 1, 2],
                [2, 48, 6, 1],
                [4, 64, 1, 2],
                [2, 96, 3, 1]
            ], **wargs)

class TinyP(Tiny):
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, activation=nn.PReLU, **wargs)

class TinyL(Tiny):
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, activation=LeakyReLU, **wargs)

class Nano(MobileFaceNet):
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, first_channel_num=8, extra_conv=False, inverted_residual_setting = [
                # t, c, n, s
                [2, 8, 3, 2],
                [4, 16, 1, 2],
                [2, 32, 6, 1],
                [4, 64, 1, 2],
                [2, 96, 3, 1]
            ], **wargs)

class NanoP(Nano):
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, activation=nn.PReLU, **wargs)

class NanoL(Nano):
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, activation=LeakyReLU, **wargs)

class Nano2(MobileFaceNet):
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, first_channel_num=8, extra_conv=False, inverted_residual_setting = [
                # t, c, n, s
                [2, 8, 3, 2],
                [4, 16, 1, 2],
                [2, 32, 6, 1],
                [4, 64, 1, 2],
                [2, 96, 3, 1]
            ], stride=2, **wargs)

class ENano2(MobileFaceNet):
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, first_channel_num=8, extra_conv=True, inverted_residual_setting = [
                # t, c, n, s
                [2, 8, 3, 2],
                [4, 16, 1, 2],
                [2, 32, 6, 1],
                [4, 64, 1, 2],
                [2, 96, 3, 1]
            ], stride=2, **wargs)
        
class Nano2P(Nano2):
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, activation=nn.PReLU, **wargs)

class Nano2L(Nano2):
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, activation=LeakyReLU, **wargs)

class ENano2L(ENano2):
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, activation=LeakyReLU, **wargs)

class Nano2C(Nano2):
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, activation=PCone, **wargs)

class BNet2(ANet2):
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, extra_conv=False, **wargs)

class BNet2P(ANet2P):
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, extra_conv=False, **wargs)

class BNet3P(ANet3P):
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, extra_conv=False, **wargs)

class BNet1(ANet1):
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, extra_conv=False, **wargs)

class BNet1P(ANet1P):
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, extra_conv=False, **wargs)

class BNet0PMax(ANet0P):
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, extra_conv=False, gap='max', **wargs)

class BNet1PMax(ANet1P):
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, extra_conv=False, gap='max', **wargs)

class BNet2PMax(ANet2P):
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, extra_conv=False, gap='max', **wargs)

class BNet1PMax2(ANet1P):
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, extra_conv=False, gap='max', **wargs)
    def forward(self, x, return_features=False):
        y = super().forward(x, return_features)
        if return_features: return y
        return F.sigmoid(y)

class BNet1PMax3(ANet1P):
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, extra_conv=False, gap='max', **wargs)
    def forward(self, x, return_features=False):
        y = super().forward(x, return_features)
        if return_features: return y
        return y.clip(0,1)

class BNet2PMax3(ANet2P):
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, extra_conv=False, gap='max', **wargs)
    def forward(self, x, return_features=False):
        y = super().forward(x, return_features)
        if return_features: return y
        return y.clip(0,1)



class BNet3PMax3(ANet3):
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, activation=nn.PReLU, extra_conv=False, gap='max', **wargs)
    def forward(self, x, return_features=False):
        y = super().forward(x, return_features)
        if return_features: return y
        return y.clip(0,1)

if __name__ == '__main__':
    utils.test_model(ANet1)
