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
    def __init__(self, out_features=512, width_multiplier=1., channel=3, size=(112,112), inverted_residual_setting=None, 
        activation=utils.FReLU, se=False, extra_conv=True):
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

        first_channel_num = 64
        last_channel_num = 512
        self.features = [init_block(channel, first_channel_num, 2, activation)]

        self.features.append(nn.Conv2d(first_channel_num, first_channel_num, 3, 1, 1,
                                       groups=first_channel_num, bias=False))
        self.features.append(nn.BatchNorm2d(64))
        self.features.append(make_activation(activation, 64))

        # Inverted Residual Blocks
        in_channel_num = first_channel_num
        size_h, size_w = self.get_input_res()
        size_h, size_w = size_h // 2, size_w // 2
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_multiplier)
            for i in range(n):
                if i == 0:
                    size_h, size_w = size_h // s, size_w // s
                    self.features.append(InvertedResidual(in_channel_num, output_channel,
                                                          s, t, outp_size=(size_h, size_w), activation=activation, se=se))
                else:
                    self.features.append(InvertedResidual(in_channel_num, output_channel,
                                                          1, t, outp_size=(size_h, size_w), activation=activation, se=se))
                in_channel_num = output_channel

        # 1x1 expand block
        self.features.append(nn.Sequential(nn.Conv2d(in_channel_num, last_channel_num, 1, 1, 0, bias=False),
                                           nn.BatchNorm2d(last_channel_num),
                                           make_activation(activation, last_channel_num)))
        self.features = nn.Sequential(*self.features)
        self.feature_info = utils.FeatureInfo([], [])
        # Depth-wise pooling
        k_size = (self.get_input_res()[0] // 16, self.get_input_res()[1] // 16)
        self.dw_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1_extra = nn.Linear(last_channel_num, out_features) if extra_conv else nn.Identity()
        if not extra_conv: print("last_channel_num:", last_channel_num)
        self.init_weights()
    def forward(self, x, return_features=False):
        x = self.features(x)
        y = self.dw_pool(x)
        y = y[:,:,0,0]
        y = self.conv1_extra(y)
        if return_features:
            return [x,y]
        return y

    def get_input_res(self):
        return self.size

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

class IRFaceNet1(MobileFaceNet):
    # input size: torch.Size([1, 1, 112, 112])
    # param: 152.148K flops: 42.982M
    # time(ms): 6.273460388183594
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, width_multiplier=0.25, channel=1, inverted_residual_setting = [
                # t, c, n, s
                [2, 64, 3, 2],
                [4, 128, 1, 2],
                [2, 128, 3, 1],
                [4, 128, 1, 2],
                [2, 128, 2, 1]
            ], **wargs)

class IRFaceNet2(MobileFaceNet):
    # input size: torch.Size([1, 1, 112, 112])
    # param: 171.108K flops: 48.270M
    # time(ms): 6.273460388183594
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, width_multiplier=0.25, channel=1, **wargs)


class IRFaceNet3(MobileFaceNet):
    # input size: torch.Size([1, 1, 112, 112])
    # param: 354.630K flops: 85.656M
    # time(ms): 6.273460388183594
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, width_multiplier=0.5, channel=1, **wargs)

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

class ANet2P(MobileFaceNet):
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
            ], activation=nn.PReLU, **wargs)

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

class ANet1P(ANet1):
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

class BNet2(ANet2):
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, extra_conv=False, **wargs)

class BNet2P(ANet2P):
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, extra_conv=False, **wargs)

class BNet1(ANet1):
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, extra_conv=False, **wargs)

class BNet1P(ANet1P):
    def __init__(self, out_features, **wargs):
        super().__init__(out_features, extra_conv=False, **wargs)

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

class Net144(MobileFaceNet):
    # input size: torch.Size([2, 1, 144, 144])
    # param: 1.044M flops: 762.606M/2
    # time(ms): 44.18373107910156
    def __init__(self, out_features):
        super().__init__(out_features, width_multiplier=1.0, size=(144,144), channel=1)

class Net144S(MobileFaceNet):
    # input size: torch.Size([2, 1, 144, 144])
    # param: 371.062K flops: 296.855M/2
    # time(ms): 3.064136505127
    def __init__(self, out_features):
        super().__init__(out_features, width_multiplier=0.5, size=(144,144), channel=1)

class Net144T(MobileFaceNet):
    # input size: torch.Size([2, 1, 144, 144])
    # param: 168.565K flops: 150.567M
    # time(ms): 3.557705879211426
    def __init__(self, out_features):
        super().__init__(out_features, width_multiplier=0.25, size=(144,144), channel=1, inverted_residual_setting = [
                # t, c, n, s
                [2, 64, 3, 2],
                [4, 128, 1, 2],
                [2, 128, 3, 1],
                [4, 128, 1, 2],
                [2, 128, 2, 1]
            ])

class Net192x176(MobileFaceNet):
    # input size: torch.Size([1, 1, 192, 176])
    # param: 1.070M flops: 603.982M
    # time(ms): 5.743598937988281
    def __init__(self, out_features):
        super().__init__(out_features, width_multiplier=1.0, size=(192,176), channel=1)

class Net192x176S2(MobileFaceNet):
    # input size: torch.Size([1, 1, 192, 176])
    # param: 3.641M flops: 2.038G
    # time(ms): 5.743598937988281
    def __init__(self, out_features):
        super().__init__(out_features, width_multiplier=2.0, size=(192,176), channel=1)

class Net192x176C4(MobileFaceNet):
    # input size: torch.Size([1, 1, 192, 176])
    # param: 3.641M flops: 2.038G
    # time(ms): 5.743598937988281
    def __init__(self, out_features):
        super().__init__(out_features, width_multiplier=4.0, size=(192,176), channel=3)

class Net192x176S4(MobileFaceNet):
    # input size: torch.Size([1, 1, 192, 176])
    # param: 13.678M flops: 7.659G
    # time(ms): 5.743598937988281
    def __init__(self, out_features):
        super().__init__(out_features, width_multiplier=4.0, size=(192,176), channel=1)

class Net192x176S05(MobileFaceNet):
    # input size: torch.Size([1, 1, 192, 176])
    # param: 397.126K flops: 230.637M
    # time(ms): 4.568338394165039
    def __init__(self, out_features):
        super().__init__(out_features, width_multiplier=0.5, size=(192,176), channel=1)

class Net192x176S025(MobileFaceNet):
    # input size: torch.Size([1, 1, 192, 176])
    # param: 213.604K flops: 129.940M
    # time(ms): 3.6984920501708984
    def __init__(self, out_features):
        super().__init__(out_features, width_multiplier=0.25, size=(192,176), channel=1)

class Net192x192S025(MobileFaceNet):
    # input size: torch.Size([1, 1, 192, 192])
    # param: 164.452K flops: 35.479M
    # time(ms): 2.242732048034668
    def __init__(self, out_features):
        super().__init__(out_features, width_multiplier=0.25, size=(192//2,192//2), channel=1)
        self.size = (192,192)
    def forward(self, x):
        x = F.avg_pool2d(x, 2)
        return super().forward(x)

class Net128x96C4(MobileFaceNet):
    # input size: torch.Size([1, 3, 128, 96])
    # param: 13.636M flops: 2.789G
    # time(ms): 5.743598937988281
    def __init__(self, out_features):
        super().__init__(out_features, width_multiplier=4.0, size=(128,96), channel=3)
        self.full_size = (144,108)

class Net128x96C(MobileFaceNet):
    # input size: torch.Size([1, 3, 128, 96])
    def __init__(self, out_features):
        super().__init__(out_features, width_multiplier=1, size=(128,96), channel=3)
        self.full_size = (144,108)

if __name__ == '__main__':
    utils.test_model(ANet2)
