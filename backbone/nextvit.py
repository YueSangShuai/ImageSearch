from einops import rearrange
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from . import utils

class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)

    def forward(self, x):
        return self.conv(x)


class MHCA(nn.Module):
    """
    Multi-Head Convolutional Attention (MHCA)
    Uniformly set head dim to 32 in all MHCA for fast inference
    speed with various date-type on TensorRT.
    Parameters
    ----------
    channel : int
        Number of channels.
    groups : int
        Number of groups.
    """
    def __init__(self, channel):
        super(MHCA, self).__init__()
        groups = channel // 32 if channel>=32 else 16
        self.grouped_conv = nn.Conv2d(channel, channel, kernel_size = 3, padding = 1, groups = groups)
        self.bn = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace = True)
        self.point_conv = nn.Conv2d(channel, channel, kernel_size = 1)

    def forward(self, x):
        return self.point_conv(self.relu(self.bn(self.grouped_conv(x))))


class Mlp(nn.Module):
    """
    Multi layer perceptron with dropout.
    Paper: https://arxiv.org/abs/2111.11418
    """
    def __init__(self, in_features, out_features, expansion_ratio = 1):
        super().__init__()
        hidden_feature = in_features * expansion_ratio
        self.bn = nn.BatchNorm2d(in_features)
        self.relu = nn.ReLU(inplace = True)
        self.conv1 = nn.Conv2d(in_features, hidden_feature, kernel_size = 1)
        self.conv2 = nn.Conv2d(hidden_feature, out_features, kernel_size = 1)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(self.bn(x))))


class NCB(nn.Module):
    """
    Next-Convolution Block (NCB)
    Parameters
    ----------
    channel : int
        Number of channels.
    """
    def __init__(self, channel, drop_path = 0.):
        super().__init__()
        self.mhca = MHCA(channel)
        self.mlp = Mlp(channel, channel, expansion_ratio = 3)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = self.drop_path(self.mhca(x)) + x
        x = self.drop_path(self.mlp(x)) + x
        return x


class E_MHSA(nn.Module):
    """
    Effecient Multi-Head Self-Attention (E-MHSA)
    Parameters
    ----------
    dim : int
        Number of input channels.
    heads : int
        Number of heads.
    inner_dim : int
        Number of hidden channels for each head.
    dropout : float
        Dropout rate.
    stride : int
        Stride of the convolutional block.
    """
    def __init__(self, dim, heads = 8, inner_dim = 64 , dropout = 0.,stride = 2):
        super().__init__()
        self.dim = dim
        self.inner_dim = inner_dim
        self.heads = heads
        self.scaled_factor = inner_dim ** -0.5

        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm2d(dim)
        self.avg_pool = nn.AvgPool2d(stride, stride = stride)

        self.fc_q = nn.Linear(dim, self.inner_dim * self.heads)
        self.fc_k = nn.Linear(dim, self.inner_dim * self.heads)
        self.fc_v = nn.Linear(dim, self.inner_dim * self.heads)
        self.fc_o = nn.Linear(self.inner_dim * self.heads, dim)


    def forward(self, x):
        b, c, h, w = x.shape
        x = self.bn(x)
        x_reshape = x.view(b, c, h * w).permute(0, 2, 1)  # [b, h * w, c]

         # Get q, k, v
        q = self.fc_q(x_reshape)
        # [b, heads, h * w, inner_dim]
        q = q.view(b, h * w, self.heads, self.inner_dim).permute(0, 2, 1, 3).contiguous()

        k = self.fc_k(x_reshape)
        k = k.view(b, self.heads * self.inner_dim, h, w)
        k = self.avg_pool(k)
        # [b, heads, h * w, inner_dim]
        k = rearrange(k, "b (head n) h w -> b head (h w) n", head = self.heads)

        v = self.fc_v(x_reshape)
        v = v.view(b, self.heads * self.inner_dim, h, w)
        v = self.avg_pool(v)
        # [b, heads, h * w, inner_dim]
        v = rearrange(v, "b (head n) h w -> b head (h w) n", head = self.heads)

        # Attention
        attn = torch.einsum('... i d, ... j d -> ... i j', q, k) * self.scaled_factor
        attn = torch.softmax(attn, dim = -1) # [b, heads, h * w, s_h * s_w], s_h = s_h // stride

        result = torch.matmul(attn, v).permute(0, 2, 1, 3)
        result = result.contiguous().view(b, h * w, self.heads * self.inner_dim)
        result = self.fc_o(result).view(b, self.dim, h, w)
        result = result + x
        return result


class NTB(nn.Module):
    """
    Next-Transposed Convolution Block (NTB)
    Parameters
    ----------
    in_channel : int
        Number of input channels.
    out_channel : int
        Number of output channels.
    shrink_ratio: int
        Shrink ratio of the channel rection.
    """
    def __init__(self, in_channel, out_channel, shrink_ratio = 0.75, spatial_reduction_ratio = 1, drop_path = 0.):
        super().__init__()
        first_part_dim = int(out_channel * shrink_ratio)
        second_part_dim = out_channel - first_part_dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.point_conv1 = nn.Conv2d(in_channel, first_part_dim, kernel_size = 1)
        self.point_conv2 = nn.Conv2d(first_part_dim, second_part_dim, kernel_size = 1)

        self.e_mhsa = E_MHSA(first_part_dim, stride = spatial_reduction_ratio)
        self.mhca = MHCA(second_part_dim)
        self.mlp = Mlp(out_channel, out_channel, expansion_ratio = 2)

    def forward(self, x):
        x = self.point_conv1(x)
        first_part = self.drop_path(self.e_mhsa(x)) + x

        seconf_part = self.point_conv2(first_part)
        seconf_part = self.mhca(seconf_part) + seconf_part

        result = torch.cat([first_part, seconf_part], dim = 1)
        result = self.drop_path(self.mlp(result)) + result
        return result


class PatchEmbed(nn.Module):
    """
    Patch Embedding (PE)
    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super(PatchEmbed, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size = 2, stride = 2)
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(self.avgpool(x))


class Stem(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Stem, self).__init__()
        self.conv1 = Conv3x3(in_channels, 64, stride = 2)
        self.conv2 = Conv3x3(64, 32, stride = 1)
        self.conv3 = Conv3x3(32, 64, stride = 1)
        self.conv4 = Conv3x3(64, out_channels, stride = 2)

    def forward(self, x):
        return self.conv4(self.conv3(self.conv2(self.conv1(x))))


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, ncb_layers, nct_layers, repeat, spatial_reduction_ratio = 1, drop_path = 0.):
        super(Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ncb_layers = ncb_layers
        self.nct_layers = nct_layers
        self.repeat = repeat
        self.spatial_reduction_ratio = spatial_reduction_ratio
        self.drop_path = drop_path

        block = []
        for num in range(repeat):
            if num != repeat - 1:
                block += self._make_layer(self.in_channels, self.in_channels,
                                          self.ncb_layers, self.nct_layers, self.drop_path)
            else:
                block += self._make_layer(self.in_channels, self.out_channels,
                                          self.ncb_layers, self.nct_layers, self.drop_path)
        self.block = nn.Sequential(*block)

    def _make_layer(self, in_channels, out_channels, ncb_layers, nct_layers, drop_path):
        self.sub_layers = []
        for _ in range(ncb_layers):
            self.sub_layers +=  [NCB(in_channels, drop_path)]
        for _ in range(nct_layers):
            self.sub_layers +=  [NTB(in_channels, out_channels, spatial_reduction_ratio = self.spatial_reduction_ratio, drop_path = drop_path)]
        return nn.Sequential(*self.sub_layers)

    def forward(self, x):
        return self.block(x)


class NextVit(nn.Module):
    def __init__(self, in_channels = 3, stage3_repeat = 2, out_features = 512, drop_path = 0., width_mult = 1.):
        super().__init__()
        self.next_vit_channel = [int(96*width_mult), int(192*width_mult), int(384*width_mult), int(768*width_mult)]
        channels = [int(64*width_mult), int(256*width_mult), int(512*width_mult), int(1024*width_mult)]
        # Next-Vit Layer
        self.stem = Stem(in_channels, channels[0])
        self.stage1 = nn.Sequential(
            PatchEmbed(channels[0], self.next_vit_channel[0]),
            Block(self.next_vit_channel[0], self.next_vit_channel[0], 1, 0, 1, 8, drop_path),
        )
        self.stage2 = nn.Sequential(
            PatchEmbed(self.next_vit_channel[0], self.next_vit_channel[1]),
            Block(self.next_vit_channel[1], channels[1], 3, 1, 1, 4, drop_path),
        )
        self.stage3 = nn.Sequential(
            PatchEmbed(channels[1], self.next_vit_channel[2]),
            Block(self.next_vit_channel[2], channels[2], 4, 1, stage3_repeat, 2, drop_path),
        )
        self.stage4 = nn.Sequential(
            PatchEmbed(channels[2], self.next_vit_channel[3]),
            Block(self.next_vit_channel[3], channels[3], 2, 1, 1, 1, drop_path),
        )

        # Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # FC
        self.fc = nn.Sequential(
            nn.Linear(channels[3], out_features),
            nn.ReLU(inplace = True),
        )
        self.feature_info = utils.FeatureInfo([self.next_vit_channel[0], channels[1], channels[2], channels[3], out_features], 
            [8, 16, 32, 64, 1024])

    def forward(self, x, return_feature = False):
        x = self.stem(x)
        y1 = self.stage1(x)
        y2 = self.stage2(y1)
        y3 = self.stage3(y2)
        y4 = self.stage4(y3)

        x = self.avg_pool(y4)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        if return_feature:
            return [y1, y2, y3, y4, x]
        return x

class NextViT_T(NextVit):
    '''
    input size: torch.Size([1, 3, 224, 224])
    param: 7.219M flops: 825.408M   
    '''
    def __init__(self, out_features):
        super().__init__(out_features=out_features, stage3_repeat = 1, drop_path = 0.02, width_mult=0.5)

class NextViT_S(NextVit):
    '''
    input size: torch.Size([1, 3, 224, 224])
    param: 31.846M flops: 1.779G
    '''
    def __init__(self, out_features):
        super().__init__(out_features=out_features, stage3_repeat = 2, drop_path = 0.1)

class NextViT_B(NextVit):
    def __init__(self, out_features):
        super().__init__(out_features=out_features, stage3_repeat = 4, drop_path = 0.2)

class NextViT_L(NextVit):
    '''
    input size: torch.Size([1, 3, 224, 224])
    param: 54.294M flops: 3.220G
    '''
    def __init__(self, out_features):
        super().__init__(out_features=out_features, stage3_repeat = 6, drop_path = 0.2)

class NextViT_H(NextVit):
    def __init__(self, out_features):
        super().__init__(out_features=out_features, stage3_repeat = 8, drop_path = 0.3, width_mult=2)

if __name__ == '__main__':
    m=utils.test_model(NextViT_L)
    torch.save(m.state_dict(), 'nextvit_l.pth')
