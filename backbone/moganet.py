import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import GELU, ReLU, FReLU, SiLU, FeatureInfo, \
    force_fp32, DropPath, PatchEmbed, constant_init, trunc_normal_init

BN = nn.BatchNorm2d

class ElementScale(nn.Module):
    """A learnable element-wise scaler."""

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale


class ChannelAggregationFFN(nn.Module):
    """An implementation of FFN with Channel Aggregation.
    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        kernel_size (int): The depth-wise conv kernel size as the
            depth-wise convolution. Defaults to 3.
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 kernel_size=3,
                 act_cfg=GELU,
                 ffn_drop=0.):
        super().__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg(feedforward_channels)

        self.fc1 = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=self.feedforward_channels,
            kernel_size=1)
        self.dwconv = nn.Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=self.feedforward_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=self.feedforward_channels)
        self.act = act_cfg(feedforward_channels)
        self.fc2 = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=embed_dims,
            kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)

        self.decompose = nn.Conv2d(
            in_channels=self.feedforward_channels,  # C -> 1
            out_channels=1, kernel_size=1,
        )
        self.sigma = ElementScale(
            self.feedforward_channels, init_value=1e-5, requires_grad=True)
        self.decompose_act = act_cfg(feedforward_channels)

    def feat_decompose(self, x):
        # x_d: [B, C, H, W] -> [B, 1, H, W]
        x = x + self.sigma(x - self.decompose_act(self.decompose(x)))
        return x

    def forward(self, x):
        # proj 1
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        # proj 2
        x = self.feat_decompose(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiOrderDWConv(nn.Module):
    """Multi-order Features with Dilated DWConv Kernel.
    Args:
        embed_dims (int): Number of input channels.
        dw_dilation (list): Dilations of three DWConv layers.
        channel_split (list): The raletive ratio of three splited channels.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 dw_dilation=[1, 2, 3,],
                 channel_split=[1, 3, 4,]):
        super().__init__()

        self.split_ratio = [i / sum(channel_split) for i in channel_split]
        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2
        self.embed_dims = embed_dims
        assert len(dw_dilation) == len(channel_split) == 3
        assert 1 <= min(dw_dilation) and max(dw_dilation) <= 3
        assert embed_dims % sum(channel_split) == 0

        # basic DW conv
        self.DW_conv0 = nn.Conv2d(
            in_channels=self.embed_dims,
            out_channels=self.embed_dims,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[0]) // 2,
            groups=self.embed_dims,
            stride=1, dilation=dw_dilation[0],
        )
        # DW conv 1
        self.DW_conv1 = nn.Conv2d(
            in_channels=self.embed_dims_1,
            out_channels=self.embed_dims_1,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[1]) // 2,
            groups=self.embed_dims_1,
            stride=1, dilation=dw_dilation[1],
        )
        # DW conv 2
        self.DW_conv2 = nn.Conv2d(
            in_channels=self.embed_dims_2,
            out_channels=self.embed_dims_2,
            kernel_size=7,
            padding=(1 + 6 * dw_dilation[2]) // 2,
            groups=self.embed_dims_2,
            stride=1, dilation=dw_dilation[2],
        )
        # a channel convolution
        self.PW_conv = nn.Conv2d(  # point-wise convolution
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1)

    def forward(self, x):
        x_0 = self.DW_conv0(x)
        x_1 = self.DW_conv1(
            x_0[:, self.embed_dims_0: self.embed_dims_0+self.embed_dims_1, ...])
        x_2 = self.DW_conv2(
            x_0[:, self.embed_dims-self.embed_dims_2:, ...])
        x = torch.cat([
            x_0[:, :self.embed_dims_0, ...], x_1, x_2], dim=1)
        x = self.PW_conv(x)
        return x


class MultiOrderGatedAggregation(nn.Module):
    """Spatial Block with Multi-order Gated Aggregation.
    Args:
        embed_dims (int): Number of input channels.
        attn_dw_dilation (list): Dilations of three DWConv layers.
        attn_channel_split (list): The raletive ratio of splited channels.
        attn_act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='GELU').
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_act_cfg=SiLU,
                 attn_force_fp32=False):
        super().__init__()

        self.embed_dims = embed_dims
        self.attn_force_fp32 = attn_force_fp32
        self.proj_1 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.gate = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.value = MultiOrderDWConv(
            embed_dims,
            dw_dilation=attn_dw_dilation,
            channel_split=attn_channel_split,
        )
        self.proj_2 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

        # activation for gating and value
        self.act_value = attn_act_cfg()
        self.act_gate = attn_act_cfg()

        # decompose
        self.sigma = ElementScale(
            embed_dims, init_value=1e-5, requires_grad=True)

    def feat_decompose(self, x):
        x = self.proj_1(x)
        # x_d: [B, C, H, W] -> [B, C, 1, 1]
        x_d = F.adaptive_avg_pool2d(x, output_size=1)
        x = x + self.sigma(x - x_d)
        x = self.act_value(x)
        return x

    @force_fp32()
    def forward_gating(self, g, v):
        g = g.to(torch.float32)
        v = v.to(torch.float32)
        return self.proj_2(self.act_gate(g) * self.act_gate(v))

    def forward(self, x):
        shortcut = x.clone()
        # proj 1x1
        x = self.feat_decompose(x)
        # gating and value branch
        g = self.gate(x)
        v = self.value(x)
        # aggregation
        if not self.attn_force_fp32:
            x = self.proj_2(self.act_gate(g) * self.act_gate(v))
        else:
            x = self.forward_gating(self.act_gate(g), self.act_gate(v))
        x = x + shortcut
        return x


class MogaBlock(nn.Module):
    """A block of MogaNet.
    Args:
        embed_dims (int): Number of input channels.
        ffn_ratio (float): The expansion ratio of feedforward network hidden
            layer channels. Defaults to 4.
        drop_rate (float): Dropout rate after embedding. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.1.
        act_cfg (dict, optional): The activation config for projections and FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='BN')``.
        init_value (float): Init value for Layer Scale. Defaults to 1e-5.
        attn_dw_dilation (list): Dilations of three DWConv layers.
        attn_channel_split (list): The raletive ratio of splited channels.
        attn_act_cfg (str): The activation config for the gating branch.
            Default: dict(type='SiLU').
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 ffn_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=GELU,
                 norm_cfg=nn.BatchNorm2d,
                 init_value=1e-5,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_act_cfg=SiLU,
                 attn_force_fp32=False):
        super().__init__()
        self.out_channels = embed_dims

        self.norm1 = norm_cfg(embed_dims)

        # spatial attention
        self.attn = MultiOrderGatedAggregation(
            embed_dims,
            attn_dw_dilation=attn_dw_dilation,
            attn_channel_split=attn_channel_split,
            attn_act_cfg=attn_act_cfg,
            attn_force_fp32=attn_force_fp32,
        )
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.norm2 = norm_cfg(embed_dims)

        # channel MLP
        mlp_hidden_dim = int(embed_dims * ffn_ratio)
        self.mlp = ChannelAggregationFFN(  # DWConv + Channel Aggregation FFN
            embed_dims=embed_dims,
            feedforward_channels=mlp_hidden_dim,
            act_cfg=act_cfg,
            ffn_drop=drop_rate,
        )

        # init layer scale
        self.layer_scale_1 = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)

    def forward(self, x):
        # spatial
        identity = x
        x = self.layer_scale_1 * self.attn(self.norm1(x))
        x = identity + self.drop_path(x)
        # channel
        identity = x
        x = self.layer_scale_2 * self.mlp(self.norm2(x))
        x = identity + self.drop_path(x)
        return x


class ConvPatchEmbed(PatchEmbed):
    """Image to Patch Embedding of VAN.
    The differences between ConvPatchEmbed & PatchEmbed:
        1. Use BN.
        2. Do not use 'flatten' and 'transpose'.
    """

    def __init__(self, *args, norm_cfg=BN, **kwargs):
        super().__init__(*args, norm_cfg=norm_cfg, **kwargs)

    def forward(self, x):
        """
        Args:
            x (Tensor): Has shape (B, C, H, W). In most case, C is 3.
        Returns:
            tuple: Contains merged results and its spatial shape.
            - x (Tensor): Has shape (B, out_h * out_w, embed_dims)
            - out_size (tuple[int]): Spatial shape of x, arrange as
              (out_h, out_w).
        """

        if self.adaptive_padding:
            x = self.adaptive_padding(x)
        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size


class StackConvPatchEmbed(nn.Module):
    """An implementation of Conv patch embedding layer.
    Args:
        in_features (int): The feature dimension.
        embed_dims (int): The output dimension of FFNs.
        kernel_size (int): The conv kernel size of middle patch embedding.
            Defaults to 3.
        stride (int): The conv stride of middle patch embedding.
            Defaults to 2.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='BN')``.
    """

    def __init__(self,
                 in_channels,
                 embed_dims,
                 kernel_size=3,
                 stride=2,
                 act_cfg=GELU,
                 norm_cfg=BN
                ):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embed_dims // 2, kernel_size=kernel_size,
                stride=stride, padding=kernel_size // 2),
                norm_cfg(embed_dims // 2),
                act_cfg(),
                nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=kernel_size,
                    stride=stride, padding=kernel_size // 2),
                norm_cfg(embed_dims),
        )

    def forward(self, x):
        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        return x, out_size


class MogaNet(nn.Module):
    """MogaNet.
    A PyTorch implement of : `Efficient Multi-order Gated Aggregation Network
    <https://arxiv.org/abs/2211.03295>`_
        v11.07, arXiv (IP120)
    Args:
        arch (str | dict): Visual Attention Network architecture.
            If use string, choose from 'tiny', 'small', 'base' and 'large'.
            If use dict, it should have below keys:
            - **embed_dims** (List[int]): The dimensions of embedding.
            - **depths** (List[int]): The number of blocks in each stage.
            - **ffn_ratios** (List[int]): The number of expansion ratio of
            feedforward network hidden layer channels.
            Defaults to 'tiny'.
        patch_sizes (List[int | tuple]): The patch size in patch embeddings.
            Defaults to [3, 3, 3, 3].
        in_channels (int): The num of input channels. Defaults to 3.
        drop_rate (float): Dropout rate after embedding. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.1.
        init_value (float): Init value for Layer Scale. Defaults to 1e-5.
        out_indices (Sequence[int]): Output from which stages.
            Default: ``(3, )``.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        stem_norm_cfg (dict): Config dict for normalization layer for all output
            features. Defaults to ``dict(type='LN')``.
        conv_norm_cfg (dict): Config dict for convolution normalization layer.
            Defaults to ``dict(type='BN')``.
        block_cfgs (Sequence[dict] | dict): The extra config of each block.
            Defaults to empty dicts.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """
    arch_zoo = {
        **dict.fromkeys(['xt', 'x-tiny'],
                        {'embed_dims': [32, 64, 96, 192],
                         'depths': [3, 3, 10, 2],
                         'ffn_ratios': [8, 8, 4, 4]}),
        **dict.fromkeys(['t', 'tiny'],
                        {'embed_dims': [32, 64, 128, 256],
                         'depths': [3, 3, 12, 2],
                         'ffn_ratios': [8, 8, 4, 4]}),
        **dict.fromkeys(['s', 'small'],
                        {'embed_dims': [64, 128, 320, 512],
                         'depths': [2, 3, 12, 2],
                         'ffn_ratios': [8, 8, 4, 4]}),
        **dict.fromkeys(['b', 'base'],
                        {'embed_dims': [64, 160, 320, 512],
                         'depths': [4, 6, 22, 3],
                         'ffn_ratios': [8, 8, 4, 4]}),
        **dict.fromkeys(['l', 'large'],
                        {'embed_dims': [64, 160, 320, 640],
                         'depths': [4, 6, 44, 4],
                         'ffn_ratios': [8, 8, 4, 4]}),
    }  # yapf: disable

    def __init__(self,
                 out_features=0,
                 arch='tiny',
                 patch_sizes=[3, 3, 3, 3],
                 in_channels=3,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 init_value=1e-5,
                 out_indices=(3, ),
                 frozen_stages=-1,
                 norm_eval=False,
                 stem_norm_cfg=BN,
                 conv_norm_cfg=BN,
                 patchembed_types=["ConvEmbed", "Conv", "Conv", "Conv",],
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_act_cfg=SiLU,
                 attn_final_dilation=True,
                 attn_force_fp32=False,
                 block_cfgs=dict()):
        super().__init__()

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {'embed_dims', 'depths', 'ffn_ratios'}
            assert isinstance(arch, dict) and set(arch) == essential_keys, \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.depths = self.arch_settings['depths']
        self.ffn_ratios = self.arch_settings['ffn_ratios']
        self.num_stages = len(self.depths)
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.attn_force_fp32 = attn_force_fp32
        self.use_layer_norm = stem_norm_cfg==nn.LayerNorm
        assert len(patchembed_types) == self.num_stages

        total_depth = sum(self.depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]  # stochastic depth decay rule

        cur_block_idx = 0
        for i, depth in enumerate(self.depths):
            if i == 0 and patchembed_types[i] == "ConvEmbed":
                assert patch_sizes[i] <= 3
                patch_embed = StackConvPatchEmbed(
                    in_channels=in_channels,
                    embed_dims=self.embed_dims[i],
                    kernel_size=patch_sizes[i],
                    stride=patch_sizes[i] // 2 + 1,
                    norm_cfg=conv_norm_cfg,
                )
            else:
                patch_embed = ConvPatchEmbed(
                    in_channels=in_channels if i == 0 else self.embed_dims[i - 1],
                    input_size=None,
                    embed_dims=self.embed_dims[i],
                    kernel_size=patch_sizes[i],
                    stride=patch_sizes[i] // 2 + 1,
                    padding=(patch_sizes[i] // 2, patch_sizes[i] // 2),
                    norm_cfg=conv_norm_cfg)

            if i == self.num_stages - 1 and not attn_final_dilation:
                attn_dw_dilation = [1, 2, 1]
            blocks = nn.ModuleList([
                MogaBlock(
                    embed_dims=self.embed_dims[i],
                    ffn_ratio=self.ffn_ratios[i],
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[cur_block_idx + j],
                    norm_cfg=conv_norm_cfg,
                    init_value=init_value,
                    attn_dw_dilation=attn_dw_dilation,
                    attn_channel_split=attn_channel_split,
                    attn_act_cfg=attn_act_cfg,
                    attn_force_fp32=attn_force_fp32,
                    **block_cfgs) for j in range(depth)
            ])
            cur_block_idx += depth
            norm = stem_norm_cfg(self.embed_dims[i])

            self.add_module(f'patch_embed{i + 1}', patch_embed)
            self.add_module(f'blocks{i + 1}', blocks)
            self.add_module(f'norm{i + 1}', norm)
        if out_features>0:
            f = self.embed_dims[-1]
            self.output_layer = nn.Sequential(
                nn.BatchNorm2d(f),
                nn.Conv2d(f, f, 3, 2, 1, groups=f), 
                nn.BatchNorm2d(f), 
                nn.PReLU(f),
                nn.Conv2d(f, f, 2, 2, 0, groups=f), 
                nn.BatchNorm2d(f), 
                nn.PReLU(f),
                nn.Conv2d(f, out_features, 2, 2), 
                nn.BatchNorm2d(out_features), 
                nn.Flatten(),
                )
        self.feature_info = FeatureInfo(self.embed_dims+[out_features], [4,8,16,32,1024])
    def init_weights(self, pretrained=None):
        super().init_weights(pretrained)

        if pretrained is None:
            if self.init_cfg is not None:
                return
            for k, m in self.named_modules():
                if isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    fan_out //= m.groups
                    m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    if not self.is_init:
                        trunc_normal_init(m, mean=0., std=0.02, bias=0)
                elif isinstance(m, (
                    nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                    constant_init(m, val=1, bias=0)

    def _freeze_stages(self):
        for i in range(0, self.frozen_stages + 1):
            # freeze patch embed
            m = getattr(self, f'patch_embed{i + 1}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

            # freeze blocks
            m = getattr(self, f'blocks{i + 1}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

            # freeze norm
            if i in self.out_indices:
                m = getattr(self, f'norm{i + 1}')
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward(self, x, return_features=False):
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f'patch_embed{i + 1}')
            blocks = getattr(self, f'blocks{i + 1}')
            norm = getattr(self, f'norm{i + 1}')

            x, hw_shape = patch_embed(x)
            for block in blocks:
                x = block(x)
            if self.use_layer_norm:
                x = x.flatten(2).transpose(1, 2)
                x = norm(x)
                x = x.reshape(-1, *hw_shape,
                            block.out_channels).permute(0, 3, 1, 2).contiguous()
            else:
                x = norm(x)
            outs.append(x)
        x = self.output_layer(F.adaptive_avg_pool2d(x,7))
        if not return_features: return x
        outs.append(x)
        return outs

    def train(self, mode=True):
        super(MogaNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                    m.eval()

MogaNetTiny = MogaNet

class MogaNetXTiny(MogaNet):
    # 12M
    def __init__(self, out_features, **kwargs):
        super().__init__(out_features, 'x-tiny', **kwargs)

class MogaNetSmall(MogaNet):
    # 96M
    def __init__(self, out_features, **kwargs):
        super().__init__(out_features, 'small', **kwargs)

class MogaNetBase(MogaNet):
    # 167M
    def __init__(self, out_features, **kwargs):
        super().__init__(out_features, 'base', **kwargs)

class MogaNetLarge(MogaNet):
    # 315M
    def __init__(self, out_features, **kwargs):
        super().__init__(out_features, 'large', **kwargs)

if __name__=="__main__":
    from . import utils
    net=utils.test_model(MogaNetLarge)                    
    torch.save(net.state_dict(), f"{net.__class__.__name__}.pth")
