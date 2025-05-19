import torch
from vit_pytorch import ViT, SimpleViT, cct, levit, scalable_vit, sep_vit, max_vit  

from .utils import FeatureInfo
from .sigmaReparam import convert_to_sn, remove_all_normalization_layers


class SViT(SimpleViT):
    def __init__(self, out_features=512, rep=False, *kwargs):
        super().__init__(
                image_size = 224,
                patch_size = 32,
                num_classes = out_features,
                dim = 1024,
                depth = 6,
                heads = 16,
                mlp_dim = 2048, *kwargs)
        if rep:
            convert_to_sn(self)
            remove_all_normalization_layers(self)
    def forward(self, x, return_features=False):
        y = super().forward(x)
        return y if not return_features else [y]

class rSViT(SViT):
    def __init__(self, out_features=512, *kwargs):
        super().__init__(out_features=out_features, rep=True, *kwargs)

class SViT1(SimpleViT):
    def __init__(self, out_features=512, rep=False, *kwargs):
        super().__init__(
                image_size = 224,
                patch_size = 16,
                num_classes = out_features,
                dim = 256,
                depth = 6,
                heads = 8,
                mlp_dim = 512, *kwargs)
        if rep:
            convert_to_sn(self)
            remove_all_normalization_layers(self)
    def forward(self, x, return_features=False):
        y = super().forward(x)
        return y if not return_features else [y]

class rSViT1(SViT1):
    def __init__(self, out_features=512, *kwargs):
        super().__init__(out_features=out_features, rep=True, *kwargs)

class CCT(cct.CCT):
    def __init__(self, out_features=512, *kwargs):
        super().__init__(
                image_size = 224,
                num_classes = out_features,
                embedding_dim = 384,
                n_conv_layers = 2,
                kernel_size = 7,
                stride = 2,
                padding = 3,
                pooling_kernel_size = 3,
                pooling_stride = 2,
                pooling_padding = 1,
                num_layers = 14,
                num_heads = 6,
                mlp_ratio = 3., *kwargs)
    def forward(self, x, return_features=False):
        y = super().forward(x)
        return y if not return_features else [y]

class LeViT(levit.LeViT):
    def __init__(self, out_features=512, *kwargs):
        super().__init__(
                image_size = 224,
                num_classes = out_features,
                stages = 3,             # number of stages
                dim = (256, 384, 512),  # dimensions at each stage
                depth = 4,              # transformer of depth 4 at each stage
                heads = (4, 6, 8),      # heads at each stage
                mlp_mult = 2,
                dropout = 0.1, 
                *kwargs)
    def forward(self, x, return_features=False):
        y = super().forward(x)
        return y if not return_features else [y]
    
class ScalableViT(scalable_vit.ScalableViT):
    def __init__(self, out_features=512, *kwargs):
        super().__init__(
                num_classes = out_features,
                dim = 32,                               # starting model dimension. at every stage, dimension is doubled
                heads = (2, 4, 8, 16),                  # number of attention heads at each stage
                depth = (2, 2, 20, 2),                  # number of transformer blocks at each stage
                ssa_dim_key = (40, 40, 40, 32),         # the dimension of the attention keys (and queries) for SSA. in the paper, they represented this as a scale factor on the base dimension per key (ssa_dim_key / dim_key)
                reduction_factor = (8, 4, 2, 1),        # downsampling of the key / values in SSA. in the paper, this was represented as (reduction_factor ** -2)
                window_size = (64, 32, None, None),     # window size of the IWSA at each stage. None means no windowing needed
                dropout = 0.1,                          # attention and feedforward dropout
                *kwargs)
    def forward(self, x, return_features=False):
        y = super().forward(x)
        return y if not return_features else [y]

class SepViT(sep_vit.SepViT):
    def __init__(self, out_features=512, *kwargs):
        super().__init__(
            num_classes = out_features,
            dim = 32,               # dimensions of first stage, which doubles every stage (32, 64, 128, 256) for SepViT-Lite
            dim_head = 32,          # attention head dimension
            heads = (1, 2, 4, 8),   # number of heads per stage
            depth = (1, 2, 6, 2),   # number of transformer blocks per stage
            window_size = 7,        # window size of DSS Attention block
            dropout = 0.1,           # dropout
            *kwargs
        )
    def forward(self, x, return_features=False):
        y = super().forward(x)
        return y if not return_features else [y]

class MaxViT(max_vit.MaxViT):
    def __init__(self, out_features=512, *kwargs):
        super().__init__(
            num_classes = out_features,
            dim_conv_stem = 64,               # dimension of the convolutional stem, would default to dimension of first layer if not specified
            dim = 96,                         # dimension of first layer, doubles every layer
            dim_head = 32,                    # dimension of attention heads, kept at 32 in paper
            depth = (2, 2, 5, 2),             # number of MaxViT blocks per stage, which consists of MBConv, block-like attention, grid-like attention
            window_size = 8,                  # window size for block and grids
            mbconv_expansion_rate = 4,        # expansion rate of MBConv
            mbconv_shrinkage_rate = 0.25,     # shrinkage rate of squeeze-excitation in MBConv
            dropout = 0.1,                     # dropout
            *kwargs
        )
    def forward(self, x, return_features=False):
        y = super().forward(x)
        return y if not return_features else [y]
        