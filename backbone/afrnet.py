import torch
from torch import nn
from typing import Optional
from .mf3 import ANet2P
from torch.nn import functional as F
from .irpe import get_rpe_config, build_rpe

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Truncated normal initialization."""
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp >= a) & (tmp <= b)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.mul_(std).add_(mean)

# Relative position encoding configuration
rpe_config = get_rpe_config(
    ratio=1.9,
    method="product",
    mode='ctx',
    shared_head=True,
    skip=0,
    rpe_on='k',
)

class Attention(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 qkv_bias: bool = False,
                 qk_scale: Optional[float] = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # Image relative position encoding
        self.rpe_q, self.rpe_k, self.rpe_v = build_rpe(
            rpe_config,
            head_dim=head_dim,
            num_heads=num_heads
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        batch_size, num_token, embed_dim = x.shape
        qkv = self.qkv(x).reshape(
            batch_size, num_token, 3, self.num_heads, embed_dim // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0].float(), qkv[1].float(), qkv[2].float()
        q *= self.scale
        attn = q @ k.transpose(-2, -1)

        # Image relative position on keys
        if self.rpe_k is not None:
            attn += self.rpe_k(q)

        # Image relative position on queries
        if self.rpe_q is not None:
            attn += self.rpe_q(k * self.scale).transpose(2, 3)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v

        # Image relative position on values
        if self.rpe_v is not None:
            out += self.rpe_v(attn)

        x = out.transpose(1, 2).reshape(batch_size, num_token, embed_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class EmbedMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x

class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, stride=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=3, stride=stride, padding=1,
            bias=True, groups=embed_dim
        )
        self.stride = stride

    def forward(self, x, height, width):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, height, width)
        x = self.proj(x) + x if self.stride == 1 else self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, in_features=1024, emb_size: int = 384):
        super().__init__()
        self.embed_mlp = EmbedMLP(in_features=in_features, hidden_features=1024, out_features=emb_size)
        self.pos_cnn = PosCNN(emb_size, emb_size)

    def forward(self, x):
        x = self.embed_mlp(x)
        x = self.pos_cnn(x, 14, 14)
        return x

class AttentionHead(nn.Module):
    def __init__(self, d_model=384, in_features=1024, nhead=12, dim_feedforward=1536):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_features=in_features, emb_size=d_model)
        self.attn = Attention(d_model, num_heads=nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(in_features=d_model, hidden_features=dim_feedforward, act_layer=nn.ReLU6, drop=0.1)
        self.encoder_norm1 = nn.BatchNorm1d(num_features=196, eps=2e-5)
        self.encoder_norm2 = nn.BatchNorm1d(num_features=d_model, eps=2e-5)
        self.linear_seq = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * 196, d_model, bias=True)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                nn.init.zeros_(m.bias) if m.bias is not None else None
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        x = self.encoder_norm1(x)
        x = self.linear_seq(x)
        x = self.encoder_norm2(x)
        return x

class ModifiedGDC(nn.Module):
    def __init__(self, image_size, in_chs, num_classes=None, dropout=0.0, stride=2, emb=512):
        super().__init__()
        patch_size = int(16 * stride)
        kernel_size = (image_size // patch_size) + (0 if image_size % patch_size == 0 else 1)
        self.conv_dw = nn.Conv2d(in_chs, in_chs, kernel_size=kernel_size, groups=in_chs, bias=False)
        self.bn1 = nn.BatchNorm2d(in_chs)
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2d(in_chs, emb, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(emb)
        self.linear = nn.Linear(emb, num_classes) if num_classes else nn.Identity()

    def forward(self, x):
        x = self.dropout(self.bn1(self.conv_dw(x)))  #
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.bn2(self.conv(x).squeeze(-1).squeeze(-1))
        x = self.linear(x)
        return x

class AFRNet(nn.Module):
    def __init__(self, out_features=384, flag2onnx=False):
        super().__init__()
        self.flag2onnx = flag2onnx
        resnet_net = ANet2P(10, width_multiplier=8)
#        print(resnet_net.feature_info)
        inch = resnet_net.feature_info.channels()[-3]
        self.feature_extraction = nn.Sequential(*list(resnet_net.children())[:-3])
        self.cnn_head = ModifiedGDC(image_size=224, in_chs=inch, emb=out_features)
        self.attention_head = AttentionHead(in_features=inch, d_model=out_features)

    def forward(self, x):
        x = self.feature_extraction(x)
#        print(x.shape) # torch.Size([2, 1024, 14, 14])
        x1 = self.cnn_head(x)
        x2 = self.attention_head(x)
        if self.flag2onnx:
            x1 = x1 / x1.norm(p=2, dim=-1, keepdim=True)
            x2 = x2 / x2.norm(p=2, dim=-1, keepdim=True)
            x = torch.cat([x1, x2], dim=-1)
            return x
        else:
            return x1, x2

if __name__ == '__main__':
    model = AFRNet(256*3)
    img = torch.randn(2, 3, 224, 224)
    out1, out2 = model(img)
    print(out1.shape, out2.shape)
