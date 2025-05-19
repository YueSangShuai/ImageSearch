import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torchvision
from einops import rearrange, reduce, repeat, pack, unpack
from einops.layers.torch import Rearrange, Reduce
from .iresnet import iresnet50, iresnet101
from .irpe import get_rpe_config, build_rpe
from typing import Optional

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp >= a) & (tmp <= b)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)

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
                 qk_scale: Optional[None] = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # image relative position encoding
        self.rpe_q, self.rpe_k, self.rpe_v = \
        build_rpe(rpe_config,
                  head_dim=head_dim,
                  num_heads=num_heads)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        
        #with torch.cuda.amp.autocast(True):
        batch_size, num_token, embed_dim = x.shape
        #qkv is [3,batch_size,num_heads,num_token, embed_dim//num_heads]
        qkv = self.qkv(x).reshape(
            batch_size, num_token, 3, self.num_heads, embed_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        #with torch.cuda.amp.autocast(False):
        q, k, v = qkv[0].float(), qkv[1].float(), qkv[2].float()
        q *= self.scale
        attn = (q @ k.transpose(-2, -1))
        
            # image relative position on keys
        if self.rpe_k is not None:
            attn += self.rpe_k(q)

        # image relative position on queries
        if self.rpe_q is not None:
            attn += self.rpe_q(k * self.scale).transpose(2, 3)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v

        # image relative position on values
        if self.rpe_v is not None:
            out += self.rpe_v(attn)

        x = out.transpose(1, 2).reshape(batch_size, num_token, embed_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class embed_mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        #x = x.flatten(2).transpose(1, 2)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# PEG  from https://arxiv.org/abs/2102.10882
class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim), )
        self.s = s

    def forward(self, x, H, W):
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size: int = 384):
        super().__init__()
        # Embed MLP
        # self.embed_mlp = nn.Sequential(
        #     Rearrange('b e h w -> b (h w) e'),
        #     nn.Linear(1024, 1024),
        #     #nn.ReLU(),
        #     nn.Linear(1024, 384)
        # )
        self.embed_mlp = embed_mlp(in_features=1024, hidden_features=1024, out_features=384)
        
        #self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        # 位置编码信息，一共有(img_size // patch_size)**2 + 1(cls token)个位置向量
        #self.positions = nn.Parameter(torch.randn(197, emb_size))
        self.pos_cnn = PosCNN(emb_size, emb_size)
        
    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.embed_mlp(x)
        x = self.pos_cnn(x, 14, 14)
        #cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # 将cls token在维度1扩展到输入上
        #x = torch.cat([cls_tokens, x], dim=1)
        # 添加位置编码
        #print(x.shape, self.positions.shape)
        #x += self.positions
        return x

class AttentionClassificationHead(nn.Module):
    def __init__(self, d_model=384, nhead=12, num_layers=1, dim_feedforward=1536):
        super(AttentionClassificationHead, self).__init__()
        
        # Embed MLP
        # self.embed_mlp = nn.Sequential(
        #     Rearrange('b e h w -> b (h w) e'),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 384)
        # )
        # self.embed_mlp.apply(self._init_weights)

        # Positional Embedding
        self.patch_embbing = PatchEmbedding()
        #register tokens
        #self.register_tokens = nn.Parameter(
        #    torch.randn(4, d_model)
        #)
        # Self-Attention + MLP
        #encoder_norm = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, num_heads=nhead, qkv_bias=False, qk_scale=None, attn_drop=0, proj_drop=0)
        #self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # self.norm1 = nn.BatchNorm1d(num_features=197, eps=2e-5)
        #self.norm2 = nn.BatchNorm1d(num_features=197, eps=2e-5)

        mlp_hidden_dim = int(d_model * 4)

        self.mlp = mlp(in_features=d_model, hidden_features=mlp_hidden_dim,
                       act_layer=nn.ReLU6, drop=0.1)
        # encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        # self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        #self.encoder_norm = nn.LayerNorm(d_model)
        self.encoder_norm1 = nn.BatchNorm1d(num_features=196, eps=2e-5)
        self.encoder_norm2 = nn.BatchNorm1d(num_features=384, eps=2e-5)

        # self.encoder_norm1 = nn.LayerNorm(d_model)
        # self.encoder_norm2 = nn.LayerNorm(d_model)

        # Za Linear
        # self.linear_seq = nn.Sequential(
            # nn.Flatten(),
            # nn.Linear(384*197, d_model, bias=True)
        # )
        self.linear_seq = nn.Sequential(
            nn.Flatten(),
            nn.Linear(384*196, d_model, bias=True)
        )
        self.apply(self._init_weights)
        # self.linear = nn.Linear(384*197, d_model, bias=True)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        batch, _, _, _ = x.shape
        x = self.patch_embbing(x)
                #repeat register token
        # r = repeat(
        #     self.register_tokens, 
        #     'n d -> b n d', 
        #     b=batch
        # )

        #pack cls token and register token
       # x, ps = pack([x, r], 'b * d ')
        
        
        #unpack cls token and register token
        
        x = x + self.attn(self.norm1(x))
        #with torch.cuda.amp.autocast(True):
        x = x + self.mlp(self.norm2(x))
       # x, _ = unpack(x, ps, 'b * d')
        
        # x = self.transformer_encoder(x)
        x = self.encoder_norm1(x)
        x = self.linear_seq(x)
        x = self.encoder_norm2(x)

        return x


class ModifiedGDC(nn.Module):
    def __init__(self, image_size, in_chs, num_classes, dropout, st=2, emb=512): # dropout implementation is in the original code but not in the paper
        super(ModifiedGDC, self).__init__()
        patch_size = int(16*st)
        if image_size % patch_size == 0:
            self.conv_dw = nn.Conv2d(in_chs, in_chs, kernel_size=(image_size//patch_size), groups=in_chs, bias=False)
        else:
            self.conv_dw = nn.Conv2d(in_chs, in_chs, kernel_size=(image_size//patch_size + 1), groups=in_chs, bias=False)
        self.bn1 = nn.BatchNorm2d(in_chs)
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2d(in_chs, emb, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(emb)
        self.linear = nn.Linear(emb, num_classes) if num_classes else nn.Identity()

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.conv(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.bn2(x)
        x = self.linear(x)
        return x

class AFRNet(nn.Module):
    def __init__(self, out_features=512, flag2onnx=True):
        super(AFRNet, self).__init__()
        self.flag2onnx = flag2onnx
        # Spatial Alignment Module
        # self.spatial_alignment = nn.Sequential(
            # nn.Conv2d(3, 16, kernel_size=7, padding=3),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(16, 24, kernel_size=5, padding=2),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(24, 32, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(32, 48, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(48, 64, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        # )

        # # Add linear layers for Spatial Alignment Module
        # self.spatial_alignment_linear = nn.Sequential(
            # nn.Flatten(),
            # nn.Linear(in_features=64*7*7, out_features=32),
            # nn.Linear(in_features=32, out_features=4)
        # )

        # Feature Extraction Branch
        #resnet_net = torchvision.models.resnet50(pretrained=True)
        resnet_net = iresnet50(pretrained=True)
        modules = list(resnet_net.children())[:-3]
        self.feature_extraction = nn.Sequential(*modules)

        # CNN Classification Head
        cnn_classification = list(resnet_net.children())[-3:-2]
        self.classification_head = nn.Sequential(
            *cnn_classification,
            nn.Flatten(),
            nn.Linear(in_features=2048*7*7, out_features=384),
        )
        self.classifier = ModifiedGDC(image_size=224, in_chs=1024, num_classes=None, dropout=0.0, st=1.0, emb=384)

        self.attention_head = AttentionClassificationHead()
        # Attention Classification Head
        # We need to implement the custom layers described in the table here

    def forward(self, x, return_featuremaps=False):
        #x = self.spatial_alignment(x)
        #x = self.spatial_alignment_linear(x.view(x.size(0), -1))
        
        x = self.feature_extraction(x)
        # x1 = self.classification_head(x)
        x1 = self.classifier(x)
        x2 = self.attention_head(x)
        if self.flag2onnx:
            x1 = x1/x1.norm(2, dim=-1, keepdim=True)
            x2 = x2/x2.norm(2, dim=-1, keepdim=True)
            x = torch.cat([x1, x2], dim=-1)
            return [x]
        else:
        # We need to add the forward pass for the rest of the modules
            return [x1, x2]


if __name__ == '__main__':
    model = AFRNet()
    img = torch.randn(2, 3, 224, 224)
    out1 = model(img)
    print(out1.shape)