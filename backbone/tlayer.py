import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Union

class ScaledDotProductAttention(nn.Module):
    """
    实现缩放点积注意力机制。
    """

    def __init__(self, dropout: float = 0.0, device=None, dtype=None):
        """
        初始化 ScaledDotProductAttention。

        Args:
            dropout (float): dropout 概率。
            device (torch.device, optional): 设备
            dtype (torch.dtype, optional): 数据类型
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.dtype = dtype

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播函数。

        Args:
            query (torch.Tensor): 查询张量，形状 (N, S_q, E) 或 (S_q, N, E)
            key (torch.Tensor): 键张量，形状 (N, S_k, E) 或 (S_k, N, E)
            value (torch.Tensor): 值张量，形状 (N, S_v, E) 或 (S_v, N, E)
             mask (torch.Tensor, optional): mask 矩阵.

        Returns:
            torch.Tensor: 注意力输出张量，形状与 value 相同。
        """
        # 1. 计算注意力权重
        dk = key.shape[-1]  # key 的维度， 用于缩放
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / (dk ** 0.5)  # 计算注意力得分

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf')) #将 mask 为 0 的部分用 -inf 填充，在softmax时会被处理成 0

        attn_weights = F.softmax(attn_weights, dim=-1)  # 应用 softmax

        # 2. 应用 dropout
        attn_weights = self.dropout(attn_weights)

        # 3. 计算加权值
        output = torch.matmul(attn_weights, value)
        return output


class MultiHeadAttention(nn.Module):
    """
    实现多头注意力机制。
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0, batch_first: bool = False, device=None, dtype=None):
        """
        初始化 MultiHeadAttention。

        Args:
            d_model (int): 输入特征的维度。
            nhead (int): 注意力头的数量。
            dropout (float): dropout 概率。
            batch_first (bool): batch first.
            device (torch.device, optional): 设备
            dtype (torch.dtype, optional): 数据类型
        """
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first
        self.head_dim = d_model // nhead  # 每个头的维度

        # 检查 d_model 是否可以被 nhead 整除
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        # 定义线性变换层
        self.in_proj = nn.Linear(d_model, 3 * d_model, device=device, dtype=dtype) # 合并 Wq, Wk, Wv
        self.Wo = nn.Linear(d_model, d_model, device=device, dtype=dtype)

        # 实例化缩放点积注意力
        self.attention = ScaledDotProductAttention(dropout, device=device, dtype=dtype)

        self.device = device
        self.dtype = dtype
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        N, S_q, E = query.shape  # batch_first=True
        # 1. 线性映射得到 q, k, v: 形状 (N, S, 3*d_model)
        qkv = self.in_proj(query)
        q, k, v = qkv.chunk(3, dim=-1)  # 分成 (N, S, d_model)

        # 2. reshape -> (N, S, nhead, head_dim) -> (N*nhead, S, head_dim)
        q = q.view(N, S_q, self.nhead, self.head_dim).permute(0,2,1,3).reshape(N*self.nhead, S_q, self.head_dim)
        k = k.view(N, S_q, self.nhead, self.head_dim).permute(0,2,1,3).reshape(N*self.nhead, S_q, self.head_dim)
        v = v.view(N, S_q, self.nhead, self.head_dim).permute(0,2,1,3).reshape(N*self.nhead, S_q, self.head_dim)

        # 3. 现在 ScaledDotProductAttention 就用 (B, S, E) 的接口来做
        #    其中 B = N * nhead
        attn_output = self.attention(q, k, v, mask=attn_mask)  # shape: (N*nhead, S, head_dim)

        # 4. 把多头合并回 (N, S, d_model)
        attn_output = attn_output.view(N, self.nhead, S_q, self.head_dim).permute(0,2,1,3)  # -> (N, S, nhead, head_dim)
        attn_output = attn_output.reshape(N, S_q, self.d_model)

        # 5. Wo 映射 -> (N, S, d_model)
        output = self.Wo(attn_output)
        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.batch_first = batch_first
        self.norm_first = norm_first
        self.bias = bias
        self.device = device
        self.dtype = dtype

        # 1. Multi-Head Attention (MHA)
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, device=device, dtype=dtype)

        # 2. Feedforward Network (FFN)
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, device=device, dtype=dtype)
        self.dropout_ffn = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, device=device, dtype=dtype)

        # 3. Layer Normalization
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, device=device, dtype=dtype)

        # 4. Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def _get_activation_fn(self, activation: Union[str, Callable[[torch.Tensor], torch.Tensor]]) -> Callable[[torch.Tensor], torch.Tensor]:
        if isinstance(activation, str):
            if activation == "relu":
                return nn.ReLU()
            elif activation == "gelu":
                return nn.GELU()
            else:
                raise ValueError(f"Unsupported activation: {activation}")
        elif isinstance(activation, Callable):
            return activation
        else:
            raise TypeError("Activation must be a string or callable.")


    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None, src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        if self.norm_first:
            return self.forward_norm_first(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        else:
            return self.forward_norm_last(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)


    def forward_norm_first(self, src: torch.Tensor, src_mask: torch.Tensor = None, src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        residual = src
        # 1. Layer Normalization 1
        src = self.norm1(src)
        # 2. Self-Attention
        attn_output = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        # 3. Dropout 1
        src = residual + self.dropout1(attn_output)

        residual = src
        # 4. Layer Normalization 2
        src = self.norm2(src)
        # 5. Feed Forward Network
        src = self.linear2(self.dropout_ffn(self._get_activation_fn(self.activation)(self.linear1(src))))
        # 6. Dropout 2
        src = residual + self.dropout2(src)
        return src

    def forward_norm_last(self, src: torch.Tensor, src_mask: torch.Tensor = None, src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        residual = src
        # 1. Self-Attention
        attn_output = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        # 2. Dropout 1
        src = residual + self.dropout1(attn_output)
        # 3. Layer Normalization 1
        src = self.norm1(src)
        
        residual = src
        # 4. Feed Forward Network
        src = self.linear2(self.dropout_ffn(self._get_activation_fn(self.activation)(self.linear1(src))))
        # 5. Dropout 2
        src = residual + self.dropout2(src)
        # 6. Layer Normalization 2
        src = self.norm2(src)
        return src

if __name__ == '__main__':
    # Example Usage
    torch.manual_seed(42)
    d_model = 512
    nhead = 8
    dim_feedforward = 2048
    dropout = 0.1
    batch_size = 32
    seq_len = 64
    layer_norm_eps = 1e-5
    # Create an instance of the custom TransformerEncoderLayer
    custom_encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, layer_norm_eps=layer_norm_eps, batch_first=True, activation="relu")
    custom_encoder_layer.eval()

    # Create a sample input tensor
    input_tensor = torch.randn(batch_size, seq_len, d_model) # batch_first=True

    with torch.no_grad():
        # Perform forward pass with your custom encoder layer
        output = custom_encoder_layer(input_tensor)

    # Check the output shape
    print("Custom Encoder Output Shape:", output.shape) #  torch.Size([32, 64, 512])

    x1 = input_tensor[0:1]
    with torch.no_grad():
        output1 = custom_encoder_layer(x1)

    print(f"{(output1-output[0:1]).norm(), output1.norm(), output[0:1].norm()}\n{output1[0,0,:8].numpy()}\n{output[0,0,:8].numpy()}")
