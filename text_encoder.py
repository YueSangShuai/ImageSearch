import torch
import torch.nn as nn
import numpy as np
import math
from torch import Tensor
from torch.nn import Parameter
from torch.nn import init
from torch.nn import functional as F

class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, shape_dim1=True, alpha_init_value=0.5, eps=1e-6):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = shape_dim1

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"

class SReLU(nn.Module):
    def forward(self, x):
        x = torch.relu(x)
        return x*x
                

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(output)

class MultiTokenAttention(nn.Module):
    """
    实现 Multi-Token Attention (MTA)
    根据论文: https://arxiv.org/abs/2404.00927 
    使用论文中描述的默认配置:
    - Key-Query Convolution: Pre-Softmax
    - Head Mixing Convolution: Post-Softmax
    - Group Normalization per Head w/ Scaling
    """
    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 kq_kernel_size_q: int = 3,
                 kq_kernel_size_k: int = 3,
                 head_group_size: int = 2, # Renamed from head_kernel_size for clarity
                 dropout: float = 0.0,
                 bias: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.kq_kernel_size_q = kq_kernel_size_q
        self.kq_kernel_size_k = kq_kernel_size_k
        # head_group_size corresponds to 'ch' in the paper
        self.head_group_size = head_group_size
        self.dropout = dropout

        # ... (parameter validation checks remain the same) ...

        if self.num_heads % self.head_group_size != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by "
                             f"head_group_size ({head_group_size})")

        self.scale = self.head_dim ** -0.5

        # --- Standard Attention Projections ---
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # --- MTA Specific Components ---

        # 1. Key-Query Convolution (Pre-Softmax)
        # ... (This part remains the same, including initialization) ...
        kq_padding_q = kq_kernel_size_q - 1
        kq_padding_k = (kq_kernel_size_k - 1) // 2
        self.key_query_conv = nn.Conv2d(
            in_channels=self.num_heads,
            out_channels=self.num_heads,
            kernel_size=(kq_kernel_size_q, kq_kernel_size_k),
            padding=(kq_padding_q, kq_padding_k),
            groups=self.num_heads,
            bias=True
        )
        # (Keep the KQ conv initialization code)
        torch.nn.init.zeros_(self.key_query_conv.weight)
        if self.key_query_conv.bias is not None:
             torch.nn.init.zeros_(self.key_query_conv.bias)
        with torch.no_grad():
            kernel_center_q = kq_kernel_size_q -1
            kernel_center_k = (kq_kernel_size_k -1) // 2
            for i in range(self.num_heads):
                 self.key_query_conv.weight[i, 0, kernel_center_q, kernel_center_k] = 1.0


        # 2. Head Mixing Operation (Post-Softmax) - Using Conv1d with kernel_size=1
        # This implements the grouped fully-connected mapping per position
        num_head_groups = self.num_heads // self.head_group_size
        self.head_mix_conv = nn.Conv1d(
            in_channels=self.num_heads,
            out_channels=self.num_heads,
            kernel_size=1,  # Kernel size 1 to act as linear map across channels
            padding=0,      # No padding needed for kernel_size 1
            groups=num_head_groups, # Grouped operation
            bias=True
        )
        # Initialize Head mixing conv kernel to be block-diagonal identity
        # Weight shape: (out_channels, in_channels // groups, kernel_size=1)
        #             = (num_heads, head_group_size, 1)
        torch.nn.init.zeros_(self.head_mix_conv.weight)
        if self.head_mix_conv.bias is not None:
             torch.nn.init.zeros_(self.head_mix_conv.bias)
        with torch.no_grad():
            for i in range(self.num_heads):
                group_idx = i // self.head_group_size
                idx_in_group = i % self.head_group_size
                # Set the diagonal element within each group's linear map to 1
                self.head_mix_conv.weight[i, idx_in_group, 0] = 1.0


        # 3. Group Normalization & Scaling (Per Head)
        # ... (This part remains the same) ...
        self.norm = DynamicTanh(self.head_dim)
        self.depth_scale = nn.Parameter(torch.ones(1))

    def _apply_causal_mask_0_where(self, attn_logits):
        # ... (This function remains the same) ...
        batch_size, num_heads, seq_len_q, seq_len_k = attn_logits.size()
        mask = torch.triu(torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=attn_logits.device), diagonal=1)
        masked_logits = attn_logits.masked_fill(mask.unsqueeze(0).unsqueeze(0), 0.0)
        return masked_logits
    def _apply_causal_mask_0(self, attn_logits):
        batch_size, num_heads, seq_len_q, seq_len_k = attn_logits.size()
        # 使用乘法替代masked_fill
        mask = torch.triu(torch.ones(seq_len_q, seq_len_k, device=attn_logits.device), diagonal=1)
        return attn_logits * (1 - mask.unsqueeze(0).unsqueeze(0))
    def forward(self,
                Q: torch.Tensor,
                K: torch.Tensor,
                V: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                **kwargs
                ) -> torch.Tensor:

        batch_size, seq_len, _ = Q.size()
        seq_q = seq_len # Assume query and key/value lengths are the same for now
        seq_k = seq_len

        # 1. Project Q, K, V & Reshape
        # ... (remains the same) ...
        query_states = self.q_proj(Q).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(K).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(V).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Initial Attention Logits
        attn_logits = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scale

        # --- MTA Modifications ---

        # 3. Apply Mask_0 (for Eq. 5)
        attn_logits_masked_0 = self._apply_causal_mask_0(attn_logits)

        # 4. Key-Query Convolution (Pre-Softmax)
        attn_logits_conv = self.key_query_conv(attn_logits_masked_0)
        # Remove padding added for query dimension causality
        # Adjust seq_len_q dim if necessary (should match original seq_q)
        attn_logits_conv = attn_logits_conv[:, :, :seq_q, :]

        # 5. Apply Standard Attention Mask (Mask_inf)
        # ... (remains the same) ...
        if attention_mask is not None:
             if attention_mask.dim() == 2:
                 attention_mask = attention_mask[:, None, None, :]
             elif attention_mask.dim() == 3:
                  attention_mask = attention_mask[:, None, :, :]
             # Adjust mask size if needed after convolution (though it shouldn't change here)
             current_seq_k = attn_logits_conv.shape[-1]
             if attention_mask.shape[-1] != current_seq_k:
                 attention_mask = attention_mask[:, :, :, :current_seq_k]

             # Add mask (it should have large negative values)
             attn_logits_conv = attn_logits_conv * attention_mask.to(attn_logits_conv.dtype)


        # 6. Softmax
        attn_weights = F.softmax(attn_logits_conv, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # 7. Head Mixing Operation (Post-Softmax) using Conv1d (kernel_size=1)
        # Reshape weights for Conv1d: (bs, n_heads, seq_q, seq_k) -> (bs*seq_q, n_heads, seq_k)
        # The convolution will apply a grouped linear map across n_heads for each seq_k position
        bsz, n_heads, current_seq_q, current_seq_k = attn_weights.size() # Use current sizes
        attn_weights_reshaped = attn_weights.permute(0, 2, 1, 3).contiguous().view(bsz * current_seq_q, n_heads, current_seq_k)

        attn_weights_mixed_reshaped = self.head_mix_conv(attn_weights_reshaped) # Output shape (bs*seq_q, n_heads, seq_k)

        # Reshape back: (bs*seq_q, n_heads, seq_k) -> (bs, seq_q, n_heads, seq_k) -> (bs, n_heads, seq_q, seq_k)
        attn_weights_mixed = attn_weights_mixed_reshaped.view(bsz, current_seq_q, n_heads, current_seq_k).permute(0, 2, 1, 3)

        # 8. Compute Attention Output
        attn_output = torch.matmul(attn_weights_mixed, value_states) # value_states shape: (bs, n_heads, seq_k, head_dim)

        # 9. Normalization & Scaling
        # ... (remains the same) ...
        attn_output = self.norm(attn_output)
        attn_output = attn_output * self.depth_scale

        # 10. Reshape and Project Out
        # ... (remains the same) ...
        attn_output = attn_output.transpose(1, 2).contiguous()
        # Ensure final reshape uses the query sequence length
        attn_output = attn_output.reshape(batch_size, current_seq_q, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=512):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiTokenAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            SReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = DynamicTanh(d_model)
        self.norm2 = DynamicTanh(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class CharsEncoder(nn.Module):
    def __init__(self, d_model=256, num_heads=8, num_layers=6, d_ff=1024, embedding_dim=256, max_seq_length=512, dropout=0.1):
        super().__init__()
        # UTF-8 byte embedding (256 possible byte values)
        self.byte_embedding0 = nn.Embedding(256, d_model)
        self.byte_embedding1 = nn.Embedding(256, d_model)
        self.byte_embedding2 = nn.Embedding(256, d_model)
        self.byte_embedding3 = nn.Embedding(256, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=9, padding=4),
            nn.BatchNorm2d(8),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 2, kernel_size=5, padding=2),
            nn.BatchNorm2d(2),
            SReLU(),
        )
        self.fc = nn.Linear(d_model*2, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.export_onnx = False
    def encoding_input(self, x):
        x = x.long()
        x0 = self.byte_embedding0(x[:, :, 0])  # (batch_size, seq_length, bytes_per_char*d_model)
        x1 = self.byte_embedding1(x[:, :, 1])
        x2 = self.byte_embedding2(x[:, :, 2])
        x3 = self.byte_embedding3(x[:, :, 3])
        x0 = self.pos_encoding(x0)
        x1 = self.pos_encoding(x1)
        x2 = self.pos_encoding(x2)
        x3 = self.pos_encoding(x3)
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        return x
    def enhance(self, x):
        d_model = self.d_model
        batch_size, seq_length, _ = x.shape
        bytes_per_char = 4
        x = x.view(batch_size, seq_length*bytes_per_char, d_model)  # (batch_size, seq_length*bytes_per_char, d_model)        
        for layer in self.encoder_layers:
            x = layer(x)
        x = x.view(batch_size, seq_length, bytes_per_char, d_model).permute(0, 2, 1, 3).contiguous() 
        # (batch_size, bytes_per_char, seq_length, d_model)
        x = self.conv1(x)
        x1 = x[:, :4]
        x2 = x[:, 4:]
        x = torch.max(x1, x2)
        x = self.dropout(x)
        x = self.conv2(x)
        x = x.view(batch_size, seq_length, d_model*2)
        x = self.dropout(x)
        ret = self.fc(x)  # (batch_size, seq_length, embedding_dim)
        return ret
    def out(self, x):
        return torch.max(x, dim=1, keepdim=False)[0]  # (batch_size, embedding_dim)
    def export(self, x=None):
        self.export_onnx = True
        if x is not None:
            return self.encoding_input(x)
    def forward(self, x):
        if self.export_onnx:
            ret = self.enhance(x)
            return ret  # 如果不要最后一层
            return self.out(ret)
        x = self.encoding_input(x) # (batch_size, seq_length, bytes_per_char*d_model)
        x = self.dropout(x)
        ret = self.enhance(x)
        return self.out(ret)
