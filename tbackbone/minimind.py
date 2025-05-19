# --- Start of Updated Code ---
import math
from typing import Any, Optional, Tuple, List, Union
import torch
import torch.nn.functional as F
from torch import nn
import os

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# --- MODIFIED FUNCTION: Returns two real tensors ---
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precomputes the frequency tensor for complex exponentials (cis) used in RoPE.
    Returns cos and sin components separately as real tensors.
    """
    # Calculate frequencies based on the formula: 1.0 / (theta^(2k / dim))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # Create a tensor representing positions [0, 1, ..., end-1]
    t = torch.arange(end, device=freqs.device)
    # Calculate the outer product of positions and frequencies: shape (end, dim / 2)
    freqs = torch.outer(t, freqs).float()
    # Compute cosine and sine values directly
    freqs_cos = torch.cos(freqs) # Shape: (end, dim / 2)
    freqs_sin = torch.sin(freqs) # Shape: (end, dim / 2)
    return freqs_cos, freqs_sin
# --- END MODIFIED FUNCTION ---

# --- MODIFIED FUNCTION: Accepts cos and sin tensors ---
def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
    """
    Applies Rotary Position Embedding (RoPE) to query (xq) and key (xk) tensors
    using precomputed cosine and sine frequencies.
    """
    # freqs_cos, freqs_sin shape: (seq_len, head_dim // 2)
    # xq shape: (bsz, seq_len, n_heads, head_dim)
    # xk shape: (bsz, seq_len, n_kv_heads, head_dim)

    # Add dimensions for broadcasting: (1, seq_len, 1, head_dim // 2)
    cos = freqs_cos.unsqueeze(0).unsqueeze(2)
    sin = freqs_sin.unsqueeze(0).unsqueeze(2)

    # Reshape xq and xk to view the last dimension as pairs
    # Shape: (bsz, seq_len, n_heads, head_dim // 2, 2)
    xq_reshaped = xq.float().reshape(*xq.shape[:-1], -1, 2)
    # Shape: (bsz, seq_len, n_kv_heads, head_dim // 2, 2)
    xk_reshaped = xk.float().reshape(*xk.shape[:-1], -1, 2)

    # Extract the two parts for rotation
    xq_part1 = xq_reshaped[..., 0]
    xq_part2 = xq_reshaped[..., 1]
    xk_part1 = xk_reshaped[..., 0]
    xk_part2 = xk_reshaped[..., 1]

    # Apply rotation using broadcasting
    xq_rotated_part1 = xq_part1 * cos - xq_part2 * sin
    xq_rotated_part2 = xq_part1 * sin + xq_part2 * cos
    xk_rotated_part1 = xk_part1 * cos - xk_part2 * sin
    xk_rotated_part2 = xk_part1 * sin + xk_part2 * cos

    # Combine the rotated parts back
    xq_rotated = torch.stack((xq_rotated_part1, xq_rotated_part2), dim=-1)
    xk_rotated = torch.stack((xk_rotated_part1, xk_rotated_part2), dim=-1)

    # Reshape back to original head_dim
    xq_out = xq_rotated.flatten(start_dim=-2)
    xk_out = xk_rotated.flatten(start_dim=-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)
# --- END MODIFIED FUNCTION ---


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, n_heads: int, dim: int, n_kv_heads: Optional[int] = None,
                 dropout: float = 0.0, max_seq_len: int = 2048, flash_attn: bool = True):
        super().__init__()
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        assert n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_local_heads = n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.dropout = dropout

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and flash_attn
        if not self.flash:
            mask = torch.full((1, 1, max_seq_len, max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask, persistent=False)

    def forward(self,
                x: torch.Tensor,
                freqs_cos: torch.Tensor, # Accept cos component
                freqs_sin: torch.Tensor, # Accept sin component
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False):

        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # Apply rotary embeddings using the modified function with cos/sin
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos=freqs_cos, freqs_sin=freqs_sin)

        # KV Caching implementation
        if past_key_value is not None:
            past_key, past_value = past_key_value
            if past_key.shape[1] > 0:
                 xk = torch.cat([past_key, xk], dim=1)
                 xv = torch.cat([past_value, xv], dim=1)

        current_key_value = (xk, xv) if use_cache else None

        # Reshape and repeat KV heads
        xq = xq.transpose(1, 2)
        key = repeat_kv(xk, self.n_rep).transpose(1, 2)
        value = repeat_kv(xv, self.n_rep).transpose(1, 2)

        full_seq_len = key.shape[-2]

        # Attention Calculation
        if self.flash and not use_cache and seq_len > 1:
            dropout_p = self.dropout if self.training else 0.0
            try:
                output = F.scaled_dot_product_attention(
                    xq, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=True
                )
            except RuntimeError as e:
                 print(f"Flash Attention failed: {e}. Falling back to manual attention.")
                 self.flash = False # Disable flash for subsequent calls if it fails
                 # Fall through to manual calculation block

        if not self.flash or use_cache or seq_len == 1:
            scores = (xq @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
            current_mask = self.mask[:, :, :seq_len, :full_seq_len]
            scores = scores + current_mask
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ value

        # Reshape output
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        output = self.resid_dropout(self.wo(output))

        return output, current_key_value


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None,
                 multiple_of: int = 256, dropout: float = 0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, n_heads: int, dim: int,
                 n_kv_heads: Optional[int] = None, dropout: float = 0.0,
                 max_seq_len: int = 2048, flash_attn: bool = True,
                 norm_eps: float = 1e-6, hidden_dim: Optional[int] = None,
                 multiple_of: int = 256):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = Attention(n_heads, dim, n_kv_heads, dropout, max_seq_len, flash_attn)
        self.feed_forward = FeedForward(dim, hidden_dim, multiple_of, dropout)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def forward(self,
                x: torch.Tensor,
                freqs_cos: torch.Tensor, # Accept cos component
                freqs_sin: torch.Tensor, # Accept sin component
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False):

        # Attention Part
        residual = x
        normed_x = self.attention_norm(x)
        h_attn, current_key_value = self.attention(
            normed_x,
            freqs_cos, # Pass cos
            freqs_sin, # Pass sin
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        h = residual + h_attn

        # FeedForward Part
        residual = h
        normed_h = self.ffn_norm(h)
        ffn_output = self.feed_forward(normed_h)
        out = residual + ffn_output

        return out, current_key_value


class MiniMindLM(nn.Module):
    def __init__(self, vocab_size: int, n_layers: int, n_heads: int, dim: int,
                 n_kv_heads: Optional[int] = None, dropout: float = 0.0,
                 max_seq_len: int = 2048, flash_attn: bool = True,
                 norm_eps: float = 1e-6, rope_theta: float = 10000.0, # Common value for RoPE theta
                 hidden_dim: Optional[int] = None, multiple_of: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.dropout = nn.Dropout(dropout)

        if n_kv_heads is None:
            n_kv_heads = n_heads
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        head_dim = dim // n_heads

        self.layers = nn.ModuleList([
            MiniMindBlock(l, n_heads, dim, n_kv_heads, dropout, max_seq_len,
                         flash_attn, norm_eps, hidden_dim, multiple_of)
            for l in range(n_layers)
        ])
        self.norm = RMSNorm(dim, eps=norm_eps)

        # --- Store cos and sin buffers ---
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=head_dim, end=max_seq_len, theta=rope_theta
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        # --- End buffer update ---

        # Print model config details
        print("-" * 20)
        print("MiniMindLM Configuration:")
        print(f"  vocab_size: {self.vocab_size}, n_layers: {self.n_layers}, n_heads: {n_heads}")
        print(f"  dim: {dim}, n_kv_heads: {n_kv_heads}, head_dim: {head_dim}")
        print(f"  max_seq_len: {self.max_seq_len}, dropout: {dropout}, flash_attn: {flash_attn}")
        print(f"  rope_theta: {rope_theta}")
        if hasattr(self, 'freqs_cos') and hasattr(self, 'freqs_sin'):
            print(f"  freqs_cos shape: {self.freqs_cos.shape}, freqs_sin shape: {self.freqs_sin.shape}")
        else:
            print("  freqs_cos/freqs_sin: Not Registered Yet")
        print("-" * 20)


    def forward(self,
                input_ids: torch.Tensor,
                past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
                use_cache: bool = False):

        bsz, seq_len = input_ids.shape

        start_pos = 0
        if past_key_values is not None and past_key_values[0] is not None:
             past_key_shape = past_key_values[0][0].shape
             if len(past_key_shape) == 4 and past_key_shape[1] > 0:
                  start_pos = past_key_shape[1]

        if start_pos + seq_len > self.max_seq_len:
             raise ValueError(
                 f"Input sequence length ({seq_len}) + cache length ({start_pos}) exceeds model max sequence length ({self.max_seq_len})"
             )

        # --- Slice cos and sin buffers ---
        freqs_cos = self.freqs_cos[start_pos : start_pos + seq_len]
        freqs_sin = self.freqs_sin[start_pos : start_pos + seq_len]
        # --- End slicing update ---

        h = self.tok_embeddings(input_ids)
        h = self.dropout(h)

        new_past_key_values = [] if use_cache else None

        for layer_idx, layer in enumerate(self.layers):
            layer_past_key_value = past_key_values[layer_idx] if past_key_values is not None else None

            h, current_key_value = layer(
                h,
                freqs_cos, # Pass sliced cos
                freqs_sin, # Pass sliced sin
                past_key_value=layer_past_key_value,
                use_cache=use_cache
            )

            if use_cache:
                new_past_key_values.append(current_key_value)

        encoded = self.norm(h)
        output_embedding = encoded[:, -1, :] # Use last token state

        if use_cache:
            return output_embedding, new_past_key_values
        else:
            return output_embedding


    def load(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
             print(f"Warning: Checkpoint file not found at {checkpoint_path}. Model weights will be random.")
             return False
        try:
            state = torch.load(checkpoint_path, map_location='cpu')
            my_state = self.state_dict()
            filtered_state = {}
            ignored_keys = []
            for k, v in state.items():
                if k in my_state:
                    if my_state[k].shape == v.shape:
                        filtered_state[k] = v
                    else:
                        print(f"Warning: Shape mismatch for key {k}. Checkpoint: {v.shape}, Model: {my_state[k].shape}. Skipping.")
                        ignored_keys.append(k + " (shape mismatch)")
                else:
                    # Allow ignoring freqs_cos/sin if checkpoint only has pos_cis (backward compatibility)
                    if k != "pos_cis": # Check if the ignored key is the old complex buffer
                         ignored_keys.append(k + " (not in model)")

            missing_keys, unexpected_keys = self.load_state_dict(filtered_state, strict=False)

            # Report issues, ignoring expected missing keys if loading from old checkpoint
            final_missing = [k for k in missing_keys if k not in ["freqs_cos", "freqs_sin"]]
            if final_missing:
                 print(f"Warning: Missing keys in checkpoint not loaded: {final_missing}")
            if ignored_keys:
                 print(f"Warning: Keys from checkpoint ignored: {ignored_keys}")
            if not final_missing and not ignored_keys:
                 print(f"Successfully loaded weights from {checkpoint_path}")
            return True

        except Exception as e:
            print(f"Error loading checkpoint from {checkpoint_path}: {e}")
            return False


class L1(MiniMindLM):
    def __init__(self, embedding_dim=512):
        dim = 512
        super().__init__(vocab_size=6400, n_layers=8, n_heads=8, dim=dim, n_kv_heads=2, dropout=0.1, hidden_dim=1408, max_seq_len=512, flash_attn=False)
        self.embedding_dim = embedding_dim
        self.fc = nn.Linear(dim, embedding_dim, bias=True)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        fn = os.path.join(script_dir, "pretrained_MiniMindLM.L1.pth")
        print(f"Attempting to load L1 pretrained weights from: {fn}")
        self.load(fn)

    def forward(self, input_ids, past_key_values=None, use_cache=False):
        if use_cache:
            hidden_state, cache = super().forward(input_ids, past_key_values=past_key_values, use_cache=use_cache)
            output_embedding = self.fc(hidden_state)
            return output_embedding, cache
        else:
            hidden_state = super().forward(input_ids, past_key_values=past_key_values, use_cache=use_cache)
            output_embedding = self.fc(hidden_state)
            return output_embedding


class L2(MiniMindLM):
    def __init__(self, embedding_dim=512):
        dim = 256
        super().__init__(vocab_size=6400, n_layers=8, n_heads=8, dim=dim, n_kv_heads=2, dropout=0, hidden_dim=512, max_seq_len=512, flash_attn=False)
        self.embedding_dim = embedding_dim
        self.fc = nn.Linear(dim, embedding_dim, bias=True)
        # Add loading logic if needed

    def forward(self, input_ids, past_key_values=None, use_cache=False):
        if use_cache:
            hidden_state, cache = super().forward(input_ids, past_key_values=past_key_values, use_cache=use_cache)
            output_embedding = self.fc(hidden_state)
            return output_embedding, cache
        else:
            hidden_state = super().forward(input_ids, past_key_values=past_key_values, use_cache=use_cache)
            output_embedding = self.fc(hidden_state)
            return output_embedding


# --- Usage Example (Mostly Unchanged) ---
if __name__ == "__main__":
    # --- Dummy Tokenizer and Files Setup ---
    tokenizer_dir = "tokenizer"
    pretrained_file = "pretrained_MiniMindLM.L1.pth"
    model_max_length = 512

    if not os.path.exists(tokenizer_dir):
        print(f"Creating dummy tokenizer directory: {tokenizer_dir}")
        os.makedirs(tokenizer_dir)
        with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w") as f:
            f.write('{"model_max_length": %d, "tokenizer_class": "BertTokenizer"}' % model_max_length)
        with open(os.path.join(tokenizer_dir, "vocab.txt"), "w") as f:
            f.write("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nHello\n,\nworld\n!\nI\nam\nvery\nhappy\n.\n</w>\n")
        with open(os.path.join(tokenizer_dir, "special_tokens_map.json"), "w") as f:
            f.write('{"unk_token": "[UNK]", "sep_token": "[SEP]", "pad_token": "[PAD]", "cls_token": "[CLS]", "mask_token": "[MASK]"}')

    if not os.path.exists(pretrained_file):
        print(f"Creating dummy pretrained weights file: {pretrained_file}")
        try:
            if '__file__' not in globals(): globals()['__file__'] = 'dummy_minimind.py'
            # Create dummy state dict *without* complex buffer to match new model structure
            dummy_model = L1(1152)
            # Save only a few keys, ensuring no complex tensors are included
            dummy_state_dict = {k: v for k, v in dummy_model.state_dict().items() if 'tok_embeddings' in k or 'fc' in k or 'freqs' not in k}
            torch.save(dummy_state_dict, pretrained_file)
            del dummy_model
            print(f"Dummy weights saved to {pretrained_file}")
        except Exception as e:
            print(f"Error creating dummy weights file: {e}")

    # --- Model Initialization ---
    print("\nInitializing L1 model...")
    try:
        if '__file__' not in globals(): globals()['__file__'] = 'dummy_minimind.py'
        model = L1(embedding_dim=1152)
        model.eval()
        print("Model initialized successfully.")
        print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    except Exception as e:
        print(f"Error initializing model: {e}")
        exit()

    # --- Tokenization ---
    input_str = ["Hello, world!", "I am very happy."]
    print(f"\nTokenizing input strings: {input_str}")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        print("Tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading tokenizer from '{tokenizer_dir}': {e}")
        exit()

    try:
        input_ids = tokenizer(
            input_str, return_tensors="pt", padding=True, truncation=True, max_length=model.max_seq_len
        ).input_ids
        print(f"Tokenization successful. Input IDs shape: {input_ids.shape}")
    except Exception as e:
        print(f"Error during tokenization: {e}")
        exit()

    # --- Model Inference ---
    print("\nRunning model inference (forward pass)...")
    try:
        with torch.no_grad():
             output = model(input_ids)
        print(f"Inference successful. Output shape: {output.shape}")
    except Exception as e:
        print(f"Error during model forward pass: {e}")
        import traceback
        traceback.print_exc()
        exit()

    # --- ONNX Export ---
    onnx_file = "model_text.onnx"
    print(f"\nAttempting ONNX export to {onnx_file}...")
    try:
        input_names = ['input_ids']
        output_names = ['last_hidden_state']
        dynamic_axes = {
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'last_hidden_state': {0: 'batch_size'}
        }
        torch.onnx.export(
            model, (input_ids,), onnx_file,
            input_names=input_names, output_names=output_names,
            dynamic_axes=dynamic_axes, export_params=True,
            opset_version=11, do_constant_folding=True, verbose=False
        )
        print(f'Successfully exported model to "{onnx_file}".')

        # Optional: Verify the ONNX model
        try:
            import onnx
            onnx_model = onnx.load(onnx_file)
            onnx.checker.check_model(onnx_model)
            print("ONNX model check passed.")
        except ImportError:
            print("ONNX Python package not found. Skipping model verification.")
        except Exception as e:
            print(f"ONNX model verification failed: {e}")

    except RuntimeError as e:
        print(f"ONNX export failed with RuntimeError: {e}")
        print("Common causes include unsupported PyTorch operations in the selected opset version (11).")
        print("Try updating the opset_version (e.g., 13, 17) if your ONNX runtime supports it.")
        print("Also, ensure Flash Attention is disabled in the model config (`flash_attn=False`).")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"An unexpected error occurred during ONNX export: {e}")
        import traceback
        traceback.print_exc()

# --- End of Updated Code ---
