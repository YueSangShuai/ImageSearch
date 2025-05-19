import torch
from torch import nn
from reformer_pytorch import ReformerLM

class L1(nn.Module): # 25.197M
    def __init__(self, embedding_dim=512, max_seq_length=1024, vocab_size=6400, cls_token_id=1,
        n_layers=8, n_heads=8, dim=512, dropout=0.1, hidden_dim=1408):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.fc = nn.Linear(dim, embedding_dim, bias=False)
        self.encoder = ReformerLM(
            num_tokens = vocab_size,
            dim = dim,              # smaller hidden dimension
            heads = n_heads,              # less heads
            dim_head = dim // n_heads,
            depth = n_layers,
            max_seq_len = max_seq_length,
            return_embeddings = True,
            ff_mult = 4,
            ff_dropout = dropout,
        )
        self.export_ = False
    def forward(self, input_ids):
        # if self.export_:
        #     h = input_ids
        #     h = h[:,0]
        # else:
        #     h = self.prepare_inputs(input_ids)
        # padding to self.max_seq_length
        slen = input_ids.shape[1]
        if slen < self.max_seq_length:
            padding = torch.zeros((input_ids.shape[0], self.max_seq_length - slen), dtype=input_ids.dtype, device=input_ids.device)
            input_ids = torch.concat([input_ids, padding], dim=1)
        else:
            input_ids = input_ids[:, :self.max_seq_length]
        h = self.encoder(input_ids)
        h = self.fc(torch.mean(h, dim=1))
        return h
    def prepare_inputs(self, input_ids):
        return input_ids
    def export(self, input_ids):
        self.export_ = True
        ret = self.prepare_inputs(input_ids)
        return  ret
        

class L2(L1):
    def __init__(self, embedding_dim=512, max_seq_length=1024, vocab_size=6400, cls_token_id=1):
        super().__init__(embedding_dim=embedding_dim, max_seq_length=max_seq_length, vocab_size=vocab_size, cls_token_id=cls_token_id, 
            n_layers=8, n_heads=8, dim=256, dropout=0, hidden_dim=512)

class L3(L1): 
    def __init__(self, embedding_dim=512, max_seq_length=1024, vocab_size=6400, cls_token_id=1):
        super().__init__(embedding_dim=embedding_dim, max_seq_length=max_seq_length, vocab_size=vocab_size, cls_token_id=cls_token_id, 
            n_layers=12, n_heads=8, dim=512, dropout=0.2, hidden_dim=1024)

class L10(L1):
    def __init__(self, embedding_dim=512, max_seq_length=1024, vocab_size=6400, cls_token_id=1):
        super().__init__(embedding_dim=embedding_dim, max_seq_length=max_seq_length, vocab_size=vocab_size, cls_token_id=cls_token_id, 
            n_layers=2, n_heads=4, dim=64, dropout=0, hidden_dim=128)

# --- Usage Example (Mostly Unchanged) ---
if __name__ == "__main__":
    import os
    # --- Dummy Tokenizer and Files Setup ---
    tokenizer_dir = "tokenizer"
    pretrained_file = "pretrained_MiniMindLM.L1.pth"
    model_max_length = 512

    # --- Model Initialization ---
    print("\nInitializing L2 model...")
    try:
        model = L2(embedding_dim=1152, max_seq_length=128, vocab_size=6400, cls_token_id=1,)
        model.eval()
        print(model)
        print("Model initialized successfully.")
        print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    except Exception as e:
        print(f"Error initializing model: {e}")
        exit()

    # --- Tokenization ---
    input_str = ["Hello, world!", "I am very happy."]
    print(f"\nTokenizing input strings: {input_str}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    print("Tokenizer loaded successfully.")

    try:
        input_ids = tokenizer(
            input_str, 
            return_tensors="pt", 
            padding=True, #'max_length',  # 强制填充到max_length
            truncation=True, 
            max_length=model.max_seq_length,
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
    input_ids = input_ids[:1]
    print(f"\nAttempting ONNX export to {onnx_file}...")
    try:
        input_names = ['data']
        output_names = ['output']
        dynamic_axes = {
            'data': {1: 'sequence_length'},
        }
        print(f"Exported input IDs shape: {input_ids.shape}")
        input_ids = model.export(input_ids)
        print(f"Exported input IDs shape: {input_ids.shape}")
        torch.onnx.export(
            model, (input_ids,), onnx_file,
            input_names=input_names, output_names=output_names,
            dynamic_axes=dynamic_axes, export_params=True,
            opset_version=17, do_constant_folding=True, verbose=False
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

        import onnxsim
        import onnx
        onnx_model = onnx.load(onnx_file)
        model_simp, checked = onnxsim.simplify(onnx_model)
        if checked: 
            onnx.save(model_simp, "model_text_s.onnx")
            onnxsim.model_info.print_simplifying_info(onnx_model, model_simp)
        # check onnx model
        print("test onnx model ... ...")
        import onnxruntime
        sess = onnxruntime.InferenceSession("model_text_s.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        result = sess.run(output_names, {'data': input_ids.detach().numpy()})
        for k, out in zip(output_names, result):
            print("%s%s:\tmin=%0.5f, max=%0.5f"%(k, list(out.shape), float(out.min()), float(out.max())))

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
