import torch
import torch.nn.functional as F
import os
import json
from model import Model
from data import PairedImageTextDataset
from torch.utils.data import Dataset, DataLoader
from torchvision import utils as tvutils
import numpy as np

def export_(model, x, export_onnx=None, export_torch=None):
    """
    Export a PyTorch model to ONNX or TorchScript format.
    
    Args:
        model: The PyTorch model to export.
        x: Input tensor for tracing.
        export_onnx: Path to save the ONNX model (optional).
        export_torch: Path to save the TorchScript model (optional).
    """
    model = model.cpu()
    model.eval()
    
    output = os.path.dirname(export_onnx) if export_onnx else os.path.dirname(export_torch)
    if hasattr(model, 'export'): x = model.export(x)


    print("input%s:\tmin=%0.5f, max=%0.5f"%(list(x.shape), float(x.min()), float(x.max())))
    dynamic_axes={}
    if len(x.shape)==4: # image
        # dynamic_axes={'data': {0:"batch_size"}, }
        for i, img in enumerate(x):
            fn=os.path.join(output, 'export_images_%d.png' % i)
            tvutils.save_image(img, fn, normalize=True)
            print("save samples to ", fn)
    else:
        text_embeddings = {}
        for p in model.named_parameters():
            print(f"\tparameter: {p[0]}, {p[1].shape}")
            if p[0].startswith('byte_embedding') or p[0].startswith('pos_encoding') or p[0].startswith('tok_embeddings'):
                text_embeddings[p[0]] = p[1].detach().cpu().numpy()
        np.save(os.path.join(output, 'input_embeddings.pt'), text_embeddings)
        dynamic_axes={'data': {1:"seq_length"}}
    y=model(x)
    if type(y)==dict:
        for k, out in y.items():
            print("output %s%s:\tmin=%0.5f, max=%0.5f"%(k, list(out.shape), float(out.min()), float(out.max())))
        output_names = list(y.keys())
    elif type(y) in (list, tuple):
        for k, out in enumerate(y):
            print("output %s%s:\tmin=%0.5f, max=%0.5f"%(k, list(out.shape), float(out.min()), float(out.max())))
        output_names = ['out%d'%i for i in range(len(y))]
    else:
        print("output %s:\tmin=%0.5f, max=%0.5f"%(list(y.shape), float(y.min()), float(y.max())))
        output_names = ["output"]

    if export_onnx:
        try:
            torch.onnx.export(model, x, export_onnx, dynamic_axes=dynamic_axes, #dynamic_axes,
                export_params=True, opset_version=11, do_constant_folding=True, verbose=False,
                input_names = ['data'], output_names=output_names)
        except Exception as e:
            torch.onnx.export(model, x, export_onnx, dynamic_axes={}, #dynamic_axes,
                export_params=True, opset_version=14, do_constant_folding=True, verbose=False,
                input_names = ['data'], output_names=output_names)
        print('export to "%s" OK.'%export_onnx)
        import onnxsim
        import onnx
        onnx_model = onnx.load(export_onnx)
        model_simp, checked = onnxsim.simplify(onnx_model)
        if checked: 
            onnx.save(model_simp, export_onnx)
            onnxsim.model_info.print_simplifying_info(onnx_model, model_simp)
        # check onnx model
        print("test onnx model ... ...")
        import onnxruntime
        sess = onnxruntime.InferenceSession(export_onnx, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        result = sess.run(output_names, {'data': x.numpy()})
        for k, out in zip(output_names, result):
            print("%s%s:\tmin=%0.5f, max=%0.5f"%(k, list(out.shape), float(out.min()), float(out.max())))
    if export_torch:
        # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
        traced_script_module = torch.jit.trace(model, x)
        traced_script_module.save(export_torch)
        print('export to "%s" OK.'%export_torch)
        output = traced_script_module(x)
        for k, out in zip(output_names, output):
            print("output %s%s:\tmin=%0.5f, max=%0.5f"%(k, list(out.shape), float(out.min()), float(out.max())))

def export(val_dataloader, model, output):
    i = 0
    model.eval()
    datas = []
    for items in val_dataloader:
        person_ids = items['id']
        image = items['image']
        caption = items['caption']
        image_embedding = model.image_encoder(image)
        caption_embedding = model.text_encoder(caption)
        break
    export_(model.image_encoder, image[:1], export_onnx=os.path.join(output, "model_image.onnx"))
    export_(model.text_encoder, caption[:1], export_onnx=os.path.join(output, "model_text.onnx"))

def load_trained_model(path, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    args = state['args']
    model = Model(args).to(device)
    model.load_state_dict(state['state_dict'])
    val_dataset = PairedImageTextDataset(args.data, args.embedding_path, text_len=args.max_seq_length, train=False, image_size=args.image_size)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        drop_last=False,
        collate_fn=val_dataset.collate_fn
    )
    return model, val_dataloader
    

if __name__ == '__main__':
    import sys
    device = "cpu"
    if len(sys.argv) < 3:
        print("Usage: python evaluation_model.py <model_path> <output>")
        sys.exit(1)
    output = sys.argv[2]
    model, val_dataloader = load_trained_model(sys.argv[1], device)
    with torch.no_grad():
        export(val_dataloader, model, output)

#
# richard@ubuntu02:/cache/richard/work/person$ python export.py logs/expModel-backbone.mf3.ANet2-224x112-512/8/35000.pt logs/expModel-backbone.mf3.ANet2-224x112-512/8/
# ... ...
# export to "logs/expModel-backbone.mf3.ANet2-224x112-512/8/model_text.onnx" OK.
# ┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
# ┃            ┃ Original Model ┃ Simplified Model ┃
# ┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
# │ Add        │ 67             │ 67               │
# │ Constant   │ 264            │ 176              │
# │ Conv       │ 14             │ 14               │
# │ MatMul     │ 49             │ 49               │
# │ Max        │ 1              │ 1                │
# │ Mul        │ 61             │ 61               │
# │ ReduceMax  │ 1              │ 1                │
# │ Relu       │ 7              │ 7                │
# │ Reshape    │ 39             │ 39               │
# │ Slice      │ 8              │ 8                │
# │ Softmax    │ 6              │ 6                │
# │ Sub        │ 6              │ 0                │
# │ Tanh       │ 18             │ 18               │
# │ Transpose  │ 37             │ 37               │
# │ Trilu      │ 6              │ 0                │
# │ Unsqueeze  │ 12             │ 0                │
# │ Model Size │ 27.8MiB        │ 15.6MiB          │
# └────────────┴────────────────┴──────────────────┘
# test onnx model ... ...
# output[1, 512]: min=-4.49576, max=41.99710
