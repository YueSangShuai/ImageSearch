import torch
import torch.nn as nn
from transformers import MobileViTModel, AutoConfig, PreTrainedModel
from safetensors.torch import load_file  # 注意这里！

class MobileViTWithHead(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.backbone = MobileViTModel(config)
        self.proj = nn.Linear(640, 768)  # 你自己加的层

    def forward(self, pixel_values, **kwargs):
        outputs = self.backbone(pixel_values, **kwargs)
        x = outputs.last_hidden_state
        x = x.mean(dim=[2, 3])  # GAP
        x = self.proj(x)
        return x

# === 1. 使用官方的 MobileViT 配置 ===
config = AutoConfig.from_pretrained("apple/mobilevit-small")

# === 2. 实例化你自己的模型 ===
model = MobileViTWithHead(config)

# === 3. 只加载主干模型（backbone）的权重 ===
state_dict = load_file("/data/yuesang/LLM/contrastors/src/ckpts/apple/mobilevit-small/epoch_0_model/model.safetensors")
state_dict = {k.replace("vision.trunk.", ""): v for k, v in state_dict.items()}
# for k, v in state_dict.items():
#     print(k)
load_info =model.backbone.load_state_dict(state_dict, strict=False)  # 不加载 proj 层

# === 4. 保存为 HuggingFace 格式模型 ===
save_dir = "./mobilevit_custom"
model.save_pretrained(save_dir)
config.save_pretrained(save_dir)

# === 3.1 打印加载统计信息 ===
total_params_in_ckpt = len(state_dict)
missing = load_info.missing_keys
unexpected = load_info.unexpected_keys

print(model)

# print(f"Total parameters in checkpoint: {total_params_in_ckpt}")
# print(f"Successfully loaded: {total_params_in_ckpt - len(unexpected)}")
# print(f"Missing keys: {len(missing)}")
# print(f"Unexpected keys: {len(unexpected)}")
# print("Missing keys (example):", missing[:10])
# print("Unexpected keys (example):", unexpected[:10])

import os
dummy_input = torch.randn(1, 3, 224, 224)  # MobileViT 默认输入尺寸

onnx_path = os.path.join(save_dir, "mobilevit_with_head.onnx")
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["pixel_values"],
    output_names=["proj_features"],
    opset_version=13,
    do_constant_folding=True,
    dynamic_axes={
        "pixel_values": {0: "batch_size"},
        "proj_features": {0: "batch_size"}
    }
)

print(f"ONNX 模型已保存到: {onnx_path}")
