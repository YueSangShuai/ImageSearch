from transformers import AutoImageProcessor
import torch
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor,AutoConfig
# ==== 加载配置和模型 ====
ckpt_path = "/data/yuesang/LLM/contrastors/nomic-vision-embv1.5"  # 替换成你的模型路径
onnx_output_path = "dino-vits8.onnx"

config = AutoConfig.from_pretrained(ckpt_path)
model=AutoModel.from_pretrained(
            ckpt_path, 
            torch_dtype=torch.float16, 
            trust_remote_code=True, 
            local_files_only=True
        )
model.eval()

# ==== 构造一个 dummy 输入（假设输入为图像） ====
image_processor = AutoImageProcessor.from_pretrained(ckpt_path)
dummy_image = torch.randn(1, 3, 224, 224)  # 假设输入为 ViT 类型模型，3通道224x224图像

# ==== 导出为 ONNX ====
torch.onnx.export(
    model,
    dummy_image,  # 如果 forward 接收的是 `pixel_values`
    onnx_output_path,
    input_names=["pixel_values"],
    output_names=["image_features"],
    dynamic_axes={"pixel_values": {0: "batch_size"}, "image_features": {0: "batch_size"}},
    opset_version=14
)

print(f"✅ Vision encoder 导出成功: {onnx_output_path}")
