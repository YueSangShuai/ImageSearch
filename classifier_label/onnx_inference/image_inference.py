from transformers import AutoImageProcessor
from PIL import Image
import numpy as np
import onnxruntime as ort
import time

# === 配置 ===
onnx_path = "/data/yuesang/LLM/VectorIE/models/nomic/nomic-embed-vision-v1.5/onnx/model_uint8.onnx"  # 替换为你的 ONNX 模型路径
model_path = "/data/yuesang/LLM/VectorIE/models/nomic/nomic-embed-vision-v1.5"       # 用于加载预处理器
device = "cpu"  # or "cpu"
image_path = "/data/yuesang/LLM/VectorIE/classifier_label/image-3.png"  # 替换为你的图像路径

# === 图像预处理 ===
image = Image.open(image_path).convert("RGB")
processor = AutoImageProcessor.from_pretrained(model_path)
inputs = processor(images=image, return_tensors="np")  # -> {"pixel_values": np.array of shape (1, 3, 224, 224)}

# === 创建 ONNX session ===
providers = ["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
session = ort.InferenceSession(onnx_path, providers=providers)

input_name = session.get_inputs()[0].name   # 通常是 'pixel_values'
output_name = session.get_outputs()[0].name # 通常是 'embeddings' or 'last_hidden_state'

onnx_inputs = {input_name: inputs["pixel_values"]}


warn_up_range=50
test_range=500
from tqdm import tqdm


ouput = session.run([output_name], onnx_inputs)
print(len(ouput))
print(ouput[0].shape)

# === 预热 ===
for _ in tqdm(range(warn_up_range)):
    _ = session.run([output_name], onnx_inputs)

# === 正式推理 + 计时 ===
start = time.time()
for _ in tqdm(range(test_range)):
    _ = session.run([output_name], onnx_inputs)
end = time.time()

avg_time = (end - start) / test_range
print(f"🌡️ 平均推理耗时: {avg_time * 1000:.2f} ms")

# === 获取最终 embedding ===
output = session.run([output_name], onnx_inputs)
embedding = output[0]  # (1, 768)

