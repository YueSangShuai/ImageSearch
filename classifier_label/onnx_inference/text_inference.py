import time
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np

# 配置
device = "cpu"  # 或 "cpu"
texts = ["Hello world 666."]
onnx_path = "/data/yuesang/LLM/VectorIE/siglip2_text_encoder.onnx"
model_path = "/data/yuesang/LLM/VectorIE/models/mexma-siglip2"

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)

# 编码输入
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='np')
print(inputs)

# 创建 ONNX 推理会话（指定使用 CUDA 或 CPU）
providers = ['CUDAExecutionProvider'] if device == "cuda" else ['CPUExecutionProvider']
session = ort.InferenceSession(onnx_path, providers=providers)

# 获取输入输出名
input_names = [i.name for i in session.get_inputs()]
output_names = [o.name for o in session.get_outputs()]


onnx_inputs = {name: inputs[name] for name in input_names}
output = session.run(output_names, onnx_inputs)

print(output)
# from sklearn.metrics.pairwise import cosine_similarity
# vectors = []
# with open("/data/yuesang/LLM/VectorIE/c++/build/output.txt", "r") as f:
#     for line in f:
#         vec = np.array([float(x) for x in line.strip().split()])
#         vectors.append(vec)

# vectors = np.stack(vectors).reshape(output[0].shape)


# vec1 = vectors.reshape(-1, vectors.shape[-1])     # 来自 C++
# vec2 = output[0].reshape(-1, output[0].shape[-1]) # 来自 Python

# print(vec1.shape)
# print(vec2.shape)

# 比较每一对向量
# similarities = cosine_similarity(vec1, vec2)

# print("余弦相似度矩阵：")
# print(similarities)
# print(output[0])
# print(similarities)
# print(vectors.shape)
# print(output[0].shape)

# warn_up_range=50
# test_range=500
# from tqdm import tqdm
# # === 预热 ===
# for _ in tqdm(range(warn_up_range)):
#     _ = session.run(output_names, onnx_inputs)

# # === 正式计时 ===
# start = time.time()
# for _ in tqdm(range(test_range)):
#     _ = session.run(output_names, onnx_inputs)
# end = time.time()

# avg_time = (end - start) / test_range
# print(f"🌡️ 平均推理耗时: {avg_time*1000:.2f} ms/次")

# # 输出一条 embedding 看看
# outputs = session.run(output_names, onnx_inputs)
# embeddings = outputs[0]
# # print("Embedding shape:", embeddings.shape)
# # print("First sentence embedding:", embeddings[0])
