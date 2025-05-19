from transformers import AutoModel, AutoTokenizer
import torch

# 加载本地模型
model = AutoModel.from_pretrained("/data/yuesang/LLM/VectorIE/models/mexma-siglip2")
tokenizer = AutoTokenizer.from_pretrained("/data/yuesang/LLM/VectorIE/models/mexma-siglip2", local_files_only=True)

# 拿到 text encoder
text_encoder = model.text_model.eval().cpu()  # 可改为 .cuda() 如需 GPU 导出

# 准备输入
text_inputs = tokenizer(text=["Hello world 666."], return_tensors="pt")
# input_ids = text_inputs["input_ids"]
# attention_mask = text_inputs["attention_mask"]

# # 导出 ONNX
# torch.onnx.export(
#     text_encoder,
#     args=(input_ids, attention_mask),
#     f="siglip2_text_encoder.onnx",
#     input_names=["input_ids", "attention_mask"],
#     output_names=["last_hidden_state"],
#     dynamic_axes={
#         "input_ids": {0: "batch", 1: "sequence"},
#         "attention_mask": {0: "batch", 1: "sequence"},
#         "last_hidden_state": {0: "batch", 1: "sequence"}
#     },
#     opset_version=15
# )



text_embedding = model.encode_texts(normalize=True, **text_inputs)
text_embedding = text_embedding.float().cpu()
pritn(text_embedding)
print("✅ 导出完成：siglip2_text_encoder.onnx")
