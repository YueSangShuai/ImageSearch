import torch
import numpy as np
import onnxruntime as ort
import cv2

# --------------------------
# 配置参数（请根据实际情况修改）
# --------------------------
ONNX_MODEL_PATH = "/data/yuesang/LLM/contrastors/emov2_hf/emov2_batch_one.onnx"  # ONNX模型路径
IMAGE_PATH = "/data/yuesang/LLM/contrastors/calling_man.png"  # 待推理的图像路径
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"  # 推理设备（CPU/GPU）


# --------------------------
# 1. OpenCV预处理（完全替代PIL）
# --------------------------
def opencv_preprocess(image_path, target_size=224):
    """
    简化版预处理：直接resize到224x224（不保持宽高比）
    包含：普通resize + RGB转换 + 归一化
    """
    # 1. 读取图像（确保路径为字符串）
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}（检查路径或图像是否损坏）")
    
    # 2. 转换为RGB格式（对应PyTorch的img.convert("RGB")）
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 3. 直接resize到224x224（不保持宽高比，使用双三次插值）
    img_resized = cv2.resize(
        img_rgb, 
        (target_size, target_size),  # 直接缩放到224x224
        interpolation=cv2.INTER_CUBIC  # 双三次插值，保证缩放质量
    )
    
    # 4. 转换为float并归一化到[0, 1]（对应PyTorch的ToTensor）
    img_float = img_resized.astype(np.float32) / 255.0
    
    # 5. 标准化（减均值，除以标准差）
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    for i in range(3):
        img_float[:, :, i] = (img_float[:, :, i] - mean[i]) / std[i]
    
    # 6. 调整通道顺序为[C, H, W]（匹配PyTorch张量格式）
    img_transposed = img_float.transpose(2, 0, 1)
    
    return img_transposed


# --------------------------
# 2. ONNX推理工具类
# --------------------------
class Emov2OnnxInferencer:
    def __init__(self, model_path, device="cuda:0"):
        if "cuda" in device and ort.get_device() == "GPU":
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': int(device.split(":")[-1]),
                    'do_copy_in_default_stream': True
                }),
                'CPUExecutionProvider'
            ]
        else:
            providers = ['CPUExecutionProvider']

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]

        print(f"[INFO] 加载ONNX模型: {model_path}")
        print(f"[INFO] 推理设备: {'GPU' if 'CUDAExecutionProvider' in providers else 'CPU'}")
        print(f"[INFO] 输入名称: {self.input_name}")
        print(f"[INFO] 输出数量: {len(self.output_names)}")

    def infer(self, image_np):
        """接收OpenCV预处理后的NumPy数组，执行推理"""
        # 增加批次维度（[3,224,224] -> [1,3,224,224]）
        input_data = np.expand_dims(image_np, axis=0)
        outputs = self.session.run(self.output_names, {self.input_name: input_data})
        result = {name: torch.from_numpy(out) for name, out in zip(self.output_names, outputs)}
        return result


# --------------------------
# 3. 辅助函数：安全获取张量值
# --------------------------
def get_tensor_value(tensor):
    squeezed = tensor.squeeze(0)  # 移除批次维度
    
    if squeezed.ndim == 0:  # 标量张量
        return squeezed.item()
    elif squeezed.ndim == 1 and squeezed.numel() == 1:  # 单元素张量
        return squeezed[0].item()
    else:  # 多元素张量（分类logits）
        max_val, max_idx = torch.max(squeezed, dim=0)
        return {
            "type": "classification",
            "class_index": max_idx.item(),
            "max_probability": max_val.item(),
            "all_values": squeezed.tolist()
        }


# --------------------------
# 4. 主推理流程（完全使用OpenCV）
# --------------------------
def main():
    # 步骤1：使用OpenCV加载并预处理图像（修复核心错误）
    try:
        # 检查路径是否为字符串（OpenCV要求路径为字符串）
        if not isinstance(IMAGE_PATH, str):
            raise TypeError(f"图像路径必须是字符串，当前类型: {type(IMAGE_PATH)}")
        
        # 使用OpenCV加载并预处理
        image_np = opencv_preprocess(IMAGE_PATH)  # 返回[3,224,224]的NumPy数组
        print(f"[INFO] 加载并预处理图像: {IMAGE_PATH}")
        print(f"[INFO] 预处理后形状: {image_np.shape}，数据类型: {image_np.dtype}")
    except Exception as e:
        print(f"[ERROR] 图像加载/预处理失败: {str(e)}")
        return

    # 步骤2：初始化ONNX推理器
    try:
        inferencer = Emov2OnnxInferencer(ONNX_MODEL_PATH, device=DEVICE)
    except Exception as e:
        print(f"[ERROR] ONNX模型加载失败: {str(e)}")
        return

    # 步骤3：执行推理
    try:
        result = inferencer.infer(image_np)  # 传入OpenCV预处理的结果
        print(f"[INFO] 推理完成，输出包含: {list(result.keys())[:5]}...")
    except Exception as e:
        print(f"[ERROR] 推理失败: {str(e)}")
        return

    # 步骤4：解析并输出结果
    print("\n" + "="*50)
    print("【推理结果解析】")
    print("="*50)

    # 1. 处理embeddings
    embeddings = result["embeddings"]
    print(f"1. embeddings: 形状={embeddings.shape}, 前5个值={embeddings[0, :5].tolist()}")

    # 2. 处理所有属性
    key_attributes = [
        "Age18-60", "AgeLess18", "AgeOver60", "Back", "Backpack", 
        "Female", "Front", "Glasses", "HandBag", "Hat", "HoldObjectsInFront", 
        "LongCoat", "LongSleeve", "LowerPattern", "LowerStripe", "ShortSleeve", 
        "Shorts", "ShoulderBag", "Side", "Skirt&Dress", "Trousers", "UpperLogo", 
        "UpperPlaid", "UpperSplice", "UpperStride", "boots", "lbeige", "lblack", 
        "lblue", "lbrown", "lgray", "lgreen", "lmulticolor", "lorange", "lpink", 
        "lpurple", "lred", "lstriped_color", "lwhite", "lyellow", "ubeige", "ublack", 
        "ublue", "ubrown", "ugray", "ugreen", "umulticolor", "uorange", "upink", 
        "upurple", "ured", "ustriped_color", "uwhite", "uyellow"
    ]
    
    print("\n2. 属性预测结果:")
    for attr in key_attributes:
        if attr in result:
            value = get_tensor_value(result[attr])
            if isinstance(value, dict) and value["type"] == "classification":
                print(f"   - {attr}: 最可能类别={value['class_index']}, 概率={value['max_probability']:.4f}")
            else:
                print(f"   - {attr}: {value:.4f}")

    # 3. 输出所有可用属性
    print("\n3. 所有可用属性（共{}个）:".format(len(result)))
    print(f"   {', '.join(result.keys())}")


if __name__ == "__main__":
    main()