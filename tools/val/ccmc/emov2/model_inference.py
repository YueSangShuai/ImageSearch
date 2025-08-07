from torchvision import transforms
import torch
import yaml
from contrastors.config import Config
from transformers import AutoTokenizer
from contrastors.models.dual_encoder import DualEncoderConfig,DualEncoder
from PIL import Image
from safetensors.torch import load_file

# 如果你使用的是 bicubic 插值
bicubic = transforms.InterpolationMode.BICUBIC
preprocess = transforms.Compose([
    transforms.Resize(size=224, interpolation=bicubic, antialias=True),  # 缩放图像到较大的边为 224
    transforms.CenterCrop(size=(224, 224)),  # 居中裁剪为 224x224
    transforms.Lambda(lambda img: img.convert("RGB")),  # 确保图像是 RGB
    transforms.ToTensor(),  # 转换为 [0, 1] 的 Tensor，形状为 [C, H, W]
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],  # 通常使用 ImageNet 的均值与方差
                         std=[0.26862954, 0.26130258, 0.27577711]),
])

def tokenize(text, tokenizer, add_eos=False, add_prefix=False):
    if add_eos:
        text = text + tokenizer.eos_token
    if add_prefix:
        text = f"search_query: {text}"
    tokenized = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")
    if add_eos:
        tokenized["input_ids"][:, -1] = tokenizer.eos_token_id

    return tokenized

def read_config(path):
    # read yaml and return contents
    with open(path, 'r') as file:
        try:
            return Config(**yaml.safe_load(file))
        except yaml.YAMLError as exc:
            print(exc)

class Emove_inference:
    def __init__(self,args):
        config=read_config(args.yaml_path)
        self.text_args=config.text_model_args
        self.device=args.device
        self.model = self.get_image_model(config,args.vision_model).to(self.device)
        self.model = self.model.to(dtype=torch.bfloat16)
        self.tokenizer=self.get_tokenizer(config)
    
    
    def get_tokenizer(self, config):
        config = config.text_model_args
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        tokenizer.model_max_length = config.seq_len

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if tokenizer.cls_token is None:
            tokenizer.add_special_tokens({"cls_token": "<s>"})

        if tokenizer.mask_token is None:
            tokenizer.add_special_tokens({"mask_token": "<mask>"})

        return tokenizer
    
    def get_image_model(self,ckpt_path,modelsafe):
        config = DualEncoderConfig(ckpt_path)
        model = DualEncoder(config)
        
        state_dict = load_file(modelsafe, device="cpu")
        # 检查模型状态字典与加载的权重匹配情况
        model_state_dict = model.state_dict()
        loaded_keys = set(state_dict.keys())
        model_keys = set(model_state_dict.keys())
        
        
        # 找出匹配的参数（名称和形状均一致）
        matched_keys = []
        mismatched_shape_keys = []
        for key in loaded_keys & model_keys:
            if state_dict[key].shape == model_state_dict[key].shape:
                matched_keys.append(key)
            else:
                mismatched_shape_keys.append(key)

        # 未在模型中找到的参数
        unmatched_loaded_keys = loaded_keys - model_keys
        # 模型中未被加载的参数
        unmatched_model_keys = model_keys - loaded_keys

        # 打印结果
        print(f"成功加载的参数（{len(matched_keys)}个）：")
        for key in sorted(matched_keys):
            print(f"  - {key} (形状: {state_dict[key].shape})")

        if mismatched_shape_keys:
            print(f"\n形状不匹配的参数（{len(mismatched_shape_keys)}个）：")
            for key in sorted(mismatched_shape_keys):
                print(f"  - {key}: 权重形状{state_dict[key].shape} vs 模型形状{model_state_dict[key].shape}")

        if unmatched_loaded_keys:
            print(f"\n权重中存在但模型中不存在的参数（{len(unmatched_loaded_keys)}个）：")
            for key in sorted(unmatched_loaded_keys):
                print(f"  - {key}")

        if unmatched_model_keys:
            print(f"\n模型中存在但权重中不存在的参数（{len(unmatched_model_keys)}个）：")
            for key in sorted(unmatched_model_keys):
                print(f"  - {key}")

        # 实际加载权重（可选，若尚未加载）
        try:
            model.load_state_dict(state_dict, strict=True)
            print("\n权重加载完成（strict=True）")
        except RuntimeError as e:
            print(f"\n加载失败：{e}")
        
        
        return model
    
    def inference_image(self, img_path):
        # 加载并预处理图像
        image = Image.open(img_path).convert("RGB")
        image = preprocess(image)  # 假设预处理后得到的是CPU上的tensor
        image = image.unsqueeze(0) 
        # 获取模型所在设备
        vision = self.model.vision
        device = next(vision.parameters()).device
        # 将输入图像转移到模型设备，保持与模型一致的数据类型
        image = image.to(device, dtype=vision.dtype)  # 使用模型的dtype确保类型匹配
        
        vision.eval()
        with torch.no_grad():  # 推理时关闭梯度计算
            image_emb = vision(image)["embedding"]
            # 无需再转移到self.device，因为模型已经在正确设备上
        
        return image_emb

    def inference_text(self, txt_info):
        # 文本 tokenize 处理
        tokenized = tokenize(txt_info, self.tokenizer, self.text_args.nomic_encoder == False, self.text_args.add_prefix)
        tokenized = {k: v.squeeze(0).to(self.device) for k, v in tokenized.items()}  # 直接转移到设备
        
        text = self.model.text
        device = next(text.parameters()).device
        # 确保输入文本数据在模型所在设备
        for k, v in tokenized.items():
            tokenized[k] = v.to(device, dtype=torch.long).unsqueeze(0)
            
        
        text.eval()
        with torch.no_grad():  # 推理时关闭梯度计算
            text_emb = text(** tokenized)["embedding"]  # 注意这里应该传入tokenized而不是原始文本
        
        return text_emb
        