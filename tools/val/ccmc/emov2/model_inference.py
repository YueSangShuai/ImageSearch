from torchvision import transforms
import torch
import yaml
from contrastors.config import Config
from contrastors.dataset.image_text_loader import get_local_image_text_dataset
from transformers import AutoTokenizer
import torch.nn.functional as F
import argparse
from contrastors.models.dual_encoder import DualEncoderConfig,DualEncoder
from transformers import PreTrainedModel
from tabulate import tabulate 
from PIL import Image


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
        self.model = self.get_image_model(args.vision_model).to(self.device)
        self.model = self.model.to(dtype=torch.bfloat16)
        self.tokenizer=AutoTokenizer.from_pretrained(self.text_args.model_name, local_files_only=True, trust_remote_code=True)
        
    def get_image_model(self,ckpt_path):
        config = DualEncoderConfig.from_pretrained(ckpt_path)
        config = DualEncoderConfig.from_pretrained(ckpt_path)
        model = DualEncoder.from_pretrained(ckpt_path, config=config)
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
        