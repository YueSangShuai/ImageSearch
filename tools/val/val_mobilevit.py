import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor,AutoConfig
from PIL import Image
import os
import argparse
from typing import List
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
from transformers import MobileViTModel, AutoConfig, PreTrainedModel
from safetensors.torch import load_file

# Label dictionaries
age_dict = {
    "ageless15": 0,
    "age16-30": 1,
    "age31-45": 2,
    "age46-60": 3,
    "ageabove60": 4
}

gender_dict = {"female": 0, "male": 1}
age_text_en = {
    "ageless15": "this is a person under 15 years old",
    "age16-30": "this is a person between 16 and 30 years old",
    "age31-45": "this is a person between 31 and 45 years old",
    "age46-60": "this is a person between 46 and 60 years old",
    "ageabove60": "this is a person above 60 years old"
}

gender_text_en = {
    "female": "this is a female",
    "male": "this is a male"
}




# 如果你使用的是 bicubic 插值
bicubic = transforms.InterpolationMode.BICUBIC

# 定义预处理 pipeline
preprocess = transforms.Compose([
    transforms.Resize(size=224, interpolation=bicubic, antialias=True),  # 缩放图像到较大的边为 224
    transforms.CenterCrop(size=(224, 224)),  # 居中裁剪为 224x224
    transforms.Lambda(lambda img: img.convert("RGB")),  # 确保图像是 RGB
    transforms.ToTensor(),  # 转换为 [0, 1] 的 Tensor，形状为 [C, H, W]
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],  # 通常使用 ImageNet 的均值与方差
                         std=[0.26862954, 0.26130258, 0.27577711]),
])



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

class Eval:
    def __init__(self,args):
        image_model_name = args.vision_model
        text_model_name = args.text_model
        
        
        config = AutoConfig.from_pretrained("apple/mobilevit-small")
        # === 2. 实例化你自己的模型 ===
        image_model = MobileViTWithHead(config)
        state_dict=load_file(image_model_name)
        image_model.load_state_dict(state_dict, strict=False)  # strict=False 可跳过不匹配项
        
        text_model = AutoModel.from_pretrained(
            text_model_name, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True, 
            local_files_only=True
        )
        
        image_model.to(args.device)
        text_model.to(args.device)
        
        image_tokenizer = None
        # image_processor = AutoImageProcessor.from_pretrained(image_model_name, local_files_only=True, trust_remote_code=True)
        
        text_tokenizer = AutoTokenizer.from_pretrained(text_model_name, local_files_only=True, trust_remote_code=True)
        text_processor = None
        
        
        self.device=args.device
        
        self.image_model=image_model
        self.text_model=text_model
        self.text_tokenizer=text_tokenizer
        self.image_processor=preprocess
        self.samples=self.get_data(args)
        self.batch_size=args.batch_size
        
    
    
    @torch.no_grad()
    def inference_once(self, image_paths: List[str]) -> torch.Tensor:
        """处理图像路径列表，返回归一化后的特征"""
        # 读取并处理图像
        images = [Image.open(img_path).convert('RGB') for img_path in image_paths]
        inputs = self.image_processor(images, return_tensors="pt").to(self.device)
        outputs = self.image_model(**inputs)
        feats = outputs.last_hidden_state[:, 0]  # [B, D]
        return F.normalize(feats, p=2, dim=-1)
    
    @torch.no_grad()
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode_text(self, query_texts: List[str]) -> torch.Tensor:
        """编码文本列表，返回GPU上的归一化特征"""
        encoded_input = self.text_tokenizer(
            query_texts, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        ).to(self.device)
        model_output = self.text_model(**encoded_input)
        text_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        text_embeddings = F.layer_norm(text_embeddings, (text_embeddings.shape[1],))
        return F.normalize(text_embeddings, p=2, dim=1)  # 保持在GPU

    
    def get_data(self,args):
        samples = []
        with open(args.dataset_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
            
        for line in lines:
            img_path = line.strip()
            txt_path = os.path.splitext(img_path)[0] + ".txt"
            if os.path.exists(txt_path):
                samples.append((img_path, txt_path))
        return samples
    
    def compute_image_text_similarity(self,image_embs: torch.Tensor, text_embs: torch.Tensor, normalize=False) -> torch.Tensor:
        assert image_embs.shape == text_embs.shape, "image and text embeddings must match in shape"
        if normalize:
            image_embs = F.normalize(image_embs, dim=-1)
            text_embs = F.normalize(text_embs, dim=-1)
        return (image_embs * text_embs).sum(dim=-1)  # [N]


    @torch.no_grad()
    def _extract_embeddings(self):
        """提取所有样本的特征（使用DataLoader加速）"""
        # 自定义Dataset
        class EvalDataset(Dataset):
            def __init__(self, samples, image_processor):
                self.samples = samples
                self.image_processor = image_processor  # 添加图像处理器
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                img_path, txt_path = self.samples[idx]
                
                # 1. 图像处理：转换为张量
                image = Image.open(img_path)
                # image_tensor = self.image_processor(image, return_tensors="pt")["pixel_values"][0]  # [C, H, W]
                image_tensor=self.image_processor(image)
                # print(self.image_processor)
                # print(preprocess)
                
                # 2. 文本处理
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                
                # 3. 标签处理
                fname = Path(img_path).name.lower()
                age_label = next((v for k,v in age_dict.items() if k in fname), -1)
                gender_label = next((v for k,v in gender_dict.items() if k in fname), -1)
                
                return {
                    "image": image_tensor,  # 已经是张量
                    "text": text,
                    "age_label": torch.tensor(age_label),
                    "gender_label": torch.tensor(gender_label)
                }

        dataset = EvalDataset(self.samples, self.image_processor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=4)

        all_image_embs, all_text_embs = [], []
        age_labels, gender_labels = [], []

        
        
        for batch in tqdm(dataloader, desc="提取特征"):
            # 从批次字典中提取数据
            images = batch["image"].to(self.device)          # [B, C, H, W] 已经在Dataset预处理过
            texts = batch["text"]                            # List[str]
            age_labels_batch = batch["age_label"].to(self.device)    # [B]
            gender_labels_batch = batch["gender_label"].to(self.device) # [B]

            # ------------------------------
            # 1. 处理图像（无需再调用image_processor）
            # ------------------------------
            # 直接输入模型（注意：假设image_model接受 pixel_values 输入）
            image_outputs = self.image_model(pixel_values=images)  # 或使用 ** 解包
            image_feats = image_outputs
            image_feats = F.normalize(image_feats, dim=-1)
            all_image_embs.append(image_feats.cpu())  # 可选移动至CPU节省显存
            # ------------------------------
            # 2. 处理文本
            # ------------------------------
            text_embs = self.encode_text(texts)  # 自动处理文本列表
            all_text_embs.append(text_embs.cpu())

            # ------------------------------
            # 3. 收集标签
            # ------------------------------
            age_labels.extend(age_labels_batch.cpu().tolist())      # 转移到CPU并转成列表
            gender_labels.extend(gender_labels_batch.cpu().tolist())


        # 合并结果
        all_image_embs = torch.cat(all_image_embs, dim=0).to(self.device)
        all_text_embs = torch.cat(all_text_embs, dim=0).to(self.device)
        age_labels = torch.tensor(age_labels, device=self.device)
        gender_labels = torch.tensor(gender_labels, device=self.device)

        return all_image_embs, all_text_embs, age_labels, gender_labels

    
    
    def get_class_embdings(self, class_templates_dict, tokenizer, model, device):
        """
        class_templates_dict: Dict[int, List[str]]
        返回: [num_classes, embedding_dim]
        """
        class_embs = []
        model.eval()
        print()
        with torch.no_grad():
            for cls_id, templates in class_templates_dict.items():
                cls_emb = self.encode_text(templates)
                class_embs.append(cls_emb)

        return torch.stack(class_embs, dim=0)  # [num_classes, emb_dim]
    
    
    
    def _compute_zero_shot_accuracy(self, image_embs, age_labels, gender_labels):

            age_text_embs = self.get_class_embdings(age_text_en, self.text_tokenizer, self.text_model, self.device).squeeze(1)
            gender_text_embs = self.get_class_embdings(gender_text_en, self.text_tokenizer, self.text_model, self.device).squeeze(1)

            
            image_embs = F.normalize(image_embs, dim=-1)

            logit_scale = (
                self.image_model.logit_scale.logit_scale.exp().float()
                if hasattr(self.image_model, "logit_scale") and hasattr(self.image_model.logit_scale, "logit_scale")
                else torch.tensor(1.0, device=image_embs.device).float()
            )

            
            age_logits = logit_scale * (image_embs.float() @ age_text_embs.T.float())
            gender_logits = logit_scale * (image_embs.float() @ gender_text_embs.T.float())


            
            age_preds = age_logits.argmax(dim=-1).cpu()
            gender_preds = gender_logits.argmax(dim=-1).cpu()

            
            age_labels = [x.item() if x.numel() == 1 else x[0].item() for x in age_labels]
            gender_labels = [x.item() if x.numel() == 1 else x[0].item() for x in gender_labels]
            age_targets = torch.tensor(age_labels)
            gender_targets = torch.tensor(gender_labels)

            
            
            age_acc = (age_preds == age_targets).float().mean().item()
            gender_acc = (gender_preds == gender_targets).float().mean().item()


            
            return age_acc, gender_acc
    
    
    def _eval_image_text_similarity(self):
        # 提取嵌入与标签
        image_embs, text_embs, age_labels, gender_labels = self._extract_embeddings()

        # 图文相似度
        sim_scores = self.compute_image_text_similarity(image_embs, text_embs)
        sim_mean = sim_scores.mean().item()
        age_acc, gender_acc = self._compute_zero_shot_accuracy(image_embs, age_labels, gender_labels)
        # 分别打印日志项
        print(f"[Eval]| zero_shot/image_text_similarity: {sim_mean:.4f}")
        print(f"[Eval] zero_shot/age_top1: {age_acc:.4f}")
        print(f"[Eval]  zero_shot/gender_top1: {gender_acc:.4f}")




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision_model", type=str, default="/data/yuesang/LLM/contrastors/mobilevit_custom/model.safetensors")
    parser.add_argument("--text_model", type=str, default="/data/yuesang/LLM/VectorIE/models/nomic/nomic-embed-text-v1.5")
    parser.add_argument("--dataset_name", type=str, default="/data/yuesang/LLM/contrastors/data/val.txt") 
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    eval_tools=Eval(args)
    eval_tools._eval_image_text_similarity()




if __name__ == "__main__":
    main()
