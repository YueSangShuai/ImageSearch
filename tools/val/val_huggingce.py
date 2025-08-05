from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
from PIL import Image
import torch
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import os



from torchvision import transforms
import torch
import yaml
from contrastors.config import Config
from contrastors.dataset.image_text_loader import get_local_image_text_dataset
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import os

def aggregate_dicts(dict_list):
    """
    将包含多个字典的列表转换为按键聚合值的新列表
    
    参数:
        dict_list: 包含多个字典的列表，如 [{a:0,b:1}, {a:1,b:2}]
        
    返回:
        聚合后的列表，如 [{a:[0,1], b:[1,2]}]
    """
    if not dict_list:
        return []
    
    # 收集所有唯一的键
    all_keys = set()
    for d in dict_list:
        all_keys.update(d.keys())
    
    # 按键聚合值
    aggregated = {}
    for key in all_keys:
        aggregated[key] = [d.get(key) for d in dict_list if key in d]
    
    # 转换为包含单个字典的列表（按照示例格式）
    return aggregated


class ImageTextDataset(Dataset):
    def __init__(self, json_file,classifer_config):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            self.classifer_config=classifer_config
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]["image_path"]
        txt_path = self.data[idx]["text_path"]
        label_dict = {}
        for cls_name in self.classifer_config.classes:
            label = self.data[idx].get(cls_name)
            label_dict[cls_name] = int(label)
            
        try:
            img = Image.open(img_path).convert("RGB")
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except Exception as e:
            print(f"加载失败：{img_path}, {txt_path}, 错误：{e}")
            img = None
            text = None

        return img_path, txt_path, img, text,label_dict


def collate_fn(batch):
    imgs, texts, img_paths, txt_paths,label_dict_list = [], [], [], [],[]
    for img_path, txt_path, img, text,label_dict in batch:
        if img is not None and text is not None:
            imgs.append(img)
            texts.append(text)
            img_paths.append(img_path)
            txt_paths.append(txt_path)
            label_dict_list.append(label_dict)
            
    return imgs, texts, img_paths, txt_paths,label_dict_list


def read_config(path):
    # read yaml and return contents
    with open(path, 'r') as file:
        try:
            return Config(**yaml.safe_load(file))
        except yaml.YAMLError as exc:
            print(exc)



class EVAL:
    def __init__(self,args):
        
        config=read_config(args.yaml_path)
        self.data_args=config.data_args
        dataset=ImageTextDataset(config.data_args.imagenet_val_path,config.classiffer_config)
        self.val_dataloader = DataLoader(
        dataset, batch_size=1024, shuffle=False, num_workers=16, collate_fn=collate_fn
    )
        self.device=args.device
        self.classifer_config=config.classiffer_config
        # 加载模型、tokenizer、processor
        self.model = AutoModel.from_pretrained(
            args.vision_model, torch_dtype=torch.bfloat16, trust_remote_code=True, optimized=True
        ).to(args.device).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(args.vision_model)
        
        

    def inference_image(self,imgs):
        img_inputs = self.processor(images=imgs, return_tensors="pt").to(self.device)
        img_inputs = {k: v.to(dtype=torch.bfloat16) for k, v in img_inputs.items()}
        with torch.inference_mode():
            img_embs = self.model.encode_images(**img_inputs, normalize=True)
            
        return img_embs
    
    def inference_text(self,texts):
        text_inputs = self.tokenizer(
            texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            text_embs = self.model.encode_texts(**text_inputs, normalize=True)
            
        return text_embs


    def get_class_embdings(self, class_templates_dict):
        """
        class_templates_dict: Dict[int, List[str]]
        返回: [num_classes, embedding_dim]
        """
        class_embs = []
        with torch.no_grad():
            for cls_id, templates in class_templates_dict.items():
                embeddings=self.inference_text(templates)
                embeddings = F.normalize(embeddings, dim=-1)
                cls_emb = embeddings.mean(dim=0)  # 每个类的多个模板平均
                cls_emb = F.normalize(cls_emb, dim=0)
                class_embs.append(cls_emb)

        return torch.stack(class_embs, dim=0)  # [num_classes, emb_dim]
    
    def compute_image_text_similarity(self,image_embs: torch.Tensor, text_embs: torch.Tensor, normalize=True) -> torch.Tensor:
        assert image_embs.shape == text_embs.shape, "image and text embeddings must match in shape"
        if normalize:
            image_embs = F.normalize(image_embs, dim=-1)
            text_embs = F.normalize(text_embs, dim=-1)
        return (image_embs * text_embs).sum(dim=-1)  # [N]
    
    def _extract_embeddings(self, dataloader):
        all_image_embs = []
        all_text_embs = []
        all_labels = {label_name: [] for label_name in self.classifer_config.classes}
        all_valid_masks = {label_name: [] for label_name in self.classifer_config.classes}  # 新增：记录有效掩码

        with torch.no_grad():
            for imgs, texts, _, _ ,label_dict_list in tqdm(dataloader, desc="Inferencing"):
                image_emb = self.inference_image(imgs).to(self.device)
                text_emb = self.inference_text(texts).to(self.device)
                all_image_embs.append(image_emb)
                all_text_embs.append(text_emb)
                
                label_dict_list=aggregate_dicts(label_dict_list)
                for label_name in self.classifer_config.classes:
                    label_value = label_dict_list.get(label_name)
                    if label_value is not None:
                        label_tensor = label_value if isinstance(label_value, torch.Tensor) else torch.tensor(label_value)
                        label_tensor = label_tensor.to(self.device)
                        all_labels[label_name].append(label_tensor)
                        
                        # 新增：创建并记录有效掩码 (-1 表示无效)
                        valid_mask = (label_tensor != -1)
                        all_valid_masks[label_name].append(valid_mask)
                      
                
        # concat embeddings
        all_image_embs = torch.cat(all_image_embs, dim=0)
        all_text_embs = torch.cat(all_text_embs, dim=0)


        # concat & gather labels
        for label_name in all_labels:
            all_labels[label_name] = torch.cat(all_labels[label_name], dim=0)
            all_valid_masks[label_name] = torch.cat(all_valid_masks[label_name], dim=0)
        
        return all_image_embs, all_text_embs, all_labels,all_valid_masks
    
    def _compute_zero_shot_accuracy(self, image_embs, all_labels: dict, all_valid_masks: dict):
        class_cfg = self.classifer_config.classes
        acc_dict = {}
        # normalize image embeddings
        image_embs = F.normalize(image_embs, dim=-1)


        for class_name, class_info in class_cfg.items():
            if class_name not in all_labels:
                continue
            
            # --------------------------
            # 核心修改1：忽略标签为-1的样本
            # 1. 原始有效掩码（如样本是否存在）
            original_valid_mask = all_valid_masks[class_name]
            # 2. 过滤标签为-1的掩码（标签!=-1才有效）
            label_not_minus1_mask = (all_labels[class_name] != -1)
            # 3. 合并掩码：原始有效且标签!=-1
            valid_mask = original_valid_mask & label_not_minus1_mask
            # --------------------------

            if valid_mask.sum() == 0:  # 无有效样本（包括-1的情况）
                acc_dict[class_name] = float('nan')
                continue

            # 应用最终掩码
            valid_image_embs = image_embs[valid_mask]
            valid_targets = all_labels[class_name][valid_mask].cpu()  # 已过滤-1

            
            
            
            text_en_dict = class_info.get("text_en")
            text_embs = self.get_class_embdings(text_en_dict)

            logits = (valid_image_embs.float() @ text_embs.T.float())
            preds = logits.argmax(dim=-1).cpu()

            acc = (preds == valid_targets).float().mean().item()
            acc_dict[class_name] = acc



        return acc_dict
        
    def _eval_image_text_similarity(self,model, dataloader, step, **kwargs):
        # 提取嵌入与标签（返回一个 dict）
        image_embs, text_embs, all_labels, all_valid_masks = self._extract_embeddings(dataloader)
       # 统计每个属性的有效/无效标签数量（保持不变）
        label_stats = {}
        for attr_name in all_labels.keys():
            # 1. 获取当前属性的标签和加载时有效掩码
            attr_labels = all_labels[attr_name]
            # 加载时有效掩码（True=加载有效，False=加载无效）
            attr_valid_mask = all_valid_masks.get(attr_name, torch.ones_like(attr_labels, dtype=torch.bool))
            attr_valid_mask = attr_valid_mask.bool()  # 确保为布尔型
            
            
            # 2. 计算核心指标（重新定义）
            total_labels = attr_labels.numel()  # 总标签数（所有样本）
            valid_samples_count = attr_valid_mask.sum().item()  # 加载时有效的标签数
            # 加载时无效的标签数（核心修改：取反掩码的总和）
            invalid_labels_count = (~attr_valid_mask).sum().item()  # ~表示“非加载有效”=“加载无效”

            # 3. 保留原有“有效标签数”（可选：若需要区分“加载有效且标签≠-1”的样本）
            # （如果不需要，可删除这两行，直接用valid_samples_count作为“有效标签数”）
            valid_label_mask = (attr_labels != -1) & attr_valid_mask  # 加载有效且标签≠-1
            valid_label_count = valid_label_mask.sum().item()

            # 4. 存入统计结果（明确标注“加载时”）
            label_stats[attr_name] = {
                "总标签数": total_labels,  # 新增：总标签数（方便核对）
                "加载时有效的标签数": valid_samples_count,  # 加载时有效（无论标签是否为-1）
                "加载时无效的标签数": invalid_labels_count,  # 替换为加载时无效的标签数
                "加载有效且标签有效的数量": valid_label_count  # 可选：加载有效且标签≠-1（原“有效标签数”）
            }
        
        valid_mask = torch.ones(len(image_embs), dtype=torch.bool, device=image_embs.device)
        # 直接使用所有样本计算相似度（只要样本总数>0）
        if len(image_embs) > 0 and len(text_embs) > 0:  # 确保有样本
            valid_image_embs = image_embs[valid_mask]  # 实际就是所有image_embs
            valid_text_embs = text_embs[valid_mask]    # 实际就是所有text_embs
            sim_scores = self.compute_image_text_similarity(valid_image_embs, valid_text_embs)
            sim_mean = sim_scores.mean().item()
        else:
            sim_mean = float('nan')  # 无样本时仍为nan（但理论上不会出现）

        zero_shot_acc_dict = self._compute_zero_shot_accuracy(image_embs, all_labels, all_valid_masks)


        # 控制台打印
        print(f"[Eval] Step {step} | zero_shot/image_text_similarity: {sim_mean:.4f}")

        for name, acc in zero_shot_acc_dict.items():
                stats = label_stats.get(name, {
                    "加载时有效的标签数": 0,
                    "加载时无效的标签数": 0,
                    "加载有效且标签有效的数量": 0
                })
                print(
                    f"[Eval] Step {step} | zero_shot/{name}_top1: {acc:.4f} "
                    f"| 加载时有效的标签数: {stats['加载时有效的标签数']} "
                    f"| 加载时无效的标签数: {stats['加载时无效的标签数']}"  # 明确标注“加载时”
                )
        
        
        for name, acc in zero_shot_acc_dict.items():
            print(f"[Eval] Step {step} | zero_shot/{name}_top1: {acc:.4f}")

            
        if zero_shot_acc_dict:  # 避免空字典导致的错误
            zero_shot_avg = sum(zero_shot_acc_dict.values()) / len(zero_shot_acc_dict)
            print(f"[Eval] Step {step} | zero_shot/平均_top1: {zero_shot_avg:.4f}")


    def val(self):
         self._eval_image_text_similarity(self.model, self.val_dataloader, 0)
    
        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision_model", type=str, default="/data/yuesang/LLM/VectorIE/models/mexma-siglip2")
    parser.add_argument("--yaml_path", type=str, default="/data/yuesang/LLM/contrastors/src/contrastors/configs/train/Mals/nomic_vits.yaml")
    parser.add_argument('--device', type=str, default='cuda:2')
    args = parser.parse_args()
    eval=EVAL(args)
    eval.val()








