from torchvision import transforms
import torch
import yaml
from contrastors.config import Config
from contrastors.dataset.image_text_loader import get_local_image_text_dataset
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import argparse
from contrastors.models.dual_encoder import DualEncoderConfig,DualEncoder
from tqdm import tqdm
from typing import Any, Dict, Optional, Tuple, Union
from contrastors.config import AugmentationCfg
from contrastors.dataset.constants import OPENAI_IMAGE_DATASET_MEAN, OPENAI_IMAGE_DATASET_STD
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    RandAugment,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
import warnings
import torch.nn as nn
from contextlib import nullcontext

class ResizeMaxSize(nn.Module):
    def __init__(self, max_size, interpolation=InterpolationMode.BICUBIC, fn='max', fill=0):
        super().__init__()
        if not isinstance(max_size, int):
            raise TypeError(f"Size should be int. Got {type(max_size)}")
        self.max_size = max_size
        self.interpolation = interpolation
        self.fn = min if fn == 'min' else min
        self.fill = fill

    def forward(self, img):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[:2]
        else:
            width, height = img.size
        scale = self.max_size / float(max(height, width))
        if scale != 1.0:
            new_size = tuple(round(dim * scale) for dim in (height, width))
            img = F.resize(img, new_size, self.interpolation)
            pad_h = self.max_size - new_size[0]
            pad_w = self.max_size - new_size[1]
            img = F.pad(img, padding=[pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2], fill=self.fill)
        return img

def _convert_to_rgb(image):
    return image.convert('RGB')

def image_transform(
    image_size: int,
    is_train: bool,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
    resize_longest_max: bool = False,
    fill_color: int = 0,
    aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
):
    mean = mean or OPENAI_IMAGE_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_IMAGE_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    if isinstance(aug_cfg, dict):
        aug_cfg = AugmentationCfg(**aug_cfg)
    else:
        aug_cfg = aug_cfg or AugmentationCfg()

    normalize = Normalize(mean=mean, std=std)
    if is_train:
        aug_cfg_dict = {k: v for k, v in aug_cfg.dict().items() if v is not None}
        train_transform = Compose(
            [
                # RandAugment(),
                RandomResizedCrop(
                    image_size,
                    scale=aug_cfg_dict.pop('scale'),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                # RandomHorizontalFlip(),
                _convert_to_rgb,
                ToTensor(),
                normalize,
            ]
        )
        if aug_cfg_dict:
            warnings.warn(f'Unused augmentation cfg items, specify `use_timm` to use ({list(aug_cfg_dict.keys())}).')
        return train_transform
    else:
        if resize_longest_max:
            transforms = [ResizeMaxSize(image_size, fill=fill_color)]
        else:
            transforms = [
                Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(image_size),
            ]
        transforms.extend(
            [
                _convert_to_rgb,
                ToTensor(),
                normalize,
            ]
        )
        return Compose(transforms)

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
        self.classifer_config=config.classiffer_config
        self.text_args=config.text_model_args
        self.tokenizer=self.get_tokenizer(config)
        self.transforms = self.get_transforms(config.transforms)
        
        val_data_info = get_local_image_text_dataset(args=self.data_args,
                                                           classifer_config=self.classifer_config,
                                                           transforms=self.transforms["val"], 
                                                           is_train=False,
                                                           tokenizer=self.tokenizer, 
                                                           epoch=0,
                                                           add_eos=self.text_args.nomic_encoder == False,
                                                           add_prefix=self.text_args.add_prefix,
                                                           )

        self.val_dataloader=val_data_info.dataloader
        self.device=args.device
        
        self.model = self.get_image_model(args.vision_model).to(self.device)
        self.model = self.model.to(dtype=torch.bfloat16)
        
    def get_transforms(self, transforms):
        train_transforms = image_transform(**transforms.dict(), is_train=True)
        val_transforms = image_transform(
            **transforms.dict(exclude={"aug_cfg", "resize_longest_max", "fill_color"}), is_train=False
        )

        return {"train": train_transforms, "val": val_transforms}


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
    

    def get_image_model(self,ckpt_path):
        config = DualEncoderConfig.from_pretrained(ckpt_path)
        model = DualEncoder.from_pretrained(ckpt_path, config=config)
        return model
    
    def get_class_embdings(self, class_templates_dict, tokenizer, model, device):
        """
        class_templates_dict: Dict[int, List[str]]
        返回: [num_classes, embedding_dim]
        """
        class_embs = []
        model.eval()
        with torch.no_grad():
            for cls_id, templates in class_templates_dict.items():
                inputs = tokenizer(templates, return_tensors="pt", padding=True, truncation=True).to(device)
                embeddings = model(**inputs)["embedding"]
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
    
    
    def _extract_embeddings(self, model, dataloader, save_dir="./extracted_data"):
        """
        提取并保存vision_inputs、text_inputs、图像嵌入和文本嵌入
        
        Args:
            model: 训练好的模型
            dataloader: 数据加载器
            save_dir: 保存目录（会自动创建）
        """
        # 初始化存储列表
        all_image_embs = []
        all_text_embs = []
        all_labels = {label_name: [] for label_name in self.classifer_config.classes}
        all_valid_masks = {label_name: [] for label_name in self.classifer_config.classes}
        
        # 新增：存储输入数据（vision_inputs和text_inputs）
        all_vision_inputs = []  # 存储每个样本的vision输入
        all_text_inputs = []    # 存储每个样本的text输入
        sample_ids = []         # 生成唯一样本ID，用于对齐

        device = model.device
        text, vision = model.text, model.vision
        text.eval()
        vision.eval()

        
        import os
        import json
        from tqdm import tqdm
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "vision_inputs"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "text_inputs"), exist_ok=True)

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader)):
                # 提取单批次数据
                vision_inputs = {k: v.to(device) for k, v in batch["vision"].items()}
                text_inputs = {k: v.to(device) for k, v in batch["text"].items()}
                labels = batch["label"]
                batch_size = next(iter(vision_inputs.values())).shape[0]  # 批次大小

                # 生成该批次样本的唯一ID（格式：批次索引_样本索引）
                current_sample_ids = [f"batch_{batch_idx}_sample_{i}" for i in range(batch_size)]
                sample_ids.extend(current_sample_ids)

                # 保存单批次的vision_inputs和text_inputs（按样本拆分）
                for i in range(batch_size):
                    # 保存vision_inputs（如pixel_values等）
                    vision_sample = {k: v[i].cpu() for k, v in vision_inputs.items()}
                    torch.save(
                        vision_sample,
                        os.path.join(save_dir, "vision_inputs", f"{current_sample_ids[i]}.pt")
                    )
                    # 保存text_inputs（如input_ids、attention_mask等）
                    text_sample = {k: v[i].cpu() for k, v in text_inputs.items()}
                    torch.save(
                        text_sample,
                        os.path.join(save_dir, "text_inputs", f"{current_sample_ids[i]}.pt")
                    )

                # 提取嵌入
                autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
                with autocast_ctx:
                    image_emb = vision(** vision_inputs)["embedding"].to(device)
                    text_emb = text(**text_inputs)["embedding"].to(device)

                all_image_embs.append(image_emb)
                all_text_embs.append(text_emb)

                # 处理标签和掩码
                for label_name in self.classifer_config.classes:
                    label_value = labels.get(label_name)
                    if label_value is not None:
                        label_tensor = label_value if isinstance(label_value, torch.Tensor) else torch.tensor(label_value)
                        label_tensor = label_tensor.to(device)
                        all_labels[label_name].append(label_tensor)
                        all_valid_masks[label_name].append(label_tensor != -1)

        # 合并所有批次的嵌入
        all_image_embs = torch.cat(all_image_embs, dim=0)
        all_text_embs = torch.cat(all_text_embs, dim=0)

        # 合并标签和掩码
        for label_name in all_labels:
            all_labels[label_name] = torch.cat(all_labels[label_name], dim=0)
            all_valid_masks[label_name] = torch.cat(all_valid_masks[label_name], dim=0)

        # 保存嵌入向量（整体保存，方便批量加载）
        torch.save(all_image_embs.cpu(), os.path.join(save_dir, "all_image_embeddings.pt"))
        torch.save(all_text_embs.cpu(), os.path.join(save_dir, "all_text_embeddings.pt"))

        # 保存样本ID与嵌入索引的映射（方便查询）
        embedding_mapping = {
            "sample_ids": sample_ids,
            "num_samples": len(sample_ids),
            "image_embedding_path": "all_image_embeddings.pt",
            "text_embedding_path": "all_text_embeddings.pt",
            "vision_inputs_dir": "vision_inputs/",
            "text_inputs_dir": "text_inputs/"
        }
        with open(os.path.join(save_dir, "embedding_mapping.json"), "w", encoding="utf-8") as f:
            json.dump(embedding_mapping, f, ensure_ascii=False, indent=2)

        print(f"数据保存完成：\n"
            f"- 总样本数：{len(sample_ids)}\n"
            f"- 图像嵌入：{os.path.join(save_dir, 'all_image_embeddings.pt')}\n"
            f"- 文本嵌入：{os.path.join(save_dir, 'all_text_embeddings.pt')}\n"
            f"- 视觉输入：{os.path.join(save_dir, 'vision_inputs')}\n"
            f"- 文本输入：{os.path.join(save_dir, 'text_inputs')}\n"
            f"- 映射关系：{os.path.join(save_dir, 'embedding_mapping.json')}")

        return all_image_embs, all_text_embs, all_labels, all_valid_masks
    
    
    # def _extract_embeddings(self, model, dataloader):
    #     all_image_embs = []
    #     all_text_embs = []
    #     all_labels = {label_name: [] for label_name in self.classifer_config.classes}
    #     all_valid_masks = {label_name: [] for label_name in self.classifer_config.classes}  # 新增：记录有效掩码

    #     device = model.device
    #     text, vision = model.text, model.vision
    #     text.eval()
    #     vision.eval()

    #     with torch.no_grad():
    #         for batch in tqdm(dataloader):
    #             vision_inputs = {k: v.to(device) for k, v in batch["vision"].items()}
    #             text_inputs = {k: v.to(device) for k, v in batch["text"].items()}
    #             labels = batch["label"]

    #             autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    #             with autocast_ctx:
    #                 image_emb = vision(**vision_inputs)["embedding"].to(device)
    #                 text_emb = text(**text_inputs)["embedding"].to(device)
                
                
    #             all_image_embs.append(image_emb)
    #             all_text_embs.append(text_emb)

    #             # 动态提取所有标签字段
    #             for label_name in self.classifer_config.classes:
    #                 label_value = labels.get(label_name)
    #                 if label_value is not None:
    #                     label_tensor = label_value if isinstance(label_value, torch.Tensor) else torch.tensor(label_value)
    #                     label_tensor = label_tensor.to(device)
    #                     all_labels[label_name].append(label_tensor)
                        
    #                     # 新增：创建并记录有效掩码 (-1 表示无效)
    #                     valid_mask = (label_tensor != -1)
    #                     all_valid_masks[label_name].append(valid_mask)

    #     # concat embeddings
    #     all_image_embs = torch.cat(all_image_embs, dim=0)
    #     all_text_embs = torch.cat(all_text_embs, dim=0)



    #     # concat & gather labels
    #     for label_name in all_labels:
    #         all_labels[label_name] = torch.cat(all_labels[label_name], dim=0)
    #         all_valid_masks[label_name] = torch.cat(all_valid_masks[label_name], dim=0)
        
    #     return all_image_embs, all_text_embs, all_labels, all_valid_masks  # 返回掩码
 
    def _compute_zero_shot_accuracy(self, image_embs, model, all_labels: dict, all_valid_masks: dict):
        text_model = model.text
        device = model.device
        tokenizer = self.tokenizer
        class_cfg = self.classifer_config.classes

        acc_dict = {}

        # 归一化图像嵌入
        image_embs = F.normalize(image_embs, dim=-1)

        # 获取logit缩放因子
        logit_scale = (
            model.logit_scale.logit_scale.exp().float()
            if hasattr(model, "logit_scale") and hasattr(model.logit_scale, "logit_scale")
            else torch.tensor(1.0, device=image_embs.device).float()
        )

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

            # 获取文本嵌入并计算logits
            text_en_dict = class_info.get("text_en")
            text_embs = self.get_class_embdings(text_en_dict, tokenizer, text_model, device)
            logits = logit_scale * (valid_image_embs.float() @ text_embs.T.float())
            preds = logits.argmax(dim=-1).cpu()

            # 计算准确率（此时无-1标签干扰）
            acc = (preds == valid_targets).float().mean().item()
            acc_dict[class_name] = acc

        return acc_dict

    def _compute_classifer(self, model, vision_embs, targets, valid_masks):
        attr_outputs = {}
        valid_attrs = {}

        if model.attribute_classes:
            for attr, classifier in model.attribute_classifiers.items():
                if attr not in targets:
                    continue

                # --------------------------
                # 核心修改2：忽略标签为-1的样本
                # 1. 原始有效掩码
                original_valid_mask = valid_masks[attr]
                # 2. 过滤标签为-1的掩码
                label_not_minus1_mask = (targets[attr] != -1)
                # 3. 合并掩码
                valid_mask = original_valid_mask & label_not_minus1_mask
                # --------------------------

                if valid_mask.sum() == 0:  # 无有效样本（包括-1）
                    continue

                # 类型对齐并应用掩码
                vision_embs_casted = vision_embs.to(dtype=next(classifier.parameters()).dtype)
                vision_embs_casted = F.normalize(vision_embs_casted, dim=-1)
                valid_vision_embs = vision_embs_casted[valid_mask]  # 已过滤-1

                # 计算logits
                logits = classifier(valid_vision_embs)
                attr_outputs[f"{attr}_logits"] = logits
                valid_attrs[attr] = model.attribute_classes[attr]

            acc_dict = {}

            for attr, num_classes in valid_attrs.items():
                logits = attr_outputs[f"{attr}_logits"]
                preds = logits.argmax(dim=-1).cpu()
                
                # 应用相同掩码获取目标（已过滤-1）
                valid_targets = targets[attr][valid_masks[attr] & (targets[attr] != -1)].cpu()

                # 计算准确率（无-1标签）
                acc = (preds == valid_targets).float().mean().item()
                acc_dict[attr] = acc

            return acc_dict

        else:
            return {}
    
        
    def _eval_image_text_similarity(self,model, dataloader, step, **kwargs):

        # 提取嵌入与标签（返回一个 dict）
        image_embs, text_embs, all_labels, all_valid_masks = self._extract_embeddings(model, dataloader)  

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
        
        
        # Zero-shot 图文分类精度（保持不变）
        zero_shot_acc_dict = self._compute_zero_shot_accuracy(image_embs, model, all_labels, all_valid_masks)

        # 多属性分类精度（保持不变）
        attr_acc_dict = self._compute_classifer(model, image_embs, all_labels, all_valid_masks)



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
            
        print("\n[属性分类精度及标签统计]")
        for attr, acc in attr_acc_dict.items():
                stats = label_stats.get(attr, {
                    "加载时有效的标签数": 0,
                    "加载时无效的标签数": 0,
                    "加载有效且标签有效的数量": 0
                })
                print(
                    f"[Eval] Step {step} | attr_/{attr}_top1: {acc:.4f} "
                    f"| 加载时有效的标签数: {stats['加载时有效的标签数']} "
                    f"| 加载时无效的标签数: {stats['加载时无效的标签数']}"  # 明确标注“加载时”
                )
        
        
        if zero_shot_acc_dict:  # 避免空字典导致的错误
            zero_shot_avg = sum(zero_shot_acc_dict.values()) / len(zero_shot_acc_dict)
            print(f"[Eval] Step {step} | zero_shot/平均_top1: {zero_shot_avg:.4f}")
            
        if attr_acc_dict:  # 避免空字典导致的错误
            zero_shot_avg = sum(attr_acc_dict.values()) / len(attr_acc_dict)
            print(f"[Eval] Step {step} | attr_shot/平均_top1: {zero_shot_avg:.4f}")

    
    def val(self):
         self._eval_image_text_similarity(self.model, self.val_dataloader, 0)
    
        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision_model", type=str, default="/data/yuesang/LLM/contrastors/src/ckpts/test/pa-100k/test/epoch_0_model")
    parser.add_argument("--yaml_path", type=str, default="/data/yuesang/LLM/contrastors/src/contrastors/configs/train/test/nomic_pa-100k.yaml")
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    eval=EVAL(args)
    eval.val()