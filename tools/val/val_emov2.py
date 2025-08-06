from torchvision import transforms
import torch
import yaml
from contrastors.config import Config
from contrastors.dataset.image_text_loader import get_local_image_text_dataset
from transformers import AutoTokenizer
import torch.nn.functional as F
import argparse
from contrastors.models.dual_encoder import DualEncoderConfig,DualEncoder
import torch.nn as nn
from transformers import PreTrainedModel
from tqdm import tqdm
import onnxruntime as ort
import numpy as np 
from tabulate import tabulate 
import math

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

def read_config(path):
    # read yaml and return contents
    with open(path, 'r') as file:
        try:
            return Config(**yaml.safe_load(file))
        except yaml.YAMLError as exc:
            print(exc)


class EMO2ForEmbedding(PreTrainedModel):
    config_class = DualEncoderConfig

    def __init__(self, config):
        super().__init__(config)
        from contrastors.models.emov2 import MODEL  # 你实际用哪个模型就 import 哪个
        self.trunk = MODEL.get_module(config.image_model_args.model_name)(pretrained=False, num_classes=1000)
        self.proj=nn.Linear(200,768)
        
        self.attribute_classifiers = nn.ModuleDict()
        self.attribute_classes = {
            attr_name: attr_cfg["nc"]
            for attr_name, attr_cfg in config.image_model_args.classifier_config["classes"].items()
        }
        hidden_dim = config.image_model_args.projection_dim
 
        
        for attr_name, num_classes in self.attribute_classes.items():
            self.attribute_classifiers[attr_name] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(hidden_dim // 2, num_classes)
            )

    def forward(self, pixel_values):
        x = self.trunk(pixel_values)['out']
        embeddings = self.proj(x)  # 投影到目标维度

        # 计算每个属性分类器的输出
        attribute_logits = {
            attr_name: classifier(embeddings)
            for attr_name, classifier in self.attribute_classifiers.items()
        }

        return {
            "embeddings": embeddings,
            "attribute_logits": attribute_logits
        }


class Emov2_Onnx:
    def __init__(self, model_path, device_id=0):
        # 设置执行提供者
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': device_id,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider',
        ]
        
        # 加载模型
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # 获取输入输出名称
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # 打印模型信息
        print(f"ONNX模型输入: {self.input_name}")
        print(f"ONNX模型输出: {self.output_names}")
        
    def forward(self, pixel_values):
        """
        参数:
            pixel_values: [batch_size, 3, 224, 224] 的PyTorch张量
        
        返回:
            包含所有输出的字典
        """
        # 转换为numpy数组
        input_data = pixel_values.cpu().numpy()
        
        # 模型推理
        ort_outputs = self.session.run(self.output_names, {self.input_name: input_data})
        
        # 构建输出字典
        output_dict = {name: torch.from_numpy(output) for name, output in zip(self.output_names, ort_outputs)}
        
        # 确保所有输出都在正确的设备上
        for name in output_dict:
            output_dict[name] = output_dict[name].to(pixel_values.device)
            
        return output_dict
    
    
    
class EVAL:
    def __init__(self,args):
        
        config=read_config(args.yaml_path)
        self.data_args=config.data_args
        self.classifer_config=config.classiffer_config
        self.text_args=config.text_model_args
        self.tokenizer=AutoTokenizer.from_pretrained(self.text_args.model_name, local_files_only=True, trust_remote_code=True)
        
        val_data_info = get_local_image_text_dataset(args=self.data_args,
                                                           classifer_config=self.classifer_config,
                                                           transforms=preprocess, 
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
        self.vision_onnx=Emov2_Onnx(args.onnx_path,self.device)
        
    
    def get_image_model(self,ckpt_path):
        config = DualEncoderConfig.from_pretrained(ckpt_path)
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
    
    def _extract_embeddings(self, model,dataloader):
        all_image_embs = []
        all_text_embs = []
        all_labels = {label_name: [] for label_name in self.classifer_config.classes}
        all_valid_masks = {label_name: [] for label_name in self.classifer_config.classes}  # 新增：记录有效掩码

        device = model.device
        text, vision = model.text, model.vision
        text.eval()
        vision.eval()

        with torch.no_grad():
            for batch in tqdm(dataloader):
                vision_inputs = {k: v.to(device) for k, v in batch["vision"].items()}
                text_inputs = {k: v.to(device) for k, v in batch["text"].items()}
                labels = batch["label"]


                image_emb = vision(**vision_inputs)["embedding"].to(device)
                text_emb = text(**text_inputs)["embedding"].to(device)

                all_image_embs.append(image_emb)
                all_text_embs.append(text_emb)

                # 动态提取所有标签字段
                for label_name in self.classifer_config.classes:
                    label_value = labels.get(label_name)
                    if label_value is not None:
                        label_tensor = label_value if isinstance(label_value, torch.Tensor) else torch.tensor(label_value)
                        label_tensor = label_tensor.to(device)
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

        return all_image_embs, all_text_embs, all_labels, all_valid_masks  # 返回掩码
    
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
            attr_avg = sum(attr_acc_dict.values()) / len(attr_acc_dict)
            print(f"[Eval] Step {step} | attr_/平均_top1: {attr_avg:.4f}")



    def _compute_classifer_onnx(self, classification_outputs, all_labels):
        acc_dict = {}

        # 遍历所有分类输出
        for output_name, logits in classification_outputs.items():
            # 从输出名称中提取属性名（例如："logits_Age18-60" -> "Age18-60"）
            if not output_name.startswith("logits_"):
                continue
                
            attr_name = output_name[7:]  # 去掉 "logits_" 前缀
            
            # 检查该属性是否有对应的标签
            if attr_name not in all_labels:
                continue
                
            # 计算预测结果
            preds = logits.argmax(dim=-1).cpu()
            targets = all_labels[attr_name].cpu()
            
            # 处理多维标签（如果有）
            if targets.ndim > 1:
                targets = targets.view(-1)
                
            # 过滤无效样本（标签为-1的样本）
            valid_mask = (targets != -1)
            valid_count = valid_mask.sum().item()
            
            if valid_count == 0:
                acc_dict[attr_name] = float('nan')
                continue
                
            # 计算准确率
            filtered_preds = preds[valid_mask]
            filtered_targets = targets[valid_mask]
            accuracy = (filtered_preds == filtered_targets).float().mean().item()
            
            acc_dict[attr_name] = accuracy
            
        return acc_dict
    
    def _extract_embeddings_onnx(self, model, dataloader):
        all_image_embs = []  # 存储图像嵌入
        all_text_embs = []   # 存储文本嵌入
        all_labels = {label_name: [] for label_name in self.classifer_config.classes}
        all_valid_masks = {label_name: [] for label_name in self.classifer_config.classes}  # 新增：记录有效掩码
        all_classification_outputs = {}  # 存储分类结果
        
        
        device = model.device
        text, vision = model.text, self.vision_onnx
        text.eval()

        with torch.no_grad():
            for batch in tqdm(dataloader):
                vision_inputs = {k: v.to(device) for k, v in batch["vision"].items()}
                text_inputs = {k: v.to(device) for k, v in batch["text"].items()}
                labels = batch["label"]

                # 获取 ONNX 模型输出（包含 embeddings 和所有分类 logits）
                onnx_outputs = vision.forward(vision_inputs["input_ids"])
                
                # 提取图像嵌入
                image_emb = onnx_outputs["embeddings"]
                all_image_embs.append(image_emb)
                
                # 提取分类结果（除了 embeddings 以外的所有输出）
                for key, value in onnx_outputs.items():
                    if key != "embeddings":  # 跳过嵌入，只处理分类结果
                        if key not in all_classification_outputs:
                            all_classification_outputs[key] = []
                        all_classification_outputs[key].append(value)
                
                # 提取文本嵌入
                text_emb = text(**text_inputs)["embedding"].to(device)
                all_text_embs.append(text_emb)

                # 处理标签
                for label_name in self.classifer_config.classes:
                    label_value = labels.get(label_name)
                    if label_value is not None:
                        label_tensor = label_value if isinstance(label_value, torch.Tensor) else torch.tensor(label_value)
                        label_tensor = label_tensor.to(device)
                        all_labels[label_name].append(label_tensor)
                        
                        # 新增：创建并记录有效掩码 (-1 表示无效)
                        valid_mask = (label_tensor != -1)
                        all_valid_masks[label_name].append(valid_mask)

        # 拼接嵌入张量
        all_image_embs = torch.cat(all_image_embs, dim=0)
        all_text_embs = torch.cat(all_text_embs, dim=0)
        
        # 拼接分类结果
        for key in all_classification_outputs:
            all_classification_outputs[key] = torch.cat(all_classification_outputs[key], dim=0)

        # 拼接标签
        for label_name in all_labels:
            all_labels[label_name] = torch.cat(all_labels[label_name], dim=0)
            all_valid_masks[label_name] = torch.cat(all_valid_masks[label_name], dim=0)

        return all_image_embs, all_text_embs, all_labels, all_valid_masks, all_classification_outputs
    
    def _eval_image_text_similarity_onnx(self,model, dataloader, step, **kwargs):


        # 提取嵌入与标签（返回一个 dict）
        all_image_embs, all_text_embs, all_labels, all_valid_masks, all_classification_outputs = self._extract_embeddings_onnx(model, dataloader)

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
        
        valid_mask = torch.ones(len(all_image_embs), dtype=torch.bool, device=all_image_embs.device)
        
        # 直接使用所有样本计算相似度（只要样本总数>0）
        if len(all_image_embs) > 0 and len(all_text_embs) > 0:  # 确保有样本
            valid_image_embs = all_image_embs[valid_mask]  # 实际就是所有image_embs
            valid_text_embs = all_text_embs[valid_mask]    # 实际就是所有text_embs
            sim_scores = self.compute_image_text_similarity(valid_image_embs, valid_text_embs)
            sim_mean = sim_scores.mean().item()
        else:
            sim_mean = float('nan')  # 无样本时仍为nan（但理论上不会出现）
        
        # Zero-shot 图文分类精度（保持不变）
        zero_shot_acc_dict = self._compute_zero_shot_accuracy(all_image_embs, model, all_labels, all_valid_masks)


        # 多属性 zero-shot 分类精度（如年龄/性别）
        attr_acc_dict = self._compute_classifer_onnx(all_classification_outputs, all_labels)



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
            attr_avg = sum(attr_acc_dict.values()) / len(attr_acc_dict)
            print(f"[Eval] Step {step} | attr_/平均_top1: {attr_avg:.4f}")

        
        
        
        return zero_shot_acc_dict,attr_acc_dict
    
    def val(self):
        zero_shot_acc_dict,attr_acc_dict=self._eval_image_text_similarity(self.model, self.val_dataloader, 0)
              
    def val_onnx(self):
       zero_shot_acc_dict,attr_acc_dict=self._eval_image_text_similarity_onnx(self.model, self.val_dataloader,0)
    
    def compare_accuracy(self):
        """对比val和val_onnx的精度差异"""
        # 获取两种方法的精度结果
        torch_zero_shot, torch_attr = self.val()
        onnx_zero_shot, onnx_attr = self.val_onnx()

        # 合并所有类别（确保覆盖两种方法的所有类别）
        all_categories = set(torch_zero_shot.keys()).union(
            torch_attr.keys(),
            onnx_zero_shot.keys(),
            onnx_attr.keys()
        )

        # 存储差异结果
        diff_results = []

        for category in sorted(all_categories):
            # 获取两种方法的精度（若类别不存在则记为NaN）
            # 1. Zero-shot 精度对比
            torch_z = torch_zero_shot.get(category, np.nan)
            onnx_z = onnx_zero_shot.get(category, np.nan)
            zero_shot_diff = abs(torch_z - onnx_z) if not np.isnan(torch_z) and not np.isnan(onnx_z) else np.nan

            # 2. Attribute 精度对比
            torch_a = torch_attr.get(category, np.nan)
            onnx_a = onnx_attr.get(category, np.nan)
            attr_diff = abs(torch_a - onnx_a) if not np.isnan(torch_a) and not np.isnan(onnx_a) else np.nan

            # 添加到结果列表
            diff_results.append({
                "类别": category,
                "PyTorch(zero-shot)": round(torch_z, 4) if not np.isnan(torch_z) else "N/A",
                "ONNX(zero-shot)": round(onnx_z, 4) if not np.isnan(onnx_z) else "N/A",
                "zero-shot绝对误差": round(zero_shot_diff, 4) if not np.isnan(zero_shot_diff) else "N/A",
                "PyTorch(attr)": round(torch_a, 4) if not np.isnan(torch_a) else "N/A",
                "ONNX(attr)": round(onnx_a, 4) if not np.isnan(onnx_a) else "N/A",
                "attr绝对误差": round(attr_diff, 4) if not np.isnan(attr_diff) else "N/A"
            })

        # 打印结果表格
        print("="*120)
        print("PyTorch与ONNX精度对比结果")
        print("="*120)
        print(tabulate(diff_results, headers="keys", tablefmt="grid"))

        # 计算平均误差（排除NaN）
        zero_shot_diffs = [d["zero-shot绝对误差"] for d in diff_results if d["zero-shot绝对误差"] != "N/A"]
        attr_diffs = [d["attr绝对误差"] for d in diff_results if d["attr绝对误差"] != "N/A"]

        print("\n" + "="*120)
        print(f"zero-shot平均绝对误差: {np.mean(zero_shot_diffs):.4f}")
        print(f"attr平均绝对误差: {np.mean(attr_diffs):.4f}")
        print(f"最大误差: {max(zero_shot_diffs + attr_diffs):.4f}")
        print("="*120)
         
        
        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision_model", type=str, default="/data/yuesang/LLM/contrastors/src/ckpts/person/Mals/emov2/epoch_50/model/")
    parser.add_argument("--yaml_path", type=str, default="/data/yuesang/LLM/contrastors/src/contrastors/configs/train/Mals/nomic_vits.yaml")
    parser.add_argument("--onnx_path", type=str, default="/data/yuesang/LLM/contrastors/emov2_hf/emov2.onnx")
    parser.add_argument('--device', type=str, default='cuda:1')
    args = parser.parse_args()
    eval=EVAL(args)
    eval.val()

