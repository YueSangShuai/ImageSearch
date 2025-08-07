from functools import partial
from pathlib import Path

import datasets
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from clip_benchmark.metrics import zeroshot_retrieval as zsr
from deepspeed.checkpoint.utils import clone_tensors_for_torch_save
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from contrastors.dataset.image_text_loader import get_imagenet, get_wds_image_text_dataset,get_local_image_text_dataset
from contrastors.dataset.transform import image_transform
from contrastors.distributed import gather
from contrastors.eval.datacomp.retr_eval import RetrievalDataset, image_captions_collage_fn_prefix
from contrastors.eval.imagenet import evaluate_imagenet
from contrastors.models.dual_encoder import DualEncoder, DualEncoderConfig
import torch.nn.functional as F
from .text_text import TextTextTrainer
from contextlib import nullcontext
import os
import csv
from transformers import PreTrainedModel,AutoModel, AutoImageProcessor
from tqdm import tqdm
import json

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0




class ImageTextTrainer(TextTextTrainer):
    def __init__(self, config, dtype):
        self.transforms = self.get_transforms(config.transforms)
        self.classifer_config=config.classiffer_config
        super(ImageTextTrainer, self).__init__(config, dtype)
        
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

    def get_model(self, config):
        model_config = DualEncoderConfig(config)
        model = DualEncoder(model_config)
        has_trainable_params = sum(p.requires_grad for p in model.parameters()) > 0
        model = model.to(f"cuda:{self.process_index}")

        if self.distributed and not self.deepspeed and has_trainable_params:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[dist.get_rank()],
            )

        models = {"model": model}

        return models

    def get_dataloaders(self, config, epoch=0):
        train_args = config.train_args
        data_config = config.data_args
        train_transforms = self.transforms["train"]
        val_transforms = self.transforms["val"]
        teacher_transformer=None
        if config.vision_model_args.distill_type!="" and "mexma-siglip2" in config.teacher_vision_model_args.model_name:
            teacher_model_name=config.teacher_vision_model_args.model_name
            teacher_transformer= AutoImageProcessor.from_pretrained(teacher_model_name)
        
        tokenizer = self.tokenizer
        gradient_accumulation_steps = train_args.gradient_accumulation_steps

        text_args = config.text_model_args
        
        if data_config.is_local:
            train_data_info = get_local_image_text_dataset(args=data_config,
                                                           classifer_config=self.classifer_config,
                                                           transforms=train_transforms, 
                                                           is_train=True,
                                                           tokenizer=tokenizer, 
                                                           epoch=epoch,
                                                           add_eos=text_args.nomic_encoder == False,
                                                           add_prefix=text_args.add_prefix,
                                                           teacher_transformer=teacher_transformer
                                                           )

            
        else:
            train_data_info = get_wds_image_text_dataset(
                data_config,
                train_transforms,
                tokenizer=tokenizer,
                is_train=True,
                epoch=epoch,
                add_eos=text_args.nomic_encoder == False,
                add_prefix=text_args.add_prefix,
                precomputed_text=text_args.precomputed,
            )
        train_dataloader = train_data_info.dataloader
        

        self.total_num_steps = int(len(train_data_info) / gradient_accumulation_steps)
        if data_config.imagenet_val_path is not None:
            # val_data_info = get_imagenet(data_config, transforms=val_transforms)
            val_data_info=get_local_image_text_dataset(args=data_config,
                                                        classifer_config=self.classifer_config,
                                                           transforms=val_transforms, 
                                                           is_train=False,
                                                           tokenizer=tokenizer, 
                                                           epoch=epoch,
                                                           add_eos=text_args.nomic_encoder == False,
                                                           add_prefix=text_args.add_prefix,
                                                           teacher_transformer=teacher_transformer
                                                           )
            val_dataloader = val_data_info.dataloader
            
        else:
            val_dataloader = None

        dataloaders = {"train": train_dataloader, "train_sampler": train_data_info, "val": val_dataloader, "test": None}

        
        data_config = config.data_args
        if data_config.eval_flickr:
            val_transforms = self.transforms["val"]

            dataset = RetrievalDataset(
                datasets.load_dataset(
                    f"nlphuji/flickr_1k_test_image_text_retrieval",
                    split="test",
                ),
                transform=val_transforms,
            )

            dataloader = DataLoader(
                dataset,
                batch_size=128,
                shuffle=False,
                num_workers=0,
                collate_fn=lambda x: image_captions_collage_fn_prefix(
                    x, prefix="search_query" if self.config.text_model_args.add_prefix else None
                ),
            )

            imagenet_dataloader = dataloaders["val"]
            dataloaders["val"] = {"imagenet": imagenet_dataloader, "flickr": dataloader}

        return dataloaders

    def save_model(self, output_dir):
        super().save_model(output_dir)
        vision_output_dir = Path(f"{output_dir}/vision")
        if not vision_output_dir.exists():
            vision_output_dir.mkdir(parents=True, exist_ok=True)


        if self.global_rank == 0:
            if self.config.vision_model_args.freeze is False and "vision" in self.model:
                
                unwrapped = self.unwrap(self.model["vision"])
                if self.deepspeed:
                    # TODO: need to add zero3 support
                    # reduces bloat when saving with deepsped
                    state_dict = clone_tensors_for_torch_save(unwrapped.state_dict())
                else:
                    state_dict = None

                unwrapped.save_pretrained(vision_output_dir, state_dict=state_dict)
        
            
            logit_scale = self.model.get("logit_scale", None)
            if isinstance(logit_scale, (nn.Module, nn.DataParallel, nn.parallel.DistributedDataParallel)) and any(
                p.requires_grad for p in logit_scale.parameters()
            ):
                unwrapped_scale = self.unwrap(logit_scale)
                torch.save(unwrapped_scale.state_dict(), f"{output_dir}/logit_scale.pt")
                      
    def forward_step(self, model, inputs, **kwargs):
        model.train()
        if self.use_grad_cache:
            raise NotImplementedError("Grad cache not supported for three towers")
        else:
            loss = self._forward_step(model, inputs, **kwargs)

        return loss

    def backward(self, loss):
        # grad cache backprops in the loss function, becomes a noop
        if not self.use_grad_cache:
            if self.deepspeed:
                self.engine.backward(loss["loss"])
                self.engine.step()
            else:
                loss["loss"].backward()

    def _forward_step(self, model, batch, **kwargs):
        text_inputs = {k: v.to(model.device) for k, v in batch["text"].items()}
        vision_inputs = {k: v.to(model.device) for k, v in batch["vision"].items()}
        if batch.get("teacher_vision") is not None:
            teacher_vision_inputs={k: v.squeeze(1).to(dtype=model.dtype).to(model.device) for k, v in batch["teacher_vision"].items()}
            outputs = model(text_inputs=text_inputs, vision_inputs=vision_inputs,targets=batch["label"],img_path=teacher_vision_inputs,tokenizer=self.tokenizer)
        else:
            outputs = model(text_inputs=text_inputs, vision_inputs=vision_inputs,targets=batch["label"],tokenizer=self.tokenizer)
        
        return outputs

    def training_step(
        self, model, batch, optimizer, scheduler, step, train_args, total_num_steps, gradient_accumulation_steps
    ):
        loss = super().training_step(
            model, batch, optimizer, scheduler, step, train_args, total_num_steps, gradient_accumulation_steps
        )

        if train_args.clamp_logits:
            if sum(p.requires_grad for p in model["model"].logit_scale.parameters()) > 0:
                with torch.no_grad():
                    torch.clamp_(model["model"].logit_scale.logit_scale, 0, np.log(train_args.logit_max))

        if train_args.wandb:
            if sum(p.requires_grad for p in model["model"].logit_scale.parameters()) > 0:
                self.log({"logit_scale": model["model"].logit_scale.logit_scale.exp().item()}, step=step)

        return loss

    def _eval_imagenet(self, model, dataloader, step, **kwargs):
        
        train_args = self.config.train_args
        text = model.text
        text.eval()
        vision = model.vision
        vision.eval()

        with torch.autocast(device_type="cuda", dtype=self.dtype):
            top1, top5, _, _ = evaluate_imagenet(
                text=text,
                vision=vision,
                tokenizer=self.tokenizer,
                dataloader=dataloader,
                return_embeddings=False,
                prefix="search_query" if self.config.text_model_args.add_prefix else None,
            )

        top1 = gather(top1)
        top5 = gather(top5)

        top1 = torch.mean(top1).item()
        top5 = torch.mean(top5).item()

        log_val = {"top1_acc": top1, "top5_acc": top5}

        if train_args.wandb:
            self.log(log_val, step=step)
        else:
            self.print(f"Step: {step} Top1: {top1} Top5: {top5}")

    def _eval_flickr(self, model, dataloader, step, **kwargs):
        tokenizer = self.tokenizer
        if self.global_rank == 0:
            tokenizer.model_max_length = 77
            tokenizer = partial(tokenizer, return_tensors="pt", truncation=True, padding="max_length")
            device = model.device
            metrics = zsr.evaluate(model, dataloader, tokenizer, recall_k_list=[1, 5, 10], device=device, amp=False)
            metrics["mean_recall@1"] = 0.5 * (metrics["text_retrieval_recall@1"] + metrics["image_retrieval_recall@1"])

            log_flickr = {"flickr/" + k: v for k, v in metrics.items()}
            if self.config.train_args.wandb:
                self.log(log_flickr, step=step)
            else:
                self.print(f"Step: {step} Flickr: {metrics}")

        dist.barrier()

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
    
    def _extract_embeddings(self, model, dataloader):
        all_image_embs = []
        all_text_embs = []
        all_labels = {label_name: [] for label_name in self.classifer_config.classes}
        all_valid_masks = {label_name: [] for label_name in self.classifer_config.classes}  # 新增：记录有效掩码

        device = model.device
        text, vision = model.text, model.vision
        text.eval()
        vision.eval()

        with torch.no_grad():
            for batch in dataloader:
                vision_inputs = {k: v.to(device) for k, v in batch["vision"].items()}
                text_inputs = {k: v.to(device) for k, v in batch["text"].items()}
                labels = batch["label"]
                autocast_ctx = torch.autocast(device_type="cuda", dtype=self.dtype) if device.type == "cuda" else nullcontext()
                with autocast_ctx:
                    image_emb = vision(**vision_inputs)["embedding"].to(device)
                    text_emb = text(**text_inputs)["embedding"].to(device)

                all_image_embs.append(image_emb)
                all_text_embs.append(text_emb)

                # 动态提取所有标签字段，并记录有效掩码
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

        # gather embeddings
        all_image_embs = gather(all_image_embs).to(device)
        all_text_embs = gather(all_text_embs).to(device)

        # concat & gather labels 和 masks
        for label_name in all_labels:
            all_labels[label_name] = torch.cat(all_labels[label_name], dim=0)
            all_labels[label_name] = gather(all_labels[label_name]).to(device)
            
            all_valid_masks[label_name] = torch.cat(all_valid_masks[label_name], dim=0)
            all_valid_masks[label_name] = gather(all_valid_masks[label_name]).to(device)

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
    
    def _eval_image_text_similarity(self, save_path, model, dataloader, step, **kwargs):
        train_args = self.config.train_args

        # 提取嵌入、标签和掩码
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

        # --------------------------
        # 修改：图文相似度使用所有样本（不用过滤）
        # --------------------------
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

        # 写入 CSV（保持不变）
        clip_csv = os.path.join(save_path, "clip_metrics.csv")
        attr_csv = os.path.join(save_path, "attr_metrics.csv")
        os.makedirs(save_path, exist_ok=True)
        
        clip_header = ["step"] + [f"clip_{k}_top1" for k in zero_shot_acc_dict.keys()]
        clip_row = [step] + [zero_shot_acc_dict[k] for k in zero_shot_acc_dict]
        write_clip_header = not os.path.exists(clip_csv)
        with open(clip_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if write_clip_header:
                writer.writerow(clip_header)
            writer.writerow(clip_row)

        attr_header = ["step"] + [f"attr_{k}_top1" for k in attr_acc_dict.keys()]
        attr_row = [step] + [attr_acc_dict[k] for k in attr_acc_dict]
        write_attr_header = not os.path.exists(attr_csv)
        with open(attr_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if write_attr_header:
                writer.writerow(attr_header)
            writer.writerow(attr_row)
                
        # 日志记录（适配新的label_stats字段）
        if train_args.wandb:
            self.log({"image_text_similarity": sim_mean}, step=step)
            
            # 计算并记录Zero-shot（clip）平均精度
            if zero_shot_acc_dict:  # 避免空字典导致的错误
                clip_acc_values = list(zero_shot_acc_dict.values())
                clip_mean_acc = sum(clip_acc_values) / len(clip_acc_values)
                self.log({"clip_mean_top1": clip_mean_acc}, step=step)
            else:
                clip_mean_acc = float('nan')
                self.log({"clip_mean_top1": clip_mean_acc}, step=step)
            
            # 记录每个Zero-shot精度
            for name, acc in zero_shot_acc_dict.items():
                self.log({
                    f"clip_{name}_top1": acc,
                }, step=step)
            
            # 计算并记录属性（attr）平均精度
            if attr_acc_dict:  # 避免空字典导致的错误
                attr_acc_values = list(attr_acc_dict.values())
                attr_mean_acc = sum(attr_acc_values) / len(attr_acc_values)
                self.log({"attr_mean_top1": attr_mean_acc}, step=step)
            else:
                attr_mean_acc = float('nan')
                self.log({"attr_mean_top1": attr_mean_acc}, step=step)
            
            # 记录每个属性精度
            for attr, acc in attr_acc_dict.items():
                self.log({
                    f"attr_{attr}_top1": acc,
                }, step=step)
                
        if is_main_process():
            print(f"[Eval] Step {step} | zero_shot/image_text_similarity: {sim_mean:.4f}")
            
            # 打印Zero-shot平均精度
            if zero_shot_acc_dict:
                clip_acc_values = list(zero_shot_acc_dict.values())
                clip_mean_acc = sum(clip_acc_values) / len(clip_acc_values)
            else:
                clip_mean_acc = float('nan')
            print(f"[Eval] Step {step} | zero_shot/mean_top1: {clip_mean_acc:.4f}")  # 新增平均精度
            
            print("\n[Zero-shot 分类精度及标签统计]")
            for name, acc in zero_shot_acc_dict.items():
                stats = label_stats.get(name, {
                    "加载时有效的标签数": 0,
                    "加载时无效的标签数": 0,
                    "加载有效且标签有效的数量": 0
                })
                print(
                    f"[Eval] Step {step} | zero_shot/{name}_top1: {acc:.4f} "
                    f"| 加载时有效的标签数: {stats['加载时有效的标签数']} "
                    f"| 加载时无效的标签数: {stats['加载时无效的标签数']}"
                )
            
            # 打印属性平均精度
            if attr_acc_dict:
                attr_acc_values = list(attr_acc_dict.values())
                attr_mean_acc = sum(attr_acc_values) / len(attr_acc_values)
            else:
                attr_mean_acc = float('nan')
            print(f"\n[Eval] Step {step} | attr/mean_top1: {attr_mean_acc:.4f}")  # 新增平均精度
            
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
                    f"| 加载时无效的标签数: {stats['加载时无效的标签数']}"
                )

    def eval_loop(self,save_path, model, dataloader, step, **kwargs):
        if isinstance(dataloader, dict):
            imagenet_dataloader = dataloader["imagenet"]
        else:
            imagenet_dataloader = dataloader
        self._eval_image_text_similarity(save_path,model, imagenet_dataloader, step, **kwargs)

        if isinstance(dataloader, dict):
            self._eval_flickr(model, dataloader["flickr"], step, **kwargs)

