import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import PreTrainedModel,AutoModel, AutoImageProcessor
import torch.nn as nn
from contrastors.distributed import gather_with_grad
from contrastors.models.biencoder import BiEncoder, LogitScale
from contrastors.loss import focal_loss,cosine_similarity_loss,image_text_loss
from collections import defaultdict
from .configuration_dual_encoder import DualEncoderConfig


class BiEncoder_teacher(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        model_name=config.teacher_vision_model_args.model_name
        model_config = DualEncoderConfig.from_pretrained(model_name)
        model = DualEncoder.from_pretrained(model_name, config=model_config)
        self.vision=model.vision
        

        
    def forward(self, input_ids, attention_mask=None, is_padded_inputs=True, normalize=True, binarize=False, **kwargs):
        image_emb = self.vision(input_ids)["embedding"]
        
        return image_emb



class BiEncoder_teacher_slip2(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        model_name=config.teacher_vision_model_args.model_name
        self.backbone= AutoModel.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, optimized=True
        )
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.linear=nn.Linear(1152,768)
        
        for param in self.backbone.parameters():
            param.requires_grad = False

    
    
    def forward(self, imgs):
        img_embs = self.backbone.encode_images(**imgs, normalize=True)
        img_embs=self.linear(img_embs)
        return img_embs




class DualEncoder(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        text_model_args = config.text_model_args
        if text_model_args.precomputed:
            assert text_model_args.freeze, "Precomputed text model must be frozen"
            self.precomputed_text = True
        else:
            self.precomputed_text = False
            
        self.vision = BiEncoder(config.image_model_args)
        
        self.text = BiEncoder(config.text_model_args)
        
        self.distill=False

        if config.image_model_args.distill_type!="":
            self.distill=True
            self.teacher_name=config.teacher_vision_model_args.model_name
            if "mexma-siglip2" in self.teacher_name:
                self.teacher_vision=BiEncoder_teacher_slip2(config)
                self.teacher_vision.eval()
            else:
                self.teacher_vision=BiEncoder_teacher(config)
                for param in self.teacher_vision.parameters():
                    param.requires_grad = False  # 核心：禁用参数的梯度计算
                # 可选：将教师网络设置为评估模式（避免BatchNorm等层的行为受训练影响）
                self.teacher_vision.eval()

            

        self.logit_scale = LogitScale(config.image_model_args)
        
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
    
    
    def encode_text(self, text, normalize=True):
        text_outputs = self.text(**text, normalize=normalize)
        return text_outputs["embedding"]

    def encode_image(self, vision, normalize=True):
        vision_outputs = self.vision(vision, normalize=normalize)

        return vision_outputs["embedding"]

    
    @staticmethod
    def multi_attribute_focal_loss(outputs, targets, attribute_classes, alpha=0.25, gamma=2.0):
        loss = 0.0
        loss_dict = {}
        for attr, num_classes in attribute_classes.items():
            logits = outputs[f"{attr}_logits"]     # (B, C)
            if isinstance(targets[attr], list):
                target = torch.tensor(targets[attr], dtype=torch.long, device=logits.device)
            else:
                target = targets[attr].to(logits.device)
                
            l = focal_loss(logits, target, alpha=alpha, gamma=gamma)
            loss_dict[f"{attr}_loss"] = l.detach()  # 保留 tensor
            loss += l
        loss_dict["classifier_loss"] = loss.detach()
        
        return loss, loss_dict
            
    
    # def multi_clip_loss():
    
    def forward(self, text_inputs, vision_inputs,targets,img_path=None):
        if self.precomputed_text:
            assert "text_embs" in text_inputs, "Precomputed text inputs must have text_embs"
            text_outputs = {"embedding": text_inputs["text_embs"]}
            vision_outputs = self.vision(**vision_inputs, normalize=False)
        else:
            text_outputs = self.text(**text_inputs, normalize=False)
            vision_outputs = self.vision(**vision_inputs, normalize=False)
        if self.distill:
            if "mexma-siglip2" in self.teacher_name:
                teacher_vision_outputs=self.teacher_vision(img_path)
            else:
                teacher_vision_outputs=self.teacher_vision(**vision_inputs, normalize=False)
        
        
        metrics = {}
        
        text_emb = F.normalize(text_outputs["embedding"], dim=-1, p=2)
        all_text_emb = gather_with_grad(text_emb)
        vision_emb = F.normalize(vision_outputs["embedding"], dim=-1, p=2)
        all_vis_emb = gather_with_grad(vision_emb)

        if self.distill:
            if "mexma-siglip2" in self.teacher_name:
                teacher_all_vis_emb = gather_with_grad(teacher_vision_outputs)
            else:
                teacher_vision_emb = F.normalize(teacher_vision_outputs, dim=-1, p=2)
                teacher_all_vis_emb = gather_with_grad(teacher_vision_emb)
        
        #图文对比损失
        image_loss,text_loss=image_text_loss(self.logit_scale,text_emb,all_text_emb,vision_emb,all_vis_emb)
        #蒸馏损失
        if self.distill:
            teacher_student_loss=cosine_similarity_loss(all_vis_emb,teacher_all_vis_emb)

        
        total_loss=0
        total_loss+=image_loss
        total_loss+=text_loss
        
        metrics = {"loss": total_loss, "image_loss": image_loss,"text_loss":text_loss}
        if self.distill:
            metrics.update({"teacher_student_loss":teacher_student_loss})

        #分类损失
        if self.attribute_classes:
            attr_outputs = {}
            valid_attrs = {}
            
            for attr, classifier in self.attribute_classifiers.items():
                if attr not in targets:
                    continue  # 如果目标中没有该属性标签，跳过
                # 计算分类器输出
                logits = classifier(vision_emb)  # (B, C)
                # 过滤掉标签为 -1 的样本
                valid_mask = (targets[attr] != -1)
                if valid_mask.sum() == 0:
                    continue  # 如果没有有效样本，跳过该属性
                # 应用掩码
                valid_logits = logits[valid_mask]
                valid_targets = targets[attr][valid_mask]
                attr_outputs[f"{attr}_logits"] = valid_logits
                valid_attrs[attr] = self.attribute_classes[attr]  # 只传递有效属性
            if valid_attrs:  # 如果存在至少一个有效的属性标签才计算损失
                attr_loss, loss_dict = self.multi_attribute_focal_loss(
                    attr_outputs,
                    {attr: targets[attr][targets[attr] != -1] for attr in valid_attrs},
                    valid_attrs,
                    alpha=0.25,
                    gamma=2.0
                )
                total_loss = total_loss + attr_loss
                metrics.update(loss_dict)

        metrics["loss"] = total_loss

        
        return metrics
