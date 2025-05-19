import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np # For dummy data
from utils import import_var
from losses import PairwiseCircleCacheLoss, RKdAngle

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.image_encoder = import_var(args.backbone)(args.feature_dim)
        self.text_encoder = import_var(args.text_backbone)(args.feature_dim)
        if args.resume:
            self.load_state_dict(torch.load(args.resume, map_location="cpu", weights_only=False)['state_dict'])
        self.id_img_loss_fn = PairwiseCircleCacheLoss(margin=0.2, gamma=30.0, cache_size=100000, size_step=10)
        self.id_cap_loss_fn = PairwiseCircleCacheLoss(margin=0.2, gamma=30.0, cache_size=100000, size_step=10)
        self.id_img_cap_loss_fn = PairwiseCircleCacheLoss(margin=0.2, gamma=30.0, cache_size=100000, size_step=10)
        self.cap_distall_loss_fn = nn.SmoothL1Loss()
        self.img_distall_loss_fn = nn.SmoothL1Loss()
        self.kd_circleloss_fun = PairwiseCircleCacheLoss(margin=0.2, gamma=30.0, cache_size=0)
        self.rkda_loss_fn = RKdAngle()
    def forward(self, batch):  # -> loss
        """
        batch:
            image: torch.Size([B, 3, 224, 112])        # 图像
            image_embedding: torch.Size([B, 1152])     # 图像特征
            caption: torch.Size([B, 200, 6400])        # 文本的词表索引
            caption_embedding: torch.Size([B, 1152])   # 文本特征 
            attributes: torch.Size([B, 200, 6400])    # 属性词的词表索引
            attributes_embedding: torch.Size([B, 1152]) # 属性词的特征
            prompt_caption: torch.Size([B, 200, 6400]) # 提示词的词表索引
            prompt_caption_embedding: torch.Size([B, 1152]) # 提示词的特征
            has_embedding: torch.Size([B])             # 是否有预训练的文本和图像特征，True/False，若为False，则 image_embedding和caption_embedding无效（或不存在）
            id: torch.Size([B])                        # 人ID
        return: losses
        """
        device = self.args.device
        image = batch['image'].to(device)
        caption = batch['caption'].to(device)
        image_embedding = self.image_encoder(image)
        image_embedding = F.normalize(image_embedding)
        caption_embedding = self.text_encoder(caption)
        caption_embedding = F.normalize(caption_embedding)
        attributes = batch['attributes'].to(device)
        attributes_embedding = self.text_encoder(attributes)
        attributes_embedding = F.normalize(attributes_embedding)
        prompt_caption = batch['prompt_caption'].to(device)
        prompt_caption_embedding = self.text_encoder(prompt_caption)
        prompt_caption_embedding = F.normalize(prompt_caption_embedding)
        if not self.training:
            return image_embedding, caption_embedding
        args = self.args
        losses = dict()
        if 'id' in batch: person_id = batch['id'].to(device)
        if args.dist_rel_weight > 0 or args.rkda_weight > 0 or args.cap_distall_weight > 0 or args.img_distall_weight > 0 or args.cap_circle_weight > 0 or args.img_circle_weight > 0:
            # 跨模态关系蒸馏 和 模态内关系蒸馏
            # 挑选出有预训练的文本和图像特征
            has_embedding = batch['has_embedding'].to(device)
            has_embedding_index = torch.where(has_embedding)[0]
            if has_embedding_index.numel() > 0:
                t_image_embedding = batch['image_embedding'].to(device)[has_embedding_index]
                t_caption_embedding = batch['caption_embedding'].to(device)[has_embedding_index]
                i_e, c_e = image_embedding[has_embedding_index], caption_embedding[has_embedding_index]
                if args.dist_rel_weight > 0:
                    # 计算跨模态相似度矩阵
                    tempture = 4  # 由于Siglip2用于通用图像识别，我们观察到其用于行人检索时，图文相似度的普遍低于0.22，因此这里设置为4
                    t_sim_matrix = (torch.matmul(t_image_embedding, t_caption_embedding.t())*tempture).clamp(max=1.0)
                    sim_matrix = torch.matmul(i_e, c_e.t())
                    losses['dist_rel'] = F.mse_loss(sim_matrix, t_sim_matrix, reduction='mean')*args.dist_rel_weight
                    # 加权MSE损失
                    # 创建权重矩阵，对角线元素乘以diag_weight
                    # weights = torch.ones_like(sim_matrix)
                    # diag_mask = torch.eye(sim_matrix.size(0), device=device).bool()
                    # weights[diag_mask] *= args.feature_dim-1
                    # loss = (F.mse_loss(sim_matrix, t_sim_matrix, reduction='none') * weights)/weights.sum()
                    # losses['dist_rel'] = loss.sum() * args.dist_rel_weight
                if args.rkda_weight > 0:
                    # 模态内关系蒸馏
                    losses['rkda_img'] = self.rkda_loss_fn(i_e, t_image_embedding)*args.rkda_weight
                    losses['rkda_cap'] = self.rkda_loss_fn(c_e, t_caption_embedding)*args.rkda_weight
                if args.cap_distall_weight > 0:
                    # 文本模态内关系蒸馏
                    losses['cap_dist'] = self.cap_distall_loss_fn(c_e, t_caption_embedding)*args.cap_distall_weight
                if args.img_distall_weight > 0:
                    # 图像模态内关系蒸馏
                    losses['img_dist'] = self.img_distall_loss_fn(i_e, t_image_embedding)*args.img_distall_weight
                s_id = person_id[has_embedding_index]
                if args.img_circle_weight > 0:
                    losses['img_c'] = self.kd_circleloss_fun(i_e, s_id, t_image_embedding)*args.img_circle_weight
                if args.cap_circle_weight > 0:
                    # 直接蒸馏文本模态三种标注的嵌入特征
                    losses['cap_c'] = self.kd_circleloss_fun(c_e, s_id, t_caption_embedding)*args.cap_circle_weight
                    losses['a_c'] = self.kd_circleloss_fun(attributes_embedding[has_embedding_index], 
                        s_id, batch['attributes_embedding'].to(device)[has_embedding_index])*args.cap_circle_weight
                    losses['p_c'] = self.kd_circleloss_fun(prompt_caption_embedding[has_embedding_index], 
                        s_id, batch['prompt_caption_embedding'].to(device)[has_embedding_index])*args.cap_circle_weight
            else:
                if args.dist_rel_weight > 0: losses['dist_rel'] = torch.tensor(0.0, device=device)
                if args.rkda_weight > 0: 
                    losses['rkda_img'] = torch.tensor(0.0, device=device)   
                    losses['rkda_cap'] = torch.tensor(0.0, device=device)

        if args.id_image_weight > 0: # 图像模态长记忆ID对比损失
            losses['id_img'] = self.id_img_loss_fn(image_embedding, person_id)*args.id_image_weight
        if args.id_cap_weight > 0: # 文本模态长记忆ID对比损失
            losses['id_cap'] = self.id_cap_loss_fn(caption_embedding, person_id)*args.id_cap_weight
        if args.id_img_cap_weight > 0: # 跨模态长记忆ID对比损失
            losses['id_img_cap'] = self.id_img_cap_loss_fn(torch.cat([image_embedding, caption_embedding],0), torch.cat([person_id, person_id],0))*args.id_img_cap_weight

        sim_matrix = torch.matmul(image_embedding, caption_embedding.t())
        pos = torch.eye(sim_matrix.size(0), device=device) # 相似度矩阵的对角线为正样本
        neg = 1 - pos
        pos_score = (sim_matrix*pos).sum()/pos.sum()
        neg_score = (sim_matrix*neg).sum()/neg.sum()
        losses['d_pos'] = 1-pos_score     # 要求正样本的相似度越高越好
        losses['s_neg'] = neg_score       # 要求负样本的相似度越低越好

        return losses