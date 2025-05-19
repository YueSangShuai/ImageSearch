import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np # For dummy data
from utils import import_var
from losses import PairwiseCircleCacheLoss, RKdAngle
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
from data import JSONLDataset
import os

tokenizer = None

def collate_fn(batch):
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("tbackbone/minimind_tokenizer")
        tokenizer.cls_token_id = 1
        
    captions = [item['text'] for item in batch] + [item['zh'] for item in batch]
    bs = len(batch)
    caption_ids = tokenizer(captions, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids        
    ret = {
        'text': captions[:bs],
        'caption': caption_ids[:bs],
        'zh_text': captions[bs:],
        'zh_caption': caption_ids[bs:],
        'idx': torch.tensor([item['idx'] for item in batch])
    }
    return ret

class Model(nn.Module):
    def __init__(self, args, state_dict=None):
        super().__init__()
        self.args = args
        global tokenizer
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained("tbackbone/minimind_tokenizer")
            tokenizer.cls_token_id = 1
        self.text_encoder = import_var(args.text_backbone)(args.feature_dim, max_seq_length=self.args.max_seq_length, 
            vocab_size=tokenizer.vocab_size, cls_token_id=tokenizer.cls_token_id,
            )
        if args.resume:
            try:
                self.text_encoder.load_state_dict(torch.load(args.resume, map_location="cpu", weights_only=False)['state_dict'], strict=False)
            except Exception as e:
                print(f"Failed to load model weights: {e}")
                self.load_state_dict(torch.load(args.resume, map_location="cpu", weights_only=False)['state_dict'], strict=False)
        if state_dict is not None:
            self.text_encoder.load_state_dict(state_dict, strict=True)
            print("Loaded model weights from state_dict")
        self.kd_circleloss_fun = PairwiseCircleCacheLoss(margin=0.2, gamma=30.0, cache_size=0)
        self.rkda_loss_fn = RKdAngle()
        self.load_teachor(args.teachor)
    def load_teachor(self, model_name):
        model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True, optimized=True, local_files_only=True)
        self.teachor_tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.teachor_model = model.to(self.args.device)
        for p in self.teachor_model.parameters():
            p.requires_grad = False
        self.teachor_model.eval()
    def state_dict(self):
        return self.text_encoder.state_dict()
    @torch.no_grad()
    def get_teachor_embedding(self, text):
        if isinstance(text, str):
            text = [text]
        text_inputs = self.teachor_tokenizer(
            text, 
            padding="max_length", 
            max_length=min(self.args.max_seq_length, self.teachor_tokenizer.model_max_length),
            truncation=True, 
            return_tensors='pt'
        ).to(self.args.device)
        text_embedding = self.teachor_model.encode_texts(normalize=True, **text_inputs)
        text_embedding = text_embedding.float()
        return text_embedding
    def forward(self, batch):  # -> loss
        """
        batch:
            caption: torch.Size([B, 200, 6400])        # 文本的词表索引
        return: losses
        """
        device = self.args.device
        caption = batch['caption'][:, :self.args.max_seq_length].to(device)
        zh_caption = batch['zh_caption'][:, :self.args.max_seq_length].to(device)
        batch_size = len(caption)
        embeddings = self.text_encoder(torch.concat([caption, zh_caption], dim=0))
        normed_embeddings = F.normalize(embeddings)
        caption_embeddings = embeddings[:batch_size]
        zh_caption_embeddings = embeddings[batch_size:]
        normed_caption_embeddings = normed_embeddings[:batch_size]
        normed_zh_caption_embeddings = normed_embeddings[batch_size:]
        normed_teachor_embeddings = self.get_teachor_embedding(batch['text'])
#        normed_zh_teachor_embeddings = self.get_teachor_embedding(batch['zh_text'])
        if not self.training: return normed_caption_embeddings
        args = self.args
        losses = dict()
        if args.teachor_sim_weight > 0:
            # 相似度蒸馏
            t_sim_matrix = torch.matmul(normed_teachor_embeddings, normed_teachor_embeddings.t())
            sim_matrix = torch.matmul(normed_caption_embeddings, normed_caption_embeddings.t())
            losses['sim'] = F.mse_loss(sim_matrix, t_sim_matrix, reduction='mean')*args.teachor_sim_weight
            Csim_matrix = torch.matmul(normed_zh_caption_embeddings, normed_zh_caption_embeddings.t())
            losses['Csim'] = F.mse_loss(Csim_matrix, t_sim_matrix, reduction='mean')*args.teachor_sim_weight
        if args.teachor_rel_weight > 0:
            # 关系蒸馏
            losses['rel'] = self.kd_circleloss_fun(
                normed_caption_embeddings, 
                batch['idx'].to(device),
                normed_teachor_embeddings)*args.teachor_rel_weight
            losses['Crel'] = self.kd_circleloss_fun(
                normed_zh_caption_embeddings, 
                batch['idx'].to(device),
                normed_teachor_embeddings)*args.teachor_rel_weight
        if args.teachor_l1_weight > 0:
            # teachor L1
            losses['l1'] = F.smooth_l1_loss(caption_embeddings, normed_teachor_embeddings*220)*args.teachor_l1_weight
            losses['Cl1'] = F.smooth_l1_loss(zh_caption_embeddings, normed_teachor_embeddings*220)*args.teachor_l1_weight
        if args.teachor_l2_weight > 0:
            # teachor L1
            losses['l2'] = F.mse_loss(caption_embeddings, normed_teachor_embeddings*220)*args.teachor_l2_weight
            losses['Cl2'] = F.mse_loss(zh_caption_embeddings, normed_teachor_embeddings*220)*args.teachor_l2_weight
        
        if args.teachor_info_weight > 0:
            # InfoNCE损失
            sim_matrix = torch.matmul(normed_caption_embeddings, normed_teachor_embeddings.T) / args.temperature
            labels = torch.arange(normed_caption_embeddings.shape[0], device=device)
            losses['info'] = F.cross_entropy(sim_matrix, labels, reduction='mean')
            Csim_matrix = torch.matmul(normed_zh_caption_embeddings, normed_teachor_embeddings.T) / args.temperature
            losses['Cinfo'] = F.cross_entropy(Csim_matrix, labels, reduction='mean')
        return losses


if __name__ == '__main__':
    # 测试 teachor embedding similarity
    data = [
        "A man with dark hair is wearing a blue shirt. He is also wearing a pair of purple pants and white shoes.",
        "A girl with long hair is wearing a red dress and a pair of white socks.",
        "A dog running in a park.",
        "a white table",
        ]
    from opts import get_args
    args = get_args()
    model = Model(args)
    embeddings = model.get_teachor_embedding(data)
    print(embeddings.shape,
        embeddings)
    dist = embeddings @ embeddings.T
    print(dist)
    
