from contextlib import nullcontext

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb

from contrastors.distributed import gather, gather_with_grad
from contrastors.rand_state import RandContext


def calculate_auxiliary_loss(router_logits, num_experts, top_k, attention_mask=None, tracker=None, name=None, step=None):
    # TODO: this might be annoying since we unpad during training
    device = router_logits[0].device
    concatenated_router_logits = torch.cat(router_logits, dim=0).to(device)

    router_weights = F.softmax(concatenated_router_logits, dim=-1)
    _, selected_experts = torch.topk(router_weights, top_k, dim=-1)

    expert_mask = F.one_hot(selected_experts, num_experts)
    if attention_mask is not None:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_router_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(router_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    else:
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
        router_prob_per_expert = torch.mean(router_weights, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))

    if tracker is not None:
        # log the percentage of tokens routed to each expert
        tokens_per_expert = tokens_per_expert.detach().cpu().numpy()
        router_prob_per_expert = router_prob_per_expert.detach().cpu().numpy()
        for k in range(top_k):
            tokens_per_expert_k = tokens_per_expert[k]

            tpe_data = [(f"expert_{i}", tokens_per_expert_k[i]) for i in range(num_experts)]
            tpe_table = wandb.Table(data=tpe_data, columns=["expert", "prob"])
            tracker.log({f"tokens_per_expert_pct_k{k}": wandb.plot.bar(tpe_table, "expert", "prob", title=f"tokens_per_expert_pct_k{k}")}, step=step)

        rpe_data = [(f"expert_{i}", router_prob_per_expert[i]) for i in range(num_experts)]
        rpe_table = wandb.Table(data=rpe_data, columns=["expert", "prob"])
        tracker.log({f"router_prob_per_expert_pct": wandb.plot.bar(rpe_table, "expert", "prob", title="router_prob_per_expert_pct")}, step=step)

    return overall_loss * num_experts


def clip_loss(
    query,
    document,
    logit_scale,
    step=None,
    gather_enabled=False,
    tracker=None,
    dataset="",
    bidirectional=False,
):
    """Calculates the InfoNCE Loss for a batch of queries and documents.
    Inspired by: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/loss.py#L66

    Assumes that query.shape[0] <= document.shape[0]
    This will work for non-square matrices as well

    params:
        query: torch.Tensor of shape N x D
        document: torch.Tensor of shape M x D where M >= N
        temperature: torch.Tensor of shape 1

    returns:
        torch.Tensor of shape 1 corresponding to the loss
    """
    if gather_enabled:
        document = gather_with_grad(document)

    device = query.device

    if query.dtype != document.dtype:
        document = document.to(query.dtype)

    labels = torch.arange(query.shape[0]).to(device)
    similarity_query_document = logit_scale(torch.matmul(query, document.T))
    num_logits = similarity_query_document.size(0)
    rank = dist.get_rank() if dist.is_initialized() else 0
    # calculate sub-batch labels
    labels = labels + rank * num_logits

    # if training with negatives
    # multiply by world size since we only gather the document embeddings
    labels = labels * (document.size(0) // (query.size(0) * dist.get_world_size()))

    if bidirectional:
        similarity_document_query = logit_scale(torch.matmul(document, query.T))
        loss = (
            F.cross_entropy(similarity_query_document, labels) + F.cross_entropy(similarity_document_query, labels)
        ) #* dist.get_world_size()
    else:
        loss = F.cross_entropy(similarity_query_document, labels) * dist.get_world_size()

    if tracker is not None:
        # this will only calculate 1/N accuracy where N is the number of gpus
        accuracy = (similarity_query_document.argmax(dim=1) == labels).float().mean()
        tracker.log({f"accuracy/accuracy_{dataset}": accuracy.detach().cpu().item()}, step=step)

    return loss


def get_chunked_embeddings(model, chunks):
    embeddings = []
    rand_states = []

    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            for chunk in chunks:
                rand_states.append(RandContext(chunk))
                emb = model(**chunk)
                embeddings.append(emb["embedding"])

    return torch.concat(embeddings, dim=0), rand_states


def accumulate_gradients(model, inputs, cache, rand_states, router_aux_coeff):
    length = len(inputs)
    sync_contexts = [model.no_sync for _ in range(length - 1)] + [nullcontext]

    for inp, grad, state, sync_context in zip(inputs, cache, rand_states, sync_contexts):
        with sync_context():
            with state:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    embedding = model(**inp)
            surrogate = torch.dot(embedding["embedding"].flatten(), grad.flatten())
            if "router_loss" in embedding and embedding["router_loss"] is not None:
                surrogate = surrogate + embedding["router_loss"] * router_aux_coeff
            surrogate.backward()


def cache_loss(tower1, tower2, query_embeddings, document_embeddings, logit_scale, bidirectional=False):
    # only require grad for embedding / representation
    query_embs = query_embeddings.detach().requires_grad_()
    document_embs = document_embeddings.detach().requires_grad_()

    # I'm not sure this works for LiT
    # TODO: this broke everything with grad cache!
    # no_tower1_sync = getattr(tower1, "no_sync", nullcontext)
    # no_tower2_sync = getattr(tower2, "no_sync", nullcontext)
    no_tower1_sync, no_tower2_sync = nullcontext, nullcontext

    with torch.autocast("cuda", dtype=torch.bfloat16):
        with no_tower1_sync():
            with no_tower2_sync():
                loss = clip_loss(query_embs, document_embs, logit_scale, gather_enabled=True, bidirectional=bidirectional)
                loss.backward()

    query_cache = query_embs.grad
    document_cache = document_embs.grad

    return query_cache, document_cache, loss.detach()


def grad_cache_loss(tower1, t1_inputs, tower2, t2_inputs, chunk_size, logit_scale, bidirectional=False, router_aux_coeff=False):
    total_bs = t1_inputs["input_ids"].shape[0]
    chunked_queries = []
    chunked_documents = []

    for chunk_start in range(0, total_bs, chunk_size):
        query_chunk = {k: v[chunk_start : chunk_start + chunk_size] for k, v in t1_inputs.items()}
        chunked_queries.append(query_chunk)

        document_chunk = {k: v[chunk_start : chunk_start + chunk_size] for k, v in t2_inputs.items()}
        chunked_documents.append(document_chunk)

    query_embs, query_rand_states = get_chunked_embeddings(tower1, chunked_queries)
    document_embs, doc_rand_states = get_chunked_embeddings(tower2, chunked_documents)

    query_cache, document_cache, loss = cache_loss(
        tower1, tower2, query_embs, document_embs, logit_scale, bidirectional=bidirectional
    )

    chunked_query_cache = query_cache.split(chunk_size)
    chunked_document_cache = document_cache.split(chunk_size)

    accumulate_gradients(tower1, chunked_queries, chunked_query_cache, query_rand_states, router_aux_coeff=router_aux_coeff)
    if tower2.training:
        accumulate_gradients(tower2, chunked_documents, chunked_document_cache, doc_rand_states, router_aux_coeff=router_aux_coeff)

    return loss


def focal_loss(logits, target, alpha=1.0, gamma=2.0, reduction='mean'):
    """
    Focal Loss for multi-class classification (function version).

    Args:
        logits (Tensor): shape (B, C), raw model outputs (before softmax)
        target (Tensor): shape (B,), ground truth class indices
        alpha (float): balance factor
        gamma (float): focusing parameter
        reduction (str): 'mean', 'sum' or 'none'

    Returns:
        Tensor: loss value
    """
    log_probs = F.log_softmax(logits, dim=1)                      # (B, C)
    probs = torch.exp(log_probs)                                  # (B, C)
    target_log_probs = log_probs[range(logits.size(0)), target]   # (B,)
    target_probs = probs[range(logits.size(0)), target]           # (B,)

    focal_weight = (1 - target_probs) ** gamma                    # (B,)
    loss = -alpha * focal_weight * target_log_probs               # (B,)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss  # shape (B,)
    
 
def image_text_loss(logit_scale,text_emb,all_text_emb,vision_emb,all_vis_emb):
    logits_per_image = logit_scale(vision_emb @ all_text_emb.T)
    logits_per_text = logit_scale(text_emb @ all_vis_emb.T)

    num_logits = logits_per_image.shape[0]
    labels = torch.arange(num_logits).to(logits_per_image.device)
    labels = labels + num_logits * dist.get_rank()

    
    image_loss=F.cross_entropy(logits_per_image, labels)/2* dist.get_world_size()
    text_loss=F.cross_entropy(logits_per_text, labels)/2** dist.get_world_size()
    
    return image_loss,text_loss




def cosine_similarity_loss(vec1, vec2, target=1.0):
    """
    余弦相似度损失：使vec1和vec2的余弦相似度接近target（默认为1，即方向一致）
    Args:
        vec1: 向量1，形状 [batch_size, dim]
        vec2: 向量2，形状 [batch_size, dim]
        target: 目标相似度（1表示方向相同，-1表示方向相反）
    Returns:
        loss: 平均损失值
    """
    # 计算余弦相似度（范围[-1, 1]）
    cos_sim = F.cosine_similarity(vec1, vec2, dim=1)
    # 转换为损失（与目标值的MSE，或使用1 - cos_sim直接作为损失）
    loss = F.mse_loss(cos_sim, torch.full_like(cos_sim, target))
    # 或：loss = (1 - cos_sim).mean()  # 若目标是让相似度最大化
    return loss
