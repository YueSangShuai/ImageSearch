import torch
import torch.nn as nn
from torch.nn import functional as F
import os, math

class FocalLoss(nn.Module):
    def __init__(self, gamma=0.0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()
    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

class SoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()

class DRKDLoss(nn.Module):
    # Dynamic Rectification Knowledge Distillation
    # https://arxiv.org/abs/2201.11319
    def __init__(self, t=1):
        super().__init__()
        self.t = t
        self.kd_loss = nn.KLDivLoss()
    def forward(self, outputs, teacher_outputs, labels):
        # 根据标签修正教师的输出
        predicted_labels = torch.argmax(teacher_outputs, 1)
        selected = predicted_labels != labels
        t_labels = labels[selected]
        t_predicted_labels = predicted_labels[selected]
        saved = teacher_outputs[selected,t_predicted_labels].clone()
        teacher_outputs[selected,t_predicted_labels] = teacher_outputs[selected,t_labels]
        teacher_outputs[selected,t_labels] = saved
        # 计算损失
        T = self.t
        loss_KD = self.kd_loss(F.log_softmax(outputs / T, dim=1), 
            F.softmax(teacher_outputs / T, dim=1))
        return T * T * loss_KD

class RKdAngle(nn.Module):
    # Relational Knowledge Distillation: Angle-wise Loss
    # 参考：https://arxiv.org/pdf/1904.05068v2.pdf
    def forward(self, student, teacher):
        # N x C
        # N x N x C

        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='mean')
        return loss


class CoupleLoss(nn.Module):
    def __init__(self, feature_root, K=100, margin=0.03):
        super().__init__()
        id_prototypes_fn = os.path.join(feature_root, "prototypes.bin")
        self.id_prototypes = torch.load(id_prototypes_fn)
        idH_fn = os.path.join(feature_root, f'idH-100.bin')
        self.idH = torch.load(idH_fn)
        self.q = margin
        self.K = min(100,K)
    def to(self, device):
        self.id_prototypes = self.id_prototypes.to(device)
        self.idH = self.idH.to(device)
    def forward(self, ftr, teachor_ftr, label):
        # ftr = F.normalize(ftr)
        # teachor_ftr = F.normalize(teachor_ftr)
        y = label
        self.id_prototypes[y]=teachor_ftr
        losses = []
        for i in range(y.shape[0]):
            yi = y[i]
            fsi = ftr[i:i+1]
            fti = teachor_ftr[i:i+1]
            gi = self.id_prototypes[self.idH[yi,:self.K]]
            tmrs = torch.mm(gi, fti.t())[:,0]
            smrs = torch.mm(gi, fsi.t())[:,0]
            loss = F.relu(smrs - tmrs - self.q)
            losses.append(loss)
        loss = torch.cat(losses)
        return loss.mean()
                
class SpaceLoss1(nn.Module):
    def __init__(self, feature_root):
        super().__init__()
        id_prototypes_fn = os.path.join(feature_root, "prototypes.bin")
        self.id_prototypes = torch.load(id_prototypes_fn)
    def to(self, device):
        self.id_prototypes = self.id_prototypes.to(device)
    def forward(self, ftr, teachor_ftr, label):
        # ftr = F.normalize(ftr)
        # teachor_ftr = F.normalize(teachor_ftr)
        y = self.id_prototypes[label]
        losses = []
        for f, tf in zip(ftr, teachor_ftr):
            tf_da = torch.mm(y, tf[:,None])[:,0]
            f_da = torch.mm(y, f[:,None])[:,0]
            loss = torch.norm(tf_da - f_da)
            losses.append(loss)
        return sum(losses)/len(losses)

class SpaceLoss(nn.Module):
    def __init__(self, feature_root, K=1000, margin=0.03):
        super().__init__()
        id_prototypes_fn = os.path.join(feature_root, "prototypes.bin")
        self.id_prototypes = torch.load(id_prototypes_fn)
        self.K = K
        self.margin = margin
    def to(self, device):
        self.id_prototypes = self.id_prototypes.to(device)
    def forward(self, ftr, teachor_ftr, label):
        # ftr = F.normalize(ftr)
        # teachor_ftr = F.normalize(teachor_ftr)
        y = self.id_prototypes[label]
        ap_distance = 1-(ftr*y).sum(dim=1)  # cos distance
        lbl = torch.randperm(self.id_prototypes.shape[0])[:self.K]
        y = self.id_prototypes[lbl]
        an_distances = 1-torch.mm(ftr, y.t()) # N x K cos distance
        an_distance = []
        for f, tf, l, an_dist in zip(ftr, teachor_ftr, label, an_distances):
            da, di = an_dist.sort(dim=0)
            an_distance.append(da[1 if di[0] == l else 0])
        an_distance = torch.tensor(an_distance, device=ap_distance.device)
        loss = F.relu(ap_distance - an_distance + self.margin)
        return loss.mean()

def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt                

class DKDLoss(nn.Module):
    """
    Decoupled Knowledge Distillation(CVPR 2022)
    https://openaccess.thecvf.com/content/CVPR2022/papers/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.pdf
    """
    def __init__(self, alpha=0.5, beta=2., temperature=1.):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
    def forward(self, logits_student, logits_teacher, target):
        alpha, beta, temperature = self.alpha, self.beta, self.temperature
        gt_mask = _get_gt_mask(logits_student, target)
        other_mask = _get_other_mask(logits_student, target)
        pred_student = F.softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        pred_student = cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = torch.log(pred_student)
        tckd_loss = (
            F.kl_div(log_pred_student, pred_teacher, size_average=False, reduction="batchmean")
            * (temperature**2)
            / target.shape[0]
        )
        pred_teacher_part2 = F.softmax(
            logits_teacher / temperature - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_student / temperature - 1000.0 * gt_mask, dim=1
        )
        nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False, reduction="batchmean")
            * (temperature**2)
            / target.shape[0]
        )
        return alpha * tckd_loss + beta * nckd_loss


class EKD(nn.Module):
    """ Evaluation-oriented knowledge distillation for deep face recognition, CVPR2022
    """

    def __init__(self):
        super().__init__()
        self.topk = 2000
        self.t = 0.01
        self.anchor = [10, 100, 1000, 10000, 100000, 1000000]
        self.momentum = 0.01
        self.register_buffer('s_anchor', torch.zeros(len(self.anchor)))
        self.register_buffer('t_anchor', torch.zeros(len(self.anchor)))

    def forward(self, g_s, g_t, labels):
        # normalize feature
        class_size = labels.size(0)
        g_s = g_s.view(class_size, -1)
        g_s = F.normalize(g_s)
        classes_eq = (labels.repeat(class_size, 1) == labels.view(-1, 1).repeat(1, class_size))
        # print("classes_eq = ", classes_eq)
        similarity_student = torch.mm(g_s, g_s.transpose(0, 1))
        s_inds = torch.triu(torch.ones(classes_eq.size(), device=g_s.device), 1).bool()

        pos_inds = classes_eq[s_inds]
        # print("pos_inds = ", pos_inds)
        neg_inds = ~classes_eq[s_inds]
        # print("neg_inds = ", neg_inds)
        s = similarity_student[s_inds]
        pos_similarity_student = torch.masked_select(s, pos_inds)
        neg_similarity_student = torch.masked_select(s, neg_inds)
        sorted_s_neg, sorted_s_index = torch.sort(neg_similarity_student, descending=True)

        with torch.no_grad():
            g_t = g_t.view(class_size, -1)
            g_t = F.normalize(g_t)
            similarity_teacher = torch.mm(g_t, g_t.transpose(0, 1))
            t = similarity_teacher[s_inds]
            pos_similarity_teacher = torch.masked_select(t, pos_inds)
            neg_similarity_teacher = torch.masked_select(t, neg_inds)
            sorted_t_neg, _ = torch.sort(neg_similarity_teacher, descending=True)
            length = sorted_s_neg.size(0)
            select_indices = [length // anchor for anchor in self.anchor]
            s_neg_thresholds = sorted_s_neg[select_indices]
            t_neg_thresholds = sorted_t_neg[select_indices]
            self.s_anchor = self.momentum * s_neg_thresholds + (1 - self.momentum) * self.s_anchor
            self.t_anchor = self.momentum * t_neg_thresholds + (1 - self.momentum) * self.t_anchor
        s_pos_kd_loss = self.relative_loss(pos_similarity_student, pos_similarity_teacher)

        s_neg_selected = neg_similarity_student[sorted_s_index[0:self.topk]]
        t_neg_selected = neg_similarity_teacher[sorted_s_index[0:self.topk]]

        s_neg_kd_loss = self.relative_loss(s_neg_selected, t_neg_selected)

        loss = s_pos_kd_loss * 0.02 + s_neg_kd_loss * 0.01

        return loss

    def sigmoid(self, inputs, temp=1.0):
        """ temperature controlled sigmoid
            takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
        """
        exponent = -inputs / temp
        # clamp the input tensor for stability
        exponent = torch.clamp(exponent, min=-50, max=50)
        y = 1.0 / (1.0 + torch.exp(exponent))
        return y

    def relative_loss(self, s_similarity, t_similarity):
        s_distance = s_similarity.unsqueeze(1) - self.s_anchor.unsqueeze(0)
        t_distance = t_similarity.unsqueeze(1) - self.t_anchor.unsqueeze(0)

        s_rank = self.sigmoid(s_distance, self.t)
        t_rank = self.sigmoid(t_distance, self.t)

        s_rank_count = s_rank.sum(axis=1, keepdims=True)
        t_rank_count = t_rank.sum(axis=1, keepdims=True)

        s_kd_loss = F.mse_loss(s_rank_count, t_rank_count)
        return s_kd_loss

class SpaceLoss2(nn.Module):
    def __init__(self, feature_root, margin=0.5):
        from triplet_losses import TripletLoss
        super().__init__()
        id_prototypes_fn = os.path.join(feature_root, "prototypes.bin")
        self.id_prototypes = torch.load(id_prototypes_fn)
        self.triplet_loss = TripletLoss(margin=0.3)
        self.margin = margin
    def to(self, device):
        self.id_prototypes = self.id_prototypes.to(device)
    def forward(self, ftr, teachor_ftr, label):
        y = self.id_prototypes[label]
        similarity = torch.mm(ftr, ftr.t())
        losses = []
        for f, y_, sim, lbl in zip(ftr, y, similarity, label):
            max_sim, max_i = sim.sort()
            for s, i in zip(max_sim, max_i):
                if i == lbl: continue
                # distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
                # distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
                # losses = F.relu(distance_positive - distance_negative + self.margin)
                losses.append(F.relu((f-y_).pow(2).sum() - (f-ftr[i]).pow(2).sum() + self.margin))
                break
        return sum(losses) / len(losses)



class CC(nn.Module):
    """ Correlation Congruence for Knowledge Distillation. ICCV 2019
    """
    def __init__(self, gamma=0.4, P_order=2):
        super().__init__()
        self.gamma = gamma
        self.P_order = P_order

    def forward(self, feat_s, feat_t):
        corr_mat_s = self.get_correlation_matrix(feat_s)
        corr_mat_t = self.get_correlation_matrix(feat_t)
        loss = F.mse_loss(corr_mat_s, corr_mat_t)
        return loss

    def get_correlation_matrix(self, feat):
        feat = F.normalize(feat)
        sim_mat = torch.matmul(feat, feat.t())
        corr_mat = torch.zeros_like(sim_mat)
        for p in range(self.P_order+1):
            corr_mat += math.exp(-2*self.gamma) * (2*self.gamma)**p / \
                        math.factorial(p) * torch.pow(sim_mat, p)
        return corr_mat        

class SP(nn.Module):
    """Similarity-Preserving Knowledge Distillation, ICCV2019
    """
    def __init__(self):
        super().__init__()

    def similarity_loss(self, f_s, f_t):
        bsz = f_s.shape[0]
        f_s = f_s.view(bsz, -1)
        f_t = f_t.view(bsz, -1)

        G_s = torch.mm(f_s, torch.t(f_s))
        G_s = F.normalize(G_s)
        G_t = torch.mm(f_t, torch.t(f_t))
        G_t = F.normalize(G_t)

        G_diff = G_t - G_s
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        return loss

    def sp_loss(self, g_s, g_t):
        return sum([self.similarity_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)])


def svd(feat, n=1):
    size = feat.shape
    assert len(size) == 4

    x = feat.view(size[0], size[1] * size[2], size[3]).float()
    u, s, v = torch.svd(x)

    u = removenan(u)
    s = removenan(s)
    v = removenan(v)

    if n > 0:
        u = F.normalize(u[:, :, :n], dim=1)
        s = F.normalize(s[:, :n], dim=1)
        v = F.normalize(v[:, :, :n], dim=1)

    return u, s, v


def removenan(x):
    x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
    return x


def align_rsv(a, b):
    cosine = torch.matmul(a.transpose(-2, -1), b)
    max_abs_cosine, _ = torch.max(torch.abs(cosine), 1, keepdim=True)
    mask = torch.where(
        torch.eq(max_abs_cosine, torch.abs(cosine)),
        torch.sign(cosine),
        torch.zeros_like(cosine),
    )
    a = torch.matmul(a, mask)
    return a, b


class KDSVD(nn.Module):
    """Self-supervised Knowledge Distillation using Singular Value Decomposition
    """
    def __init__(self, k=1.0):
        super().__init__()
        self.k = k
    def forward(self, g_s, g_t):
        v_sb = None
        v_tb = None
        losses = []
        for i, f_s, f_t in zip(range(len(g_s)), g_s, g_t):
            u_t, s_t, v_t = svd(f_t, self.k)
            u_s, s_s, v_s = svd(f_s, self.k + 3)
            v_s, v_t = align_rsv(v_s, v_t)
            s_t = s_t.unsqueeze(1)
            v_t = v_t * s_t
            v_s = v_s * s_t

            if i > 0:
                s_rbf = torch.exp(-(v_s.unsqueeze(2) - v_sb.unsqueeze(1)).pow(2) / 8)
                t_rbf = torch.exp(-(v_t.unsqueeze(2) - v_tb.unsqueeze(1)).pow(2) / 8)

                l2loss = (s_rbf - t_rbf.detach()).pow(2)
                l2loss = torch.where(
                    torch.isfinite(l2loss), l2loss, torch.zeros_like(l2loss)
                )
                losses.append(l2loss.sum())

            v_tb = v_t
            v_sb = v_s

        bsz = g_s[0].shape[0]
        losses = [l / bsz for l in losses]
        return sum(losses)

class PairwiseCircleloss(nn.Module):
    def __init__(self, margin=0.4, gamma=30.0):
        super().__init__()
        self.margin = margin
        self.gamma = gamma
    def forward(self, 
        embedding: torch.Tensor,
        targets: torch.Tensor) -> torch.Tensor:
        margin = self.margin
        gamma = self.gamma
        embedding = F.normalize(embedding, dim=1)
        dist_mat = torch.matmul(embedding, embedding.t())

        N = dist_mat.size(0)

        is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
        is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

        # Mask scores related to itself
        is_pos = is_pos - torch.eye(N, N, device=is_pos.device)

        s_p = dist_mat * is_pos
        s_n = dist_mat * is_neg

        alpha_p = torch.clamp_min(-s_p.detach() + 1 + margin, min=0.)
        alpha_n = torch.clamp_min(s_n.detach() + margin, min=0.)
        delta_p = 1 - margin
        delta_n = margin

        logit_p = - gamma * alpha_p * (s_p - delta_p) + (-99999999.) * (1 - is_pos)
        logit_n = gamma * alpha_n * (s_n - delta_n) + (-99999999.) * (1 - is_neg)

        loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()

        return loss

class PairwiseCirclelossKD(nn.Module):
    def __init__(self, margin=0.4, gamma=30.0):
        super().__init__()
        self.margin = margin
        self.gamma = gamma
    def forward(self, 
        embedding_s: torch.Tensor,
        embedding_t: torch.Tensor,
        targets: torch.Tensor) -> torch.Tensor:
        margin = self.margin
        gamma = self.gamma
        embedding = torch.cat([embedding_s, embedding_t], dim=0)
        targets = torch.cat([targets, targets], dim=0)
        #embedding = F.normalize(embedding, dim=1)
        dist_mat = torch.matmul(embedding, embedding.t())

        N = dist_mat.size(0)

        is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
        is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

        # Mask scores related to itself
        is_pos = is_pos - torch.eye(N, N, device=is_pos.device)

        s_p = dist_mat * is_pos
        s_n = dist_mat * is_neg

        alpha_p = torch.clamp_min(-s_p.detach() + 1 + margin, min=0.)
        alpha_n = torch.clamp_min(s_n.detach() + margin, min=0.)
        delta_p = 1 - margin
        delta_n = margin

        logit_p = - gamma * alpha_p * (s_p - delta_p) + (-99999999.) * (1 - is_pos)
        logit_n = gamma * alpha_n * (s_n - delta_n) + (-99999999.) * (1 - is_neg)

        loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()

        return loss

class PairwiseCosface(nn.Module):
    def __init__(self, margin=0.4, gamma=30.0) -> None:
        super().__init__()
        self.margin = margin
        self.gamma = gamma
    def forward(self, 
        embedding: torch.Tensor,
        targets: torch.Tensor) -> torch.Tensor:
        margin = self.margin
        gamma = self.gamma
        # Normalize embedding features
        embedding = F.normalize(embedding, dim=1)

        dist_mat = torch.matmul(embedding, embedding.t())

        N = dist_mat.size(0)
        is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
        is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

        # Mask scores related to itself
        is_pos = is_pos - torch.eye(N, N, device=is_pos.device)

        s_p = dist_mat * is_pos
        s_n = dist_mat * is_neg

        logit_p = -gamma * s_p + (-99999999.) * (1 - is_pos)
        logit_n = gamma * (s_n + margin) + (-99999999.) * (1 - is_neg)

        loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()

        return loss

class PairwiseCosfaceKD(nn.Module):
    def __init__(self, margin=0.4, gamma=30.0) -> None:
        super().__init__()
        self.margin = margin
        self.gamma = gamma
    def forward(self, 
        embedding_s: torch.Tensor,
        embedding_t: torch.Tensor,
        targets: torch.Tensor) -> torch.Tensor:
        margin = self.margin
        gamma = self.gamma
        embedding = torch.cat([embedding_s, embedding_t], dim=0)
        targets = torch.cat([targets, targets], dim=0)
        # Normalize embedding features
        #embedding = F.normalize(embedding, dim=1)

        dist_mat = torch.matmul(embedding, embedding.t())

        N = dist_mat.size(0)
        n = N//2
        is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
        is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

        # Mask scores related to itself
        is_pos = is_pos - torch.eye(N, N, device=is_pos.device)

        s_p = (dist_mat * is_pos)[:n]
        s_n = (dist_mat * is_neg)[:n]

        logit_p = -gamma * s_p + (-99999999.) * (1 - is_pos[:n])
        logit_n = gamma * (s_n + margin) + (-99999999.) * (1 - is_neg[:n])

        loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()

        return loss



class PairwiseSoftMargin(nn.Module):
    def __init__(self, margin=0.4, gamma=30.0) -> None:
        super().__init__()
        self.margin = margin
        self.gamma = gamma
        self.loss_fun = nn.SoftMarginLoss()
    def forward(self, 
        embedding: torch.Tensor,
        targets: torch.Tensor) -> torch.Tensor:
        margin = self.margin
        gamma = self.gamma
        # Normalize embedding features
        embedding = F.normalize(embedding, dim=1)

        dist_mat = torch.matmul(embedding, embedding.t()) # cos similarity, -1,1 
        # 0,1-> -1,1 (x-0.5)*2 | -1,1->-3,1
        dist_mat = dist_mat*4.0

        N = dist_mat.size(0)
        is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
        is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

        # Mask scores related to itself
        is_pos = is_pos - torch.eye(N, N, device=is_pos.device)

        labels = torch.zeros(dist_mat.shape, device=dist_mat.device).int()
        labels[is_neg>0] = -1
        labels[is_pos>0] = 1
        selected = labels!=0
        loss = self.loss_fun(dist_mat[selected], labels[selected])
        return loss


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
class VICReg(nn.Module):
    def __init__(self, sim_coeff, std_coeff, cov_coeff):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
    def forward(self, embedding_s, embedding_t, targets):
        batch_size, num_features = embedding_s.shape
        x, y = embedding_s, embedding_t
        repr_loss = F.mse_loss(x, y)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(num_features)

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        return loss
        
class PairwiseTripletKD(nn.Module):
    def __init__(self, margin=0.4, gamma=30.0) -> None:
        super().__init__()
        self.margin = margin
        self.gamma = gamma
    def forward(self, 
        embedding_s: torch.Tensor,
        embedding_t: torch.Tensor,
        targets: torch.Tensor) -> torch.Tensor:
        margin = self.margin
        gamma = self.gamma
        embedding = torch.cat([embedding_s, embedding_t], dim=0)
        targets = torch.cat([targets, targets], dim=0)
        # Normalize embedding features
        #embedding = F.normalize(embedding, dim=1)

        dist_mat = torch.matmul(embedding, embedding.t())  # cosine distance

        N = dist_mat.size(0)
        n = N//2
        is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
        is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

        # Mask scores related to itself
        is_pos = is_pos - torch.eye(N, N, device=is_pos.device)
        is_neg[:,n:] = 0  # remove embedding_s with embedding_t pair

        s_p = ((2-dist_mat) * is_pos)[:n]
        sp_value, sp_index = torch.max(s_p, dim=1)
        s_n = ((dist_mat+1) * is_neg)[:n]
        sn_value, sn_index = torch.max(s_n, dim=1)
        sn_value = 2-(sn_value-1)
        loss = F.relu(sp_value - sn_value + self.margin).mean()
        return loss

class FtrBinLoss(nn.Module):
    def __init__(self, bin_count=32, min_value=-99, max_value=102, alpha=0.01):
        super().__init__()
        self.alpha = alpha
        self.start = min_value
        self.end = max_value
        self.bin_count = bin_count
        self.step = (self.end - self.start) / (self.bin_count - 1)  # 根据 bin_count 动态计算步长
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1,)
        self.reg_criterion = nn.MSELoss()
        
    def forward(self, predict_bins, targets):
        # 计算分类标签
        b = predict_bins.size(0)
        bin_labels = ((targets - self.start) / self.step).clamp(0, self.bin_count - 1).long()
        bin_loss = self.criterion(predict_bins.reshape(-1, self.bin_count), bin_labels.view(-1))  # 分类损失
        if self.alpha == 0: return bin_loss
        # 连续值预测并计算回归损失
        predicted_continuous = self.predict(predict_bins)
        reg_loss = self.reg_criterion(predicted_continuous, targets)
        
        # 总损失：分类损失 + 加权回归损失
        total_loss = bin_loss + self.alpha * reg_loss
        return total_loss

    def predict(self, predict_bins):
        idx_tensor = torch.arange(self.bin_count, dtype=torch.float32, device=predict_bins.device)
        return torch.sum(F.softmax(predict_bins, dim=1) * idx_tensor, dim=1) * self.step + self.start


class FtrBinLossKD(nn.Module):
    def __init__(self, bin_count=128, min_value=-0.49, max_value=0.49, temperature=1., is_kd=False):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.bin_count = bin_count
        self.t = temperature
        self.kd = is_kd
    def forward(self, logits, teachor_embeddings, labels):
        N,C = teachor_embeddings.shape
        bin_labels = ((F.normalize(teachor_embeddings, p=2, dim=1)-self.min_value)*self.bin_count/(self.max_value-self.min_value)).clamp(min=0, max=self.bin_count-1).long()
        if not self.kd:
            predict_bins = logits.reshape(N*C, self.bin_count)
            bin_losses = F.cross_entropy(predict_bins, bin_labels.view(-1)) 
            return bin_losses
        one_hot = torch.zeros((N*C,self.bin_count), device=logits.device)
        one_hot.scatter_(1, bin_labels.view(-1, 1).long(), 1)
        outputs = logits.view(N, -1) # N C*bin_count)
        teacher_outputs = one_hot.view(N, -1)+0.1
        T = self.t
        loss_KD = F.kl_div(F.log_softmax(outputs / T, dim=1), 
            F.softmax(teacher_outputs / T, dim=1), reduction="batchmean")
        return T * T * loss_KD * C

class SphereFace2LossKD(nn.Module):
    """ reference: <SphereFace2: Binary Classification is All You Need
                    for Deep Face Recognition>
        margin='C' -> SphereFace2-C
        margin='A' -> SphereFace2-A
        marign='M' -> SphereFAce2-M
    """
    def __init__(self, num_class, magn_type='C', alpha=0.7, s=40., m=0.4, t=3., lw=10.):
        super().__init__()
        self.magn_type = magn_type

        # alpha is the lambda in paper Eqn. 5
        self.alpha = alpha
        self.r = s
        self.m = m
        self.t = t
        self.lw = lw
        self.num_class = num_class
        r=s

        # init bias
        z = alpha / ((1. - alpha) * (num_class - 1.))
        if magn_type == 'C':
            ay = r * (2. * 0.5**t - 1. - m)
            ai = r * (2. * 0.5**t - 1. + m)
        elif magn_type == 'A':
            theta_y = min(math.pi, math.pi/2. + m)
            ay = r * (2. * ((math.cos(theta_y) + 1.) / 2.)**t - 1.)
            ai = r * (2. * 0.5**t - 1.)
        elif magn_type == 'M':
            theta_y = min(math.pi, m * math.pi/2.)
            ay = r * (2. * ((math.cos(theta_y) + 1.) / 2.)**t - 1.)
            ai = r * (2. * 0.5**t - 1.)
        else:
            raise NotImplementedError

        temp = (1. - z)**2 + 4. * z * math.exp(ay - ai)
        b = (math.log(2. * z) - ai
             - math.log(1. - z +  math.sqrt(temp)))
        self.b = b
        # self.b = nn.Parameter(torch.Tensor(1))
        # nn.init.constant_(self.b, b)
        print("SphereFace2LossKD.b:", b)
    def forward(self, embedding_s, embedding_t, label):
        # weight = F.normalize(embedding_t, dim=1)
        # embedding_s = F.normalize(embedding_s, dim=1)
        #delta theta with margin
        cos_theta = embedding_s.mm(embedding_t.t())
        one_hot = torch.zeros_like(cos_theta)
        #y = label.view(-1, 1)
        y = []
        for i in range(embedding_s.shape[0]):
            lbl = int(label[i])
            if lbl in y:
                y.append(y.index(lbl))
            else:
                y.append(len(y))
        y = torch.tensor(y, dtype=torch.long, device=embedding_s.device).view(-1, 1)
        #y = torch.arange(0, embedding_s.shape[0], dtype=torch.long, device=embedding_s.device).view(-1,1)
        one_hot.scatter_(1, y, 1.)
        with torch.no_grad():
            if self.magn_type == 'C':
                g_cos_theta = 2. * ((cos_theta + 1.) / 2.).pow(self.t) - 1.
                g_cos_theta = g_cos_theta - self.m * (2. * one_hot - 1.)
            elif self.magn_type == 'A':
                theta_m = torch.acos(cos_theta.clamp(-1+1e-5, 1.-1e-5))
                theta_m.scatter_(1, y, self.m, reduce='add')
                theta_m.clamp_(1e-5, 3.14159)
                g_cos_theta = torch.cos(theta_m)
                g_cos_theta = 2. * ((g_cos_theta + 1.) / 2.).pow(self.t) - 1.
            elif self.magn_type == 'M':
                m_theta = torch.acos(cos_theta.clamp(-1+1e-5, 1.-1e-5))
                m_theta.scatter_(1, y, self.m, reduce='multiply')
                m_theta.clamp_(1e-5, 3.14159)
                g_cos_theta = torch.cos(m_theta)
                g_cos_theta = 2. * ((g_cos_theta + 1.) / 2.).pow(self.t) - 1.
            else:
                raise NotImplementedError
            d_theta = g_cos_theta - cos_theta
        
        logits = self.r * (cos_theta + d_theta) + self.b
        weight = self.alpha * one_hot + (1. - self.alpha) * (1. - one_hot)
        weight = self.lw * self.num_class / self.r * weight
        loss = F.binary_cross_entropy_with_logits(
                logits, one_hot, weight=weight)

        return loss
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'out_features=' + str(self.num_class) \
               + ', s=' + str(self.r) \
               + ', m=' + str(self.m) \
               + ', alpha=' + str(self.alpha) \
               + ', lw=' + str(self.lw) \
               + ', t=' + str(self.t) \
               + ')'

class InfoNCE(nn.Module):
    """
    https://github.com/RElbers/info-nce-pytorch/blob/main/info_nce/__init__.py
    
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.
    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113
    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.
    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.
    Returns:
         Value of the InfoNCE Loss.
     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)

def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = [None if x is None else F.normalize(x, dim=-1) for x in (query, positive_key, negative_keys)]
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ (negative_keys.transpose(-2, -1))

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ (negative_keys.transpose(-2, -1))
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ (positive_key.transpose(-2, -1))

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)

class PADL2Loss(nn.Module):
    """
    "Prime-Aware Adaptive Distillation"
    """
    def __init__(self, feat_dim, teachor_dim, dropout=0.5):
        super().__init__()
        self.var_estimator = nn.Sequential(
            nn.Linear(feat_dim, teachor_dim),
            nn.BatchNorm1d(teachor_dim)
        )
        self.align = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, teachor_dim),
            nn.BatchNorm1d(teachor_dim)
        ) if (feat_dim != teachor_dim) or (dropout>0) else nn.Identity()
    def forward(self, student_embed_outputs, teacher_embed_outputs):
        log_variances = 2*torch.sigmoid(self.var_estimator(student_embed_outputs))-1  # 限定其范围使得训练稳定
        student_embed_outputs = self.align(student_embed_outputs)
        squared_losses = torch.mean(
            (teacher_embed_outputs - student_embed_outputs) ** 2 / (1e-6 + torch.exp(log_variances))
            + log_variances+1, dim=1
        )
        return squared_losses.mean()

class MultiSimilarityLoss(nn.Module):
    def __init__(self, scale_pos=1.0, scale_neg=1.0, threshold=0.5, margin=0.1, use_cosine=True):
        super().__init__()
        self.thresh = threshold
        self.margin = margin
        self.use_cosine = use_cosine
        self.scale_pos = scale_pos
        self.scale_neg = scale_neg
    def forward(self, feats, targets):
        norm_feats = F.normalize(feats, dim=1)
        batch_size = norm_feats.size(0)
        sim_mat = torch.matmul(norm_feats, torch.t(norm_feats))
        epsilon = 1e-5
        dist_mat = torch.matmul(norm_feats, norm_feats.t())

        N = dist_mat.size(0)
        is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
        is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

        # Mask scores related to itself
        is_pos = is_pos - torch.eye(N, N, device=is_pos.device)

        pos_pair = dist_mat[is_pos>0]
        neg_pair = dist_mat[is_neg>0]
        pos_loss = 0
        if is_pos.sum()>0:
            pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))/is_pos.sum()
        neg_loss = 0
        if is_neg.sum()>0:
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))/is_neg.sum()
        return pos_loss+neg_loss

        for i in range(batch_size):
            pos_pair_ = sim_mat[i][labels == labels[i]]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = sim_mat[i][labels != labels[i]]

            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
            pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)]

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            # weighting step
            pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        loss = sum(loss) / batch_size
        return loss        

class NKDLoss(nn.Module):
    """
    https://github.com/yzd-v/cls_KD/blob/master/mmcls/distillation/losses/nkd.py
    PyTorch version of NKD: `Rethinking Knowledge Distillation via Cross-Entropy` 
    """

    def __init__(self, temp=1.0, alpha=1.5):
        super(NKDLoss, self).__init__()
        self.temp = temp
        self.alpha = alpha

    def forward(self, logit_s, logit_t, gt_label):
        if len(gt_label.size()) > 1:
            label = torch.max(gt_label, dim=1, keepdim=True)[1]
        else:
            label = gt_label.view(len(gt_label), 1)

        # N*class
        y_i = F.softmax(logit_s, dim=1)
        t_i = F.softmax(logit_t, dim=1)
        # N*1
        y_t = torch.gather(y_i, 1, label)
        w_t = torch.gather(t_i, 1, label).detach()

        mask = torch.zeros_like(logit_s).scatter_(1, label, 1).bool()
        logit_s = logit_s - 1000 * mask
        logit_t = logit_t - 1000 * mask
        
        # N*class
        T_i = F.softmax(logit_t/self.temp, dim=1)
        S_i = F.softmax(logit_s/self.temp, dim=1)
        # N*1
        T_t = torch.gather(T_i, 1, label)
        S_t = torch.gather(S_i, 1, label)
        # N*class 
        np_t = T_i/(1-T_t)
        np_s = S_i/(1-S_t)
        np_t[T_i==T_t] = 0
        np_s[T_i==T_t] = 1

        soft_loss = - (w_t * torch.log(y_t)).mean() 
        distributed_loss =  (np_t * torch.log(np_s)).sum(dim=1).mean()
        distributed_loss = - self.alpha * (self.temp**2) * distributed_loss

        return soft_loss + distributed_loss         

class TFNKDLoss(nn.Module):
    """PyTorch version of tf-NKD: `Rethinking Knowledge Distillation via Cross-Entropy` """
    def __init__(self):
        super().__init__()
    def forward(self, logit_s, gt_label):
        if len(gt_label.size()) > 1:
            value, label = torch.sort(gt_label, descending=True, dim=-1)
        else:
            label = gt_label.view(len(gt_label),1)
            value = torch.ones_like(label)
        # N*class
        y_i = F.softmax(logit_s, dim=1)
        s_loss = {}
        if len(gt_label.size()) > 1:
            t_len = 2
        else:
            t_len = 1
        for i in range(t_len):
            y_t = torch.gather(y_i, 1, label[:,i].unsqueeze(-1))
            w_t = y_t + value[:,i].unsqueeze(-1) - y_t.mean()
            w_t[value[:,i].unsqueeze(-1)==0] = 0
            w_t = w_t.detach()
            s_loss['loss_tf_nkd'+str(i)] = - (w_t*torch.log(y_t)).mean()
        return s_loss


class TKDSelfLoss(nn.Module):
    # loss function for self training: Tf-KD_{self}
    # https://github.com/yuanli2333/Teacher-free-Knowledge-Distillation/blob/master/my_loss_function.py
    def __init__(self, alpha=0.95, temperature=20, multiplier=1.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.multiplier = multiplier
    def forward(self, logit_s, logit_t, labels):
        T = self.temperature
        loss_CE = F.cross_entropy(logit_s, labels)
        D_KL = nn.KLDivLoss()(F.log_softmax(logit_s/T, dim=1), F.softmax(logit_t/T, dim=1)) * (T * T) * self.multiplier  # multiple is 1.0 in most of cases, some cases are 10 or 50
        KD_loss =  (1. - self.alpha)*loss_CE + self.alpha*D_KL
        return KD_loss


class TSelfRegLoss(nn.Module):
    # loss function for mannually-designed regularization: Tf-KD_{reg}
    # https://github.com/yuanli2333/Teacher-free-Knowledge-Distillation/blob/master/my_loss_function.py
    def __init__(self, alpha=0.95, temperature=20, multiplier=1.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.multiplier = multiplier
    def forward(self, logits, labels):
        alpha = self.alpha
        T = self.temperature
        correct_prob = 0.99    # the probability for correct class in u(k)
        loss_CE = F.cross_entropy(logits, labels)
        K = logits.size(1)
        teacher_soft = torch.ones_like(logits).cuda()
        teacher_soft = teacher_soft*(1-correct_prob)/(K-1)  # p^d(k)
        for i in range(logits.shape[0]):
            teacher_soft[i ,labels[i]] = correct_prob
        loss_soft_regu = nn.KLDivLoss()(F.log_softmax(logits, dim=1), F.softmax(teacher_soft/T, dim=1))*self.multiplier
        KD_loss = (1. - alpha)*loss_CE + alpha*loss_soft_regu
        return KD_loss 

class AlignL2Loss(nn.Module):
    def __init__(self, feat_dim, teachor_dim, dropout=0.5):
        super().__init__()
        self.align = nn.Sequential(
            nn.Linear(feat_dim, teachor_dim),
            nn.BatchNorm1d(teachor_dim),
            nn.PReLU(teachor_dim),
            nn.Dropout(dropout),
            nn.Linear(teachor_dim, teachor_dim),
            nn.BatchNorm1d(teachor_dim),
            nn.PReLU(teachor_dim),
        )
    def forward(self, student_embed_outputs, teacher_embed_outputs):
#        print(student_embed_outputs.shape, teacher_embed_outputs.shape)
        student_embed_outputs = self.align(student_embed_outputs)
        return F.mse_loss(student_embed_outputs, teacher_embed_outputs)

class MGDLoss(nn.Module):
    """PyTorch version of `Masked Generative Distillation`
   
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map. 
        alpha_mgd (float, optional): Weight of dis_loss. Defaults to 0.00007
        lambda_mgd (float, optional): masked ratio. Defaults to 0.5
    """
    def __init__(self,
                 student_channels,
                 teacher_channels,
                 alpha_mgd=0.00007,
                 lambda_mgd=0.15,
                 ):
        super().__init__()
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd
    
        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

        self.generation = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
            )


    def forward(self,
                preds_S,
                preds_T):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """
        shape = preds_S.shape
        if len(shape)==2:
            preds_S = preds_S.view(shape[0], shape[1], 1, 1)
            preds_T = preds_T.view(shape[0], preds_T.shape[1], 1, 1)

        if self.align is not None:
            preds_S = self.align(preds_S)
    
        loss = self.get_dis_loss(preds_S, preds_T)*self.alpha_mgd
            
        return loss

    def get_dis_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        device = preds_S.device
        mat = torch.rand((N,C,1,1)).to(device)
        # mat = torch.rand((N,1,H,W)).to(device)
        mat = torch.where(mat < self.lambda_mgd, 0, 1).to(device)

        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation(masked_fea)

        dis_loss = loss_mse(new_fea, preds_T)/N

        return dis_loss

class ContrastiveLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x, y):
        target = torch.arange(0, y.size(0), int(y.size(0) / x.size(0)), device=x.device)
        scores = torch.matmul(x, y.transpose(0, 1))
        return F.cross_entropy(scores, target=target)
        
class CombinedPairwiseCacheLoss(nn.Module):
    def __init__(self, margin=0.4, gamma=30.0, cache_size=10000, ignore_hard_positives=False, size_step=10):
        super().__init__()
        self.margin = margin
        self.gamma = gamma
        self.size = cache_size
        self.cache_features = None
        self.cache_labels = None
        self.ignore_hard_positives = ignore_hard_positives  # 控制是否忽略硬正样本
        self.index = 0
        self.size_step = size_step
    def forward(self, 
                embedding: torch.Tensor,
                targets: torch.Tensor, t_embedding: torch.Tensor = None) -> torch.Tensor:
        self.index += 1
        margin = self.margin
        gamma = self.gamma
        N = embedding.size(0)
        if t_embedding is not None:
            embedding = torch.cat([embedding, t_embedding], dim=0)
            targets = torch.cat([targets, targets], dim=0)
        embedding = F.normalize(embedding, dim=1)

        # Cache features
        if self.cache_features is None or self.size==0:
            self.cache_features = embedding.detach()
            self.cache_labels = targets.detach()
        else:
            size = self.index*self.size_step
            if size>self.size: 
                size=self.size
            if size<N: # 保证cache的大小不小于N
                size=N
            self.cache_features = torch.cat([embedding.detach(), self.cache_features], dim=0)[:size]
            self.cache_labels = torch.cat([targets.detach(), self.cache_labels], dim=0)[:size]

        dist_mat = torch.matmul(embedding, self.cache_features.t())
        N, M = dist_mat.shape

        is_pos = targets.view(N, 1).expand(N, M).eq(self.cache_labels.view(M, 1).expand(M, N).t()).float()
        is_neg = targets.view(N, 1).expand(N, M).ne(self.cache_labels.view(M, 1).expand(M, N).t()).float()

        # Mask scores related to itself
        is_pos[:,:N] = is_pos[:,:N] - torch.eye(N, N, device=is_pos.device)

        s_p = dist_mat * is_pos
        s_n = dist_mat * is_neg

        # Ignore hard positive if enabled
        if self.ignore_hard_positives:
            is_not_hard_pos = dist_mat > 0.1  # 定义硬正样本的阈值
            s_p = s_p * is_not_hard_pos

        alpha_p = torch.clamp_min(-s_p.detach() + 1 + margin, min=0.)
        alpha_n = torch.clamp_min(s_n.detach() + margin, min=0.)
        delta_p = 1 - margin
        delta_n = margin

        logit_p = - gamma * alpha_p * (s_p - delta_p) + (-99999999.) * (1 - is_pos)
        logit_n = gamma * alpha_n * (s_n - delta_n) + (-99999999.) * (1 - is_neg)

        loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()

        return loss
    
class PairwiseCircleCacheLoss(CombinedPairwiseCacheLoss):
    def __init__(self, margin=0.4, gamma=30.0, cache_size=10000, size_step=10):
        super().__init__(margin=margin, gamma=gamma, cache_size=cache_size, ignore_hard_positives=False, size_step=size_step)

class PairwiseCosCacheLoss(CombinedPairwiseCacheLoss):
    def __init__(self, margin=0.4, gamma=30.0, cache_size=10000, size_step=10):
        super().__init__(margin=margin, gamma=gamma, cache_size=cache_size, ignore_hard_positives=True, size_step=size_step)
        

class PKT(nn.Module):
    """Probabilistic Knowledge Transfer for deep representation learning
    Code from author: https://github.com/passalis/probabilistic_kt"""
    def __init__(self):
        super(PKT, self).__init__()

    def forward(self, f_s, f_t, label=None):
        return self.cosine_similarity_loss(f_s, f_t)

    @staticmethod
    def cosine_similarity_loss(output_net, target_net, eps=0.0000001):
        # Normalize each vector by its norm
        output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
        output_net = output_net / (output_net_norm + eps)
        output_net[output_net != output_net] = 0

        target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
        target_net = target_net / (target_net_norm + eps)
        target_net[target_net != target_net] = 0

        # Calculate the cosine similarity
        model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
        target_similarity = torch.mm(target_net, target_net.transpose(0, 1))

        # Scale cosine similarity to 0..1
        model_similarity = (model_similarity + 1.0) / 2.0
        target_similarity = (target_similarity + 1.0) / 2.0

        # Transform them into probabilities
        model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
        target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)

        # Calculate the KL-divergence
        loss = torch.mean(target_similarity * torch.log((target_similarity + eps) / (model_similarity + eps)))

        return loss

class KLDivergence(nn.Module):
    """A measure of how one probability distribution Q is different from a
    second, reference probability distribution P.

    Args:
        tau (float): Temperature coefficient. Defaults to 1.0.
        reduction (str): Specifies the reduction to apply to the loss:
            ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``.
            ``'none'``: no reduction will be applied,
            ``'batchmean'``: the sum of the output will be divided by
                the batchsize,
            ``'sum'``: the output will be summed,
            ``'mean'``: the output will be divided by the number of
                elements in the output.
            Default: ``'batchmean'``
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(
        self,
        tau=1.0,
        reduction='batchmean',
    ):
        super(KLDivergence, self).__init__()
        self.tau = tau

        accept_reduction = {'none', 'batchmean', 'sum', 'mean'}
        assert reduction in accept_reduction, \
            f'KLDivergence supports reduction {accept_reduction}, ' \
            f'but gets {reduction}.'
        self.reduction = reduction

    def forward(self, preds_S, preds_T):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W) or shape (N, C).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W) or shape (N, C).

        Return:
            torch.Tensor: The calculated loss value.
        """
        preds_T = preds_T.detach()
        softmax_pred_T = F.softmax(preds_T / self.tau, dim=1)
        logsoftmax_preds_S = F.log_softmax(preds_S / self.tau, dim=1)
        loss = (self.tau**2) * F.kl_div(
            logsoftmax_preds_S, softmax_pred_T, reduction=self.reduction)
        return loss

class ContrastSigmoidLoss(nn.Module):
    def __init__(self, temperature=1.0, bias_init=-10.0):
        super().__init__()
        self.temperature = temperature
        self.bias = nn.Parameter(torch.tensor(bias_init))
    def forward(self, image_embeddings, labels):
        batch_size = image_embeddings.size(0)

        # Normalize the embeddings
        image_embeddings = F.normalize(image_embeddings, p=2, dim=1)

        # Compute pairwise similarity
        logits = torch.matmul(image_embeddings, image_embeddings.t()) * self.temperature + self.bias

        # Create labels matrix: 1 for positive pairs (same class), -1 for negative pairs (different classes)
        labels_matrix = (labels.unsqueeze(0) == labels.unsqueeze(1)).float() * 2 - 1

        # Compute sigmoid loss
        loss = -torch.mean(torch.log(torch.sigmoid(labels_matrix * logits)))
        
        return loss

class BiLD(nn.Module):
    """
    BiLD: Bi-directional Logits Difference Loss for Large Language Model Distillation
    https://github.com/fpcsong/BiLD/blob/main/distil_losses/BILD.py

    """
    def __init__(self, top_k=8, temperature=3, student_led=False):
        super().__init__()
        self.top_k = top_k
        self.temperature = temperature
        self.student_led = student_led

    def forward(self, logits_s, logits_t):
        """
        Bi-directional Logits Difference loss.

        Args:
            logits_s (torch.Tensor): the student logits, shape (batch_size, seq_len, vocab_size).
            logits_t (torch.Tensor): the teacher logits, shape (batch_size, seq_len, vocab_size).
            top_k (int, optional): choose top-k logits for calculating loss, defaults to 8.
            temperature (int, optional): the temperature, defaults to 3.
            student_led (bool, optional): if true, calculate student-led logits difference loss (t-LD), else t-LD.
        """
        top_k=self.top_k
        temperature=self.temperature
        student_led=self.student_led
        # print(logits_t.shape, logits_s.shape, top_k, temperature, student_led) # torch.Size([64, 28430]) torch.Size([64, 28430]) 8 3.0 False
        pair_num = top_k * (top_k-1) // 2

        if not student_led:
            # select top-k teacher logits & corresponding student logits
            with torch.no_grad():
                select_logits_t, select_pos = torch.topk(logits_t, k=top_k, dim=-1)
            select_logits_s = torch.gather(logits_s, 1, select_pos)
        else:
            # select top-k student logits & corresponding teacher logits
            select_logits_s, select_pos = torch.topk(logits_s, k=top_k, dim=-1)
            with torch.no_grad():
                select_logits_t = torch.gather(logits_t, 1, select_pos)

        scaled_logits_t = select_logits_t / temperature
        scaled_logits_s = select_logits_s / temperature

        # calculate logit difference
        def get_prob_diff(logits):
            b, v = logits.size()
            i, j = torch.triu_indices(v, v, offset=1)

            logits_diff = logits[..., i] - logits[..., j]

            return logits_diff

        logits_diff_t = get_prob_diff(scaled_logits_t)
        logits_diff_s = get_prob_diff(scaled_logits_s)

        logits_diff_t = F.softmax(logits_diff_t, dim=-1)

        loss = F.kl_div(F.log_softmax(logits_diff_s, dim=-1), logits_diff_t, reduction='batchmean')

        return loss