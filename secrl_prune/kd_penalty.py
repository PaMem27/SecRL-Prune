"""
Knowledge-distillation penalty on teacher Top-K target distribution.

This variant uses the student's FULL softmax and compares against the teacher's
distribution restricted to the top-K tokens plus an OTHER bucket that collects
the remaining mass (1 - sum_k). The teacher Top-K probabilities are expected to
be raw (i.e., not renormalized over the K tokens).
"""
from typing import Literal

import torch
import torch.nn.functional as F

Reduction = Literal["mean", "sum"]


def kd_kl_on_topk(
    student_logits_seq,        # len T, each [V] or [1, V]
    teacher_topk_ids_seq,      # len T, each [k] (Long)
    teacher_topk_probs_seq,    # len T, each [k] (Float), RAW (not renormed)
    temp: float = 1.0,
    eps: float = 1e-8,
    reduction: "Reduction" = "mean",
):
    assert len(student_logits_seq) == len(teacher_topk_ids_seq) == len(teacher_topk_probs_seq)
    T = len(student_logits_seq)
    kls = []

    denom = float(max(temp, eps))

    for t in range(T):
        logits_t = student_logits_seq[t]
        if logits_t.dim() == 2:  # [1, V] -> [V]
            logits_t = logits_t[0]
        logits_t_f = logits_t.float()

        # Student: full softmax & gather top-K
        log_pS_full = F.log_softmax(logits_t_f / denom, dim=-1)     # [V]
        ids_k = teacher_topk_ids_seq[t].long()                       # [k]
        log_pS_k = log_pS_full.gather(dim=-1, index=ids_k)          # [k]
        pS_k = log_pS_k.exp()                                        # [k]
        sum_pS_k = pS_k.sum()
        pS_other = torch.clamp(1.0 - sum_pS_k, min=eps)
        
        # Teacher: sanitize first, then compute OTHER= max(1 - sum_k, 0), then renorm on (K ∪ OTHER)
        pT_k_raw = torch.nan_to_num(
            teacher_topk_probs_seq[t].to(logits_t_f.dtype),
            nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(min=0.0)
        sum_pT_k_raw = pT_k_raw.sum()
        pT_other_raw = (1.0 - sum_pT_k_raw).clamp(min=0.0)

        denomT = torch.clamp(sum_pT_k_raw + pT_other_raw, min=eps)
        pT_k = (pT_k_raw / denomT).clamp(min=eps)
        pT_other = (pT_other_raw / denomT).clamp(min=eps)

        # KL(T || S) over Top-K and OTHER
        kl_topk = torch.sum(pT_k * (torch.log(pT_k) - torch.log(torch.clamp(pS_k, min=eps))))
        kl_other = pT_other * (torch.log(pT_other) - torch.log(pS_other))
        kl_t = kl_topk + kl_other

        kls.append(torch.nan_to_num(kl_t, nan=0.0, posinf=1e6, neginf=1e6))

    kls = torch.stack(kls)  # [T]
    out = kls.mean() if reduction == "mean" else kls.sum()
    return torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=1e6)
