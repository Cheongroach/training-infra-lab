"""GRPO clipped surrogate with explicit global reduction and KL contract."""

from __future__ import annotations

import torch

from .contracts import UpdateBatch


def collate(batch: UpdateBatch, indices, device, pad_id=0):
    trajectories = [batch.trajectories[i] for i in indices]
    if not trajectories:
        raise ValueError("Collate needs at least one trajectory")
    length = max(len(t.input_ids) for t in trajectories)
    ids = torch.full((len(indices), length), pad_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    mask = torch.zeros((len(indices), length - 1), device=device)
    old, reference = torch.zeros_like(mask), torch.zeros_like(mask)
    for row, t in enumerate(trajectories):
        n = len(t.input_ids)
        ids[row, :n] = torch.tensor(t.input_ids, device=device)
        attention[row, :n] = 1
        mask[row, :n - 1] = torch.tensor(t.response_mask, device=device)
        old[row, :n - 1] = torch.tensor(t.old_log_probs, device=device)
        if t.reference_log_probs:
            reference[row, :n - 1] = torch.tensor(t.reference_log_probs, device=device)
    return {
        "input_ids": ids, "attention_mask": attention, "response_mask": mask,
        "old_log_probs": old, "reference_log_probs": reference,
        "advantages": torch.tensor([batch.advantages()[i] for i in indices], device=device),
    }


def surrogate_sum(log_probs, data, reduction, *, clip=0.2, beta=0.0):
    """Returns local numerator. The executor applies world/global denominator.

    beta=0 is an explicit no-reference GRPO objective. A caller enabling beta
    must first validate that EVERY trajectory contains reference log-probs.
    """
    if not 0 < clip < 1 or beta < 0:
        raise ValueError("Invalid clipping or KL coefficient")
    mask = data["response_mask"].bool()
    # Exclude padding BEFORE exp: masked infinity times zero is still NaN.
    delta = torch.where(mask, log_probs - data["old_log_probs"], 0.0)
    ratio = delta.exp()
    advantage = data["advantages"][:, None]
    objective = -torch.minimum(
        ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage,
    )
    if beta:
        ref_delta = torch.where(mask, data["reference_log_probs"] - log_probs, 0.0)
        objective = objective + beta * (ref_delta.exp() - ref_delta - 1)
    objective = torch.where(mask, objective, 0.0)
    if reduction == "sequence_mean":
        return (objective.sum(-1) / mask.sum(-1).clamp_min(1)).sum()
    if reduction == "token_mean":
        return objective.sum()
    raise ValueError(reduction)
