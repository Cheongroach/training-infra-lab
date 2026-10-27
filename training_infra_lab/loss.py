"""Exact supervised-token CE with bounded temporary logits.

The objective is always a SUM. The trainer divides by the number of supervised
tokens across ALL ranks and accumulated microbatches, not by a local mean.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def full_logits_sum(hidden, head, labels):
    logits = head(hidden)
    return F.cross_entropy(
        logits[:, :-1].contiguous().float().flatten(0, 1),
        labels[:, 1:].contiguous().flatten(),
        ignore_index=-100,
        reduction="sum",
    )


def supervised_dense_sum(hidden, head, labels):
    targets = labels[:, 1:].reshape(-1)
    selected = targets.ne(-100)
    states = hidden[:, :-1].reshape(-1, hidden.size(-1))[selected]
    if not selected.any():
        return head(hidden[:, :1]).float().sum() * 0
    return F.cross_entropy(head(states).float(), targets[selected], reduction="sum")


def supervised_chunk_sum(hidden, head, labels, chunk_tokens=256):
    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be positive")
    targets = labels[:, 1:].reshape(-1)
    selected = targets.ne(-100)
    states = hidden[:, :-1].reshape(-1, hidden.size(-1))[selected]
    targets = targets[selected]
    if targets.numel() == 0:
        # Preserve a zero gradient for both the backbone and output parameters.
        return head(hidden[:, :1]).float().sum() * 0

    def chunk_loss(x, y):
        return F.cross_entropy(head(x).float(), y, reduction="sum")

    loss = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, targets.numel(), chunk_tokens):
        x = states[start : start + chunk_tokens]
        y = targets[start : start + chunk_tokens]
        # Recompute each chunk in backward. Otherwise autograd retains every
        # chunk's logits and splitting alone does not bound peak memory.
        if torch.is_grad_enabled():
            value = checkpoint(chunk_loss, x, y, use_reentrant=False)
        else:
            value = chunk_loss(x, y)
        loss = loss + value
    return loss


class SupervisedModel(torch.nn.Module):
    def __init__(self, causal_lm, mode="full", chunk_tokens=256):
        super().__init__()
        self.backbone = causal_lm.model
        self.head = causal_lm.lm_head
        self.mode = mode
        self.chunk_tokens = chunk_tokens

    def forward(self, input_ids, attention_mask, labels):
        hidden = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
        ).last_hidden_state
        if self.mode == "full":
            return full_logits_sum(hidden, self.head, labels)
        if self.mode == "selected":
            return supervised_dense_sum(hidden, self.head, labels)
        if self.mode == "chunk":
            return supervised_chunk_sum(hidden, self.head, labels, self.chunk_tokens)
        raise ValueError(self.mode)
