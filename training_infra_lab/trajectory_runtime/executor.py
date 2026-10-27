"""Collective-safe actor update. An in-step failure must fail the WHOLE job."""

from __future__ import annotations

import time

import torch
import torch.distributed as dist

from .contracts import digest
from .rl import collate, surrogate_sum


def synchronize(device):
    if device.type != "cpu":
        getattr(torch, device.type).synchronize()


def agree(batch, plan, *, beta, clip, world):
    """Every rank reports local validation before any FSDP forward collective."""
    error = None
    try:
        plan.validate(batch)
        if plan.world_size != world:
            raise ValueError("Plan world size differs from process group")
        if beta and any(not t.reference_log_probs for t in batch.trajectories):
            raise ValueError("KL requested without reference probabilities")
        if not 0 < clip < 1 or beta < 0:
            raise ValueError("Invalid objective configuration")
        key = digest([plan.sha256, beta, clip])
    except Exception as exc:
        error, key = str(exc), None
    records = [{"error": error, "digest": key}]
    if dist.is_initialized():
        records = [None] * world
        dist.all_gather_object(records, {"error": error, "digest": key})
    if any(r["error"] for r in records) or len({r["digest"] for r in records}) != 1:
        raise ValueError(f"Collective plan agreement failed: {records}")


def execute_update(model, optimizer, batch, plan, device, *, beta=0.0, clip=0.2,
                   pad_id=0, max_grad_norm=1.0, autocast_dtype=None):
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    agree(batch, plan, beta=beta, clip=clip, world=world)
    if world > 1 and not hasattr(model, "set_requires_gradient_sync"):
        raise ValueError("Multi-rank executor requires an FSDP2 root")
    if plan.defer_sync and not hasattr(model, "set_requires_gradient_sync"):
        raise ValueError("Deferred sync requires FSDP2")
    inputs = []
    for slot in plan.slots:
        indices = slot[rank]
        if indices:
            data = collate(batch, indices, device, pad_id)
        else:
            # Same model traversal with a valid two-token row. Never an all-mask mean.
            data = {
                "input_ids": torch.tensor([batch.trajectories[0].input_ids[:2]], device=device),
                "attention_mask": torch.ones((1, 2), dtype=torch.long, device=device),
                "response_mask": torch.ones((1, 1), device=device),
                "old_log_probs": torch.zeros((1, 1), device=device),
                "reference_log_probs": torch.zeros((1, 1), device=device),
                "advantages": torch.zeros(1, device=device),
            }
        inputs.append((data, bool(indices)))
    if dist.is_initialized():
        dist.barrier()
    synchronize(device)
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    local_numerator = torch.zeros((), device=device)
    slot_times = []
    for s, (data, active) in enumerate(inputs):
        if hasattr(model, "set_requires_gradient_sync"):
            model.set_requires_gradient_sync(not plan.defer_sync or s == len(inputs) - 1)
        before = time.perf_counter()
        with torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            log_probs = model(**{key: data[key] for key in (
                "input_ids", "attention_mask", "response_mask",
            )})
            numerator = surrogate_sum(log_probs, data, batch.reduction, clip=clip, beta=beta)
            numerator = numerator * int(active)
            # FSDP averages across ranks; preserve the GLOBAL sequence/token mean.
            loss = numerator * world / batch.denominator
        loss.backward()
        local_numerator += numerator.detach()
        # Diagnostic timing includes synchronization; performance comparisons
        # must use the same instrumentation on every policy.
        synchronize(device)
        slot_times.append(time.perf_counter() - before)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_grad_norm, error_if_nonfinite=True, foreach=False,
    )
    optimizer.step()
    synchronize(device)
    elapsed = time.perf_counter() - start
    totals = torch.tensor([elapsed, float(local_numerator)], device=device)
    if dist.is_initialized():
        dist.all_reduce(totals[:1], op=dist.ReduceOp.MAX)
        dist.all_reduce(totals[1:])
    return {
        "update_id": batch.update_id, "policy_version": batch.policy_version,
        "batch_sha256": batch.sha256, "plan_sha256": plan.sha256,
        "strategy": plan.strategy, "seconds": totals[0].item(),
        "loss": totals[1].item() / batch.denominator,
        "grad_norm": grad_norm.full_tensor().item() if hasattr(grad_norm, "full_tensor") else grad_norm.item(),
        "rank": rank, "slot_seconds": slot_times, "slots": len(inputs),
        "dummy_slots": sum(not active for _, active in inputs),
        "response_tokens": sum(sum(t.response_mask) for t in batch.trajectories),
        "input_tokens": sum(len(t.input_ids) for t in batch.trajectories),
        "timing_scope": "resident-batch forward/backward/clip/Adam plus slot synchronization",
    }
