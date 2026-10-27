"""Bounded FSDP2 inference residency for this runtime's BF16/reshard=True policy."""

from contextlib import contextmanager

import torch
import torch.distributed as dist

from .rewards import STRICT_RULE


@contextmanager
def rollout_residency(actor, device, *, enabled=False, reserve_bytes=8 * 2**30):
    """Retain gathered BF16 weights only inside a no-grad generation phase.

    The caller owns the training policy: every FSDP group normally reshards
    after forward. Admission reserves one full BF16 model plus a fixed budget
    for cache/activations. This is not an OOM guarantee for arbitrary lengths.
    All ranks enter this context and take the same branch.
    """
    decision = {"requested": enabled, "admitted": False}
    if not enabled:
        yield decision
        return
    if device.type != "cuda" or torch.is_grad_enabled():
        raise ValueError("Residency requires CUDA and a no-grad scope")
    if reserve_bytes < 0:
        raise ValueError("Negative memory reserve")
    from torch.distributed.fsdp import FSDPModule

    modules = [m for m in actor.modules() if isinstance(m, FSDPModule)]
    if not modules or modules[0] is not actor:
        raise ValueError("Residency requires an FSDP2 root")
    # DTensor.numel() reports the global parameter size.
    full_weights = sum(p.numel() for p in actor.parameters()) * 2
    free, _ = torch.cuda.mem_get_info(device)
    reclaimable = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    available = free + reclaimable
    admitted = torch.tensor(int(available >= full_weights + reserve_bytes), device=device)
    if dist.is_initialized():
        dist.all_reduce(admitted, op=dist.ReduceOp.MIN)
    decision.update(admitted=bool(admitted.item()), full_bf16_weight_bytes=full_weights,
                    available_bytes=available, reserve_bytes=reserve_bytes)
    if not decision["admitted"]:
        yield decision
        return
    actor.set_reshard_after_forward(False)
    try:
        yield decision
    finally:
        # FSDPModule.reshard() is explicitly nonrecursive in PyTorch 2.9.
        for module in reversed(modules):
            module.reshard()
        actor.set_reshard_after_forward(True)


@torch.no_grad()
def benchmark_rollout(actor, tokenizer, row, device, output, *, repeats=3, tokens=128,
                      reward_rule=STRICT_RULE):
    """Paired generation only; restore RNG and never advance the optimizer.

    Each repeat uses one seed for both policies. Timings include phase entry,
    sampling, and resharding; profiler calls are separate from these samples.
    """
    import json
    import random
    import time
    from dataclasses import asdict
    from .rollout import sample_group
    from .executor import synchronize

    rank = dist.get_rank()
    original_rng = torch.cuda.get_rng_state(device)
    records = []
    output.mkdir(parents=True, exist_ok=True)
    try:
        for repeat in range(-1, repeats):
            order = [False, True]
            random.Random(431 + repeat).shuffle(order)
            expected = None
            for enabled in order:
                torch.cuda.manual_seed(5000 + 17 * rank + repeat)
                dist.barrier()
                synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                start = time.perf_counter()
                with rollout_residency(actor, device, enabled=enabled) as decision:
                    trajectories, details = sample_group(
                        actor, tokenizer, row, prompt_id="residency-benchmark", policy_version=0,
                        group_size=4, device=device, max_new_tokens=tokens, reward_rule=reward_rule)
                synchronize(device)
                elapsed = torch.tensor(time.perf_counter() - start, device=device)
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                values = [asdict(t) for t in trajectories]
                if expected is None:
                    expected = values
                    same = True
                else:
                    same = expected == values
                vote = torch.tensor(int(same), device=device)
                dist.all_reduce(vote, op=dist.ReduceOp.MIN)
                record = {"rank": rank, "repeat": repeat, "warmup": repeat < 0,
                          "residency": decision, "seconds": elapsed.item(),
                          "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                          "exact_trajectories_and_log_probs": bool(vote.item()),
                          "details": details, "max_new_tokens": tokens,
                          "scope": "paired generation including entry/exit; one prompt per rank"}
                records.append(record)
                (output / f"rank{rank}.json").write_text(json.dumps(records, indent=2))
                if not vote.item():
                    raise RuntimeError("Residency changed sampled tokens/rewards/log-probabilities")
                if enabled and not decision["admitted"]:
                    raise RuntimeError("Residency benchmark candidate was rejected by memory admission")
        from .profiling import profile_call
        for enabled in (False, True):
            torch.cuda.manual_seed(888 + rank)

            def generate():
                _, details = sample_group(
                    actor, tokenizer, row, prompt_id="profile", policy_version=0,
                    group_size=4, device=device, max_new_tokens=16, resident=enabled, reward_rule=reward_rule)
                return details

            profile_call(generate, output / f"profile-resident{enabled}-rank{rank}", device,
                         {"resident": enabled, "max_new_tokens": 16}, phase="generation")
        return records
    finally:
        torch.cuda.set_rng_state(original_rng, device)
