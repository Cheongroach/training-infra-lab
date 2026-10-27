"""Bounded synchronous GRPO and paired actor replay on identical real trajectories."""

import argparse
import hashlib
import json
import os
import random
import shutil
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from .contracts import Trajectory, UpdateBatch, digest
from .distributed import fail_worker
from .executor import execute_update, synchronize
from .model import Actor, make_lm, kernel_evidence
from .planner import plan_update
from .probe import fsdp_api, emit
from .profiling import profile_call
from .rl import collate
from .rollout import sample_group
from .rewards import RULES, STRICT_RULE


def local(tensor):
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def tensors(actor, optimizer, gradients=False):
    for i, parameter in enumerate(actor.parameters()):
        yield f"{i}-parameter", local(parameter)
        if gradients:
            yield f"{i}-gradient", local(parameter.grad)
        for key, value in optimizer.state[parameter].items():
            yield f"{i}-{key}", local(value)


def scratch_save(path, actor, optimizer, gradients=False):
    """Tensor-at-a-time scratch copies, NOT durable checkpoints or a recovery claim."""
    path.mkdir(parents=True, exist_ok=False)
    for name, value in tensors(actor, optimizer, gradients):
        torch.save(value.detach().cpu(), path / (name + ".pt"))


@torch.no_grad()
def scratch_restore(path, actor, optimizer):
    optimizer.zero_grad(set_to_none=True)
    for name, value in tensors(actor, optimizer):
        saved = torch.load(path / (name + ".pt"), weights_only=True, map_location="cpu")
        value.copy_(saved.to(value.device))


def scratch_compare(path, actor, optimizer):
    totals = defaultdict(lambda: [0.0, 0.0, 0.0])
    # Compare every local shard of every tensor, not a sample/fingerprint alone.
    for name, value in tensors(actor, optimizer, gradients=True):
        field = name.split("-", 1)[1]
        expected = torch.load(path / (name + ".pt"), weights_only=True, map_location="cpu").reshape(-1)
        actual = value.detach().cpu().reshape(-1)
        for begin in range(0, actual.numel(), 2**20):
            a, b = actual[begin:begin + 2**20].double(), expected[begin:begin + 2**20].double()
            totals[field][0] += (a - b).square().sum().item()
            totals[field][1] += b.square().sum().item()
            totals[field][2] = max(totals[field][2], (a - b).abs().max().item())
    return {key: {"relative_l2": (v[0] / max(v[1], 1e-30)) ** 0.5, "max_abs": v[2]}
            for key, v in totals.items()}


@torch.no_grad()
def replay_probabilities(actor, batch, device, pad_id):
    rank, world = dist.get_rank(), dist.get_world_size()
    # Global batch always has group_size * world * groups_per_rank trajectories.
    indices = list(range(rank, len(batch.trajectories), world))
    differences = []
    for i in indices:
        data = collate(batch, [i], device, pad_id)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            log = actor(**{k: data[k] for k in ("input_ids", "attention_mask", "response_mask")})
        mask = data["response_mask"].bool()
        differences.extend((log - data["old_log_probs"])[mask].cpu().tolist())
    gathered = [None] * world
    dist.all_gather_object(gathered, differences)
    values = torch.tensor([x for row in gathered for x in row])
    return {
        "tokens": values.numel(), "rms": values.square().mean().sqrt().item(),
        "p99_abs": values.abs().quantile(0.99).item(), "max_abs": values.abs().max().item(),
        "outside_clip_fraction": ((values.exp() < 0.8) | (values.exp() > 1.2)).float().mean().item(),
    }


def measured_update(actor, optimizer, batch, device, *, strategy, micro, budget, pad_id):
    dist.barrier()
    synchronize(device)
    start = time.perf_counter()
    plan = plan_update(batch, dist.get_world_size(), strategy=strategy, micro_batch=micro,
                       max_padded_tokens=budget, activation_checkpoint=True)
    planned = time.perf_counter() - start
    torch.cuda.reset_peak_memory_stats(device)
    result = execute_update(actor, optimizer, batch, plan, device, pad_id=pad_id,
                            autocast_dtype=torch.bfloat16)
    total = torch.tensor(time.perf_counter() - start, device=device)
    dist.all_reduce(total, op=dist.ReduceOp.MAX)
    result.update(planning_seconds=planned, total_seconds=total.item(), micro_batch=micro,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return result, plan


def paired_replay(actor, optimizer, batch, args, device, pad_id):
    rank = dist.get_rank()
    scratch = args.scratch / f"rank-{rank}"
    scratch.mkdir(parents=True, exist_ok=False)
    base, expected = scratch / "base", scratch / "expected"
    scratch_save(base, actor, optimizer)
    records = []
    # Both baselines have sorted local KK partitions. No claim of full verl/packing comparison.
    choices = [(strategy, micro) for strategy in ("verl", "slot_sorted")
               for micro in args.replay_microbatches]
    output = args.output / f"replay-rank{rank}.json"
    try:
        # Warm all candidate shapes before any measured repeat, resetting every time.
        for strategy, micro in choices:
            scratch_restore(base, actor, optimizer)
            measured_update(actor, optimizer, batch, device, strategy=strategy, micro=micro,
                            budget=args.token_budget, pad_id=pad_id)
        for repeat in range(args.repeats):
            order = choices.copy()
            random.Random(1700 + repeat).shuffle(order)
            for strategy, micro in order:
                scratch_restore(base, actor, optimizer)
                result, plan = measured_update(
                    actor, optimizer, batch, device, strategy=strategy, micro=micro,
                    budget=args.token_budget, pad_id=pad_id,
                )
                if not expected.exists():
                    scratch_save(expected, actor, optimizer, gradients=True)
                    result["reference"] = True
                else:
                    errors = scratch_compare(expected, actor, optimizer)
                    result["errors"] = errors
                    # Predeclared BF16 tolerances. No automatic relaxation on failure.
                    valid = all(errors[key]["relative_l2"] <= 0.08
                                for key in ("gradient", "exp_avg", "exp_avg_sq"))
                    valid &= errors["parameter"]["max_abs"] <= 3e-6
                    votes = torch.tensor(int(valid), device=device)
                    dist.all_reduce(votes, op=dist.ReduceOp.MIN)
                    result["numerically_accepted"] = bool(votes.item())
                result.update(repeat=repeat, plan=asdict(plan),
                              workload="real_on_policy_actor_replay", beta=0.0)
                records.append(result)
                output.write_text(json.dumps(records, indent=2))
                emit("real_replay_step", **{k: v for k, v in result.items() if k != "plan"})
                if result.get("numerically_accepted") is False:
                    raise RuntimeError("Paired real replay failed numerical gate; see per-rank evidence")
        if args.profile:
            for strategy in ("verl", "slot_sorted"):
                scratch_restore(base, actor, optimizer)

                def update():
                    result, _ = measured_update(
                        actor, optimizer, batch, device, strategy=strategy, micro=args.replay_microbatches[-1],
                        budget=args.token_budget, pad_id=pad_id,
                    )
                    return result

                profile_call(
                    update, args.output / f"profile-{strategy}-rank{rank}", device,
                    {"batch_sha256": batch.sha256, "strategy": strategy,
                     "micro_batch": args.replay_microbatches[-1]},
                )
        # Experiments never advance the live policy; a single explicit update follows.
        scratch_restore(base, actor, optimizer)
        return records
    finally:
        # Only remove this invocation's scratch. Evidence is in args.output.
        shutil.rmtree(scratch)


@torch.no_grad()
def optimizer_nonzero_counts(actor, optimizer, device):
    """Direct moment evidence, outside the timed optimizer update."""
    values = torch.stack([
        torch.stack([torch.count_nonzero(local(optimizer.state[p][field]))
                     for p in actor.parameters()]).sum()
        for field in ("exp_avg", "exp_avg_sq")
    ]).to(device)
    dist.all_reduce(values)
    return dict(zip(("exp_avg", "exp_avg_sq"), values.cpu().tolist()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--updates", type=int, default=2)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--groups-per-rank", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--token-budget", type=int, default=2048)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--fp32-head", action="store_true")
    p.add_argument("--resident-rollout", action="store_true")
    p.add_argument("--benchmark-residency", action="store_true")
    p.add_argument("--replay-microbatches", type=int, nargs="+", default=[1, 2])
    p.add_argument("--skip-replay", action="store_true", help="Live updates only; no actor comparison")
    p.add_argument("--reward-rule", choices=RULES, default=STRICT_RULE)
    args = p.parse_args()
    if args.updates < 2 or args.repeats < 1 or args.groups_per_rank < 1:
        raise ValueError("Need a warm update, fresh trajectories, and a nonempty comparison")
    if args.skip_replay and args.profile:
        raise ValueError("Actor profiling requires the independent replay stage")
    if args.replay_microbatches not in ([1], [1, 2]):
        raise ValueError("Only the predeclared [1] or [1, 2] replay matrix is supported")
    if not 0 < args.adam_eps < 1:
        raise ValueError("Invalid Adam epsilon")
    torch.set_num_threads(1)
    rank, local_rank, world = (int(os.environ[k]) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=600))
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        from transformers import AutoTokenizer
        rows = [json.loads(line) for line in args.data.read_text().splitlines()]
        if len(rows) < args.updates * world * args.groups_per_rank:
            raise ValueError("Not enough fresh prompt groups")
        data_sha = hashlib.sha256(args.data.read_bytes()).hexdigest()
        torch.manual_seed(71)
        lm = make_lm(args.model, kernel="fla", dtype=torch.bfloat16)
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        actor = Actor(lm, fp32_head=args.fp32_head).to(device=device, dtype=torch.float32)
        del lm
        paths = kernel_evidence(actor)
        parameters = sum(p.numel() for p in actor.parameters())
        shard, policy = fsdp_api()
        mp = policy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for layer in actor.backbone.layers:
            shard(layer, mp_policy=mp, reshard_after_forward=True)
        shard(actor, mp_policy=mp, reshard_after_forward=True)
        optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-6, eps=args.adam_eps, foreach=False)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        torch.manual_seed(700 + rank)
        environment = {
            "model": str(args.model), "parameters": parameters, "world": world,
            "data_sha256": data_sha, "kernel_paths": paths, "beta": 0.0,
            "reduction": "sequence_mean", "group_size": args.group_size,
            "groups_per_rank": args.groups_per_rank, "max_new_tokens": args.max_new_tokens,
            "seed_per_rank": 700 + rank, "lr": 1e-6, "adam_eps": args.adam_eps, "activation_checkpoint": True,
            "defer_sync": False, "packing": False, "torch": torch.__version__,
            "fp32_head": args.fp32_head, "resident_rollout": args.resident_rollout,
            "replay_microbatches": args.replay_microbatches, "updates": args.updates,
            "reward_rule": args.reward_rule, "actor_replay": not args.skip_replay,
            "dataset": json.loads(args.data.with_name("data_manifest.json").read_text()).get("dataset", "gsm8k"),
        }
        (args.output / f"environment-rank{rank}.json").write_text(json.dumps(environment, indent=2))
        emit("grpo_loaded", rank=rank, **environment)
        # Two scratch states + gradients, all ranks share this filesystem.
        required = parameters * 32
        if not args.skip_replay and shutil.disk_usage(args.scratch.parent).free < required:
            raise RuntimeError(f"Paired replay scratch needs {required} free bytes")
        if args.benchmark_residency:
            from .residency import benchmark_rollout
            benchmark_rollout(actor, tokenizer, rows[rank], device, args.output / "residency-benchmark",
                              reward_rule=args.reward_rule)
        history = []
        for update in range(args.updates):
            optimizer.zero_grad(set_to_none=True)
            local_trajectories, rollouts = [], []
            for group in range(args.groups_per_rank):
                index = (update * args.groups_per_rank + group) * world + rank
                prompt_id = rows[index].get("source_id", f"{environment['dataset']}/train/{index}/{digest(rows[index])[:12]}")
                ts, info = sample_group(
                    actor, tokenizer, rows[index], prompt_id=prompt_id, policy_version=update,
                    group_size=args.group_size, device=device, max_new_tokens=args.max_new_tokens,
                    resident=args.resident_rollout, reward_rule=args.reward_rule,
                )
                local_trajectories.extend(asdict(t) for t in ts)
                rollouts.append(info)
            gathered = [None] * world
            dist.all_gather_object(gathered, local_trajectories)
            batch = UpdateBatch(tuple(Trajectory.from_dict(t) for part in gathered for t in part),
                                update, update, args.group_size)
            if rank == 0:
                (args.output / f"batch-{update}.json").write_text(json.dumps(asdict(batch)))
            (args.output / f"rollout-{update}-rank{rank}.json").write_text(json.dumps(rollouts, indent=2))
            probabilities = replay_probabilities(actor, batch, device, pad_id)
            advantages = batch.advantages()
            signal_groups = len({t.prompt_group_id for t, a in zip(batch.trajectories, advantages) if a != 0})
            info = {"update": update, "batch_sha256": batch.sha256, "probability_replay": probabilities,
                    "signal_groups": signal_groups, "groups": world * args.groups_per_rank,
                    "mean_reward": sum(t.reward for t in batch.trajectories) / len(batch.trajectories)}
            emit("grpo_batch", rank=rank, **info)
            (args.output / f"batch-diagnostics-{update}-rank{rank}.json").write_text(json.dumps(info, indent=2))
            if probabilities["rms"] > 0.05 or probabilities["p99_abs"] > 0.2 or probabilities["outside_clip_fraction"] > 0.01:
                raise RuntimeError("Behavior/replay probability gate failed")
            if update == 1 and not args.skip_replay:
                paired_replay(actor, optimizer, batch, args, device, pad_id)
            result, plan = measured_update(actor, optimizer, batch, device, strategy="verl", micro=1,
                                           budget=args.token_budget, pad_id=pad_id)
            moments = optimizer_nonzero_counts(actor, optimizer, device)
            info.update(update_result=result, plan=asdict(plan), optimizer_nonzero_counts=moments,
                        real_gradient_signal=signal_groups > 0,
                        note="Zero-advantage groups retained; no reward shaping or resampling")
            history.append(info)
            (args.output / f"history-rank{rank}.json").write_text(json.dumps(history, indent=2))
            emit("grpo_update", update=update, signal_groups=signal_groups,
                 optimizer_nonzero_counts=moments, **result)
        emit("grpo_complete", rank=rank, updates=len(history))
    except BaseException:
        fail_worker()
    else:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
