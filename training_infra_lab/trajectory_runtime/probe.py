"""Bounded real-weight backend probe; synthetic tokens are explicitly labeled."""

import argparse
import importlib.metadata
import json
import os
import platform
import time
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from .contracts import Trajectory, UpdateBatch
from .distributed import fail_worker
from .executor import execute_update
from .model import Actor, kernel_evidence, make_lm
from .planner import plan_update


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}), flush=True)


def fsdp_api():
    try:
        from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    except ImportError:
        from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
    return fully_shard, MixedPrecisionPolicy


def synthetic_batch(world, length, vocab, step):
    generator = torch.Generator().manual_seed(100 + step)
    ts = []
    for i in range(world * 2):
        n = length if i % 3 == 0 else max(8, length // (1 + i % 3))
        ids = tuple(torch.randint(2, vocab, (n,), generator=generator).tolist())
        mask = (0,) * (n // 2 - 1) + (1,) * (n - n // 2)
        ts.append(Trajectory(str(i), str(i // 2), step, ids, mask,
                             (-10.0,) * (n - 1), float(i % 2)))
    return UpdateBatch(tuple(ts), step, step, 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cuda", "npu"), default="cuda")
    p.add_argument("--kernel", choices=("fla", "torch"), default="fla")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--lengths", default="128,512,1024")
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--activation-checkpoint", action="store_true")
    p.add_argument("--defer-sync", action="store_true")
    args = p.parse_args()
    if args.device == "npu":
        import torch_npu  # noqa: F401
    rank, local, world = [int(os.environ.get(k, d)) for k, d in (
        ("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1"),
    )]
    accelerator = getattr(torch, args.device)
    accelerator.set_device(local)
    device = torch.device(args.device, local)
    torch.set_num_threads(1)
    torch.manual_seed(17)
    dist.init_process_group("nccl" if args.device == "cuda" else "hccl",
                            timeout=timedelta(seconds=600))
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        begin = time.monotonic()
        lm = make_lm(args.model, tiny=args.tiny, kernel=args.kernel, dtype=torch.bfloat16)
        vocab = lm.config.vocab_size
        if args.activation_checkpoint:
            lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        actor = Actor(lm)
        del lm
        parameters = sum(p.numel() for p in actor.parameters())
        paths = kernel_evidence(actor)
        actor.to(device=device, dtype=torch.float32)
        shard, policy = fsdp_api()
        precision = policy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for layer in actor.backbone.layers:
            shard(layer, mp_policy=precision, reshard_after_forward=True)
        shard(actor, mp_policy=precision, reshard_after_forward=True)
        optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-6, foreach=False)
        environment = {
            "rank": rank, "world": world, "host": platform.node(), "device": accelerator.get_device_name(local),
            "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
            "fla_core": importlib.metadata.version("fla-core") if args.kernel == "fla" else None,
            "parameters": parameters, "tiny": args.tiny, "kernel_paths": paths,
            "load_seconds": time.monotonic() - begin,
            "workload": "synthetic_kernel_probe_not_real_RL",
            "activation_checkpoint": args.activation_checkpoint, "defer_sync": args.defer_sync,
        }
        emit("backend_loaded", **environment)
        (args.output / f"environment-rank{rank}.json").write_text(json.dumps(environment, indent=2))
        measurements = []
        for length in map(int, args.lengths.split(",")):
            for step in range(args.steps):
                batch = synthetic_batch(world, length, vocab, step)
                plan = plan_update(batch, world, strategy="verl", max_padded_tokens=length,
                                   defer_sync=args.defer_sync, activation_checkpoint=args.activation_checkpoint)
                accelerator.reset_peak_memory_stats()
                result = execute_update(actor, optimizer, batch, plan, device, autocast_dtype=torch.bfloat16)
                result.update(length=length, repeat=step, measured=step > 0,
                              peak_allocated_bytes=accelerator.max_memory_allocated(),
                              peak_reserved_bytes=accelerator.max_memory_reserved())
                measurements.append(result)
                (args.output / f"measurements-rank{rank}.json").write_text(json.dumps(measurements, indent=2))
                emit("backend_step", **result)
            if rank == 0:
                (args.output / f"plan-{length}.json").write_text(json.dumps(asdict(plan), indent=2))
        emit("backend_probe_complete", rank=rank)
    except BaseException:
        fail_worker()
    else:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
