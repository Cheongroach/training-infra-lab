"""Bounded diagnostic instrumentation; no synchronization or training changes."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time


def actor_runtime_env(env, actor_class, worker_index):
    result = {"env_vars": dict(env)}
    if (env.get("GRPO_NSYS_ENABLED") == "1"
            and actor_class.__name__ == "MegatronTrainRayActor" and worker_index == 0):
        result["_nsight"] = {
            "t": "cuda,nvtx,osrt", "sample": "none", "cpuctxsw": "none",
            "capture-range": "cudaProfilerApi", "capture-range-end": "stop",
            "cuda-graph-trace": "graph",
            "o": "/tmp/grpo-smoke/results/nsys/learner-rank0-%p",
        }
    return result


@contextmanager
def profile_train_range(rollout_id):
    import torch
    import torch.distributed as dist
    rank = dist.get_rank() if dist.is_initialized() else 0
    selected = os.environ.get("GRPO_NSYS_ENABLED") == "1" and rank == 0 and rollout_id == 1
    path = Path("/tmp/grpo-smoke/results/nsys")
    path.mkdir(parents=True, exist_ok=True)
    started = False
    begin, wall = time.monotonic_ns(), time.time_ns()
    try:
        if selected:
            torch.cuda.cudart().cudaProfilerStart()
            started = True
        torch.cuda.nvtx.range_push(f"grpo/actor_train/rollout={rollout_id}/rank={rank}")
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    finally:
        if started:
            torch.cuda.cudart().cudaProfilerStop()
        with (path / f"phases-rank{rank}.jsonl").open("a") as f:
            f.write(json.dumps({"stage": "actor_train", "rollout_id": rollout_id,
                "rank": rank, "pid": os.getpid(), "wall_start_ns": wall,
                "monotonic_start_ns": begin, "monotonic_end_ns": time.monotonic_ns(),
                "profiled": selected, "timing_kind": "host scope; includes async CUDA launches"}) + "\n")
