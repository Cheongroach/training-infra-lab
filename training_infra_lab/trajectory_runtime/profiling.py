"""Separate diagnostic traces from the uninstrumented performance samples."""

import json
import time
from pathlib import Path

import torch
import torch.distributed as dist

from .executor import synchronize


def profile_call(callback, output: Path, device, identity, *, phase="actor_update"):
    """The caller owns restoration of model/Adam before and after this update.

    Operator device times may overlap and are not an additive wall-time split.
    All ranks must call this function with the same collective-bearing callback.
    """
    if device.type not in ("cpu", "cuda"):
        raise ValueError("This profiler adapter has only CPU/CUDA support")
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    output.mkdir(parents=True, exist_ok=False)
    if dist.is_initialized():
        dist.barrier()
    synchronize(device)
    start = time.perf_counter()
    with torch.profiler.profile(
        activities=activities, record_shapes=False, profile_memory=False, with_stack=False,
    ) as profiler:
        with torch.profiler.record_function("trajectory_runtime::" + phase):
            result = callback()
        synchronize(device)
    seconds = time.perf_counter() - start
    profiler.export_chrome_trace(str(output / "trace.json.gz"))
    operators = [{
        "name": event.key, "calls": event.count,
        "self_cpu_us": event.self_cpu_time_total,
        "total_cpu_us": event.cpu_time_total,
        "self_device_us": event.self_device_time_total,
        "total_device_us": event.device_time_total,
    } for event in profiler.key_averages()]
    operators.sort(key=lambda e: e["self_device_us"] if device.type == "cuda" else e["self_cpu_us"],
                   reverse=True)
    evidence = {
        "identity": identity, "rank": dist.get_rank() if dist.is_initialized() else 0,
        "device": str(device), "torch": torch.__version__, "instrumented_seconds": seconds,
        "update_result": result, "operators": operators,
        "scope": f"diagnostic {phase}; excluded from paired throughput samples",
        "time_semantics": "operator times overlap; do not sum them as a wall-time decomposition",
    }
    (output / "operators.json").write_text(json.dumps(evidence, indent=2))
    return evidence
