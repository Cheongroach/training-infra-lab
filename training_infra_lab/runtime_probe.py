"""Bounded accelerator/collective checks before scheduling training experiments."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import time
from datetime import timedelta


def emit(event: str, **values) -> None:
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def main() -> None:
    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    backend = os.environ.get("LAB_DEVICE", "cuda")
    if backend == "npu":
        import torch_npu  # noqa: F401

    accelerator = getattr(torch, backend)
    accelerator.set_device(local_rank)
    device = torch.device(backend, local_rank)
    versions = {}
    for package in ("torch", "torch-npu", "transformers", "accelerate", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        from torch.distributed.fsdp import fully_shard  # noqa: F401

        has_fsdp2 = True
    except ImportError:
        has_fsdp2 = False
    emit(
        "environment",
        rank=rank,
        host=platform.node(),
        architecture=platform.machine(),
        python=platform.python_version(),
        versions=versions,
        device=backend,
        device_name=accelerator.get_device_name(local_rank),
        device_count=accelerator.device_count(),
        fsdp2_import=has_fsdp2,
        distributed_environment={
            key: os.environ[key]
            for key in (
                "RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                "ARNOLD_ID", "ARNOLD_WORKER_NUM", "ARNOLD_WORKER_GPU",
                "METIS_TASK_INDEX", "METIS_WORKER_0_HOST", "METIS_WORKER_0_PORT",
            )
            if key in os.environ
        },
    )
    initialized = False
    try:
        if world > 1:
            dist.init_process_group(
                "hccl" if backend == "npu" else "nccl",
                timeout=timedelta(seconds=120),
            )
            initialized = True
            value = torch.tensor([rank + 1.0], device=device)
            dist.all_reduce(value)
            expected = world * (world + 1) / 2
            assert value.item() == expected, (value.item(), expected)
            source = torch.full((128,), rank + 1.0, device=device)
            gathered = torch.empty(world * 128, device=device)
            dist.all_gather_into_tensor(gathered, source)
            reduced = torch.empty_like(source)
            dist.reduce_scatter_tensor(reduced, gathered)
            assert torch.all(reduced == world * (rank + 1)).item()
            emit("collectives_passed", rank=rank, world_size=world)

        torch.manual_seed(42)
        x = torch.randn(64, 128, device=device, requires_grad=True)
        weight = torch.randn(128, 128, device=device, requires_grad=True)
        with torch.autocast(device_type=backend, dtype=torch.bfloat16):
            loss = (x @ weight).float().square().mean()
        loss.backward()
        assert torch.isfinite(loss).item()
        assert torch.isfinite(x.grad).all().item()
        assert torch.isfinite(weight.grad).all().item()
        accelerator.synchronize()
        emit("bf16_backward_passed", rank=rank, loss=loss.item())

        if has_fsdp2 and world > 1:
            from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

            model = torch.nn.Sequential(
                torch.nn.Linear(128, 256),
                torch.nn.GELU(),
                torch.nn.Linear(256, 128),
            ).to(device)
            fully_shard(
                model,
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                ),
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            start = time.monotonic()
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=backend, dtype=torch.bfloat16):
                    fsdp_loss = model(x.detach()).float().square().mean()
                fsdp_loss.backward()
                optimizer.step()
            accelerator.synchronize()
            assert torch.isfinite(fsdp_loss).item()
            emit(
                "fsdp2_optimizer_passed", rank=rank,
                loss=fsdp_loss.item(), elapsed_seconds=time.monotonic() - start,
            )
        emit("probe_passed", rank=rank)
    finally:
        if initialized:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
