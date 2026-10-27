"""DCP recovery at optimizer boundaries, with explicit sampler and RNG state."""

import hashlib
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict


def save_checkpoint(path, model, optimizer, next_step, contract, accelerator):
    path = Path(path)
    rank = dist.get_rank()
    begin = time.monotonic()
    model_state, optim_state = get_state_dict(model, optimizer)
    dcp.save({"model": model_state, "optimizer": optim_state}, checkpoint_id=str(path))
    torch.save({
        "cpu_rng": torch.get_rng_state(),
        "device_rng": accelerator.get_rng_state(),
    }, path / f"rng-rank-{rank}.pt")
    dist.barrier()
    if rank == 0:
        (path / "progress.json").write_text(json.dumps({
            "next_step": next_step, "world_size": dist.get_world_size(),
            "contract": contract,
        }, indent=2))
    dist.barrier()
    return time.monotonic() - begin


def load_checkpoint(path, model, optimizer, contract, accelerator):
    path = Path(path)
    progress = json.loads((path / "progress.json").read_text())
    assert progress["world_size"] == dist.get_world_size(), "This RNG replay gate requires the same world size"
    assert progress["contract"] == contract, "Resume training/data configuration changed"
    model_state, optim_state = get_state_dict(model, optimizer)
    state = {"model": model_state, "optimizer": optim_state}
    dcp.load(state, checkpoint_id=str(path))
    set_state_dict(model, optimizer, model_state_dict=state["model"], optim_state_dict=state["optimizer"])
    rng = torch.load(path / f"rng-rank-{dist.get_rank()}.pt", map_location="cpu", weights_only=True)
    torch.set_rng_state(rng["cpu_rng"])
    accelerator.set_rng_state(rng["device_rng"])
    return progress["next_step"]


def state_fingerprints(model, optimizer):
    """Hash every local parameter and Adam state byte, outside the timed region."""
    parameters = hashlib.sha256()
    moments = hashlib.sha256()

    def update(digest, key, value):
        digest.update(key.encode())
        if isinstance(value, torch.Tensor):
            if hasattr(value, "to_local"):
                value = value.to_local()
            value = value.detach().cpu().contiguous()
            digest.update(str((value.shape, value.dtype)).encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            digest.update(str(value).encode())

    # FSDP may expose temporary unsharded views through model.named_parameters
    # (notably for tied weights on torch 2.7). The optimizer owns the persistent
    # shards. Inspect those, without inserting keys into its defaultdict state.
    owned = [p for group in optimizer.param_groups for p in group["params"]]
    owned_ids = {id(p) for p in owned}
    assert len(owned_ids) == len(owned), "Optimizer contains duplicate parameters"
    assert all(id(p) in owned_ids for p in optimizer.state), "Orphan optimizer state"
    print(json.dumps({"event": "optimizer_parameter_view_audit", "rank": dist.get_rank(),
                      "owned_parameters": len(owned),
                      "temporary_model_views": sum(id(p) not in owned_ids for p in model.parameters()),
                      "orphan_optimizer_states": sum(id(p) not in owned_ids for p in optimizer.state)}), flush=True)
    for index, param in enumerate(owned):
        name = f"parameter-{index}"
        update(parameters, name, param)
        for key, value in sorted(optimizer.state.get(param, {}).items()):
            update(moments, f"{name}/{key}", value)
    output = [None] * dist.get_world_size()
    dist.all_gather_object(output, {
        "rank": dist.get_rank(), "parameters": parameters.hexdigest(),
        "optimizer": moments.hexdigest(),
    })
    return output
