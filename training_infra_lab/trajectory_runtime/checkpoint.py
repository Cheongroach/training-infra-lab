"""Synchronous FSDP/DCP adapter for LocalCommitStore; fixed-world recovery."""
import random
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

from .contracts import digest


def rank_world():
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def root_call(callback):
    rank, _ = rank_world()
    result = [None]
    if rank == 0:
        try:
            result[0] = {"value": callback()}
        except Exception as error:
            result[0] = {"error": str(error)}
    if dist.is_initialized():
        dist.broadcast_object_list(result, src=0)
    if "error" in result[0]:
        raise RuntimeError(result[0]["error"])
    return result[0]["value"]


def save(store, epoch, model, optimizer, metadata, device, scheduler=None):
    rank, world = rank_world()
    if metadata["world_size"] != world:
        raise ValueError("Checkpoint world size mismatch")
    path = Path(root_call(lambda: str(store.stage(epoch, metadata["next_update"]))))
    model_state, optim_state = get_state_dict(model, optimizer)
    dcp.save({"model": model_state, "optimizer": optim_state}, checkpoint_id=path / "tensors")
    rng = {"cpu": torch.get_rng_state(), "python": random.getstate()}
    if device.type != "cpu":
        rng["accelerator"] = getattr(torch, device.type).get_rng_state()
    torch.save({"rng": rng, "scheduler": scheduler.state_dict() if scheduler else None},
               path / f"rank-{rank}.pt")
    if dist.is_initialized():
        dist.barrier()
    root_call(lambda: store.commit(path, epoch, metadata))
    return path


def restore(store, model, optimizer, contract, device, scheduler=None):
    rank, world = rank_world()
    snapshot = root_call(store.read)
    metadata, path = snapshot["metadata"], Path(snapshot["path"])
    if metadata["world_size"] != world or digest(metadata["contract"]) != digest(contract):
        raise ValueError("Restore world size or algorithm/data/model contract differs")
    # All ranks access the same verified immutable files. Mutating committed
    # snapshot files is outside this store's ownership contract.
    local = torch.load(path / f"rank-{rank}.pt", map_location="cpu", weights_only=True)
    if (scheduler is None) != (local["scheduler"] is None):
        raise ValueError("Scheduler presence differs")
    model_state, optim_state = get_state_dict(model, optimizer)
    state = {"model": model_state, "optimizer": optim_state}
    dcp.load(state, checkpoint_id=path / "tensors")
    set_state_dict(model, optimizer, model_state_dict=state["model"], optim_state_dict=state["optimizer"])
    if scheduler:
        scheduler.load_state_dict(local["scheduler"])
    torch.set_rng_state(local["rng"]["cpu"])
    random.setstate(local["rng"]["python"])
    if device.type != "cpu":
        getattr(torch, device.type).set_rng_state(local["rng"]["accelerator"])
    return metadata
