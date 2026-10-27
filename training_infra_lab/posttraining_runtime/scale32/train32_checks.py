"""Record actual learner topology at the same boundary as the full state hashes."""

import json
import os
from pathlib import Path
import socket

from recovery_checks import before_step as snapshot_before_step


def before_step(args, rollout_id, step_id, model, optimizer, scheduler):
    import torch
    import torch.distributed as dist
    from miles.backends.training_utils.parallel import get_parallel_state

    parallel = get_parallel_state()
    topology = {
        "rank": dist.get_rank(), "world_size": dist.get_world_size(),
        "tp_rank": parallel.tp.rank, "tp_size": parallel.tp.size,
        "dp_rank": parallel.effective_dp.rank, "dp_size": parallel.effective_dp.size,
        "pp_rank": parallel.pp.rank, "pp_size": parallel.pp.size,
        "cp_rank": parallel.cp.rank, "cp_size": parallel.cp.size,
        "hostname": socket.gethostname(), "arnold_id": int(os.environ["ARNOLD_ID"]),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device": torch.cuda.current_device(),
    }
    if (topology["world_size"], topology["tp_size"], topology["dp_size"],
            topology["pp_size"], topology["cp_size"]) != (16, 2, 8, 1, 1):
        raise ValueError("Unexpected 32-card learner topology")
    snapshot_before_step(args, rollout_id, step_id, model, optimizer, scheduler)
    root = Path(args.save_debug_rollout_data).parent
    path = root / f'pre-update-{rollout_id}-{step_id}-rank-{topology["rank"]}.json'
    report = json.loads(path.read_text())
    report["parallel"] = topology
    path.write_text(json.dumps(report, indent=2, allow_nan=False))
