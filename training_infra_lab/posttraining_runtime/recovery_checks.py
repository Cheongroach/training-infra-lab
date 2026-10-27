"""Per-rank pre-update and post-save fingerprints for the bounded recovery probe."""
import hashlib
import json
from pathlib import Path
import pickle
import random

_LIVE = None


def _rng_fingerprints():
    import numpy as np
    import torch
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

    def tensor_hash(value):
        return hashlib.sha256(value.cpu().contiguous().numpy().tobytes()).hexdigest()

    return {
        'python': hashlib.sha256(pickle.dumps(random.getstate(), protocol=4)).hexdigest(),
        'numpy': hashlib.sha256(pickle.dumps(np.random.get_state(), protocol=4)).hexdigest(),
        'torch_cpu': tensor_hash(torch.get_rng_state()),
        'torch_cuda': [tensor_hash(state) for state in torch.cuda.get_rng_state_all()],
        'cuda_tracker': {key: tensor_hash(state) for key, state in get_cuda_rng_tracker().get_states().items()},
    }


def _snapshot(args, rollout_id, model, optimizer, scheduler):
    import torch.distributed as dist
    from miles.backends.megatron_utils.local_weight_checksum import (
        _compute_weight_checksum_state, _transform_tensor_to_hash,
    )
    from miles.backends.training_utils.model_companion import ModelCompanionWeightVersionUtils
    state = _compute_weight_checksum_state(model=model, optimizer=optimizer).model_dump(mode='json')
    return {'rollout_id': rollout_id, 'rank': dist.get_rank(), 'world_size': dist.get_world_size(),
            'policy_version': ModelCompanionWeightVersionUtils.weight_version(model),
            'state': state, 'scheduler': _transform_tensor_to_hash(scheduler.state_dict()),
            'rng': _rng_fingerprints(),
            'load_options': {key: getattr(args, key) for key in
                             ('finetune', 'no_load_optim', 'no_load_rng', 'use_checkpoint_opt_param_scheduler')}}


def before_step(args, rollout_id, step_id, model, optimizer, scheduler):
    global _LIVE
    _LIVE = (args, model, optimizer, scheduler)
    report = _snapshot(args, rollout_id, model, optimizer, scheduler)
    root = Path(args.save_debug_rollout_data).parent
    path = root / f'pre-update-{rollout_id}-{step_id}-rank-{report["rank"]}.json'
    if path.exists():
        raise ValueError('Recovery checksum probe was invoked twice at the same boundary')
    path.write_text(json.dumps(report, indent=2, allow_nan=False))


def after_load(args, start_rollout_id, model, optimizer, scheduler):
    """Fingerprint immediately after load, before rollout or recomputation."""
    if start_rollout_id != 2:
        raise ValueError('Expected to restore the completed rollout-1 checkpoint')
    report = _snapshot(args, start_rollout_id, model, optimizer, scheduler)
    root = Path(args.save_debug_rollout_data).parent
    path = root / f'loaded-1-rank-{report["rank"]}.json'
    if path.exists():
        raise ValueError('Duplicate post-load fingerprint')
    path.write_text(json.dumps(report, indent=2, allow_nan=False))


def after_save(args, rollout_id, checkpoint_dir, hf_checkpoint_dir):
    if _LIVE is None or _LIVE[0].save_debug_rollout_data != args.save_debug_rollout_data:
        raise ValueError('Post-save checksum probe has no matching learner state')
    _, model, optimizer, scheduler = _LIVE
    report = _snapshot(args, rollout_id, model, optimizer, scheduler)
    report['checkpoint_directory'] = checkpoint_dir
    root = Path(args.save_debug_rollout_data).parent
    (root / f'saved-{rollout_id}-rank-{report["rank"]}.json').write_text(
        json.dumps(report, indent=2, allow_nan=False))
