"""Strict checkpoint-bound engine fingerprints for a cold-process recovery."""
import hashlib
import json
from pathlib import Path


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def normalize_engine_results(results):
    """Use full raw SHA256, including fused parameters; require two TP1 engines."""
    if len(results) != 2:
        raise ValueError('Expected two restored rollout engines')
    engines, cells = [], set()
    for meta, body in results:
        cell_id = meta.cell_id if hasattr(meta, 'cell_id') else meta['cell_id']
        if cell_id in cells:
            raise ValueError('Duplicate engine identity')
        cells.add(cell_id)
        if body.get('success') is not True or len(body.get('ranks', [])) != 1:
            raise ValueError('Missing successful TP1 raw checksums')
        rank = body['ranks'][0]
        roles = rank['parallelism_info']
        if not roles or {r['rank'] for r in roles} != {0}:
            raise ValueError('Unexpected engine parallelism')
        tensors = rank['checksums']
        if len(tensors) != 290 or any(
            len(v) != 64 or any(c not in '0123456789abcdef' for c in v)
            for v in tensors.values()
        ):
            raise ValueError('Expected 290 complete raw tensor SHA256 digests')
        engines.append({'cell_id': cell_id, 'tensor_checksums': tensors})
    if engines[0]['tensor_checksums'] != engines[1]['tensor_checksums']:
        raise ValueError('Restored engine replicas disagree')
    return sorted(engines, key=lambda x: x['cell_id'])


def compare_engine_commit(saved, current, inventory):
    if (saved.get('completed_rollout_id') != 1 or saved.get('policy_version') != 2
            or saved.get('checkpoint_inventory_sha256') != canonical_hash(inventory)
            or inventory.get('verified') is not True
            or inventory.get('completed_rollout_id') != 1):
        raise ValueError('Engine checksum commit does not match the checkpoint')
    for items in (saved['engines'], current):
        if len(items) != 2 or len({e['cell_id'] for e in items}) != 2:
            raise ValueError('Expected two distinct committed/restored engines')
        for engine in items:
            tensors = engine['tensor_checksums']
            if len(tensors) != 290 or any(
                len(v) != 64 or any(c not in '0123456789abcdef' for c in v)
                for v in tensors.values()
            ):
                raise ValueError('Incomplete committed/restored raw SHA256 evidence')
    previous = saved['engines']
    if len(previous) != 2 or len(current) != 2 or previous[0]['tensor_checksums'] != previous[1]['tensor_checksums']:
        raise ValueError('Saved engine replicas disagree')
    for replicas in (previous, current):
        if len({e['cell_id'] for e in replicas}) != 2 or any(
                len(e['tensor_checksums']) != 290 or any(
                    len(h) != 64 or any(c not in '0123456789abcdef' for c in h)
                    for h in e['tensor_checksums'].values()) for e in replicas):
            raise ValueError('Incomplete raw engine tensor fingerprints')
    expected = previous[0]['tensor_checksums']
    for engine in current:
        actual = engine['tensor_checksums']
        if actual != expected:
            mismatches = sorted(k for k in actual.keys() | expected.keys()
                                if actual.get(k) != expected.get(k))
            raise ValueError(f'Restored committed engine tensor mismatch: {mismatches[:20]}')


async def capture_committed_engines(args, controller):
    """Called after rollout 1 has been saved AND its version 2 published."""
    if args.start_rollout_id != 0 or not args.check_weight_update_equal:
        raise ValueError('Unexpected checkpoint engine capture boundary')
    root = Path(args.save_debug_rollout_data).parent.parent
    results = await controller.check_weights(action='raw_checksum',
        allow_quant_error=False, selector=args.check_weight_update_selector, skip_list=None)
    engines = normalize_engine_results(results)
    # The full file inventory is calculated by the next driver stage.
    report = {'completed_rollout_id': 1, 'policy_version': 2, 'engines': engines}
    path = root / 'checkpoint' / 'committed-engine-checksums.json'
    if path.exists():
        raise ValueError('Duplicate committed engine fingerprint')
    path.write_text(json.dumps(report, indent=2))


async def verify_initial_resume(args, controller):
    """Compare current engines to the trained commit, never to original HF."""
    if args.start_rollout_id != 2 or not args.check_weight_update_equal:
        raise ValueError('Unexpected recovery initial publication boundary')
    root = Path(args.save_debug_rollout_data).parent.parent
    inventory = json.loads((root / 'checkpoint-inventory.json').read_text())
    saved = json.loads((root / 'checkpoint/committed-engine-checksums.json').read_text())
    for rank in range(2):
        previous = json.loads((root / f'checkpoint/saved-1-rank-{rank}.json').read_text())
        loaded = json.loads((root / f'resume/loaded-1-rank-{rank}.json').read_text())
        if (previous['rollout_id'] != 1 or loaded['rollout_id'] != 2
                or previous['rank'] != rank or loaded['rank'] != rank
                or previous['policy_version'] != 2 or loaded['policy_version'] != 2
                or previous['world_size'] != 2 or loaded['world_size'] != 2):
            raise ValueError('Restored learner boundary mismatch')
        for field in ('state', 'scheduler', 'rng'):
            if not previous[field] or previous[field] != loaded[field]:
                raise ValueError(f'Restored learner rank {rank} {field} mismatch')
        options = loaded['load_options']
        if (options['finetune'] or options['no_load_optim'] or options['no_load_rng']
                or not options['use_checkpoint_opt_param_scheduler']):
            raise ValueError('Restored learner flags mismatch')
    results = await controller.check_weights(action='raw_checksum',
        allow_quant_error=False, selector=args.check_weight_update_selector, skip_list=None)
    engines = normalize_engine_results(results)
    compare_engine_commit(saved, engines, inventory)
    proof = {'verified': True, 'policy_version': 2, 'expected_resume_rollout_id': 2,
             'engines': engines, 'checkpoint_inventory_sha256': canonical_hash(inventory),
             'comparison': 'All 290 raw engine tensor SHA256 values against saved trained commit'}
    (root / 'resume/initial-publication-verified.json').write_text(json.dumps(proof, indent=2))
