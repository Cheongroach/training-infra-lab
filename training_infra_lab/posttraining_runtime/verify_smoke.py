"""Audit local artifacts from our frozen GRPO smoke job, never infer success from exit alone."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import torch


def optimizer_moments(state):
    """Read Adam moment hashes, excluding param-group counters and hyperparameters."""
    return [
        {key: value for key, value in optimizer['state_dict'].get('state', {}).items()}
        for optimizer in state['optimizer_hashes']
    ]


def audit_qualification(results, rid, samples, seen, check):
    from qualification import MAX_CANDIDATES, assess_candidate
    directory = results / 'qualification'
    path = directory / f'step-{rid}.json'
    check(path.is_file(), f'missing qualification manifest {rid}')
    if not path.is_file():
        return {}
    manifest = json.loads(path.read_text())
    attempts = manifest['attempts']
    check(manifest.get('passed') is True and manifest['rollout_id'] == rid
          and manifest['weight_version'] == rid
          and manifest['max_candidates'] == MAX_CANDIDATES
          and 1 <= len(attempts) <= MAX_CANDIDATES, f'qualification contract {rid}')
    admitted = []
    for number, record in enumerate(attempts):
        name = f'candidate-{rid}-{number}.json'
        check(record['attempt'] == number and record['file'] == name,
              f'qualification sequence {rid}/{number}')
        candidate_path = directory / name
        check(candidate_path.is_file(), f'missing candidate {rid}/{number}')
        if not candidate_path.is_file():
            continue
        blob = candidate_path.read_bytes()
        check(hashlib.sha256(blob).hexdigest() == record['sha256'],
              f'candidate hash {rid}/{number}')
        candidate = json.loads(blob)
        report = assess_candidate(candidate['samples'], candidate['issued'], seen,
                                  rollout_id=rid, weight_version=rid)
        check(report == record['assessment'], f'candidate assessment {rid}/{number}')
        ids = [s['index'] for s in candidate['samples']]
        check(ids == record['sample_indices'], f'candidate identities {rid}/{number}')
        seen.update(ids)
        last = number == len(attempts) - 1
        check(record['decision'] == ('admitted' if last else 'rejected')
              and (report['passed'] if last else report['retryable']),
              f'candidate decision {rid}/{number}')
        if last:
            admitted = candidate['samples']
    check(admitted == samples, f'admitted candidate/learner rollout mismatch {rid}')
    expected = {f'candidate-{rid}-{i}.json' for i in range(len(attempts))}
    check({p.name for p in directory.glob(f'candidate-{rid}-*.json')} == expected,
          f'unaccounted candidates {rid}')
    return {'candidate_batches': len(attempts), 'rejected_batches': len(attempts) - 1}


def audit(root, source_manifest, *, require_final_publication=False):
    result = {'passed': False, 'performance_claim': False, 'updates': [], 'final_update_published': False}
    errors = []

    def check(value, message):
        if not value:
            errors.append(message)

    for name, digest in source_manifest.items():
        path = root / 'code' / name
        check(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest, f'source: {name}')
    if errors:
        result['errors'] = errors
        return result
    results = root / 'results'
    gated = 'rollout_gate.py' in source_manifest
    bounded = 'bounded_rollout.py' in source_manifest
    max_response = 4096 if gated else 2048
    result['rollout_gate_required'] = gated
    check((results / 'driver-completed.json').is_file(), 'driver did not complete')
    events = []
    for path in sorted((results / 'events').rglob('*.jsonl')):
        for line in path.read_text().splitlines():
            events.append(json.loads(line))
    result['event_types'] = dict(Counter(e['type'] for e in events))
    check(bool(events), 'no audit events')
    by_type = defaultdict(list)
    for e in events:
        by_type[e['type']].append(e)
    for name in ['weight_transfer_failed', 'inference_engine_checksum_collection_failed', 'explicitly_dropped_samples']:
        check(not by_type[name], f'failure event: {name}')
    publications = {(-1 if e['rollout_id'] is None else e['rollout_id']): e for e in by_type['weight_update_result']}
    engine_events = {e['rollout_id']: e for e in by_type['inference_engine_weight_checksum']}
    if require_final_publication:
        check(2 in publications, 'missing final weight publication')
        check(2 in engine_events, 'missing final engine checksum')
    previous_local = {}
    previous_engine = None
    qualification_seen = set()
    for rid in [-1, 0, 1, 2]:
        pub = publications.get(rid)
        if rid < 2:
            check(pub is not None, f'missing publication {rid}')
        if pub:
            check(not pub['failed_cell_ids'] and len(pub['updated_cell_ids']) == 2, f'incomplete publication {rid}')
            check(pub['published_version'] == rid + 1, f'wrong published version {rid}')
        e = engine_events.get(rid)
        if rid < 2:
            check(e is not None, f'missing engine checksum {rid}')
        if e:
            snapshots = e['engine_snapshots']
            check(len(snapshots) == 2, f'engine count {rid}')
            check(e['weight_version'] == rid + 1, f'engine version {rid}')
            if len(snapshots) == 2:
                current = snapshots[0]['tensor_checksums']
                check(current == snapshots[1]['tensor_checksums'], f'engine weights disagree {rid}')
                if previous_engine is not None:
                    changed = [k for k in current if k in previous_engine and current[k] != previous_engine[k] and 'companion' not in k and 'witness' not in k]
                    check(bool(changed), f'no model weight movement on engines {rid}')
                previous_engine = current
        if rid == -1:
            continue
        summary = {'rollout_id': rid}
        if gated:
            gate_file = results / f'rollout-gate-{rid}.json'
            check(gate_file.is_file(), f'missing rollout gate {rid}')
            if gate_file.is_file():
                gate = json.loads(gate_file.read_text())
                check(gate.get('passed') is True and gate.get('rollout_id') == rid
                      and gate.get('mixed_reward_groups', 0) > 0, f'rollout gate rejected {rid}')
        rollout_file = results / f'rollout-{rid}.pt'
        train_file = results / f'train-{rid}-rank-0.pt'
        check(rollout_file.is_file(), f'missing rollout {rid}')
        check(train_file.is_file(), f'missing train dump {rid}')
        if not rollout_file.is_file() or not train_file.is_file():
            result['updates'].append(summary)
            continue
        # These are artifacts generated by our own hash-verified submitted code.
        samples = torch.load(rollout_file, map_location='cpu', weights_only=False)['samples']
        td = torch.load(train_file, map_location='cpu', weights_only=False)['rollout_data']
        if bounded:
            summary.update(audit_qualification(results, rid, samples, qualification_seen, check))
        groups = defaultdict(list)
        ids = set()
        rewards = []
        for s in samples:
            ids.add(s['index'])
            groups[s['group_index']].append(s)
            n = s['response_length']
            check(0 < n <= max_response and len(s['tokens']) > n, f'token contract {rid}/{s["index"]}')
            check(s['status'] in ['completed', 'truncated'], f'sample status {rid}/{s["index"]}')
            lp = s.get('rollout_log_probs')
            check(lp is not None and len(lp) == n and all(math.isfinite(v) and v <= 1e-5 for v in lp), f'behavior probabilities {rid}/{s["index"]}')
            reward = s['reward'].get('score') if gated and isinstance(s['reward'], dict) else s['reward']
            check(isinstance(reward, (int, float)) and math.isfinite(reward), f'reward {rid}/{s["index"]}')
            if gated:
                check(reward in (-1.0, 1.0), f'DAPO reward {rid}/{s["index"]}')
                check(isinstance(s['prompt'], str) and s['prompt'].endswith('<think>\n\n</think>\n\n'),
                      f'non-thinking prompt prefix {rid}/{s["index"]}')
            rewards.append(reward)
            spans = [span for call in s['weight_versions'] for span in call['spans']]
            check(bool(spans) and all(int(span['version']) == rid for span in spans), f'behavior version {rid}/{s["index"]}')
            cursor = len(s['tokens']) - n
            for span in spans:
                check(span['abs_start'] == cursor and span['abs_end'] > cursor, f'version span gap {rid}/{s["index"]}')
                cursor = span['abs_end']
            check(cursor == len(s['tokens']), f'incomplete version coverage {rid}/{s["index"]}')
        check(len(samples) == len(ids) == 16 and len(groups) == 4 and all(len(g) == 4 for g in groups.values()), f'incomplete groups {rid}')
        check(set(td['sample_indices']) == ids and len(td['sample_indices']) == 16, f'trainer sample ownership {rid}')
        for name in ['tokens', 'response_lengths', 'loss_masks']:
            check(name in td and len(td[name]) == 16, f'train field {rid}/{name}')
        by_index = {s['index']: s for s in samples}
        for i, index in enumerate(td['sample_indices']):
            if index not in by_index:
                continue
            sample = by_index[index]
            check(torch.as_tensor(td['tokens'][i]).tolist() == sample['tokens'], f'trainer tokens {rid}/{index}')
            check(td['response_lengths'][i] == sample['response_length'], f'trainer length {rid}/{index}')
            check(torch.equal(torch.as_tensor(td['rollout_log_probs'][i], dtype=torch.float32),
                              torch.as_tensor(sample['rollout_log_probs'], dtype=torch.float32)),
                  f'trainer behavior probabilities {rid}/{index}')
        nonzero = 0
        for name in ['advantages', 'log_probs', 'rollout_log_probs']:
            check(name in td and len(td[name]) == 16, f'train field {rid}/{name}')
            for values, n, mask in zip(td.get(name, []), td['response_lengths'], td['loss_masks']):
                t = torch.as_tensor(values).float().flatten()
                m = torch.as_tensor(mask).bool().flatten()
                check(t.numel() == n and m.numel() == n and bool(torch.isfinite(t).all()), f'train tensor contract {rid}/{name}')
                if name == 'advantages' and t.numel() == m.numel():
                    nonzero += int((t[m] != 0).sum())
        summary['nonzero_advantage_tokens'] = nonzero
        check(nonzero > 0, f'no unmasked advantage {rid}')
        summary['reward_mean'] = sum(rewards) / len(samples)
        summary['mixed_reward_groups'] = sum(
            len({s['reward']['score'] if gated else s['reward'] for s in group}) > 1
            for group in groups.values()
        )
        if gated:
            from rollout_gate import assess_batch
            recomputed = assess_batch(samples, max_response=max_response)
            check(recomputed['passed'], f'recomputed rollout gate rejected {rid}')
        summary['truncated_samples'] = sum(s['status'] == 'truncated' for s in samples)
        grad = [e['metrics']['train/grad_norm'] for e in by_type['metric'] if e.get('rollout_id') == rid and 'train/grad_norm' in e['metrics']]
        summary['grad_norm'] = grad
        check(bool(grad) and all(math.isfinite(g) and g > 0 for g in grad), f'no positive finite gradient {rid}')
        ended = [e for e in by_type['train_group_step_end'] if e['rollout_id'] == rid and e['role'] == 'actor']
        # One TP2 learner cell reports one outcome for each of its two ranks.
        check(len(ended) == 1 and len(ended[0]['cell_outcomes']) == 1
              and all(v == ['normal', 'normal'] for v in ended[0]['cell_outcomes'].values()),
              f'optimizer outcome {rid}')
        local = [e for e in by_type['train_engine_local_weight_checksum'] if e['rollout_id'] == rid]
        ranks = {e['source']['rank_within_cell'] for e in local}
        check(ranks == {0, 1} and len(local) == 2, f'local weight coverage {rid}')
        for e in local:
            rank = e['source']['rank_within_cell']
            state = e['state']
            check(bool(state['param_hashes']) and bool(state['optimizer_hashes']), f'empty optimizer hashes {rid}/{rank}')
            check(any(optimizer_moments(state)), f'empty optimizer moments {rid}/{rank}')
            if rank in previous_local:
                old = previous_local[rank]
                check(old['optimizer_hashes'] != state['optimizer_hashes'], f'optimizer unchanged {rid}/{rank}')
                check(optimizer_moments(old) != optimizer_moments(state), f'optimizer moments unchanged {rid}/{rank}')
                check(any(k in old['param_hashes'] and h != old['param_hashes'][k] for k, h in state['param_hashes'].items() if 'companion' not in k and 'witness' not in k), f'weights unchanged {rid}/{rank}')
            previous_local[rank] = state
        result['updates'].append(summary)
    final_pub, final_engine = publications.get(2), engine_events.get(2)
    result['final_update_published'] = bool(
        final_pub and final_engine and final_pub['published_version'] == 3
        and not final_pub['failed_cell_ids'] and len(final_pub['updated_cell_ids']) == 2
        and final_engine['weight_version'] == 3
        and len(final_engine['engine_snapshots']) == 2
        and final_engine['engine_snapshots'][0]['tensor_checksums']
            == final_engine['engine_snapshots'][1]['tensor_checksums'])
    result['errors'] = errors
    result['passed'] = not errors
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--source-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    try:
        result = audit(args.root, json.loads(args.source_manifest.read_text()))
    except Exception as exc:
        result = {'passed': False, 'performance_claim': False, 'errors': [f'{type(exc).__name__}: {exc}']}
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
