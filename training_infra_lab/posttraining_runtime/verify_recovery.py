"""Audit both cold-process phases and final version-3 generation, not process exit alone."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import torch

from verify_smoke import audit as audit_smoke


def read_events(folder):
    return [json.loads(line) for path in sorted((folder / 'events').rglob('*.jsonl'))
            for line in path.read_text().splitlines()]


def event_identity(event):
    return (event['timestamp'], event['type'],
            json.dumps(event['source'], sort_keys=True, separators=(',', ':')))


def deduplicate(events):
    unique = {}
    for event in events:
        identity = event_identity(event)
        if identity in unique and unique[identity] != event:
            raise ValueError('Restored event identity has changed contents')
        unique[identity] = event
    return list(unique.values())


def compose_events(before, after):
    """Checkpoint replay is history; only a new transfer ID proves a restored publication."""
    before = deduplicate(before)
    all_events = deduplicate(before + after)
    old_ids = {event_identity(e) for e in before}
    fresh = [e for e in all_events if event_identity(e) not in old_ids]
    publications = [e for e in fresh if e['type'] == 'weight_update_result' and e['rollout_id'] is None]
    if len(publications) != 1:
        raise ValueError('Missing unique resumed initial publication')
    publication = publications[0]
    uid = publication['debug_weight_update_id']
    if not uid or any(e.get('debug_weight_update_id') == uid for e in before):
        raise ValueError('Resumed transfer reused a historical update ID')
    joined = [e for e in all_events if e.get('debug_weight_update_id') == uid
              and e['type'] in ('weight_update_result', 'inference_engine_weight_checksum')]
    if len(joined) != 2 or sum(e['type'] == 'inference_engine_weight_checksum' for e in joined) != 1:
        raise ValueError('Resumed publication/checksum ID coverage')
    engines = next(e for e in joined if e['type'] == 'inference_engine_weight_checksum')
    previous = [e for e in before if e['type'] == 'inference_engine_weight_checksum' and e['rollout_id'] == 1]
    if len(previous) != 1:
        raise ValueError('Missing last committed engine snapshot')
    prior = previous[0]
    snapshots = engines['engine_snapshots']
    old_snapshots = prior['engine_snapshots']
    parse_time = lambda e: datetime.fromisoformat(e['timestamp'].replace('Z', '+00:00'))
    if (publication['candidate_version'] != 2 or publication['published_version'] != 2
            or publication['failed_cell_ids'] or len(set(publication['updated_cell_ids'])) != 2
            or engines['rollout_id'] != 1 or engines['weight_version'] != 2 or prior['weight_version'] != 2
            or len(snapshots) != 2 or len(old_snapshots) != 2 or not old_snapshots[0]['tensor_checksums']
            or any(s['tensor_checksums'] != old_snapshots[0]['tensor_checksums'] for s in old_snapshots + snapshots)
            or {s['cell_id']: s['workers_hash'] for s in snapshots} != publication['snapshot_cell_id_to_hashes']
            or set(publication['updated_cell_ids']) != {s['cell_id'] for s in snapshots}
            or engines['debug_trainer_load_state_timestamp'] != publication['debug_trainer_load_state_timestamp']
            or publication['debug_trainer_load_state_timestamp'] <= prior['debug_trainer_load_state_timestamp']
            or parse_time(publication) <= max(map(parse_time, before))
            or parse_time(engines) < parse_time(publication)):
        raise ValueError('Resumed engines did not load the last committed model with fresh provenance')
    removed = {event_identity(publication), event_identity(engines)}
    composed = [e for e in all_events if event_identity(e) not in removed]
    return composed, {
        'debug_weight_update_id': uid, 'published_version': 2,
        'replayed_events': len(before) + len(after) - len(all_events),
        'fresh_publication_verified': True,
    }



def new_phase_events(before, after):
    """Resume imports prior audit history; reject alterations, drop exact inherited rows."""
    def key(event):
        return event_identity(event)
    previous = {key(e): e for e in before}
    if len(previous) != len(before):
        raise ValueError('Duplicate event identity in checkpoint phase')
    seen, result = set(), []
    for event in after:
        identity = key(event)
        if identity in seen:
            raise ValueError('Duplicate event identity in resume phase')
        seen.add(identity)
        if identity in previous:
            if event != previous[identity]:
                raise ValueError('Altered inherited checkpoint audit event')
        else:
            result.append(event)
    return result


def model_parameters(state):
    return {key: value for key, value in state['param_hashes'].items()
            if 'companion' not in key and 'witness' not in key}


def compare_restored(saved, restored):
    errors = []
    if (saved['rollout_id'] != 1 or restored['rollout_id'] != 2
            or saved['rank'] != restored['rank'] or saved['world_size'] != restored['world_size']
            or restored['world_size'] != 2 or saved['policy_version'] != 2
            or restored['policy_version'] != 2):
        errors.append('rank/rollout/policy version restoration')
    for field, left, right in (
        ('model parameters', model_parameters(saved['state']), model_parameters(restored['state'])),
        ('optimizer', saved['state']['optimizer_hashes'], restored['state']['optimizer_hashes']),
        ('scheduler', saved['scheduler'], restored['scheduler']),
        ('RNG', saved['rng'], restored['rng']),
    ):
        if not left or left != right:
            errors.append(field + ' restoration mismatch')
    options = restored['load_options']
    if (options['finetune'] or options['no_load_optim'] or options['no_load_rng']
            or not options['use_checkpoint_opt_param_scheduler']):
        errors.append('resume load flags')
    return errors


def audit(root, manifest):
    result = {'passed': False, 'performance_claim': False, 'errors': [], 'restored_ranks': []}
    errors = result['errors']
    for name, sha in manifest.items():
        path = root / 'code' / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            errors.append('source: ' + name)
    if errors:
        return result
    results = root / 'results'
    first, resumed = results / 'checkpoint', results / 'resume'
    for folder, expected in ((first, [0, 1]), (resumed, [2])):
        path = folder / 'driver-completed.json'
        if not path.exists():
            errors.append('phase driver did not complete: ' + folder.name)
            continue
        proof = json.loads(path.read_text())
        if proof.get('driver_returned') is not True or proof.get('expected_rollout_ids') != expected:
            errors.append('phase rollout range: ' + folder.name)
    for rank in range(2):
        saved_path = first / f'saved-1-rank-{rank}.json'
        restored_path = resumed / f'loaded-1-rank-{rank}.json'
        if not saved_path.exists() or not restored_path.exists():
            errors.append(f'missing restore fingerprints rank {rank}')
            continue
        problems = compare_restored(json.loads(saved_path.read_text()), json.loads(restored_path.read_text()))
        errors.extend(f'rank {rank}: {problem}' for problem in problems)
        result['restored_ranks'].append({'rank': rank, 'passed': not problems})
    restoration_path = resumed / 'qualification-restored.json'
    if not restoration_path.exists():
        errors.append('missing sampler restoration proof')
    else:
        proof = json.loads(restoration_path.read_text())
        if (not proof['restored'] or proof['completed_rollout_id'] != 1
                or proof['next_rollout_id'] != 2 or proof['seen_samples'] <= 0
                or proof['cursor']['sample_index'] != proof['seen_samples']):
            errors.append('sampler restoration boundary')
        result['sampler_restored_samples'] = proof['seen_samples']
    inventory_path = results / 'checkpoint-inventory.json'
    if not inventory_path.exists():
        errors.append('missing checkpoint inventory')
    else:
        inventory = json.loads(inventory_path.read_text())
        if (inventory.get('verified') is not True or inventory['completed_rollout_id'] != 1
                or inventory['expected_resume_rollout_id'] != 2
                or sum(value['bytes'] for value in inventory['files'].values()) != inventory['bytes']
                or not 8_000_000_000 < inventory['bytes'] < 96 * 1024**3
                or not any(name.startswith('iter_0000001/') for name in inventory['files'])
                or not any(name.startswith('rollout/1/') for name in inventory['files'])):
            errors.append('checkpoint inventory boundary')
        result['checkpoint_bytes'] = inventory['bytes']
    before = read_events(first)
    after = read_events(resumed)
    try:
        events, result['resumed_initial_publication'] = compose_events(before, after)
        result['inherited_events_verified'] = result['resumed_initial_publication']['replayed_events']
    except ValueError as exc:
        errors.append(str(exc))
    raw_proof = resumed / 'initial-publication-verified.json'
    saved_engines = first / 'committed-engine-checksums.json'
    if not raw_proof.exists() or not saved_engines.exists():
        errors.append('missing raw committed/restored engine proof')
    else:
        from resume_publication_guard import compare_engine_commit
        proof = json.loads(raw_proof.read_text())
        if proof.get('verified') is not True or proof.get('policy_version') != 2:
            errors.append('raw engine proof version')
        try:
            compare_engine_commit(json.loads(saved_engines.read_text()), proof['engines'],
                                  json.loads(inventory_path.read_text()))
            result['raw_restored_engine_tensors'] = 290
        except (ValueError, KeyError) as exc:
            errors.append('raw engine checkpoint guard: ' + str(exc))
    if errors:
        return result
    with tempfile.TemporaryDirectory(dir='/tmp', prefix='miles-recovery-audit-') as tmp:
        composed = Path(tmp)
        shutil.copytree(root / 'code', composed / 'code')
        output = composed / 'results'
        (output / 'events').mkdir(parents=True)
        shutil.copytree(resumed / 'qualification', output / 'qualification')
        for folder in (first, resumed):
            for pattern in ('train-*.pt', 'rollout-[0-9]*.pt', 'rollout-gate-*.json'):
                for path in folder.glob(pattern):
                    target = output / path.name
                    if target.exists():
                        raise ValueError('Duplicate committed dump in the two phases')
                    shutil.copyfile(path, target)
        # The resumed initial transfer is version 2, not the fresh baseline version 0.
        # It was checked separately above; all later failures and updates remain in the audit.
        (output / 'events/composed.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
        (output / 'driver-completed.json').write_text(json.dumps({'two_phases_verified': True}))
        semantic = audit_smoke(composed, manifest, require_final_publication=True)
        result['semantic_audit'] = semantic
        errors.extend(semantic['errors'])
    probe_path = resumed / 'rollout-eval_2.pt'
    if not probe_path.exists():
        errors.append('missing final-version generation probe')
    else:
        probe = torch.load(probe_path, map_location='cpu', weights_only=False)['samples']
        if len(probe) != 4:
            errors.append('final generation probe count')
        for sample in probe:
            cursor = len(sample['tokens']) - sample['response_length']
            spans = [span for call in sample['weight_versions'] for span in call['spans']]
            if not spans:
                errors.append('missing final generation behavior version')
            for span in spans:
                if (str(span['version']) != '3' or span['abs_start'] != cursor
                        or span['abs_end'] <= cursor):
                    errors.append('final generation version/coverage')
                cursor = span['abs_end']
            if cursor != len(sample['tokens']):
                errors.append('final generation incomplete coverage')
        result['publication_probe_samples'] = len(probe)
    result['passed'] = not errors
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--source-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    try:
        result = audit(a.root, json.loads(a.source_manifest.read_text()))
    except Exception as exc:
        result = {'passed': False, 'performance_claim': False, 'errors': [f'{type(exc).__name__}: {exc}']}
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
