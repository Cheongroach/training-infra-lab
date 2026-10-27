"""Hash the job-local trained checkpoint before the cold-process resume."""
import argparse
import hashlib
import json
from pathlib import Path


def verify_inventory(checkpoint, report):
    current = {}
    for path in sorted(checkpoint.rglob('*')):
        if path.is_symlink():
            raise ValueError('Checkpoint symlink appeared before restore')
        if path.is_file():
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            current[str(path.relative_to(checkpoint))] = {
                'bytes': path.stat().st_size, 'sha256': digest}
    if report.get('verified') is not True or current != report['files']:
        raise ValueError('Checkpoint changed between commit and restore')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    checkpoint = a.root / 'training-checkpoint'
    if (checkpoint / 'latest_checkpointed_iteration.txt').read_text().strip() != '1':
        raise ValueError('Expected trained rollout 1, not an initialization/release checkpoint')
    if not (checkpoint / 'iter_0000001').is_dir() or not (checkpoint / 'rollout/1').is_dir():
        raise ValueError('Both actor and rollout checkpoint directories are required')
    files = {}
    for path in sorted(checkpoint.rglob('*')):
        if path.is_symlink():
            raise ValueError('Checkpoint symlinks are not supported by the probe')
        if path.is_file():
            with path.open('rb') as f:
                sha = hashlib.file_digest(f, 'sha256').hexdigest()
            files[str(path.relative_to(checkpoint))] = {'bytes': path.stat().st_size, 'sha256': sha}
    size = sum(v['bytes'] for v in files.values())
    if not 0 < size < 96 * 1024**3:
        raise ValueError('Unexpected checkpoint size for the bounded Qwen3-4B probe')
    report = {'verified': True, 'completed_rollout_id': 1, 'expected_resume_rollout_id': 2,
              'checkpoint_scope': 'job-local filesystem, cold process restart',
              'raw_checkpoint_in_result_archive': False, 'bytes': size, 'files': files}
    (a.root / 'results/checkpoint-inventory.json').write_text(json.dumps(report, indent=2))
    from resume_publication_guard import canonical_hash
    engine_path = a.root / 'results/checkpoint/committed-engine-checksums.json'
    engine = json.loads(engine_path.read_text())
    if engine['completed_rollout_id'] != 1 or engine['policy_version'] != 2:
        raise ValueError('Engine fingerprint is not the last committed model')
    engine['checkpoint_inventory_sha256'] = canonical_hash(report)
    engine_path.write_text(json.dumps(engine, indent=2))
    print(json.dumps({'event': 'checkpoint_inventory', 'files': len(files), 'bytes': size,
                      'completed_rollout_id': 1}), flush=True)


if __name__ == '__main__':
    main()
