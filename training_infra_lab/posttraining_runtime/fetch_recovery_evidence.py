"""Collect terminal evidence and verify the frozen archive and all runtime patches before loading PT."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--publication-probe', action='store_true')
    a = p.parse_args()
    folder = a.folder
    cli = '/home/tiger/.merlin-cli/bin/merlin-cli'

    def call(group, command, body, name):
        with (folder / (name + '.stderr')).open('wb') as stderr:
            raw = subprocess.check_output([cli, '--control-plane', 'cn', 'job-v2', group, command,
                                           '--json', json.dumps(body)], timeout=90, stderr=stderr)
        (folder / name).write_bytes(raw)
        reply = json.loads(raw)
        assert reply.get('ResponseBase', {}).get('code', 0) == 0
        return reply

    run = call('runs', 'get', {'sid': a.run}, 'terminal-run.json')['data']
    assert run['status'] in ('done', 'success', 'failed', 'stopped', 'failed_to_launch')
    trial = run['runtime_context']['trial_id']
    exit_info = call('run-trials', 'exit-info', {
        'trials_list': {'filter': {'job_run_sid_any': [a.run]}},
        'pods_list': {'filter': {'trial_sid_any': [trial]}}}, 'exit-info.json')
    call('run-trials', 'get', {'sid': trial}, 'terminal-trial.json')
    timeline = call('run-timelines', 'list', {
        'filter': {'job_run_sid': a.run}, 'page_size': 100}, 'timeline.json')
    if run['status'] == 'done':
        pods = exit_info.get('pods', {}).get('data', [])
        assert pods and all(pod['status'] == 3 and pod['exit_status'] == 0 for pod in pods)
        assert any(event['event_type'] == 'job_success' for event in timeline['data'])
    (folder / 'terminal-evidence.json').write_text(json.dumps({
        'run_sid': a.run, 'status': run['status'], 'trial_id': trial,
        'pods': [{k: pod.get(k) for k in ('sid', 'pod_name', 'status', 'exit_status', 'start_at', 'stop_at')}
                 for pod in exit_info.get('pods', {}).get('data', [])],
        'timeline': [{k: event.get(k) for k in ('event_type', 'created_at', 'created_by')}
                     for event in timeline['data']],
    }, indent=2))
    target = folder / 'artifacts.tar.gz'
    if not target.exists():
        uri = ('hdfs://haruna/home/byte_search_lifeservice/zhangyang.exp/'
               'training_infra_lab/miles-grpo/' + a.tag + '/artifacts.tar.gz')
        subprocess.run(['hdfs', 'dfs', '-get', uri, str(target)], timeout=180, check=True,
                       env=dict(os.environ, HADOOP_CONF_SAFE_REFRESH_RUNNING='true',
                                PYTHONDONTWRITEBYTECODE='1'),
                       stderr=(folder / 'archive-fetch.stderr').open('wb'))
    dest = folder / 'artifacts'
    dest.mkdir(exist_ok=False)
    names, size = set(), 0
    with tarfile.open(target, 'r:gz') as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            assert not name.is_absolute() and '..' not in name.parts
            assert name.parts[0] in ('code', 'results') and member.name not in names
            names.add(member.name)
            assert len(names) < 20_000
            path = dest / member.name
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            assert member.isfile()
            size += member.size
            assert size < 2 * 1024**3
            path.parent.mkdir(parents=True, exist_ok=True)
            blob = archive.extractfile(member).read()
            assert len(blob) == member.size
            path.write_bytes(blob)
    manifest = json.loads((folder / 'source-manifest.json').read_text())
    assert json.loads((dest / 'results/source-manifest.json').read_text()) == manifest
    for name, sha in manifest.items():
        assert hashlib.sha256((dest / 'code' / name).read_bytes()).hexdigest() == sha, name
    patches = {}
    patch_modules = [('patch_broadcast_ipv6', 'runtime-patch.json', 'ORIGINAL_SHA256')]
    if a.publication_probe:
        patch_modules.append(('patch_publication_probe', 'publication-probe-patch.json', 'ORIGINAL_SHA'))
    else:
        patch_modules.append(('patch_recovery_probe', 'recovery-patch.json', 'ORIGINAL_SHA'))
    if 'patch_ray_launcher.py' in manifest:
        patch_modules.append(('patch_ray_launcher', 'ray-launcher-patch.json', 'ORIGINAL_SHA'))
    for module_name, report_name, sha_attr in patch_modules:
        spec = importlib.util.spec_from_file_location(module_name, dest / 'code' / (module_name + '.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        report = json.loads((dest / 'results' / report_name).read_text())
        # The local upstream mirror is inside this repository, not a standalone git checkout.
        original = (a.upstream / module.RELATIVE_PATH).read_bytes()
        assert hashlib.sha256(original).hexdigest() == getattr(module, sha_attr)
        patched = module.patched_source(original.decode()).encode()
        assert report['base_commit'] == module.MILES_SHA and report['path'] == module.RELATIVE_PATH
        assert report['before_sha256'] == hashlib.sha256(original).hexdigest()
        assert report['after_sha256'] == hashlib.sha256(patched).hexdigest()
        if module_name == 'patch_recovery_probe':
            original_driver = (a.upstream / 'train.py').read_bytes()
            assert hashlib.sha256(original_driver).hexdigest() == module.TRAIN_SHA
            assert report['driver_path'] == 'train.py'
            assert report['driver_before_sha256'] == module.TRAIN_SHA
            assert report['driver_after_sha256'] == hashlib.sha256(
                module.patched_driver(original_driver.decode()).encode()).hexdigest()
        patches[module_name] = report
    diagnostics = {}
    if 'ray_readiness.py' in manifest:
        for diagnostic_manifest in (dest / 'results/ray-diagnostics').glob('*/manifest.json'):
            report = json.loads(diagnostic_manifest.read_text())
            assert report['max_files'] == 64 and report['max_bytes_per_file'] == 256 * 1024
            assert len(report['files']) <= 64
            for name, meta in report['files'].items():
                assert PurePosixPath(name).name == name
                content = (diagnostic_manifest.parent / name).read_bytes()
                assert len(content) == meta['bytes'] <= 256 * 1024
                assert hashlib.sha256(content).hexdigest() == meta['sha256']
            diagnostics[diagnostic_manifest.parent.name] = report
    (folder / 'archive-audit.json').write_text(json.dumps({
        'safe_archive': True, 'files': len(names), 'uncompressed_bytes': size,
        'source_hashes_verified': len(manifest), 'patches_verified': patches,
        'archive_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'ray_diagnostics_verified': diagnostics,
    }, indent=2))
    print('ARCHIVE_AND_RUNTIME_PATCHES_VERIFIED', flush=True)
    verifier = Path(__file__).with_name('verify_publication_probe.py' if a.publication_probe else 'verify_recovery.py')
    subprocess.run(['python', str(verifier), str(dest), '--source-manifest',
                    str(folder / 'source-manifest.json'), '--output', str(folder / 'verified.json')],
                   check=True)


if __name__ == '__main__':
    main()
