"""Package a bounded four-H20 probe; writes a request but never submits it."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-vid', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    if not args.tag.replace('-', '').isalnum():
        raise ValueError('tag must contain letters, numbers and hyphens only')
    if (args.output / 'create.json').exists():
        raise ValueError('Use a new output directory to preserve frozen submissions')
    root = Path(__file__).resolve().parent
    sources = {n: (root / n).read_bytes() for n in
               ('environment_probe.py', 'run_environment_probe.sh')}
    hashes = {n: hashlib.sha256(b).hexdigest() for n, b in sources.items()}
    blobs = {n: base64.b64encode(b).decode() for n, b in sources.items()}
    archive_dir = f'/mnt/hdfs/lab_owner/training_infra_lab/miles-preflight/{args.tag}'
    mount_marker = {'tag': args.tag, 'sources': hashes}
    inner = f'''set -euo pipefail
export PROBE_ROOT=/tmp/miles-preflight
mkdir -p "$PROBE_ROOT/code" "$PROBE_ROOT/results"
exec > >(tee "$PROBE_ROOT/results/entry.log") 2>&1
echo PROBE_ENTRY_REACHED
finish() {{
 rc=$?
 trap - EXIT
 set +e
 timeout --signal=TERM --kill-after=5s 45s /bin/bash -c {shlex.quote('mkdir -p ' + shlex.quote(archive_dir) + ' && tar -czf ' + shlex.quote(archive_dir + '/artifacts.tar.gz') + ' -C /tmp/miles-preflight code results')}
 copy_rc=$?
 echo "ORIGINAL_EXIT_CODE=$rc ARTIFACT_COPY_RC=$copy_rc"
 if [[ "$rc" == 0 && "$copy_rc" != 0 ]]; then rc=$copy_rc; fi
 exit "$rc"
}}
trap finish EXIT
trap 'exit 124' TERM
python3 - <<'UNPACK'
import base64,hashlib,json
from pathlib import Path
root=Path('/tmp/miles-preflight/code')
blobs={blobs!r}
hashes={hashes!r}
for name,blob in blobs.items():
 content=base64.b64decode(blob)
 assert hashlib.sha256(content).hexdigest()==hashes[name]
 (root/name).write_bytes(content)
Path('/tmp/miles-preflight/results/source-manifest.json').write_text(json.dumps(hashes,indent=2))
archive=Path({archive_dir!r})
if json.loads((archive/'mount-marker.json').read_text()) != {mount_marker!r}:
 raise RuntimeError('HDFS input marker does not match this frozen submission')
(archive/'bootstrap.json').write_text(json.dumps({{'entry_reached':True,'sources':hashes}}))
UNPACK
bash "$PROBE_ROOT/code/run_environment_probe.sh" "$PROBE_ROOT/results"
'''
    script = '#!/bin/bash\nexec timeout --signal=TERM --kill-after=20s 420s /bin/bash -c ' + shlex.quote(inner) + '\n'
    # Stable location and queue from the current task's resource query.
    payload = {
        'name': 'miles-preflight-' + args.tag,
        'namespace': '/user/zhangyang.exp', 'job_type': 'train',
        'description': '4 H20 environment/NCCL gate; no model loading; 420s execution cap',
        'attachments': [{'kind': 'hdfs_volume', 'hdfs_volume_attachment': {'hdfs_volumes': [{
            'access_mode': 'RW', 'mnt': '/mnt/hdfs/lab_owner',
            'path': 'hdfs://haruna/home/byte_search_lifeservice/zhangyang.exp',
            'roles': ['worker'], 'extra': ''}]}}],
        'job_config': {'job_template_config': {
            'image_meta': {'image_source': 'vid', 'image_vid': args.image_vid},
            'entrypoint_full_script': script, 'env_map': {'OMP_NUM_THREADS': '1', 'PYTHONUNBUFFERED': '1'}}},
        'resource_config': {'resource_backend': 'arnold', 'arnold_resource_config': {
            'group_id': 2079, 'cluster_ids': [22], 'cluster_select_mode': 'single',
            'roles': [{'name': 'worker', 'num': 1, 'gpu': 4, 'gpu_type': 'NVIDIA_H20',
                       'cpu': 32, 'memory_mb': 131072, 'ports': 1,
                       'queue_name': 'compute-124-lf-cloudnative-ai-search.douyin.life.nlp-guarantee',
                       'advanced': {'pod_no_restart_state': 'any',
                                    'is_complete_trial_on_task_no_restart': True,
                                    'is_fail_trial_on_pod_failure': True}}]}}
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in [('create.json', payload), ('source-manifest.json', hashes), ('mount-marker.json', mount_marker)]:
        (args.output / name).write_text(json.dumps(data, indent=2) + '\n')
    (args.output / 'entry.sh').write_text(script)
    print(json.dumps({'output': str(args.output), 'gpus': 4, 'submitted': False}))


if __name__ == '__main__':
    main()
