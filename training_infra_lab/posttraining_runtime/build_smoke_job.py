"""Freeze a bounded, self-contained four-H20 GRPO request; never submits."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--with-ipv6-patch', action='store_true')
    p.add_argument('--recovery', action='store_true')
    p.add_argument('--publication-probe', action='store_true')
    p.add_argument('--ray-readiness', action='store_true')
    p.add_argument('--profile', choices=('legacy', 'dapo-gated', 'dapo-bounded'), default='legacy')
    args = p.parse_args()
    if args.ray_readiness and not args.publication_probe:
        raise ValueError('The Ray correction requires the complete publication probe')
    if args.recovery and (args.profile != 'dapo-bounded' or not args.with_ipv6_patch):
        raise ValueError('Recovery requires the bounded profile and IPv6 patch')
    if args.publication_probe and (args.recovery or args.profile != 'dapo-bounded'
                                   or not args.with_ipv6_patch):
        raise ValueError('The complete probe requires a full bounded job with IPv6 patch')
    if not args.tag.replace('-', '').isalnum():
        raise ValueError('Invalid tag')
    if (args.output / 'create.json').exists():
        raise ValueError('Use a new submission directory')
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'model-manifest.json').read_text())
    asset = '/mnt/hdfs/lab_owner/training_infra_lab/assets/Qwen3-4B/' + manifest['revision']
    archive = '/mnt/hdfs/lab_owner/training_infra_lab/miles-grpo/' + args.tag
    names = ('run_smoke.py', 'smoke-prompts.jsonl', 'model-manifest.json')
    if args.profile in ('dapo-gated', 'dapo-bounded'):
        if not args.with_ipv6_patch:
            raise ValueError('The validated DAPO job requires the IPv6 patch')
        names += ('rollout_gate.py',)
    if args.profile == 'dapo-bounded':
        names += ('qualification.py', 'bounded_rollout.py')
    if args.with_ipv6_patch:
        names += ('patch_broadcast_ipv6.py', 'broadcast-ipv6.patch')
    if args.recovery:
        names += ('qualification_checkpoint.py', 'recoverable_rollout.py', 'recovery_checks.py',
                  'patch_recovery_probe.py', 'checkpoint_inventory.py', 'resume_publication_guard.py')
    if args.publication_probe:
        names += ('patch_publication_probe.py',)
    if args.ray_readiness:
        names += ('ray_readiness.py', 'patch_ray_launcher.py')
    patch_command = ('sudo /opt/sglang/bin/python3 "$SMOKE_ROOT/code/patch_broadcast_ipv6.py" --report "$SMOKE_ROOT/results/runtime-patch.json"' if args.with_ipv6_patch else ':')
    if args.recovery:
        patch_command += '\nsudo /opt/sglang/bin/python3 "$SMOKE_ROOT/code/patch_recovery_probe.py" --report "$SMOKE_ROOT/results/recovery-patch.json"'
    if args.publication_probe:
        patch_command += '\nsudo /opt/sglang/bin/python3 "$SMOKE_ROOT/code/patch_publication_probe.py" --report "$SMOKE_ROOT/results/publication-probe-patch.json"'
    if args.ray_readiness:
        patch_command += '\nsudo /opt/sglang/bin/python3 "$SMOKE_ROOT/code/patch_ray_launcher.py" --report "$SMOKE_ROOT/results/ray-launcher-patch.json"'
    train_command = f'python3 "$SMOKE_ROOT/code/run_smoke.py" train --root "$SMOKE_ROOT" --data "$SMOKE_ROOT/code/smoke-prompts.jsonl" --profile {shlex.quote(args.profile)}'
    if args.publication_probe:
        train_command += ' --publication-probe'
    if args.ray_readiness:
        train_command += ' --ray-readiness'
    if args.recovery:
        train_command = f"""python3 - <<'DISK'
import shutil
from pathlib import Path
space=shutil.disk_usage('/tmp/grpo-smoke')
assert space.free >= 96*1024**3, 'The local checkpoint probe requires 96 GiB free after model preparation'
DISK
{train_command} --phase checkpoint
python3 "$SMOKE_ROOT/code/checkpoint_inventory.py" --root "$SMOKE_ROOT"
{train_command} --phase resume"""
    train_command = '/bin/bash -euo pipefail -c ' + shlex.quote(train_command)
    sources = {n: (root / n).read_bytes() for n in names}
    hashes = {n: hashlib.sha256(b).hexdigest() for n, b in sources.items()}
    blobs = {n: base64.b64encode(b).decode() for n, b in sources.items()}
    marker = {'tag': args.tag, 'sources': hashes}
    snapshot_command = (
        'timeout 10s python3 "$SMOKE_ROOT/code/ray_readiness.py" --snapshot-only'
        if args.ray_readiness else ':'
    )
    diagnostic_env = ('export RAY_DIAGNOSTIC_DIR="$SMOKE_ROOT/results/ray-diagnostics"'
                      if args.ray_readiness else ':')
    copy_cmd = 'mkdir -p ' + shlex.quote(archive) + ' && tar -czf ' + shlex.quote(archive + '/artifacts.tar.gz') + ' -C /tmp/grpo-smoke code results'
    inner = f'''set -euo pipefail
export SMOKE_ROOT=/tmp/grpo-smoke
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MASTER_ADDR=127.0.0.1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
{diagnostic_env}
mkdir -p "$SMOKE_ROOT/code" "$SMOKE_ROOT/results" "$SMOKE_ROOT/models"
exec > >(tee "$SMOKE_ROOT/results/entry.log") 2>&1
echo GRPO_ENTRY_REACHED
finish() {{
 rc=$?
 trap - EXIT
 set +e
 {snapshot_command}
 timeout --signal=TERM --kill-after=5s 90s /bin/bash -c {shlex.quote(copy_cmd)}
 copy_rc=$?
 echo "ORIGINAL_EXIT_CODE=$rc ARTIFACT_COPY_RC=$copy_rc"
 if [[ "$rc" == 0 && "$copy_rc" != 0 ]]; then rc=$copy_rc; fi
 exit "$rc"
}}
trap finish EXIT
trap 'exit 124' TERM
python3 - <<'UNPACK'
import base64,hashlib,json,shutil,time
from pathlib import Path
root=Path('/tmp/grpo-smoke')
blobs={blobs!r}
hashes={hashes!r}
for name,blob in blobs.items():
 content=base64.b64decode(blob)
 assert hashlib.sha256(content).hexdigest()==hashes[name]
 (root/'code'/name).write_bytes(content)
(root/'results/source-manifest.json').write_text(json.dumps(hashes,indent=2))
archive=Path({archive!r})
assert json.loads((archive/'mount-marker.json').read_text())=={marker!r}
(archive/'bootstrap.json').write_text(json.dumps({{'entry_reached':True,'sources':hashes}}))
manifest=json.loads((root/'code/model-manifest.json').read_text())
dst=root/'models/Qwen3-4B'
dst.mkdir()
for name,meta in manifest['files'].items():
 src=Path({asset!r})/name
 target=dst/name
 target.parent.mkdir(parents=True,exist_ok=True)
 shutil.copyfile(src,target)
 assert target.stat().st_size==meta['bytes'],name
 with target.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
 assert digest==meta['sha256'],name
 print('ASSET_VERIFIED',name,meta['bytes'],flush=True)
(root/'results/model-manifest.json').write_text(json.dumps(manifest,indent=2))
(root/'results/assets-verified.json').write_text(json.dumps({{'verified':True,'unix':time.time()}}))
UNPACK
{patch_command}
timeout --signal=TERM --kill-after=20s 600s python3 "$SMOKE_ROOT/code/run_smoke.py" prepare --root "$SMOKE_ROOT" --data "$SMOKE_ROOT/code/smoke-prompts.jsonl" --staged-model
echo GRPO_CHECKPOINT_PREPARED
timeout --signal=TERM --kill-after=45s 1800s {train_command}
echo GRPO_DRIVER_COMPLETE
'''
    script = '#!/bin/bash\nexec timeout --signal=TERM --kill-after=110s 2700s /bin/bash -c ' + shlex.quote(inner) + '\n'
    payload = json.loads((root / 'job-v4/create.json').read_text())
    payload['name'] = 'miles-grpo-' + args.tag
    payload['description'] = (
        '4 H20, 2 learner TP2 + 2 rollout TP1, 3 GRPO updates; 2700s entry cap; '
        + args.profile + ('; cold-process checkpoint resume and final publication' if args.recovery else '')
        + ('; four final-version natural-EOS probes at 4096 tokens' if args.publication_probe else '')
        + ('; bounded CPU Ray Jobs readiness gate and launcher cleanup fix' if args.ray_readiness else '')
    )
    payload['job_config']['job_template_config']['entrypoint_full_script'] = script
    args.output.mkdir(parents=True, exist_ok=True)
    for n, value in (('create.json', payload), ('source-manifest.json', hashes), ('mount-marker.json', marker)):
        (args.output / n).write_text(json.dumps(value, indent=2) + '\n')
    (args.output / 'entry.sh').write_text(script)
    print(json.dumps({'output': str(args.output), 'gpus': 4, 'submitted': False}))


if __name__ == '__main__':
    main()
