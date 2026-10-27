"""Build immutable source bundles and bounded Merlin training requests."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import tarfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=("cuda", "npu"), required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--stage", choices=("smoke", "benchmark", "scale", "recovery", "profile", "sync"), default="smoke")
    p.add_argument("--devices", type=int, default=2)
    p.add_argument("--nodes", type=int, default=1)
    p.add_argument("--seconds", type=int, default=2400)
    p.add_argument("--cluster", choices=("hl", "lf"), default="hl")
    args = p.parse_args()
    assert args.tag.replace("-", "").replace("_", "").isalnum()
    assert args.device == "cuda" or args.cluster == "hl"
    assert args.nodes == 1 or (args.device == "cuda" and args.devices == 8)
    folder = Path(__file__).parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    blob = io.BytesIO()
    manifest = {}
    with tarfile.open(fileobj=blob, mode="w:gz") as tar:
        for path in sorted(folder.glob("*.py")):
            content = path.read_bytes()
            manifest[path.name] = hashlib.sha256(content).hexdigest()
            info = tarfile.TarInfo(path.name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    bundle = blob.getvalue()
    digest = hashlib.sha256(bundle).hexdigest()
    encoded = base64.b64encode(bundle).decode()
    if args.stage in ("smoke", "profile"):
        cases = [(mode, "fixed", False, 1) for mode in ("full", "chunk")]
        sequence, steps, warmup, global_batch = 1024, 8, 2, 8
        if args.stage == "profile":
            cases = [(mode, "dynamic", False, 1) for mode in ("full", "chunk")]
    elif args.stage == "benchmark":
        cases = [(mode, "fixed", False, 1) for mode in ("full", "chunk")]
        for repeat in (1, 2):
            order = ("full", "selected", "chunk") if repeat == 1 else ("chunk", "selected", "full")
            cases.extend((mode, "dynamic", False, repeat) for mode in order)
            cases.append(("chunk", "dynamic", True, repeat))
        sequence, steps, warmup, global_batch = 4096, 16, 4, 8
    else:
        cases = [("chunk", "dynamic", False, r) for r in (1, 2)]
        sequence, steps, warmup, global_batch = 4096, 20, 4, 16
    assert global_batch % (args.nodes * args.devices) == 0
    launcher = (
        f"python3 -m torch.distributed.run --standalone --nproc_per_node={args.devices}"
        if args.nodes == 1 else
        f'python3 -m torch.distributed.run --nnodes={args.nodes} --nproc_per_node={args.devices} '
        '--node_rank="$ARNOLD_ID" --master_addr="$METIS_WORKER_0_HOST" '
        '--master_port="$METIS_WORKER_0_PORT"'
    )
    commands = []
    for mode, padding, checkpoint, repeat in cases:
        name = f"{mode}-{padding}-ac{int(checkpoint)}-s{sequence}-r{repeat}"
        flags = "--activation-checkpoint" if checkpoint else ""
        if args.stage == "profile":
            flags += " --profile-step 5"
        commands.append(f"""{launcher} train.py \\
    --assets /tmp/training_lab/assets --output "/tmp/training_lab/results/{name}" \\
    --mode {mode} --padding {padding} --seq-length {sequence} --steps {steps} --warmup {warmup} \\
    --global-batch {global_batch} --micro-batch 1 --require-hosts {args.nodes} {flags} \\
    2>&1 | tee "/tmp/training_lab/results/{name}.log"
""")
    training_commands = "\n".join(commands)
    if args.stage == "sync":
        commands = []
        sync_cases = [(False, 1, False), (True, 1, False),
                      (True, 2, False), (False, 2, False)]
        if args.device == "cuda":
            sync_cases.extend([(False, 1, True), (True, 1, True)])
        for deferred, repeat, profile in sync_cases:
            name = f"selected-sync{'last' if deferred else 'each'}-r{repeat}"
            flags = "--defer-gradient-sync" if deferred else ""
            if profile:
                name += "-prof"
                flags += " --profile-step 5"
            commands.append(f"""{launcher} train.py \
    --assets /tmp/training_lab/assets --output /tmp/training_lab/results/{name} \
    --mode selected --padding dynamic --seq-length 4096 --steps 16 --warmup 4 \
    --global-batch 8 --micro-batch 1 {flags} \
    2>&1 | tee /tmp/training_lab/results/{name}.log
""")
        training_commands = "\n".join(commands)
    if args.stage == "recovery":
        assert args.nodes == 1
        commands = []
        for name, extra in (
            ("continuous", ""),
            ("split", "--stop-after 3 --save-checkpoint --checkpoint-dir /tmp/training_lab/checkpoint"),
            ("resumed", "--resume /tmp/training_lab/checkpoint --save-checkpoint --checkpoint-dir /tmp/training_lab/checkpoint-final"),
        ):
            commands.append(f"""{launcher} train.py \\
    --assets /tmp/training_lab/assets --output /tmp/training_lab/results/{name} \\
    --mode chunk --padding dynamic --seq-length 1024 --steps 6 --warmup 1 \\
    --global-batch 8 --micro-batch 1 --fingerprint --deterministic {extra} \\
    2>&1 | tee /tmp/training_lab/results/{name}.log
""")
        commands.append("""python3 - <<'VERIFY_RECOVERY'
import json
from pathlib import Path
root=Path('/tmp/training_lab/results')
a=json.loads((root/'continuous/result.json').read_text())
b=json.loads((root/'resumed/result.json').read_text())
split=json.loads((root/'split/result.json').read_text())
assert split['state_fingerprints']==b['start_fingerprints'], 'DCP did not restore exact parameter/Adam bytes'
assert b['start_step']==3 and b['next_step']==6
assert a['state_fingerprints']==b['state_fingerprints'], 'Parameter/Adam byte hashes differ after resume'
assert a['eval_final']==b['eval_final'], 'Final evaluation differs after resume'
report={'event':'checkpoint_replay_gate_passed','world_size':a['world_size'],'next_step':6,
        'parameter_and_adam_byte_identical':True,'loaded_state_byte_identical':True,
        'deterministic_algorithms':True,'eval_final':a['eval_final']}
(root/'recovery_gate.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
VERIFY_RECOVERY
mkdir -p "$LAB_OUTPUT/checkpoint"
cp -r /tmp/training_lab/checkpoint-final/. "$LAB_OUTPUT/checkpoint/"
""")
        training_commands = "\n".join(commands)
    assets_name = "assets" if args.device == "npu" else "assets-cuda"
    inner = f"""set -eo pipefail
export LAB_DEVICE={args.device} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
export NCCL_DEBUG=INFO
export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export HTTPS_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export NO_PROXY=.byted.org,localhost,127.0.0.1,::1
mkdir -p /tmp/training_lab/code /tmp/training_lab/results
export LAB_STORE=/mnt/hdfs/lab_owner/training_infra_lab/20261019
export LAB_OUTPUT="$LAB_STORE/runs/{args.tag}"
if [[ {args.nodes} -gt 1 ]]; then LAB_OUTPUT="$LAB_OUTPUT/node-$ARNOLD_ID"; fi
mkdir -p "$LAB_OUTPUT"
finish() {{
    rc=$?
    trap - EXIT
    set +e
    tar -czf "$LAB_OUTPUT/artifacts.tar.gz" -C /tmp/training_lab results code
    copy_rc=$?
    echo "ARTIFACT_COPY_RC=$copy_rc ORIGINAL_EXIT_CODE=$rc"
    if [[ "$rc" == 0 && "$copy_rc" != 0 ]]; then rc=$copy_rc; fi
    exit "$rc"
}}
trap finish EXIT
trap 'exit 124' TERM
python3 - <<'UNPACK'
import base64, hashlib, io, tarfile, json
from pathlib import Path
b=base64.b64decode('{encoded}')
assert hashlib.sha256(b).hexdigest()=='{digest}'
with tarfile.open(fileobj=io.BytesIO(b)) as t:
    for member in t.getmembers():
        assert member.isfile() and Path(member.name).name == member.name
        Path('/tmp/training_lab/code', member.name).write_bytes(t.extractfile(member).read())
Path('/tmp/training_lab/results/source_manifest.json').write_text(json.dumps({manifest!r},indent=2))
UNPACK
cd /tmp/training_lab/code
if [[ "$LAB_DEVICE" == npu ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
else
    python3 -m pip install --index-url https://bytedpypi.byted.org/simple \\
        --target /tmp/training_lab/app --timeout 30 --retries 1 \\
        transformers==4.57.1 tokenizers==0.22.2 huggingface-hub==0.36.0 \\
        safetensors==0.7.0 numpy==1.26.4 pyarrow==18.1.0 \\
        2>&1 | tee /tmp/training_lab/results/install.log
    export PYTHONPATH=/tmp/training_lab/app:${{PYTHONPATH:-}}
fi
python3 -m pip freeze > /tmp/training_lab/results/environment.txt
python3 transformer_gate.py 2>&1 | tee /tmp/training_lab/results/transformer_gate.log
{launcher} correctness.py 2>&1 | tee /tmp/training_lab/results/correctness.log
if [[ ! -f "$LAB_STORE/{assets_name}/manifest.json" ]]; then
    if [[ {args.nodes} -gt 1 ]]; then echo "Prepare assets in a single-node job first"; exit 2; fi
    python3 prepare_assets.py --root "$LAB_STORE/{assets_name}" 2>&1 | tee /tmp/training_lab/results/prepare.log
fi
mkdir -p /tmp/training_lab/assets
python3 - <<'COPY_ASSETS'
import hashlib, json, os, shutil
from pathlib import Path
source = Path(os.environ['LAB_STORE']) / '{assets_name}'
target = Path('/tmp/training_lab/assets')
manifest = json.loads((source / 'manifest.json').read_text())
for name, expected in manifest['files'].items():
    destination = target / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / name, destination)
    with destination.open('rb') as file:
        assert hashlib.file_digest(file, 'sha256').hexdigest() == expected, name
shutil.copyfile(source / 'manifest.json', target / 'manifest.json')
print(json.dumps({{'event': 'assets_copied_and_verified'}}), flush=True)
COPY_ASSETS
{training_commands}
echo LAB_SUITE_COMPLETE
"""
    # The deadline encloses environment setup, downloads, gates and all runs.
    script = f"#!/bin/bash\nexec timeout --signal=TERM --kill-after=45s {args.seconds}s /bin/bash -s <<'LAB_INNER'\n{inner}\nLAB_INNER\n"
    queue_type = "ascend" if args.device == "npu" else "ai"
    payload = {
        "name": f"training-infra-{args.tag}",
        "namespace": "/user/zhangyang.exp", "job_type": "train",
        "description": f"Full-parameter Qwen2.5-1.5B SFT {args.stage}; source sha256 {digest}; {args.seconds}s limit.",
        "attachments": [{
            "kind": "hdfs_volume",
            "hdfs_volume_attachment": {"hdfs_volumes": [{
                "access_mode": "RW", "mnt": "/mnt/hdfs/lab_owner",
                "path": "hdfs://haruna/home/byte_search_lifeservice/zhangyang.exp",
                "roles": ["worker"], "extra": "",
            }]},
        }],
        "job_config": {"job_template_config": {
            "entrypoint_full_script": script,
            "image_meta": {
                "image_source": "vid",
                "image_vid": "d6nakm3c77u7jl17ea90" if args.device == "npu" else "d7iu6hbc77u4ebjdshsg",
            },
            "env_map": {"OMP_NUM_THREADS": "1"},
        }},
        "resource_config": {
            "resource_backend": "arnold",
            "arnold_resource_config": {
                "group_id": 2079, "cluster_ids": [20 if args.cluster == "hl" else 22],
                "cluster_select_mode": "single",
                "roles": [{
                    "name": "worker", "num": args.nodes, "gpu": args.devices,
                    "gpu_type": "ASCEND910B4_64G" if args.device == "npu" else "NVIDIA_H20",
                    "cpu": 24, "memory_mb": 131072, "ports": 1,
                    "queue_name": f"compute-124-{args.cluster}-cloudnative-{queue_type}-search.douyin.life.nlp-guarantee",
                    "advanced": {
                        "pod_no_restart_state": "any",
                        "is_complete_trial_on_task_no_restart": True,
                        "is_fail_trial_on_pod_failure": True,
                    },
                }],
            },
        },
    }
    (args.output_dir / "create.json").write_text(json.dumps(payload, indent=2))
    (args.output_dir / "precheck.json").write_text(json.dumps({
        "node_type": "merlin_job_run", "check_point": "create",
        "job_run_context": payload,
    }, indent=2))
    (args.output_dir / "bundle.tar.gz").write_bytes(bundle)
    (args.output_dir / "source_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"directory": str(args.output_dir), "source_sha256": digest}))


if __name__ == "__main__":
    main()
