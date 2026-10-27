"""Freeze the runtime and asset manifest into a bounded Merlin probe request."""

import argparse
import base64
import hashlib
import io
import json
import tarfile
from pathlib import Path

from .rewards import RULES, STRICT_RULE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--assets-manifest", type=Path, required=True)
    p.add_argument("--devices", type=int, default=4)
    p.add_argument("--seconds", type=int, default=2400)
    p.add_argument("--cluster", choices=("hl", "lf"), default="hl")
    p.add_argument("--stage", choices=("probe", "grpo", "precision", "backend", "replay-diagnostic"), default="probe")
    p.add_argument("--device", choices=("cuda", "npu"), default="cuda")
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--data", type=Path)
    p.add_argument("--batches", type=Path)
    p.add_argument("--data-manifest", type=Path)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--fp32-head", action="store_true")
    p.add_argument("--resident-rollout", action="store_true")
    p.add_argument("--benchmark-residency", action="store_true")
    p.add_argument("--updates", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--replay-microbatches", type=int, nargs="+", default=[1, 2])
    p.add_argument("--skip-replay", action="store_true")
    p.add_argument("--reward-rule", choices=RULES, default=STRICT_RULE)
    p.add_argument("--token-budget", type=int, default=2048)
    args = p.parse_args()
    if not args.tag.replace("-", "").isalnum() or args.devices not in (2, 4, 8):
        raise ValueError("Invalid tag/devices")
    if args.device == "npu" and (args.stage != "backend" or args.cluster != "hl"):
        raise ValueError("NPU currently supports only the bounded HL backend probe")
    if not 0 < args.adam_eps < 1:
        raise ValueError("Invalid Adam epsilon")
    root = Path(__file__).parent
    files = {f"trajectory_runtime/{p.name}": p.read_bytes() for p in root.glob("*.py")}
    files["asset_manifest.json"] = args.assets_manifest.read_bytes()
    if args.stage == "grpo":
        if not args.data or not args.data_manifest:
            raise ValueError("GRPO requires a pinned dataset and manifest")
        files["data.jsonl"] = args.data.read_bytes()
        files["data_manifest.json"] = args.data_manifest.read_bytes()
        if hashlib.sha256(files["data.jsonl"]).hexdigest() != json.loads(files["data_manifest.json"])["sha256"]:
            raise ValueError("Dataset manifest mismatch")
    if args.stage == "replay-diagnostic":
        if not args.batches:
            raise ValueError("Frozen diagnostics require archived batches")
        for name in ("batch-0.json", "batch-1.json", "batches-manifest.json"):
            files["batches/" + name] = (args.batches / name).read_bytes()
    source = {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
    manifest_digest = source["asset_manifest.json"]
    blob = io.BytesIO()
    with tarfile.open(fileobj=blob, mode="w:gz") as archive:
        for name, content in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    bundle = blob.getvalue()
    bundle_sha = hashlib.sha256(bundle).hexdigest()
    encoded = base64.b64encode(bundle).decode()
    tiny_command = (
        '-m trajectory_runtime.probe --tiny --kernel fla --lengths 64,128 --steps 2 '
        '--output "$RUN_ROOT/results/tiny-fla"'
        if args.stage == "probe" else
        f'-m trajectory_runtime.device_gates --adam-eps {args.adam_eps} --output "$RUN_ROOT/results/device-gates"'
    )
    if args.stage == "precision":
        tiny_command = '-m trajectory_runtime.precision_diagnostic --output "$RUN_ROOT/results/precision"'
    if args.stage in ("backend", "replay-diagnostic"):
        kernel = "torch" if args.device == "npu" else "fla"
        tiny_command = (
            f'-m trajectory_runtime.probe --tiny --device {args.device} --kernel {kernel} '
            '--activation-checkpoint --lengths 32,64 --steps 2 '
            '--output "$RUN_ROOT/results/tiny-backend"'
        )
    workload_command = (
        '-m trajectory_runtime.probe --model "$RUN_ROOT/model" --kernel fla '
        '--activation-checkpoint --lengths 128,512,1024 --steps 3 '
        '--output "$RUN_ROOT/results/qwen35-9b-fla"'
        if args.stage == "probe" else
        '-m trajectory_runtime.grpo --model "$RUN_ROOT/model" --data "$RUN_ROOT/code/data.jsonl" '
        '--scratch "$RUN_ROOT/replay-scratch" --output "$RUN_ROOT/results/grpo" '
        f'--updates {args.updates} --group-size 4 --groups-per-rank 1 --max-new-tokens {args.max_new_tokens} '
        f'--repeats 3 --adam-eps {args.adam_eps} --replay-microbatches {" ".join(map(str, args.replay_microbatches))}'
        f' --reward-rule {args.reward_rule} --token-budget {args.token_budget}'
    )
    if args.stage == "replay-diagnostic":
        workload_command = (
            '-m trajectory_runtime.replay_diagnostic --model "$RUN_ROOT/model" '
            '--batches "$RUN_ROOT/code/batches" --scratch "$RUN_ROOT/diagnostic-scratch" '
            f'--output "$RUN_ROOT/results/replay-diagnostic" --adam-eps {args.adam_eps}'
        )
    for flag in ("fp32_head", "resident_rollout", "benchmark_residency", "skip_replay"):
        if getattr(args, flag):
            if args.stage != "grpo":
                raise ValueError(f"{flag} is only supported for GRPO")
            workload_command += " --" + flag.replace("_", "-")
    if args.profile:
        if args.stage != "grpo":
            raise ValueError("Profiling is currently supported by the GRPO replay driver")
        workload_command += " --profile"
    inner = f"""set -euo pipefail
if [[ "{args.device}" == npu ]]; then
  set +u
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  set -u
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export HTTPS_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export NO_PROXY=.byted.org,localhost,127.0.0.1,::1
export RUN_ROOT=/tmp/trajectory-runtime
mkdir -p "$RUN_ROOT/code" "$RUN_ROOT/results" "$RUN_ROOT/model"
export RESULT_STORE=/mnt/hdfs/lab_owner/training_infra_lab/trajectory-runtime/runs/{args.tag}
mkdir -p "$RESULT_STORE"
finish() {{
  rc=$?
  trap - EXIT
  set +e
  tar -czf "$RESULT_STORE/artifacts.tar.gz" -C "$RUN_ROOT" code results
  copy_rc=$?
  echo "ARTIFACT_COPY_RC=$copy_rc ORIGINAL_EXIT_CODE=$rc"
  if [[ "$rc" == 0 && "$copy_rc" != 0 ]]; then rc=$copy_rc; fi
  exit "$rc"
}}
trap finish EXIT
trap 'exit 124' TERM
python3 - <<'UNPACK'
import base64,hashlib,io,tarfile,json
from pathlib import Path
root=Path('/tmp/trajectory-runtime/code')
blob=base64.b64decode('{encoded}')
assert hashlib.sha256(blob).hexdigest()=='{bundle_sha}'
with tarfile.open(fileobj=io.BytesIO(blob)) as t:
 for member in t.getmembers():
  assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
  path=root/member.name; path.parent.mkdir(parents=True,exist_ok=True)
  path.write_bytes(t.extractfile(member).read())
Path('/tmp/trajectory-runtime/results/source_manifest.json').write_text(json.dumps({source!r},indent=2))
UNPACK
cd "$RUN_ROOT/code"
python3 -m pip install --index-url https://bytedpypi.byted.org/simple --target "$RUN_ROOT/app" \\
  --timeout 30 --retries 1 transformers==5.2.0 numpy==1.26.4 \\
  huggingface-hub==1.4.1 tokenizers==0.22.2 safetensors==0.7.0 einops==0.8.1 \\
  2>&1 | tee "$RUN_ROOT/results/install.log"
if [[ "{args.device}" == cuda ]]; then
python3 -m pip install --index-url https://bytedpypi.byted.org/simple --target "$RUN_ROOT/app" \\
  --no-deps --timeout 30 --retries 1 flash-linear-attention==0.4.2 fla-core==0.4.2 \\
  2>&1 | tee "$RUN_ROOT/results/install-fla.log"
fi
export PYTHONPATH="$RUN_ROOT/app:$RUN_ROOT/code:${{PYTHONPATH:-}}"
python3 -m pip freeze > "$RUN_ROOT/results/environment.txt"
CUDA_VISIBLE_DEVICES="" python3 -m unittest trajectory_runtime.test_runtime \\
  trajectory_runtime.test_state_cost trajectory_runtime.test_grpo trajectory_runtime.test_analysis \
  trajectory_runtime.test_rewards trajectory_runtime.test_residency_analysis trajectory_runtime.test_training_audit -v \\
  2>&1 | tee "$RUN_ROOT/results/cpu-tests.log"
python3 -m trajectory_runtime.failure_probe 2>&1 | tee "$RUN_ROOT/results/failure-probe.log"
python3 -m torch.distributed.run --standalone --nproc_per_node={args.devices} \\
  {tiny_command} 2>&1 | tee "$RUN_ROOT/results/device-gates.log"
if [[ "{args.stage}" == "precision" || "{args.stage}" == "backend" ]]; then
  echo TINY_DIAGNOSTIC_COMPLETE
  exit 0
fi
python3 - <<'ASSETS'
import hashlib,json,shutil
from pathlib import Path
manifest=json.loads(Path('asset_manifest.json').read_text())
source=Path('/mnt/hdfs/lab_owner/trainserve_v2/assets/Qwen3.5-9B/{manifest_digest}')
target=Path('/tmp/trajectory-runtime/model')
for name,entry in manifest['files'].items():
 path=target/name
 shutil.copyfile(source/'model'/name,path)
 with path.open('rb') as f:
  assert hashlib.file_digest(f,'sha256').hexdigest()==entry['sha256'],name
 assert path.stat().st_size==entry['bytes'],name
 print(json.dumps({{'event':'asset_verified','name':name}}),flush=True)
print(json.dumps({{'event':'model_assets_ready','repo_id':manifest['repo_id']}}),flush=True)
ASSETS
python3 -m torch.distributed.run --standalone --nproc_per_node={args.devices} \\
  {workload_command} 2>&1 | tee "$RUN_ROOT/results/workload.log"
echo TRAJECTORY_{args.stage.upper()}_COMPLETE
"""
    script = f"#!/bin/bash\nexec timeout --signal=TERM --kill-after=45s {args.seconds}s /bin/bash -s <<'INNER'\n{inner}\nINNER\n"
    model_label = "tiny Qwen3.5" if args.stage in ("precision", "backend") else "Qwen3.5-9B"
    image = "da2nitf1enl89uldequ0" if args.device == "npu" else "d7iu6hbc77u4ebjdshsg"
    queue_type = "ascend" if args.device == "npu" else "ai"
    payload = {
        "name": "trajectory-runtime-" + args.tag, "namespace": "/user/zhangyang.exp", "job_type": "train",
        "description": f"{model_label} FSDP2 {args.stage}; bounded numeric gates and workload; {args.seconds}s limit; {bundle_sha}",
        "attachments": [{"kind": "hdfs_volume", "hdfs_volume_attachment": {"hdfs_volumes": [{
            "access_mode": "RW", "mnt": "/mnt/hdfs/lab_owner",
            "path": "hdfs://haruna/home/byte_search_lifeservice/zhangyang.exp",
            "roles": ["worker"], "extra": "",
        }]}}],
        "job_config": {"job_template_config": {
            "entrypoint_full_script": script,
            "image_meta": {"image_source": "vid", "image_vid": image},
            "env_map": {"OMP_NUM_THREADS": "1"},
        }},
        "resource_config": {"resource_backend": "arnold", "arnold_resource_config": {
            "group_id": 2079, "cluster_ids": [20 if args.cluster == "hl" else 22],
            "cluster_select_mode": "single", "roles": [{
                "name": "worker", "num": 1, "gpu": args.devices,
                "gpu_type": "ASCEND910B4_64G" if args.device == "npu" else "NVIDIA_H20",
                "cpu": 16 if args.stage == "backend" else 32,
                "memory_mb": 65536 if args.stage == "backend" else 196608, "ports": 1,
                "queue_name": f"compute-124-{args.cluster}-cloudnative-{queue_type}-search.douyin.life.nlp-guarantee",
                "advanced": {"pod_no_restart_state": "any", "is_complete_trial_on_task_no_restart": True,
                             "is_fail_trial_on_pod_failure": True},
            }],
        }},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, obj in (("create.json", payload), ("source_manifest.json", source)):
        (args.output / name).write_text(json.dumps(obj, indent=2))
    (args.output / "bundle.tar.gz").write_bytes(bundle)
    (args.output / "entry.sh").write_text(script)
    print(json.dumps({"output": str(args.output), "bundle_sha256": bundle_sha, "assets_sha256": manifest_digest}))


if __name__ == "__main__":
    main()
