"""Freeze the four-node diagnostic payload with a bounded compressed bootstrap."""

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import zlib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--publication-benchmark", action="store_true")
    parser.add_argument("--converted-cache", type=Path)
    args = parser.parse_args()
    if args.output.exists() or not args.tag.replace("-", "").isalnum():
        raise ValueError("Use a new output and an alphanumeric tag")
    root = Path(__file__).resolve().parent
    proof = json.loads((args.assets / "assets-verified.json").read_text())
    if not proof.get("verified") or not proof.get("uploaded"):
        raise ValueError("The 8B model must be verified and uploaded before freezing this job")
    paths = {
        "run_train32.py": root / "run_train32.py",
        "train32_checks.py": root / "train32_checks.py",
        "multinode_entry.py": root / "multinode_entry.py",
        "publication_benchmark.py": root / "publication_benchmark.py",
        "format_contract.py": root / "format_contract.py",
        "rollout_gate.py": root / "format_rollout_gate.py",
        "original_rollout_gate.py": root.parent / "rollout_gate.py",
        "model-manifest-8b.json": args.assets / "model-manifest.json",
        "math_dapo_utils.py": root.parents[1] / "research/posttraining_20261022/upstream/miles/miles/rollout/rm_hub/math_dapo_utils.py",
        **{name: root.parent / name for name in
           ("bounded_rollout.py", "qualification.py", "recovery_checks.py",
            "patch_broadcast_ipv6.py", "broadcast-ipv6.patch", "patch_publication_probe.py",
            "patch_ray_launcher.py", "ray_readiness.py")},
        **{name: root / "format-data" / name for name in ("explicit-prompts.jsonl", "format-manifest.json")},
    }
    if args.converted_cache:
        cache = json.loads(args.converted_cache.read_text())
        model = json.loads((args.assets / "model-manifest.json").read_text())
        if (not cache.get("verified") or not cache.get("source_conversion_verified")
                or cache["model_identity"] != model["revision"]
                or not cache["remote_mount"].startswith("/mnt/hdfs/lab_owner/")
                or not cache["files"]
                or any(m["bytes"] < 0 or len(m["sha256"]) != 64
                       for m in cache["files"].values())):
            raise ValueError("Converted cache must bind to verified terminal preparation and this model")
        paths["converted-cache.json"] = args.converted_cache
    sources = {name: path.read_bytes() for name, path in paths.items()}
    for name, blob in sources.items():
        if name.endswith(".py"):
            ast.parse(blob)
    hashes = {name: hashlib.sha256(blob).hexdigest() for name, blob in sources.items()}
    blobs = {name: base64.b64encode(blob).decode() for name, blob in sources.items()}
    packed = base64.b64encode(zlib.compress(json.dumps(blobs).encode(), level=9)).decode()
    remote = "/mnt/hdfs/lab_owner/training_infra_lab/miles-grpo/" + args.tag
    unpack = f"""import base64, hashlib, json, zlib
from pathlib import Path
root=Path("/tmp/{args.tag}")
blobs=json.loads(zlib.decompress(base64.b64decode({packed!r})))
hashes={hashes!r}
for name, blob in blobs.items():
    content=base64.b64decode(blob)
    assert hashlib.sha256(content).hexdigest()==hashes[name],name
    (root/"code"/name).write_bytes(content)
(root/"results/source-manifest.json").write_text(json.dumps(hashes,indent=2))
"""
    ast.parse(unpack)
    inner = f"""set -euo pipefail
: "${{ARNOLD_ID:?missing node index}}"
RUN_DIR=/tmp/{args.tag}
mkdir -p "$RUN_DIR/code" "$RUN_DIR/results"
exec > >(tee "$RUN_DIR/results/entry.log") 2>&1
finish() {{
 rc=$?
 trap - EXIT
 set +e
 printf '%s\\n' "$rc" > "$RUN_DIR/results/original-exit-code.txt"
 timeout --signal=TERM --kill-after=5s 90s /bin/bash -c \
  'mkdir -p "$1/node-$2" && tar -czf "$1/node-$2/artifacts.tar.gz" -C "$3" code results' \
  archive {shlex.quote(remote)} "$ARNOLD_ID" "$RUN_DIR"
 copy_rc=$?
 echo "ORIGINAL_EXIT_CODE=$rc ARTIFACT_COPY_RC=$copy_rc"
 if [[ "$rc" == 0 && "$copy_rc" != 0 ]]; then rc=$copy_rc; fi
 exit "$rc"
}}
trap finish EXIT
trap 'exit 124' TERM
actual_mount=$(findmnt -n -T /mnt/hdfs/lab_owner -o TARGET)
[[ "$actual_mount" == /mnt/hdfs/lab_owner ]]
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export GRPO_PUBLICATION_BENCHMARK={int(args.publication_benchmark)}
export PYTHONPATH="$RUN_DIR/code:/root/miles:/root/Megatron-LM"
export NCCL_DEBUG=INFO CUDA_DEVICE_MAX_CONNECTIONS=1
python3 - <<'UNPACK'
{unpack}UNPACK
python3 "$RUN_DIR/code/multinode_entry.py" --root "$RUN_DIR" \
 --remote {shlex.quote(remote)} --asset {shlex.quote(proof["remote_mount"])}
echo TRAIN32_NODE_DRIVER_COMPLETE
"""
    if len(inner.encode()) >= 96 * 1024:
        raise ValueError("Bootstrap exceeds the conservative Linux argument-size bound")
    script = "#!/bin/bash\nexec timeout --signal=TERM --kill-after=110s 5400s /bin/bash -c " + shlex.quote(inner) + "\n"
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    baseline = json.loads(args.baseline.read_text())
    baseline = baseline.get("data", baseline)
    payload = {name: baseline[name] for name in ("namespace", "attachments", "resource_config")}
    payload.update(name="miles-grpo-" + args.tag, job_type="train",
                   description="Qwen3-8B 32-H20 diagnostic: four-node Ray, 16 learner TP2/DP8 + 16 rollout on 8 TP2 engines, three real updates, full parameter/Adam and publication audit; no speedup or quality claim")
    template = baseline["job_config"]["job_template_config"]
    payload["job_config"] = {"job_template_config": {
        "image_meta": template["image_meta"], "entrypoint_full_script": script,
        "env_map": {"OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
                    "CUDA_DEVICE_MAX_CONNECTIONS": "1", "NCCL_DEBUG": "INFO"},
    }}
    role = payload["resource_config"]["arnold_resource_config"]["roles"][0]
    role.update(num=4, gpu=8, cpu=64, memory_mb=262144, ports=1)
    role["advanced"] = {"pod_no_restart_state": "any", "is_fail_trial_on_pod_failure": True,
                        "is_complete_trial_on_task_no_restart": False}
    args.output.mkdir(parents=True)
    manifest = {"remote_output": remote, "source_hashes": hashes, "allocated_gpus": 32,
                "learner_gpus": 16, "rollout_gpus": 16, "entry_seconds_bound": 5400,
                "shell_argument_bytes": len(inner.encode()), "submitted": False,
                "performance_run": args.publication_benchmark, "capability_claim": False,
                "converted_cache_source_run": cache["source_run"] if args.converted_cache else None,
                "model_identity": json.loads(sources["model-manifest-8b.json"])["revision"]}
    for name, value in [
        ("create.json", payload), ("source-manifest.json", hashes), ("manifest.json", manifest),
        ("precheck-request.json", {"node_type": "merlin_job_run", "check_point": "create",
                                  "job_run_context": payload}),
    ]:
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")
    (args.output / "entry.sh").write_text(script)
    print(json.dumps({"output": str(args.output), "shell_argument_bytes": len(inner.encode()),
                      "gpus": 32, "submitted": False}))


if __name__ == "__main__":
    main()
