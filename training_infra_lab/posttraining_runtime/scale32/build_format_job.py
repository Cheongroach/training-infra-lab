"""Build a bounded format probe on the original Qwen3-4B; never submits."""

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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    if not args.tag.replace("-", "").isalnum() or args.output.exists():
        raise ValueError("Use a new output and an alphanumeric tag")
    root = Path(__file__).resolve().parent
    upstream = root.parents[1] / "research/posttraining_20261022/upstream/miles"
    source_paths = {
        "format_contract.py": root / "format_contract.py",
        "model-manifest.json": root.parent / "model-manifest.json",
        "math_dapo_utils.py": upstream / "miles/rollout/rm_hub/math_dapo_utils.py",
        **{p.name: p for p in (root / "format-data").iterdir() if p.is_file()},
    }
    sources = {name: path.read_bytes() for name, path in source_paths.items()}
    for name, content in sources.items():
        if name.endswith(".py"):
            ast.parse(content)
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in sources.items()}
    blobs = {name: base64.b64encode(content).decode() for name, content in sources.items()}
    packed = base64.b64encode(zlib.compress(json.dumps(blobs).encode(), level=9)).decode()
    manifest = json.loads(sources["model-manifest.json"])
    mount = "/mnt/hdfs/lab_owner"
    archive = f"{mount}/training_infra_lab/miles-grpo/{args.tag}"
    asset = f"{mount}/training_infra_lab/assets/Qwen3-4B/{manifest['revision']}"
    unpack = f"""import base64, hashlib, json, shutil, subprocess, zlib
from pathlib import Path
root=Path("/tmp/{args.tag}")
blobs=json.loads(zlib.decompress(base64.b64decode({packed!r})))
hashes={hashes!r}
for name, blob in blobs.items():
    content=base64.b64decode(blob)
    assert hashlib.sha256(content).hexdigest()==hashes[name]
    (root/"code"/name).write_bytes(content)
(root/"results/source-manifest.json").write_text(json.dumps(hashes,indent=2))
assert subprocess.check_output(["git","-C","/root/miles","rev-parse","HEAD"],text=True).strip()=="2233d795dcf167be489d1859971c9233393f04c3"
manifest=json.loads((root/"code/model-manifest.json").read_text())
for name, meta in manifest["files"].items():
    source=Path({asset!r})/name
    target=root/"models/Qwen3-4B"/name
    target.parent.mkdir(exist_ok=True,parents=True)
    shutil.copyfile(source,target)
    assert target.stat().st_size==meta["bytes"]
    with target.open("rb") as handle:
        assert hashlib.file_digest(handle,"sha256").hexdigest()==meta["sha256"],name
(root/"results/assets-verified.json").write_text(json.dumps({{"verified":True,"revision":manifest["revision"]}}))
"""
    ast.parse(unpack)
    inner = f"""set -euo pipefail
RUN_DIR=/tmp/{args.tag}
mkdir -p "$RUN_DIR/code" "$RUN_DIR/results"
exec > >(tee "$RUN_DIR/results/entry.log") 2>&1
finish() {{
 rc=$?
 trap - EXIT
 set +e
 printf '%s\\n' "$rc" > "$RUN_DIR/results/original-exit-code.txt"
 timeout --signal=TERM --kill-after=5s 90s /bin/bash -c \
  'mkdir -p "$1" && tar -czf "$1/artifacts.tar.gz" -C "$2" code results' \
  archive {shlex.quote(archive)} "$RUN_DIR"
 copy_rc=$?
 echo "ORIGINAL_EXIT_CODE=$rc ARTIFACT_COPY_RC=$copy_rc"
 if [[ "$rc" == 0 && "$copy_rc" != 0 ]]; then rc=$copy_rc; fi
 exit "$rc"
}}
trap finish EXIT
trap 'exit 124' TERM
actual_mount=$(findmnt -n -T {mount} -o TARGET)
[[ "$actual_mount" == {shlex.quote(mount)} ]]
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export PYTHONPATH=/root/miles:/root/Megatron-LM:${{PYTHONPATH:-}}
python3 - <<'UNPACK'
{unpack}UNPACK
timeout --signal=TERM --kill-after=30s 1000s python3 "$RUN_DIR/code/format_contract.py" \
 generate --root "$RUN_DIR" --data "$RUN_DIR/code" --output "$RUN_DIR/results/generation"
echo FORMAT_PROBE_DRIVER_COMPLETE
"""
    script = "#!/bin/bash\nexec timeout --signal=TERM --kill-after=110s 1500s /bin/bash -c " + shlex.quote(inner) + "\n"
    if len(inner.encode()) >= 96 * 1024:
        raise ValueError("Embedded shell argument exceeds the conservative Linux argument bound")
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    baseline = json.loads(args.baseline.read_text())
    baseline = baseline.get("data", baseline)
    payload = {key: baseline[key] for key in ("namespace", "attachments", "resource_config")}
    payload.update(name="miles-grpo-" + args.tag, job_type="train",
                   description="Bounded original-vs-explicit Qwen3-4B output protocol probe; 4 H20 allocated, SGLang TP2; no learner or capability claim")
    template = baseline["job_config"]["job_template_config"]
    payload["job_config"] = {"job_template_config": {
        "image_meta": template["image_meta"], "entrypoint_full_script": script,
        "env_map": {"OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"},
    }}
    role = payload["resource_config"]["arnold_resource_config"]["roles"][0]
    role.update(num=1, gpu=4, cpu=32, memory_mb=131072, ports=1)
    role["advanced"] = {"pod_no_restart_state": "any", "is_fail_trial_on_pod_failure": True}
    args.output.mkdir(parents=True)
    for name, data in [
        ("create.json", payload), ("source-manifest.json", hashes),
        ("precheck-request.json", {"node_type": "merlin_job_run", "check_point": "create",
                                  "job_run_context": payload}),
        ("manifest.json", {"remote_output": archive, "source_hashes": hashes,
                           "allocated_gpus": 4, "engine_gpus": 2, "submitted": False}),
    ]:
        (args.output / name).write_text(json.dumps(data, indent=2) + "\n")
    (args.output / "entry.sh").write_text(script)
    print(json.dumps({"output": str(args.output), "allocated_gpus": 4, "engine_tp": 2,
                      "submitted": False}))


if __name__ == "__main__":
    main()
