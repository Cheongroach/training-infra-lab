"""Build a bounded 4-node H20 environment probe; this command never submits."""

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    if not args.tag.replace("-", "").isalnum():
        raise ValueError("Tag must contain letters, digits and hyphens")
    if args.output.exists():
        raise ValueError("Use a new output directory")
    source = Path(__file__).resolve().parent.parent / "environment_probe.py"
    content = source.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    ast.parse(content)
    baseline = json.loads(args.baseline.read_text())
    payload = baseline.get("data", baseline)
    if payload["sid"] != "49b6f8781696e68e":
        raise ValueError("Unexpected baseline")
    template = payload["job_config"]["job_template_config"]
    if template["image_meta"]["image_vid"] != "dbdt1t71enl9fha0bn40":
        raise ValueError("Unexpected image")
    payload = {
        "name": "miles-grpo-" + args.tag,
        "namespace": payload["namespace"],
        "job_type": "train",
        "description": (
            "32 H20 on 4 nodes; bounded NCCL/BF16 environment preflight; "
            "no model training, quality, RDMA speedup or scaling-efficiency claim"
        ),
        "attachments": payload["attachments"],
        "job_config": {"job_template_config": {
            "image_meta": template["image_meta"],
            "env_map": {"OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
                        "NCCL_DEBUG": "INFO", "CUDA_DEVICE_MAX_CONNECTIONS": "1"},
        }},
        "resource_config": payload["resource_config"],
    }
    role = payload["resource_config"]["arnold_resource_config"]["roles"][0]
    role.update(num=4, gpu=8, cpu=64, memory_mb=262144, ports=1)
    role["advanced"] = {
        "pod_no_restart_state": "any",
        "is_fail_trial_on_pod_failure": True,
        "is_complete_trial_on_task_no_restart": False,
    }
    mount = "/mnt/hdfs/lab_owner"
    remote = mount + "/training_infra_lab/miles-grpo/" + args.tag
    bootstrap = f"""import base64, hashlib, json, os, socket
from pathlib import Path
blob = base64.b64decode({base64.b64encode(content).decode()!r})
assert hashlib.sha256(blob).hexdigest() == {digest!r}
Path("environment_probe.py").write_bytes(blob)
node = int(os.environ["ARNOLD_ID"])
assert int(os.environ["ARNOLD_WORKER_NUM"]) == 4
assert 0 <= node < 4
host = os.environ["METIS_WORKER_0_HOST"].strip("[]")
port = int(os.environ["METIS_WORKER_0_PORT"])
assert host and 0 < port < 65536
Path("master-address.txt").write_text(host)
Path("master-port.txt").write_text(str(port))
Path("results/bootstrap.json").write_text(json.dumps({{
    "node": node, "nodes": 4, "host": socket.gethostname(),
    "source_sha256": {digest!r}, "expected_world_size": 32,
    "scope": "environment_preflight_only",
}}))
"""
    ast.parse(bootstrap)
    inner = f"""set -euo pipefail
: "${{ARNOLD_ID:?missing node index}}"
: "${{ARNOLD_WORKER_NUM:?missing worker count}}"
: "${{METIS_WORKER_0_HOST:?missing master host}}"
: "${{METIS_WORKER_0_PORT:?missing master port}}"
RUN_DIR=/tmp/{args.tag}
mkdir -p "$RUN_DIR/results"
cd "$RUN_DIR"
exec > >(tee results/entry.log) 2>&1
archive={shlex.quote(remote)}
finish() {{
 rc=$?
 trap - EXIT
 set +e
 printf '%s\\n' "$rc" > results/original-exit-code.txt
 timeout --signal=TERM --kill-after=5s 90s /bin/bash -c \
   'mkdir -p "$1/node-$2" && tar -czf "$1/node-$2/artifacts.tar.gz" results environment_probe.py' \
   archive "$archive" "$ARNOLD_ID"
 copy_rc=$?
 echo "ORIGINAL_EXIT_CODE=$rc ARTIFACT_COPY_RC=$copy_rc"
 if [[ "$rc" == 0 && "$copy_rc" != 0 ]]; then rc=$copy_rc; fi
 exit "$rc"
}}
trap finish EXIT
trap 'exit 124' TERM
python3 - <<'BOOTSTRAP'
{bootstrap}BOOTSTRAP
actual_mount=$(findmnt -n -T {mount} -o TARGET)
[[ "$actual_mount" == {shlex.quote(mount)} ]]
[[ -d {shlex.quote(mount)} ]]
export MASTER_ADDR=$(cat master-address.txt)
export MASTER_PORT=$(cat master-port.txt)
export PYTHONPATH=/root/miles:/root/Megatron-LM:${{PYTHONPATH:-}}
export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE="$RUN_DIR/results/nccl.%h.%p.log"
timeout --signal=TERM --kill-after=15s 480s \
 /opt/sglang/bin/torchrun --nnodes=4 --nproc-per-node=8 \
 --node-rank="$ARNOLD_ID" --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
 environment_probe.py --require-gpu --expected-world-size=32 --output="$RUN_DIR/results"
echo MULTINODE_PREFLIGHT_DRIVER_COMPLETE
"""
    script = (
        "#!/bin/bash\nexec timeout --signal=TERM --kill-after=110s 720s "
        "/bin/bash -c " + shlex.quote(inner) + "\n"
    )
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    subprocess.run(["bash", "-n"], input=inner, text=True, check=True)
    payload["job_config"]["job_template_config"]["entrypoint_full_script"] = script
    args.output.mkdir(parents=True)
    for name, data in [
        ("create.json", payload),
        ("precheck-request.json", {"node_type": "merlin_job_run", "check_point": "create",
                                  "job_run_context": payload}),
        ("manifest.json", {"source_sha256": digest, "nodes": 4,
                           "gpus_per_node": 8, "world_size": 32,
                           "remote_output": remote, "submitted": False,
                           "scope": "environment_preflight_only"}),
    ]:
        (args.output / name).write_text(json.dumps(data, indent=2) + "\n")
    (args.output / "entry.sh").write_text(script)
    print(json.dumps({"output": str(args.output), "nodes": 4, "gpus": 32,
                      "source_sha256": digest, "submitted": False}))


if __name__ == "__main__":
    main()
