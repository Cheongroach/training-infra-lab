"""Four H20 devices across two nodes: verify launch and NCCL before scaling."""

import argparse
import base64
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("runtime_probe.py").read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    encoded = base64.b64encode(source).decode()
    script = f"""#!/bin/bash
set -eo pipefail
export LAB_DEVICE=cuda OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NCCL_DEBUG=INFO
python3 - <<'PY'
import base64, hashlib
from pathlib import Path
s=base64.b64decode('{encoded}')
assert hashlib.sha256(s).hexdigest()=='{digest}'
Path('/tmp/multinode_probe.py').write_bytes(s)
PY
: "${{ARNOLD_ID:?missing node index}}"
: "${{METIS_WORKER_0_HOST:?missing master address}}"
: "${{METIS_WORKER_0_PORT:?missing master port}}"
exec timeout --signal=TERM --kill-after=30s 600s python3 -m torch.distributed.run \\
    --nnodes=2 --nproc_per_node=2 --node_rank="$ARNOLD_ID" \\
    --master_addr="$METIS_WORKER_0_HOST" --master_port="$METIS_WORKER_0_PORT" \\
    /tmp/multinode_probe.py
"""
    payload = {
        "name": "training-infra-h20-two-node-probe",
        "namespace": "/user/zhangyang.exp", "job_type": "train",
        "description": "2 nodes x 2 H20, NCCL/FSDP2, 600s workload bound.",
        "attachments": [],
        "job_config": {"job_template_config": {
            "entrypoint_full_script": script,
            "image_meta": {"image_source": "vid", "image_vid": "d7iu6hbc77u4ebjdshsg"},
            "env_map": {"OMP_NUM_THREADS": "1"},
        }},
        "resource_config": {
            "resource_backend": "arnold",
            "arnold_resource_config": {
                "group_id": 2079, "cluster_ids": [22], "cluster_select_mode": "single",
                "roles": [{
                    "name": "worker", "num": 2, "gpu": 2, "cpu": 24,
                    "gpu_type": "NVIDIA_H20", "memory_mb": 131072, "ports": 1,
                    "queue_name": "compute-124-lf-cloudnative-ai-search.douyin.life.nlp-guarantee",
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


if __name__ == "__main__":
    main()
