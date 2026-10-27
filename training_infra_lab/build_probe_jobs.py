"""Build reviewable, self-contained Merlin requests; this script never submits."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("runtime_probe.py").read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    encoded = base64.b64encode(source).decode("ascii")
    for device, image, gpu_type, queue in (
        (
            "cuda", "d7iu6hbc77u4ebjdshsg", "NVIDIA_H20",
            "compute-124-hl-cloudnative-ai-search.douyin.life.nlp-guarantee",
        ),
        (
            "npu", "d6nakm3c77u7jl17ea90", "ASCEND910B4_64G",
            "compute-124-hl-cloudnative-ascend-search.douyin.life.nlp-guarantee",
        ),
    ):
        script = f"""#!/bin/bash
set -eo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export LAB_DEVICE={device}
if [[ "$LAB_DEVICE" == npu && -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
python3 - <<'PY'
import base64, hashlib
from pathlib import Path
source = base64.b64decode('{encoded}')
assert hashlib.sha256(source).hexdigest() == '{digest}'
Path('/tmp/training_infra_runtime_probe.py').write_bytes(source)
print('PROBE_SOURCE_SHA256={digest}', flush=True)
PY
exec timeout --signal=TERM --kill-after=30s 360s python3 -m torch.distributed.run --standalone --nproc_per_node=2 /tmp/training_infra_runtime_probe.py
"""
        payload = {
            "name": f"training-infra-{device}-probe-{args.run_tag}",
            "namespace": "/user/zhangyang.exp",
            "job_type": "train",
            "description": "Two-device runtime and FSDP2 probe; 360-second workload limit.",
            "attachments": [],
            "job_config": {
                "job_template_config": {
                    "entrypoint_full_script": script,
                    "image_meta": {"image_source": "vid", "image_vid": image},
                    "env_map": {"OMP_NUM_THREADS": "1", "PYTHONNOUSERSITE": "1"},
                },
            },
            "resource_config": {
                "resource_backend": "arnold",
                "arnold_resource_config": {
                    "group_id": 2079,
                    "cluster_ids": [20],
                    "cluster_select_mode": "single",
                    "roles": [{
                        "name": "worker", "num": 1, "gpu": 2,
                        "gpu_type": gpu_type, "cpu": 24, "memory_mb": 131072,
                        "queue_name": queue, "ports": 1,
                        "advanced": {
                            "pod_no_restart_state": "any",
                            "is_complete_trial_on_task_no_restart": True,
                            "is_fail_trial_on_pod_failure": True,
                        },
                    }],
                },
            },
        }
        request_path = args.output_dir / f"{device}.create.json"
        request_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        precheck = {
            "node_type": "merlin_job_run",
            "check_point": "create",
            "job_run_context": payload,
        }
        (args.output_dir / f"{device}.precheck.json").write_text(
            json.dumps(precheck, ensure_ascii=False, indent=2),
        )
        print(f"{device}: {request_path}; source_sha256={digest}")


if __name__ == "__main__":
    main()
