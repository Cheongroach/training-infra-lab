#!/usr/bin/env bash
# Run inside a candidate Miles image with exactly four visible GPUs.
# The caller ships this directory; no devbox-specific user path is required.
set -euo pipefail

probe_code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
probe_output_dir="${1:?Usage: run_environment_probe.sh OUTPUT_DIRECTORY}"
mkdir -p "$probe_output_dir"
probe_output_dir="$(cd -- "$probe_output_dir" && pwd)"

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET

# Each invocation is bounded independently; no environment installation or
# model download consumes the limited GPU preflight window.
timeout --signal=TERM --kill-after=15s 120s \
  python3 "$probe_code_dir/environment_probe.py" \
    --output "$probe_output_dir/dependencies" \
    2>&1 | tee "$probe_output_dir/dependencies.log"

timeout --signal=TERM --kill-after=15s 180s \
  python3 -m torch.distributed.run --standalone --nproc_per_node=4 \
    "$probe_code_dir/environment_probe.py" \
    --output "$probe_output_dir/gpu" --require-gpu \
    --expected-world-size 4 \
    2>&1 | tee "$probe_output_dir/gpu.log"

python3 - "$probe_output_dir" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
reports = [json.loads((root / "gpu" / f"environment-rank-{r}.json").read_text())
           for r in range(4)]
if any(not r["passed"] or r["rank"] != index or
       r["gpu_checks"]["world_size"] != 4
       for index, r in enumerate(reports)):
    raise SystemExit("Incomplete or failed four-rank preflight")
summary = {
    "scope": "single_node_environment_and_nccl_correctness",
    "passed": True,
    "ranks": 4,
    "rdma_validated": False,
    "real_grpo_validated": False,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary))
PY
