"""Bounded Gloo worker failure injection; this does not validate NCCL recovery."""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import timedelta

import torch.distributed as dist

from .distributed import fail_worker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        try:
            dist.init_process_group("gloo", timeout=timedelta(seconds=60))
            dist.barrier()
            if dist.get_rank() == 1:
                raise RuntimeError("INJECTED_RANK_LOCAL_FAILURE")
            dist.barrier()
            raise AssertionError("Peer continued past the failed collective")
        except BaseException:
            fail_worker()
    start = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
         "-m", "trajectory_runtime.failure_probe", "--worker"],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True, timeout=30,
    )
    elapsed = time.monotonic() - start
    if result.returncode == 0 or "INJECTED_RANK_LOCAL_FAILURE" not in result.stderr or elapsed >= 25:
        raise RuntimeError(f"Failure propagation did not terminate promptly: {result.returncode}, {elapsed}\n"
                           + result.stderr[-6000:])
    print(json.dumps({
        "event": "worker_failure_probe_pass", "backend": "gloo", "world": 2,
        "seconds_including_process_start": elapsed, "returncode": result.returncode,
        "original_exception_visible": True, "scope": "CPU process lifecycle; not NCCL/HCCL recovery",
    }), flush=True)


if __name__ == "__main__":
    main()
