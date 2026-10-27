"""Bounded preflight for Miles: separate dependency checks from GPU validation.

Run under torchrun with --require-gpu for a real NCCL/BF16 check. This does not
load model weights, generate trajectories, or claim an RDMA performance result.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import timedelta


MODULES = ("torch", "transformers", "sglang", "ray", "megatron.core",
           "transformer_engine.pytorch", "flash_attn", "flash_attn_interface", "miles")


def command(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=15)
        return {"returncode": result.returncode, "stdout": result.stdout[-12000:],
                "stderr": result.stderr[-3000:]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": f"{type(error).__name__}: {error}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--expected-world-size", type=int, default=4)
    parser.add_argument("--expected-miles-commit",
                        default="2233d795dcf167be489d1859971c9233393f04c3")
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    report = {"scope": "environment_only", "rank": rank,
              "python": sys.version, "uid": os.getuid(), "modules": {}, "errors": []}
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / f"environment-rank-{rank}.json"
    def progress(stage):
        report["stage"] = stage
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(output)
        print(json.dumps({"event": "probe_progress", "rank": rank, "stage": stage}), flush=True)

    progress("started")
    torch = None
    try:
        for name in MODULES:
            progress("import:" + name)
            try:
                module = importlib.import_module(name)
                path = getattr(module, "__file__", None)
                if name == "torch":
                    torch = module
                    if args.require_gpu:
                        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
                version = getattr(module, "__version__", None)
                if version is None:
                    try:
                        version = importlib.metadata.version(name.split(".")[0])
                    except importlib.metadata.PackageNotFoundError:
                        pass
                entry = {"path": path, "version": str(version)}
                # Record the owning checkout, when present, without dumping env.
                if path:
                    git = command(["git", "-C", str(Path(path).parent),
                                   "rev-parse", "HEAD"])
                    entry["git"] = git
                if name == "miles":
                    found = entry.get("git", {}).get("stdout", "").strip()
                    if found != args.expected_miles_commit:
                        report["errors"].append("Miles checkout revision missing or mismatched")
                report["modules"][name] = entry
                if name == "torch":
                    torch = module
            except Exception as error:
                report["modules"][name] = {"error": f"{type(error).__name__}: {error}"}
                report["errors"].append(f"Cannot import {name}")
        progress("device_metadata")
        report["nvidia_smi"] = command([
            "nvidia-smi", "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader"])
        report["topology"] = command(["nvidia-smi", "topo", "-m"])
        report["rdma_devices"] = command(["ibv_devinfo", "-l"])
        report["rdma_note"] = "Device presence is not proof of NCCL RDMA use."
        if args.require_gpu:
            if torch is None:
                raise RuntimeError("PyTorch unavailable")
            if report["errors"]:
                raise RuntimeError("Dependency checks failed before GPU collectives")
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")
            world = int(os.environ.get("WORLD_SIZE", "1"))
            if world != args.expected_world_size:
                raise RuntimeError(f"Expected {args.expected_world_size} ranks, got {world}")
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            report["cuda"] = {
                "runtime": torch.version.cuda,
                "nccl": torch.cuda.nccl.version(),
                "device": torch.cuda.get_device_name(local_rank),
                "capability": list(torch.cuda.get_device_capability(local_rank)),
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
            if "H20" not in report["cuda"]["device"]:
                raise RuntimeError("This experiment requires H20 devices")
            if not report["cuda"]["bf16_supported"]:
                raise RuntimeError("BF16 is not supported on the selected device")
            progress("nccl_init")
            torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=60))
            value = torch.tensor([rank + 1.0], device="cuda", dtype=torch.float32)
            torch.distributed.all_reduce(value)
            expected = world * (world + 1) / 2
            if value.item() != expected:
                raise RuntimeError("NCCL all-reduce produced the wrong sum")
            # Actual forward/backward, not just a device-allocation probe.
            x = torch.ones((128, 128), device="cuda", dtype=torch.bfloat16,
                           requires_grad=True)
            loss = (x @ x).float().mean()
            loss.backward()
            expected_grad = torch.full_like(x, 2 / 128)
            if loss.item() != 128 or not torch.equal(x.grad, expected_grad):
                raise RuntimeError("BF16 matmul/backward check failed")
            torch.cuda.synchronize()
            report["gpu_checks"] = {
                "nccl_sum": value.item(), "world_size": world,
                "bf16_loss": loss.item(), "gradient_matches_expected": True,
            }
            # Clean shutdown on success only. On failure torchrun terminates
            # peer ranks, rather than waiting in a mismatched cleanup collective.
            torch.distributed.destroy_process_group()
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    report["stage"] = "finished"
    report["passed"] = not report["errors"]
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"event": "environment_probe", "rank": rank,
                      "passed": report["passed"], "output": str(output),
                      "errors": report["errors"]}), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
