"""Audit a returned probe archive without importing the GPU runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

MILES_SHA = "2233d795dcf167be489d1859971c9233393f04c3"


def audit(archive, manifest):
    errors = []
    ranks = []
    with tarfile.open(archive, "r:gz") as bundle:
        def read(name):
            member = bundle.getmember(name)
            if not member.isfile() or member.size > 10_000_000:
                raise ValueError(f"Invalid evidence member: {name}")
            return bundle.extractfile(member).read()

        def load(name):
            return json.loads(read(name))

        try:
            if load("results/source-manifest.json") != manifest:
                errors.append("Returned source manifest differs from submitted manifest")
            for name, digest in manifest.items():
                if hashlib.sha256(read("code/" + name)).hexdigest() != digest:
                    errors.append("Source hash mismatch: " + name)
            summary = load("results/summary.json")
            if summary != {
                "scope": "single_node_environment_and_nccl_correctness",
                "passed": True, "ranks": 4,
                "rdma_validated": False, "real_grpo_validated": False,
            }:
                errors.append("Unexpected completion summary")
            dependency = load("results/dependencies/environment-rank-0.json")
            if dependency.get("passed") is not True or dependency.get("errors"):
                errors.append("Dependency stage did not pass")
            for rank in range(4):
                report = load(f"results/gpu/environment-rank-{rank}.json")
                checks = report.get("gpu_checks", {})
                cuda = report.get("cuda", {})
                modules = report.get("modules", {})
                if report.get("rank") != rank or report.get("passed") is not True or report.get("errors"):
                    errors.append(f"Rank {rank} did not pass")
                if checks != {"nccl_sum": 10.0, "world_size": 4,
                              "bf16_loss": 128.0, "gradient_matches_expected": True}:
                    errors.append(f"Rank {rank} failed numerical contract")
                if "H20" not in cuda.get("device", "") or cuda.get("bf16_supported") is not True:
                    errors.append(f"Rank {rank} device contract mismatch")
                revision = modules.get("miles", {}).get("git", {}).get("stdout", "").strip()
                if revision != MILES_SHA:
                    errors.append(f"Rank {rank} Miles revision mismatch")
                if any("error" in entry for entry in modules.values()) or len(modules) != 9:
                    errors.append(f"Rank {rank} has incomplete dependencies")
                ranks.append({"rank": rank, "device": cuda.get("device"),
                              "cuda": cuda.get("runtime"), "nccl": cuda.get("nccl"),
                              "python": report.get("python"), "uid": report.get("uid"),
                              "modules": modules, "gpu_checks": checks})
        except (KeyError, ValueError, OSError) as error:
            errors.append(f"Incomplete archive: {error}")
    return {"passed": not errors, "scope": "single_node_environment_and_nccl_correctness",
            "errors": errors, "ranks": ranks, "rdma_validated": False,
            "real_grpo_validated": False,
            "archive_sha256": hashlib.sha256(Path(archive).read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.archive, json.loads(args.manifest.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "ranks"}))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
