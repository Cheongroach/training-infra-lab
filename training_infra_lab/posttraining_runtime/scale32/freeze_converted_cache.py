"""Bind a converted cache to terminal source identity and verified preparation evidence."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folder = args.folder
    verified = json.loads((folder / "verified.json").read_text())
    audit_path = folder / "training-audit.json"
    audit = json.loads(audit_path.read_text())
    if not verified["terminal"] or len(verified["node_archives"]) != 4:
        raise ValueError("Converted cache reuse requires terminal evidence from all source nodes")
    manifest = json.loads((folder / "manifest.json").read_text())
    hashes = json.loads((folder / "source-manifest.json").read_text())
    for node in range(4):
        directory = folder / f"node-{node}"
        for name, expected in hashes.items():
            if hashlib.sha256((directory / "code" / name).read_bytes()).hexdigest() != expected:
                raise ValueError("Source node code hash mismatch")
        bootstrap = json.loads((directory / "results/node-bootstrap.json").read_text())
        assets = json.loads((directory / "results/assets-verified.json").read_text())
        if bootstrap["arnold_id"] != node or not assets["verified"]:
            raise ValueError("Source node identity or model assets mismatch")
    # Preparation can be reused after a later rollout admission failure. This
    # proves conversion/copying only, without asserting an optimizer step.
    observations = []
    for path in folder.glob("live-phase*-pod-0.json"):
        value = json.loads(path.read_text())
        control = value.get("control", {})
        if all(control.get(f"models-ready-{node}.json", {}).get("ready") for node in range(4)):
            observations.append(value)
    if not observations:
        raise ValueError("No observed all-node converted checkpoint readiness")
    observed = max(observations, key=lambda value: value["observed_unix"])
    remote = manifest["remote_output"]
    uri = remote.replace(
        "/mnt/hdfs/lab_owner", "hdfs://haruna/home/byte_search_lifeservice/zhangyang.exp", 1)
    response = subprocess.run(
        ["hdfs", "dfs", "-cat", uri + "/control/converted-manifest.json"],
        capture_output=True, check=True, timeout=60,
        env={**os.environ, "HADOOP_CONF_SAFE_REFRESH_RUNNING": "true",
             "PYTHONDONTWRITEBYTECODE": "1"},
    )
    receipt = json.loads(response.stdout)
    files = receipt["files"]
    if (not files or files["latest_checkpointed_iteration.txt"]["bytes"] != 7
            or any(Path(name).is_absolute() or ".." in Path(name).parts for name in files)
            or any(meta["bytes"] < 0 or len(meta["sha256"]) != 64 for meta in files.values())):
        raise ValueError("Invalid frozen converted checkpoint manifest")
    if observed["control"]["converted-manifest.json"]["files"] != files:
        raise ValueError("Current converted manifest differs from source runtime observation")
    cache = {
        "verified": True, "source_conversion_verified": True,
        "source_training_verified": bool(verified["training_verified"] and audit["passed"]),
        "source_run": verified["run"], "source_trial": verified["trial"],
        "source_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "source_conversion_observed_unix": observed["observed_unix"],
        "source_conversion_evidence": {
            "models_ready": {str(node): observed["control"][f"models-ready-{node}.json"]
                             for node in range(4)},
            "node_archive_sha256": {node: meta["sha256"]
                                   for node, meta in verified["node_archives"].items()},
            "entry_log_matches": verified["entry_log_matches"],
        },
        "source_manifest": hashes,
        "source_converted_manifest_sha256": hashlib.sha256(response.stdout).hexdigest(),
        "model_identity": manifest["model_identity"],
        "remote_mount": remote + "/converted-checkpoint",
        "files": files,
        "reuse_contract": "Every new node copies and rehashes all files before model readiness",
    }
    args.output.write_text(json.dumps(cache, indent=2) + "\n")
    print(json.dumps({"source_run": verified["run"], "files": len(files),
                      "bytes": sum(meta["bytes"] for meta in files.values())}))


if __name__ == "__main__":
    main()
