"""Collect one submitted four-node probe and verify all 32 rank artifacts."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import time
from urllib import request

CLI = "/home/tiger/.merlin-cli/bin/merlin-cli"
TERMINAL = {"done", "success", "failed", "stopped", "failed_to_launch"}
POD_FIELDS = ("sid", "pod_name", "status", "exit_status", "schedule_at", "start_at", "stop_at")


def unpack(archive, dest, allowed_roots=("results", "environment_probe.py")):
    dest.mkdir(exist_ok=True)
    names, total = set(), 0
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            path = PurePosixPath(member.name)
            if (path.is_absolute() or ".." in path.parts or member.name in names
                    or not (member.isfile() or member.isdir())):
                raise ValueError("Unexpected archive member")
            if path.parts[0] not in allowed_roots:
                raise ValueError("Unexpected archive root")
            names.add(member.name)
            total += member.size
            if len(names) > 4096 or total > 256 * 1024**2:
                raise ValueError("Archive exceeds probe bounds")
            output = dest / member.name
            if member.isdir():
                output.mkdir(parents=True, exist_ok=True)
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                with handle.extractfile(member) as src, output.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    return {"members": len(names), "uncompressed_bytes": total,
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}


def collect(run_sid, folder):
    def query(group, command, body, name):
        result = subprocess.run(
            [CLI, "--control-plane", "cn", "job-v2", group, command, "--json", json.dumps(body)],
            capture_output=True, timeout=45, check=True,
        )
        (folder / (name + ".json")).write_bytes(result.stdout)
        (folder / (name + ".stderr")).write_bytes(result.stderr)
        reply = json.loads(result.stdout)
        if reply.get("ResponseBase", {}).get("code", 0):
            raise RuntimeError("Platform query failed")
        return reply

    run = query("runs", "get", {"sid": run_sid}, "current")["data"]
    trial = run["runtime_context"]["trial_id"]
    pods = query("run-trial-pods", "list", {"filter": {"trial_sid_any": [trial]}}, "pods")["data"]
    state = {"observed_at": datetime.now(timezone.utc).isoformat(), "run_sid": run_sid,
             "trial_sid": trial, "status": run["status"],
             "trial_status": run["runtime_context"]["trial_status"],
             "pods": [{k: p.get(k) for k in POD_FIELDS} for p in pods],
             "terminal": run["status"] in TERMINAL, "verified": False}
    for pod in pods:
        node = int(pod["pod_name"].rsplit("-", 1)[1])
        logs = query("run-trial-local-logs", "get",
                     {"trial_pod_sid": pod["sid"], "stream": "stdout"}, f"node-{node}-local-log")
        url = logs.get("data", {}).get("url")
        if url:
            with request.urlopen(url, timeout=30) as response:
                blob = response.read()
            (folder / f"node-{node}-stdout.log").write_bytes(blob)
    if not state["terminal"]:
        (folder / "status.json").write_text(json.dumps(state, indent=2))
        print(json.dumps(state), flush=True)
        return False
    query("run-trials", "exit-info", {
        "trials_list": {"filter": {"job_run_sid_any": [run_sid]}},
        "pods_list": {"filter": {"trial_sid_any": [trial]}},
    }, "exit-info")
    query("run-timelines", "list", {"filter": {"job_run_sid": run_sid}, "page_size": 100}, "timeline")
    manifest = json.loads((folder / "manifest.json").read_text())
    expected = json.loads((folder / "create.checked.json").read_text())
    actual_entry = run["job_config"]["job_template_config"]["entrypoint_full_script"]
    expected_entry = expected["job_config"]["job_template_config"]["entrypoint_full_script"]
    if actual_entry != expected_entry:
        raise ValueError("Submitted entrypoint mismatch")
    uri = manifest["remote_output"].replace(
        "/mnt/hdfs/lab_owner", "hdfs://haruna/home/byte_search_lifeservice/zhangyang.exp", 1)
    errors, rank_reports, node_reports, entry_matches = [], {}, {}, {}
    gd_rdma_logs = 0
    cost_s = 0
    if len(pods) != 4:
        errors.append("Expected four Pod records")
    for pod in pods:
        node = int(pod["pod_name"].rsplit("-", 1)[1])
        archive = folder / f"node-{node}.tar.gz"
        if not archive.exists():
            with (folder / f"node-{node}-fetch.stderr").open("wb") as stderr:
                subprocess.run(["hdfs", "dfs", "-get", f"{uri}/node-{node}/artifacts.tar.gz", str(archive)],
                               check=True, timeout=60, stderr=stderr,
                               env={**os.environ, "HADOOP_CONF_SAFE_REFRESH_RUNNING": "true",
                                    "PYTHONDONTWRITEBYTECODE": "1"})
        destination = folder / f"node-{node}"
        node_reports[node] = unpack(archive, destination)
        if hashlib.sha256((destination / "environment_probe.py").read_bytes()).hexdigest() != manifest["source_sha256"]:
            errors.append(f"Node {node}: source hash mismatch")
        bootstrap = json.loads((destination / "results/bootstrap.json").read_text())
        if bootstrap["node"] != node or bootstrap["nodes"] != 4:
            errors.append(f"Node {node}: bootstrap identity mismatch")
        # The platform log query order can differ from ARNOLD_ID. Reconcile by
        # exact entry content, rather than assuming the downloaded log label.
        entry = (destination / "results/entry.log").read_bytes()
        matches = [
            path.name for path in folder.glob("node-*-stdout.log")
            if entry and path.read_bytes().count(entry) == 1
        ]
        entry_matches[node] = matches
        if len(matches) != 1:
            errors.append(f"Node {node}: entry log does not match one actual Pod stdout")
        gd_rdma_logs += sum(
            "via NET/soft_bonding_v8/" in path.read_text() and "/GDRDMA" in path.read_text()
            for path in (destination / "results").glob("nccl.*.log")
        )
        if (destination / "results/original-exit-code.txt").read_text().strip() != "0":
            errors.append(f"Node {node}: entrypoint failed")
        for path in (destination / "results").glob("environment-rank-*.json"):
            report = json.loads(path.read_text())
            rank = report["rank"]
            if rank in rank_reports or rank // 8 != node:
                errors.append(f"Node {node}: duplicate or foreign rank {rank}")
            rank_reports[rank] = report
            expected_checks = {"world_size": 32, "nccl_sum": 528.0, "bf16_loss": 128.0,
                               "gradient_matches_expected": True}
            if not report.get("passed") or report.get("gpu_checks") != expected_checks:
                errors.append(f"Rank {rank}: incomplete or failed GPU checks")
        if pod.get("start_at") and pod.get("stop_at"):
            def dt(value):
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            cost_s += (dt(pod["stop_at"]) - dt(pod["start_at"])).total_seconds()
        if pod.get("status") != 3 or pod.get("exit_status") != 0:
            errors.append(f"Node {node}: unsuccessful terminal Pod")
    if set(rank_reports) != set(range(32)):
        errors.append("Missing rank reports")
    if len({matches[0] for matches in entry_matches.values() if len(matches) == 1}) != 4:
        errors.append("Node entry logs do not map to four distinct Pod stdout streams")
    state.update(verified=not errors, errors=errors, rank_count=len(rank_reports),
                 node_archives=node_reports, gpu_hours=cost_s * 8 / 3600,
                 entry_log_matches=entry_matches, gdrdma_channel_log_files=gd_rdma_logs,
                 cost_scope="Sum of terminal Pod residence, including preparation",
                 scope="Environment, NCCL and BF16 preflight only; no model training",
                 rdma_performance_claim=False, training_claim=False)
    (folder / "status.json").write_text(json.dumps(state, indent=2))
    (folder / "verified.json").write_text(json.dumps(state, indent=2))
    print(json.dumps(state), flush=True)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    deadline = time.monotonic() + (2 * 3600 if args.watch else 0)
    while True:
        try:
            if collect(args.run, args.folder) or not args.watch:
                break
        except Exception as exc:
            print(json.dumps({"collection_error": f"{type(exc).__name__}: {exc}"}), flush=True)
            if not args.watch:
                raise
        if time.monotonic() >= deadline:
            break
        time.sleep(45)


if __name__ == "__main__":
    main()
