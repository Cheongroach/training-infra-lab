"""Collect four physical-node archives and the shared diagnostic results."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from urllib import request

from collect_preflight import CLI, POD_FIELDS, TERMINAL, unpack


def collect(sid, folder):
    def query(group, command, body, name):
        response = subprocess.run(
            [CLI, "--control-plane", "cn", "job-v2", group, command, "--json", json.dumps(body)],
            check=True, capture_output=True, timeout=45)
        (folder / f"{name}.json").write_bytes(response.stdout)
        (folder / f"{name}.stderr").write_bytes(response.stderr)
        reply = json.loads(response.stdout)
        if reply.get("ResponseBase", {}).get("code", 0):
            raise RuntimeError("Platform query failed")
        if "data" in reply:
            return reply["data"]
        if group == "run-trials" and command == "exit-info":
            for section in ("trials", "pods"):
                if reply.get(section, {}).get("ResponseBase", {}).get("code", 0):
                    raise RuntimeError(f"Platform {section} query failed")
            return reply
        raise RuntimeError(f"Unexpected response shape for {group} {command}")

    run = query("runs", "get", {"sid": sid}, "current")
    trial = run["runtime_context"]["trial_id"]
    pods = query("run-trial-pods", "list", {"filter": {"trial_sid_any": [trial]}}, "pods")
    state = {
        "observed_at": datetime.now(timezone.utc).isoformat(), "run": sid, "trial": trial,
        "status": run["status"], "terminal": run["status"] in TERMINAL,
        "pods": [{k: p.get(k) for k in POD_FIELDS} for p in pods],
        "training_verified": False, "performance_claim": False,
    }
    for number, pod in enumerate(pods):
        log = query("run-trial-local-logs", "get",
                    {"trial_pod_sid": pod["sid"], "stream": "stdout"}, f"pod-{number}-local-log")
        if log.get("url"):
            with request.urlopen(log["url"], timeout=30) as response:
                (folder / f"pod-{number}-stdout.log").write_bytes(response.read())
    if not state["terminal"]:
        (folder / "status.json").write_text(json.dumps(state, indent=2))
        print(json.dumps(state), flush=True)
        return False
    query("run-trials", "exit-info", {
        "trials_list": {"filter": {"job_run_sid_any": [sid]}},
        "pods_list": {"filter": {"trial_sid_any": [trial]}},
    }, "exit-info")
    query("run-timelines", "list", {"filter": {"job_run_sid": sid}, "page_size": 100}, "timeline")
    expected = json.loads((folder / "create.checked.json").read_text())
    if run["job_config"]["job_template_config"]["entrypoint_full_script"] != \
            expected["job_config"]["job_template_config"]["entrypoint_full_script"]:
        raise ValueError("Submitted entrypoint mismatch")
    manifest = json.loads((folder / "manifest.json").read_text())
    uri = manifest["remote_output"].replace(
        "/mnt/hdfs/lab_owner", "hdfs://haruna/home/byte_search_lifeservice/zhangyang.exp", 1)
    hashes = json.loads((folder / "source-manifest.json").read_text())
    env = {**os.environ, "HADOOP_CONF_SAFE_REFRESH_RUNNING": "true", "PYTHONDONTWRITEBYTECODE": "1"}

    def fetch(remote, local, name):
        if local.exists():
            return
        with (folder / f"{name}-fetch.stderr").open("wb") as stderr:
            subprocess.run(["hdfs", "dfs", "-get", uri + "/" + remote, str(local)],
                           check=True, timeout=60, stderr=stderr, env=env)

    errors, archives, mappings = [], {}, {}
    if len(pods) != 4 or any(p.get("status") != 3 or p.get("exit_status") != 0 for p in pods):
        errors.append("Unsuccessful terminal cluster")
    for node in range(4):
        archive = folder / f"node-{node}.tar.gz"
        try:
            fetch(f"node-{node}/artifacts.tar.gz", archive, f"node-{node}")
        except subprocess.CalledProcessError:
            errors.append(f"Node archive unavailable {node}")
            continue
        dest = folder / f"node-{node}"
        archives[node] = unpack(archive, dest, ("code", "results"))
        for name, digest in hashes.items():
            if hashlib.sha256((dest / "code" / name).read_bytes()).hexdigest() != digest:
                errors.append(f"Node {node}: frozen source mismatch {name}")
        bootstrap = json.loads((dest / "results/node-bootstrap.json").read_text())
        if bootstrap["arnold_id"] != node or bootstrap["nodes"] != 4:
            errors.append(f"Node bootstrap mismatch {node}")
        entry = (dest / "results/entry.log").read_bytes()
        matches = [p.name for p in folder.glob("pod-*-stdout.log")
                   if entry and p.read_bytes().count(entry) == 1]
        mappings[node] = matches
        if len(matches) != 1:
            errors.append(f"Node {node}: entry does not match one actual Pod stdout")
        if (dest / "results/original-exit-code.txt").read_text().strip() != "0":
            errors.append(f"Node {node}: original entry failed")
        if not (dest / "results/node-completed.json").is_file():
            errors.append(f"Node {node}: driver did not complete")
    if len({m[0] for m in mappings.values() if len(m) == 1}) != 4:
        errors.append("Physical-node archives do not map to four distinct actual Pod logs")
    shared = folder / "shared-results"
    if not shared.exists():
        incomplete = folder / "shared-results.incomplete"
        if incomplete.exists():
            shutil.rmtree(incomplete)
        try:
            fetch("results", incomplete, "shared-results")
            incomplete.rename(shared)
        except subprocess.CalledProcessError:
            errors.append("Shared diagnostic results unavailable")
    audit_path = folder / "training-audit.json"
    if shared.is_dir() and (folder / "node-0/code").is_dir():
        audit_command = subprocess.run([
            "python3", str(Path(__file__).with_name("audit_train32.py")),
            "--results", str(shared), "--code", str(folder / "node-0/code"),
            "--source-manifest", str(folder / "source-manifest.json"),
            "--output", str(audit_path)], capture_output=True, timeout=120)
        (folder / "audit.stdout").write_bytes(audit_command.stdout)
        (folder / "audit.stderr").write_bytes(audit_command.stderr)
        if audit_path.is_file():
            audit = json.loads(audit_path.read_text())
            state["training_audit"] = audit
            if not audit["passed"]:
                errors.append("Independent semantic training audit failed")
        else:
            errors.append("Independent semantic training audit did not complete")
        if manifest.get("performance_run"):
            benchmark_path = folder / "publication-benchmark-audit.json"
            benchmark_command = subprocess.run([
                "python3", str(Path(__file__).with_name("audit_publication_benchmark.py")),
                "--results", str(shared), "--output", str(benchmark_path)],
                capture_output=True, timeout=120)
            (folder / "benchmark-audit.stdout").write_bytes(benchmark_command.stdout)
            (folder / "benchmark-audit.stderr").write_bytes(benchmark_command.stderr)
            if benchmark_path.is_file():
                state["publication_benchmark"] = json.loads(benchmark_path.read_text())
                if not state["publication_benchmark"]["passed"]:
                    errors.append("Independent publication benchmark audit failed")
            else:
                errors.append("Publication benchmark audit did not complete")
    else:
        errors.append("Insufficient artifacts for semantic audit")
    def dt(value):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    cost = sum((dt(p["stop_at"]) - dt(p["start_at"])).total_seconds() * 8 / 3600
               for p in pods if p.get("start_at") and p.get("stop_at"))
    state.update(errors=errors, training_verified=not errors, node_archives=archives,
                 entry_log_matches=mappings, gpu_hours=cost,
                 cost_scope="Terminal Pod residence including initialization and conversion")
    (folder / "verified.json").write_text(json.dumps(state, indent=2))
    (folder / "status.json").write_text(json.dumps(state, indent=2))
    print(json.dumps(state), flush=True)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    deadline = time.monotonic() + 3 * 3600
    while True:
        try:
            if collect(args.run, args.folder) or not args.watch:
                return
        except Exception as exc:
            print(json.dumps({"collection_error": f"{type(exc).__name__}: {exc}"}), flush=True)
            if not args.watch:
                raise
        if time.monotonic() >= deadline:
            return
        time.sleep(45)


if __name__ == "__main__":
    main()
