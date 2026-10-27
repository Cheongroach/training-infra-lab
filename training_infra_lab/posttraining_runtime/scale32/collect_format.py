"""Collect and independently recompute the frozen output-protocol probe."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
from urllib import request

from collect_preflight import CLI, TERMINAL, unpack
from format_contract import audit_response


def collect(sid, folder):
    def query(group, command, body, name):
        result = subprocess.run(
            [CLI, "--control-plane", "cn", "job-v2", group, command, "--json", json.dumps(body)],
            capture_output=True, check=True, timeout=45,
        )
        (folder / (name + ".json")).write_bytes(result.stdout)
        (folder / (name + ".stderr")).write_bytes(result.stderr)
        data = json.loads(result.stdout)
        if data.get("ResponseBase", {}).get("code", 0):
            raise RuntimeError("Platform query error")
        return data

    run = query("runs", "get", {"sid": sid}, "current")["data"]
    trial = run["runtime_context"]["trial_id"]
    pods = query("run-trial-pods", "list", {"filter": {"trial_sid_any": [trial]}}, "pods")["data"]
    state = {"observed_at": datetime.now(timezone.utc).isoformat(), "run": sid,
             "status": run["status"], "trial": trial, "terminal": run["status"] in TERMINAL,
             "summary": None, "audit_complete": False, "protocol_passed": False}
    for pod in pods:
        response = query("run-trial-local-logs", "get",
                         {"trial_pod_sid": pod["sid"], "stream": "stdout"}, "local-log")
        url = response.get("data", {}).get("url")
        if url:
            with request.urlopen(url, timeout=30) as reply:
                (folder / "stdout.log").write_bytes(reply.read())
    if not state["terminal"]:
        (folder / "status.json").write_text(json.dumps(state, indent=2))
        print(json.dumps(state), flush=True)
        return False
    query("run-trials", "exit-info", {
        "trials_list": {"filter": {"job_run_sid_any": [sid]}},
        "pods_list": {"filter": {"trial_sid_any": [trial]}},
    }, "exit-info")
    query("run-timelines", "list", {"filter": {"job_run_sid": sid}, "page_size": 100}, "timeline")
    if len(pods) == 1 and pods[0].get("exit_status") == 126:
        state["errors"] = ["Entrypoint exited 126 before generation; inspect stdout and stderr"]
        pod = pods[0]
        start = datetime.fromisoformat(pod["start_at"].replace("Z", "+00:00"))
        stop = datetime.fromisoformat(pod["stop_at"].replace("Z", "+00:00"))
        state["gpu_hours"] = (stop - start).total_seconds() * 4 / 3600
        (folder / "status.json").write_text(json.dumps(state, indent=2))
        (folder / "verified.json").write_text(json.dumps(state, indent=2))
        print(json.dumps(state), flush=True)
        return True
    manifest = json.loads((folder / "manifest.json").read_text())
    archive = folder / "artifacts.tar.gz"
    if not archive.exists():
        uri = manifest["remote_output"].replace(
            "/mnt/hdfs/lab_owner", "hdfs://haruna/home/byte_search_lifeservice/zhangyang.exp", 1)
        with (folder / "fetch.stderr").open("wb") as stderr:
            subprocess.run(["hdfs", "dfs", "-get", uri + "/artifacts.tar.gz", str(archive)],
                           check=True, timeout=60, stderr=stderr,
                           env={**os.environ, "HADOOP_CONF_SAFE_REFRESH_RUNNING": "true",
                                "PYTHONDONTWRITEBYTECODE": "1"})
    destination = folder / "artifacts"
    state["archive"] = unpack(archive, destination, ("results", "code"))
    hashes = json.loads((folder / "source-manifest.json").read_text())
    for name, digest in hashes.items():
        if hashlib.sha256((destination / "code" / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Source archive hash mismatch")
    expected = json.loads((folder / "create.checked.json").read_text())
    if run["job_config"]["job_template_config"]["entrypoint_full_script"] != expected["job_config"]["job_template_config"]["entrypoint_full_script"]:
        raise ValueError("Entrypoint mismatch")
    errors = []
    if len(pods) != 1 or pods[0]["status"] != 3 or pods[0]["exit_status"] != 0:
        errors.append("Unsuccessful terminal Pod")
    if (destination / "results/original-exit-code.txt").read_text().strip() != "0":
        errors.append("Entrypoint failed")
    generation = destination / "results/generation"
    if not (generation / "summary.json").is_file():
        errors.append("Generation probe did not complete")
    else:
        contract = json.loads((generation / "generation-contract.json").read_text())
        if contract["scorer_sha256"] != hashes["math_dapo_utils.py"]:
            raise ValueError("Scorer provenance mismatch")
        # Execute the local frozen scorer, after checking artifact equality.
        scorer_file = Path(__file__).resolve().parents[2] / "research/posttraining_20261022/upstream/miles/miles/rollout/rm_hub/math_dapo_utils.py"
        if hashlib.sha256(scorer_file.read_bytes()).hexdigest() != hashes["math_dapo_utils.py"]:
            raise ValueError("Local scorer revision mismatch")
        spec = importlib.util.spec_from_file_location("local_frozen_dapo", scorer_file)
        scorer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scorer)
        results = []
        for variant, index, repeat, temperature in contract["tasks"]:
            result = json.loads((generation / f"{variant}-{index}-{repeat}.json").read_text())
            if (result["variant"], result["source_index"], result["repeat"], result["temperature"]) != (variant, index, repeat, temperature):
                errors.append("Response identity mismatch")
            if result["reward"] != scorer.compute_score(result["text"], result["label"]):
                errors.append(f"Reward mismatch {variant}/{index}/{repeat}")
            assessment = audit_response(result, set(contract["eos_token_ids"]))
            if assessment != result["audit"]:
                errors.append(f"Response audit mismatch {variant}/{index}/{repeat}")
            results.append(result)
        state["summary"] = json.loads((generation / "summary.json").read_text())
        state["protocol_passed"] = not errors and all(
            r["audit"]["passed"] for r in results if r["variant"] == "explicit")
        state["responses_recomputed"] = len(results)
        state["audit_complete"] = not errors
    entry = (destination / "results/entry.log").read_bytes()
    if (folder / "stdout.log").read_bytes().count(entry) != 1:
        errors.append("Entry log does not match actual Pod stdout")
    state["errors"] = errors
    state["audit_complete"] = state["audit_complete"] and not errors
    def dt(value):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    state["gpu_hours"] = sum(
        (dt(p["stop_at"]) - dt(p["start_at"])).total_seconds() * 4 / 3600
        for p in pods if p.get("start_at") and p.get("stop_at")
    )
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
    deadline = time.monotonic() + 2 * 3600
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
