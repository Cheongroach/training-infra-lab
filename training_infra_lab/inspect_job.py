"""Read-only run -> trial -> pod -> log-proxy evidence collection."""

import argparse
import json
import subprocess
import urllib.request
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sid")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cli", default="/home/tiger/.merlin-cli/bin/merlin-cli")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    def call(group, command, body):
        output = subprocess.check_output([
            args.cli, "--control-plane", "cn", "job-v2", group, command,
            "--json", json.dumps(body),
        ], timeout=60)
        result = json.loads(output)
        if result.get("ResponseBase", {}).get("code", 0) != 0:
            raise RuntimeError(result)
        return result

    run = call("runs", "get", {"sid": args.sid})
    (args.output / "run.json").write_text(json.dumps(run, indent=2))
    data = run["data"]
    trial = data.get("runtime_context", {}).get("trial_id")
    print(json.dumps({"sid": args.sid, "status": data["status"], "trial": trial}), flush=True)
    if not trial:
        return
    pods = call("run-trial-pods", "list", {"filter": {"trial_sid_any": [trial]}})
    (args.output / "pods.json").write_text(json.dumps(pods, indent=2))
    for pod in pods["data"]:
        print(json.dumps({k: pod.get(k) for k in ("sid", "pod_name", "status", "exit_status")}), flush=True)
        for stream in ("stdout", "stderr"):
            source = call("run-trial-local-logs", "get", {"trial_pod_sid": pod["sid"], "stream": stream})
            url = source["data"].get("url")
            if not url:
                print(f"{stream}: no live log URL", flush=True)
                continue
            with urllib.request.urlopen(url, timeout=30) as response:
                text = response.read().decode(errors="replace")
            (args.output / f"{pod['sid']}.{stream}.log").write_text(text)
            # Omit entrypoint echo / source bundle from the concise terminal view.
            marker = "==================== main script start ===================="
            tail = text.split(marker)[-1].splitlines()
            events = [line for line in tail if line.startswith('{"event"')]
            errors = [line for line in tail if any(token in line for token in (
                "Traceback", "Error:", "Exception:", "AssertionError",
                "ORIGINAL_EXIT_CODE", "LAB_SUITE_COMPLETE",
            ))]
            print(f"{stream}: {len(text)} bytes", flush=True)
            selected = events[-6:] + errors[-8:] if events else tail[-8:]
            print("\n".join(line[:1600] for line in selected)[-7000:], flush=True)


if __name__ == "__main__":
    main()
