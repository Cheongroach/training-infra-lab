"""Precheck a frozen request, then explicitly submit the reviewed result."""
import argparse
import json
import subprocess
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("action", choices=("check", "submit"))
p.add_argument("directory", type=Path)
a = p.parse_args()
folder = a.directory.resolve()
cli = "/home/tiger/.merlin-cli/bin/merlin-cli"

def call(group, command, filename, output, extra=()):
    raw = subprocess.check_output([cli, "--control-plane", "cn", "job-v2", group, command,
                                   "--from-file", filename, *extra], cwd=folder, timeout=120,
                                   stderr=(folder / (output + ".stderr")).open("wb"))
    (folder / output).write_bytes(raw)
    if "--dry-run" in extra:
        return {}
    value = json.loads(raw)
    assert value.get("ResponseBase", {}).get("code", 0) == 0, value
    return value

payload = json.loads((folder / "create.json").read_text())
if a.action == "check":
    call("runs", "create", "create.json", "dry-run.json", ("--dry-run",))
    wrapped = {"node_type": "merlin_job_run", "check_point": "create", "job_run_context": payload}
    (folder / "precheck.json").write_text(json.dumps(wrapped, indent=2))
    checked = call("check-items", "check", "precheck.json", "precheck-result.json")
    print(json.dumps(checked, ensure_ascii=False), flush=True)
    assert checked["data"]["status"] != "error", "Precheck rejected"
    payload.setdefault("request_context", {})["precheck_sid"] = checked["data"]["sid"]
    (folder / "checked.json").write_text(json.dumps(payload, indent=2))
    (folder / "checked-at.json").write_text(json.dumps({"unix": time.time()}))
else:
    checked = json.loads((folder / "precheck-result.json").read_text())
    assert checked["data"]["status"] != "error"
    assert time.time() - json.loads((folder / "checked-at.json").read_text())["unix"] < 1800
    assert json.loads((folder / "precheck.json").read_text())["job_run_context"] == payload
    payload.setdefault("request_context", {})["precheck_sid"] = checked["data"]["sid"]
    assert json.loads((folder / "checked.json").read_text()) == payload
    assert not (folder / "create-result.json").exists(), "Submission already attempted"
    print(json.dumps(call("runs", "create", "checked.json", "create-result.json"), ensure_ascii=False), flush=True)
