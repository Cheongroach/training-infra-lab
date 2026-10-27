"""Collect completed worker artifacts and bind them to the submitted source."""

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def collect(job, archive, output):
    run = json.loads((job / "evidence/run.json").read_text())["data"]
    pods = json.loads((job / "evidence/pods.json").read_text())["data"]
    if run["status"] not in ("done", "success", "failed", "stopped"):
        raise ValueError("Collect only terminal jobs; live archives may be incomplete")
    if not pods or any(not p.get("stop_at") for p in pods):
        raise ValueError("Missing terminal pod evidence")
    manifest = json.loads((job / "source_manifest.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    # Keep the original archive as evidence; never execute its contents.
    shutil.copyfile(archive, output / "artifacts.tar.gz")
    with tarfile.open(output / "artifacts.tar.gz", "r:gz") as tar:
        for member in tar.getmembers():
            relative = Path(member.name)
            if (relative.is_absolute() or ".." in relative.parts or
                    not relative.parts or relative.parts[0] not in ("code", "results")):
                raise ValueError(f"Unexpected archive member: {member.name}")
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"Nonregular archive member: {member.name}")
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target)
    for name, expected in manifest.items():
        if sha256(output / "code" / name) != expected:
            raise ValueError(f"Submitted/worker source mismatch: {name}")
    actual_manifest = json.loads((output / "results/source_manifest.json").read_text())
    if actual_manifest != manifest:
        raise ValueError("Submitted/worker manifest mismatch")
    request = json.loads((job / "create.json").read_text())
    precheck = json.loads((job / "precheck-result.json").read_text())["data"]
    record = {
        "control_plane": "cn", "run": run["sid"],
        "url": "https://ml.bytedance.net/development/instance/jobs/" + run["sid"],
        "trial": run.get("runtime_context", {}).get("trial_id"),
        "status": run["status"],
        "pods": [{k: p.get(k) for k in (
            "sid", "pod_name", "status", "exit_status", "start_at", "stop_at",
        )} for p in pods],
        "image_vid": request["job_config"]["job_template_config"]["image_meta"]["image_vid"],
        "resource_config": request["resource_config"],
        "archive_sha256": sha256(output / "artifacts.tar.gz"),
        "bundle_sha256": sha256(job / "bundle.tar.gz"),
        "submitted_source_verified": True,
        "precheck_status": precheck["status"],
        "precheck_messages": [x for x in precheck["results"] if x["status"] != "success"],
        "log_source": "log-proxy and worker artifact archive",
    }
    (output / "job_evidence.json").write_text(json.dumps(record, indent=2))
    shutil.copyfile(job / "source_manifest.json", output / "source_manifest.json")
    print(json.dumps(record, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collect(args.job, args.archive, args.output)


if __name__ == "__main__":
    main()
