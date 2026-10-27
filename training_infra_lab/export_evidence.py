"""Export selected job evidence without platform payloads or signed log URLs."""

import argparse
import json
import re
import shutil
from pathlib import Path


def events(text):
    decoder = json.JSONDecoder()
    for line in text.splitlines():
        if not line.startswith('{"event":'):
            continue
        while line.startswith('{"event":'):
            try:
                value, end = decoder.raw_decode(line)
            except json.JSONDecodeError:
                break
            yield value
            line = line[end:].lstrip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("tags", nargs="+")
    args = p.parse_args()
    for tag in args.tags:
        src, dst = args.jobs / tag, args.output / tag
        ev = src / "evidence"
        if not (ev / "run.json").exists():
            continue
        dst.mkdir(parents=True, exist_ok=True)
        run = json.loads((ev / "run.json").read_text())["data"]
        pods = json.loads((ev / "pods.json").read_text())["data"]
        request = json.loads((src / "create.json").read_text())
        precheck = json.loads((src / "precheck-result.json").read_text())["data"]
        record = {
            "tag": tag, "control_plane": "cn", "sid": run["sid"],
            "url": "https://ml.bytedance.net/development/instance/jobs/" + run["sid"],
            "status": run["status"], "created_at": run.get("created_at"),
            "trial_id": run.get("runtime_context", {}).get("trial_id"),
            "image_vid": request["job_config"]["job_template_config"]["image_meta"]["image_vid"],
            "resource_config": request["resource_config"],
            "precheck_status": precheck["status"],
            "precheck_messages": [x for x in precheck["results"] if x["status"] != "success"],
            "log_source": "log-proxy", "pods": [],
        }
        extracted, network = [], []
        for pod in pods:
            info = {k: pod.get(k) for k in ("sid", "pod_name", "host", "gpuv", "status", "exit_status", "start_at", "stop_at")}
            text = (ev / f"{pod['sid']}.stdout.log").read_text(errors="replace")
            info["suite_complete"] = "\nLAB_SUITE_COMPLETE\n" in text
            info["artifact_copy"] = [l for l in text.splitlines() if re.match(r"ARTIFACT_COPY_RC=", l)]
            record["pods"].append(info)
            for value in events(text):
                event = value["event"]
                if "gate_passed" in event or event in ("training_environment", "optimizer_parameter_view_audit", "checkpoint_restored"):
                    extracted.append({"pod_sid": pod["sid"], **value})
            for token in ("Soft-bonding Using", "Using network", "via NET/"):
                network.extend(l for l in text.splitlines() if token in l and "NCCL INFO" in l)
        (dst / "job_evidence.json").write_text(json.dumps(record, indent=2))
        (dst / "validation_events.json").write_text(json.dumps(extracted, indent=2))
        if network:
            unique = list(dict.fromkeys(network))
            # Keep examples of each transport kind, including actual channels.
            lines = [l for l in unique if "via NET/" not in l][:8]
            lines += [l for l in unique if "via NET/" in l][:16]
            (dst / "network_evidence.txt").write_text("\n".join(lines) + "\n")
        for filename in ("source_manifest.json", "bundle.tar.gz"):
            if (src / filename).exists():
                shutil.copyfile(src / filename, dst / filename)
        print(tag, record["status"], "events", len(extracted))


if __name__ == "__main__":
    main()
