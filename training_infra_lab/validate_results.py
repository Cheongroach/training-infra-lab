"""Audit published experiment evidence; not an additional training benchmark."""
import hashlib
import json
import math
import tarfile
from pathlib import Path

root = Path(__file__).parent / "results"
rows = json.loads((root / "all_results.json").read_text())
tags = ["h20-benchmark-v1", "npu-benchmark-v1", "h20-scale2-v1", "h20-scale8-v1",
        "h20-scale16-v2", "h20-recovery-v2", "npu-recovery-v3", "h20-sync-v1", "npu-sync-v1"]
checks = []
for tag in tags:
    record = json.loads((root / tag / "job_evidence.json").read_text())
    assert record["status"] == "done", (tag, record["status"])
    assert record["pods"] and all(p["exit_status"] == 0 and p["suite_complete"] for p in record["pods"]), tag
    events = json.loads((root / tag / "validation_events.json").read_text())
    env = [e for e in events if e["event"] == "training_environment"]
    assert env and all(e["parameters"] == e["trainable_parameters"] == 1543714304 for e in env)
    manifest_hash = hashlib.sha256((root / "assets_manifest.json").read_bytes()).hexdigest()
    assert all(e["assets_manifest_sha256"] == manifest_hash for e in env)
    source = json.loads((root / tag / "source_manifest.json").read_text())
    with tarfile.open(root / tag / "bundle.tar.gz") as tar:
        for name, digest in source.items():
            assert hashlib.sha256(tar.extractfile(name).read()).hexdigest() == digest, (tag, name)
    checks.append(f"{tag}: terminal success, full parameters, assets, source hashes")

for row in rows:
    measured = [s for s in row["steps"] if s["measured"]]
    duration = sum(s["seconds"] for s in measured)
    assert len(measured) == row["measured_steps"]
    assert not any(s.get("profiled") for s in measured)
    assert math.isclose(sum(s["nonpad_tokens"] for s in measured) / duration, row["nonpad_tokens_per_second"])
    assert math.isclose(sum(s["supervised_tokens"] for s in measured) / duration, row["supervised_tokens_per_second"])
    assert all(math.isfinite(s["loss"]) and s["seconds"] > 0 for s in row["steps"])
checks.append(f"{len(rows)} result records: finite losses, timing and throughput recomputed")

for scope in (["h20-benchmark-v1", "npu-benchmark-v1", "h20-sync-v1", "npu-sync-v1"],
              ["h20-scale2-v1", "h20-scale8-v1", "h20-scale16-v2"]):
    selected = [r for r in rows if r["tag"] in scope and not any(s.get("profiled") for s in r["steps"])]
    signatures = {tuple((s["step"], s["measured"], s["nonpad_tokens"], s["supervised_tokens"]) for s in r["steps"]) for r in selected}
    assert len(signatures) == 1, scope
    checks.append(f"{scope}: identical step/token budgets")

for tag in ("h20-recovery-v2", "npu-recovery-v3"):
    folder = root / tag / "results"
    a, b, c = [json.loads((folder / phase / "result.json").read_text()) for phase in ("continuous", "split", "resumed")]
    assert b["state_fingerprints"] == c["start_fingerprints"]
    assert a["state_fingerprints"] == c["state_fingerprints"]
    assert a["eval_final"] == c["eval_final"]
    assert [s["loss"] for s in a["steps"]] == [s["loss"] for s in b["steps"] + c["steps"]]
    assert c["start_step"] == 3 and c["next_step"] == 6
    gate = json.loads((folder / "recovery_gate.json").read_text())
    assert gate["loaded_state_byte_identical"] and gate["parameter_and_adam_byte_identical"]
    checks.append(f"{tag}: loaded/final parameter+Adam bytes and all step losses equal")

record = json.loads((root / "h20-scale16-v2" / "job_evidence.json").read_text())
assert len({p["host"] for p in record["pods"]}) == 2
assert "GDRDMA" in (root / "h20-scale16-v2" / "network_evidence.txt").read_text()
checks.append("16 H20: two physical hosts; GDRDMA channel evidence present")
output = {"status": "passed", "checks": checks}
(root / "evidence_audit.json").write_text(json.dumps(output, indent=2))
print(json.dumps(output, indent=2))
