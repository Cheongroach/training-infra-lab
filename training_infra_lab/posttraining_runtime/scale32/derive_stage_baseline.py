"""Recompute a diagnostic baseline from audited samples and exact metric identities."""

import argparse
import json
import math
from pathlib import Path


def derive(folder):
    verified = json.loads((folder / "verified.json").read_text())
    audit = json.loads((folder / "training-audit.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    if not verified.get("terminal") or not verified.get("training_verified") or not audit["passed"]:
        raise ValueError("Stage baselines require independently verified terminal training")
    events = [json.loads(line) for path in sorted((folder / "shared-results/events").rglob("*.jsonl"))
              for line in path.read_text().splitlines() if line.strip()]
    rows = []
    rollout_gpus = manifest["rollout_gpus"]
    for update in audit["updates"]:
        number = update["rollout_id"]
        rollout = [e for e in events if e.get("type") == "metric" and e.get("rollout_id") == number
                   and e.get("source", {}).get("component") == "rollout_executor"
                   and "perf/rollout_time" in e.get("metrics", {})]
        learner = [e for e in events if e.get("type") == "metric" and e.get("rollout_id") == number
                   and e.get("source", {}).get("component") == "actor"
                   and e["source"].get("rank_within_cell") == 0
                   and "perf/actor_train_time" in e.get("metrics", {})]
        if len(rollout) != 1 or len(learner) != 1:
            raise ValueError(f"Non-unique rollout/actor rank-0 metric identity: {number}")
        rm, lm = rollout[0]["metrics"], learner[0]["metrics"]
        samples, tokens = update["admitted_samples"], update["response_tokens"]
        seconds = rm["perf/rollout_time"]
        rate = tokens / seconds / rollout_gpus
        if not (rm["rollout/num_training_samples"] == samples
                and math.isclose(rm["rollout/response_len/mean"] * samples, tokens)
                and math.isclose(rm["perf/tokens_per_gpu_per_sec"], rate, rel_tol=1e-9)):
            raise ValueError(f"Audited samples do not match recorded rollout throughput: {number}")
        row = {
            "rollout_id": number, "admitted_samples": samples,
            "response_tokens": tokens, "nonzero_advantage_tokens": update["nonzero_advantage_tokens"],
            "nonzero_advantage_fraction": update["nonzero_advantage_tokens"] / tokens,
            "rollout_seconds": seconds, "response_tokens_per_rollout_gpu_second": rate,
            "actor_train_seconds_rank0": lm["perf/actor_train_time"],
            "log_probs_seconds_rank0": lm["perf/log_probs_time"],
            "train_wait_seconds_rank0": lm["perf/train_wait_time"],
            "publication_seconds_rank0": lm["perf/update_weights_time"],
            "rollout_metric_timestamp": rollout[0]["timestamp"],
            "learner_metric_timestamp": learner[0]["timestamp"],
        }
        if any(not math.isfinite(v) or v < 0 for v in row.values()
               if isinstance(v, (int, float))):
            raise ValueError(f"Invalid numeric stage observation: {number}")
        rows.append(row)
    tokens = sum(row["response_tokens"] for row in rows)
    signal = sum(row["nonzero_advantage_tokens"] for row in rows)
    samples = sum(row["admitted_samples"] for row in rows)
    gpu_hours = verified["gpu_hours"]
    result = {
        "passed": True, "rows": rows,
        "totals": {
            "admitted_samples": samples, "admitted_response_tokens": tokens,
            "nonzero_advantage_tokens": signal, "nonzero_advantage_fraction": signal / tokens,
            "terminal_pod_gpu_hours": gpu_hours,
            "admitted_samples_per_total_gpu_hour": samples / gpu_hours,
            "admitted_response_tokens_per_total_gpu_second": tokens / (gpu_hours * 3600),
            "response_tokens_per_rollout_gpu_second_pooled": tokens / (
                rollout_gpus * sum(row["rollout_seconds"] for row in rows)),
        },
        "scope": "Three-update diagnostic baseline; rank-0 learner timers are not global critical-path timers",
        "cost_scope": "All 32 terminal Pod residence times, including preparation and final evaluation",
        "token_scope": "Admitted response tokens only; no TP/DP double counting; excludes prompts and evaluation",
        "signal_scope": "Nonzero advantage tokens indicate a learning signal, not correct answers or quality improvement",
        "speedup_claim": False, "quality_improvement_claim": False,
    }
    (folder / "stage-baseline.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, required=True)
    print(json.dumps(derive(parser.parse_args().folder)["totals"]))
