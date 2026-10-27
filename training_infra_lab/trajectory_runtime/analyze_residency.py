"""Validate all ranks and paired workloads before summarizing generation speed."""
import argparse
import json
import math
import statistics
from pathlib import Path


def analyze(root, world=4):
    ranks = []
    for rank in range(world):
        rows = json.loads((root / f"rank{rank}.json").read_text())
        values = {(r["repeat"], r["residency"]["requested"]): r for r in rows}
        expected = {(repeat, enabled) for repeat in range(-1, 3) for enabled in (False, True)}
        if len(rows) != len(expected) or set(values) != expected:
            raise ValueError("Incomplete/duplicated generation comparison")
        for (repeat, enabled), row in values.items():
            if row["rank"] != rank or row["warmup"] != (repeat == -1):
                raise ValueError("Rank/warmup identity mismatch")
            if not row["exact_trajectories_and_log_probs"]:
                raise ValueError("Generation equivalence failed")
            if enabled and not row["residency"]["admitted"]:
                raise ValueError("Candidate was not actually resident")
            if not math.isfinite(row["seconds"]) or row["seconds"] <= 0:
                raise ValueError("Invalid generation timing")
            if row["max_new_tokens"] != 128:
                raise ValueError("Workload changed")
        for repeat in range(-1, 3):
            a, b = (values[(repeat, enabled)] for enabled in (False, True))
            for key in ("trajectories", "decode_steps", "wasted_decode_rows", "sampling", "reward_rule"):
                if a["details"][key] != b["details"][key]:
                    raise ValueError("Paired workload differs")
        ranks.append(values)
    pairs = []
    for repeat in range(3):
        pair = {"repeat": repeat}
        for enabled, label in ((False, "baseline"), (True, "resident")):
            rows = [r[(repeat, enabled)] for r in ranks]
            seconds = max(r["seconds"] for r in rows)
            tokens = sum(t["response_tokens"] for r in rows for t in r["details"]["trajectories"])
            pair[label] = {"seconds": seconds, "response_tokens": tokens,
                           "tokens_per_second": tokens / seconds,
                           "peak_allocated_gib": max(r["peak_allocated_bytes"] for r in rows) / 2**30}
        if pair["baseline"]["response_tokens"] != pair["resident"]["response_tokens"]:
            raise ValueError("Token denominator changed")
        pair["speedup"] = pair["baseline"]["seconds"] / pair["resident"]["seconds"]
        pairs.append(pair)
    profiles = []
    for rank in range(world):
        for enabled in (False, True):
            p = json.loads((root / f"profile-resident{enabled}-rank{rank}/operators.json").read_text())
            if p["rank"] != rank or p["identity"] != {"resident": enabled, "max_new_tokens": 16}:
                raise ValueError("Profile identity mismatch")
            profiles.append({"rank": rank, "resident": enabled,
                             "decode_steps": p["update_result"]["decode_steps"],
                             "collectives": [o for o in p["operators"] if o["name"].startswith("nccl:")]})
    return {"world": world, "paired_samples": pairs,
            "median_paired_speedup": statistics.median(p["speedup"] for p in pairs),
            "exact_sampling_passed": True, "profiles": profiles,
            "scope": "128-token generation only; 3 paired repetitions in one job; one prompt per rank",
            "excluded": "optimizer updates, startup, weights copying, warmup, independent 16-token profiles",
            "timing": "includes memory admission, first all-gather, sampling and final reshard",
            "limitations": "not full RL throughput, convergence, or independent-job replication"}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = analyze(args.directory)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
