"""Recompute paired actor-replay metrics, rejecting partial/unaligned evidence."""
import argparse
import json
import math
import statistics
from pathlib import Path

from .contracts import UpdateBatch
from .planner import Plan


def validate_record(record, batch, world):
    plan = Plan(**record["plan"])
    plan.validate(batch)
    if (plan.world_size != world or plan.sha256 != record["plan_sha256"] or
            plan.strategy != record["strategy"] or record["batch_sha256"] != batch.sha256):
        raise ValueError("Measurement does not match its batch/plan")
    if (record["input_tokens"] != sum(len(t.input_ids) for t in batch.trajectories) or
            record["response_tokens"] != sum(sum(t.response_mask) for t in batch.trajectories)):
        raise ValueError("Incorrect throughput token count")
    for field in ("seconds", "total_seconds", "peak_allocated_bytes"):
        if not math.isfinite(record[field]) or record[field] <= 0:
            raise ValueError(f"Invalid measurement: {field}")
    if record.get("reference") is True:
        return
    if record.get("numerically_accepted") is not True:
        raise ValueError("Numerical gate failed or is missing")
    errors = record["errors"]
    limits = [errors[k]["relative_l2"] for k in ("gradient", "exp_avg", "exp_avg_sq")]
    if (any(not math.isfinite(x) or not 0 <= x <= 0.08 for x in limits) or
            not 0 <= errors["parameter"]["max_abs"] <= 3e-6):
        raise ValueError("Stored error metrics violate the predeclared gate")


def analyze(directory):
    files = sorted(directory.glob("replay-rank*.json"))
    if not files:
        raise ValueError("No replay measurements")
    environments = [json.loads((directory / f"environment-rank{r}.json").read_text())
                    for r in range(len(files))]
    world = environments[0]["world"]
    if len(files) != world or any(e["world"] != world for e in environments):
        raise ValueError("Incomplete rank evidence")
    common = [{k: v for k, v in e.items() if k != "seed_per_rank"} for e in environments]
    if any(e != common[0] for e in common):
        raise ValueError("Ranks disagree on environment")
    batch = UpdateBatch.from_dict(json.loads((directory / "batch-1.json").read_text()))
    batches = [json.loads((directory / f"batch-diagnostics-1-rank{r}.json").read_text())
               for r in range(world)]
    if any(b != batches[0] for b in batches) or batches[0]["batch_sha256"] != batch.sha256:
        raise ValueError("Ranks disagree on batch identity")
    histories = [json.loads((directory / f"history-rank{r}.json").read_text()) for r in range(world)]
    if any([h["update"] for h in history] != list(range(environments[0].get("updates", 2)))
           for history in histories):
        raise ValueError("Missing completed synchronous updates")
    signal_groups = len({t.prompt_group_id for t, a in zip(batch.trajectories, batch.advantages()) if a})
    if signal_groups != batches[0]["signal_groups"]:
        raise ValueError("Incorrect gradient-signal diagnostic")
    rows = [json.loads((directory / f"replay-rank{r}.json").read_text()) for r in range(world)]
    maps = [{(v["repeat"], v["strategy"], v["micro_batch"]): v for v in rank} for rank in rows]
    if any(len(m) != len(row) for m, row in zip(maps, rows)):
        raise ValueError("Duplicate measurement keys")
    if any(m.keys() != maps[0].keys() for m in maps):
        raise ValueError("Missing rank measurements")
    repeats = sorted({k[0] for k in maps[0]})
    choices = {(s, m) for _, s, m in maps[0]}
    micros = environments[0].get("replay_microbatches", [1, 2])
    if micros not in ([1], [1, 2]):
        raise ValueError("Unrecognized predeclared replay matrix")
    if len(repeats) < 3 or choices != {(s, m) for s in ("verl", "slot_sorted") for m in micros}:
        raise ValueError("Incomplete prescribed comparison")
    if set(maps[0]) != {(r, s, m) for r in repeats for s, m in choices}:
        raise ValueError("Incomplete repeat/configuration matrix")
    references = [{k for k, v in m.items() if v.get("reference") is True} for m in maps]
    if len(references[0]) != 1 or any(keys != references[0] for keys in references):
        raise ValueError("Reference update must be unique and aligned across ranks")
    combined = {}
    for key in maps[0]:
        records = [m[key] for m in maps]
        if any(v["batch_sha256"] != batches[0]["batch_sha256"] for v in records):
            raise ValueError("Comparison changes update input")
        if len({v["plan_sha256"] for v in records}) != 1:
            raise ValueError("Ranks disagree on execution plan")
        for record in records:
            validate_record(record, batch, world)
        combined[key] = {
            "seconds": max(v["total_seconds"] for v in records),
            "peak_allocated_gib": max(v["peak_allocated_bytes"] for v in records) / 2**30,
            "input_tokens": records[0]["input_tokens"],
            "response_tokens": records[0]["response_tokens"],
        }
    summaries = []
    for strategy, micro in sorted(choices):
        values = [combined[(r, strategy, micro)] for r in repeats]
        times = [v["seconds"] for v in values]
        summaries.append({
            "strategy": strategy, "micro_batch": micro,
            "seconds_per_repeat": times, "median_seconds": statistics.median(times),
            "min_seconds": min(times), "max_seconds": max(times),
            "median_input_tokens_per_second": statistics.median(v["input_tokens"] / v["seconds"] for v in values),
            "median_response_tokens_per_second": statistics.median(v["response_tokens"] / v["seconds"] for v in values),
            "peak_allocated_gib": max(v["peak_allocated_gib"] for v in values),
        })
    pairs = [{"micro_batch": micro, "baseline": "verl", "candidate": "slot_sorted",
              "speedups_per_repeat": [
                  combined[(r, "verl", micro)]["seconds"] / combined[(r, "slot_sorted", micro)]["seconds"]
                  for r in repeats]} for micro in micros]
    return {
        "batch_sha256": batches[0]["batch_sha256"], "world": world,
        "signal_groups": batches[0]["signal_groups"], "groups": batches[0]["groups"],
        "mean_reward": batches[0]["mean_reward"],
        "numerical_gate_passed": True,
        "useful_gradient_test": batches[0]["signal_groups"] > 0,
        "configurations": summaries, "paired_comparisons": pairs,
        "scope": "one real batch, 3 randomized repeats in one process group; not independent job trials",
        "timing": "planning, plan agreement, collation, forward/backward/clip/Adam, slot synchronization",
        "excluded": "rollout, setup, scratch copies/restoration, numerical verification, checkpoints",
        "limitations": "no packing, no full verl comparison, short responses, no convergence claim",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = analyze(args.directory)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
