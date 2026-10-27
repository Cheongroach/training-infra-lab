"""Recompute equal-work publication timings and bind them to terminal training artifacts."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from audit_train32 import adam_moments, model_hashes
from publication_benchmark import fingerprint, schedule


def audit(results):
    root = results / "publication-benchmark"
    errors = []

    def check(value, message):
        if not value:
            errors.append(message)

    events = [json.loads(line) for p in sorted((results / "events").rglob("*.jsonl"))
              for line in p.read_text().splitlines() if line.strip()]
    learner_states = {
        event["source"]["rank_within_cell"]: event["state"]
        for event in events
        if event["type"] == "train_engine_local_weight_checksum"
        and event["rollout_id"] == 2 and event["source"]["component"] == "actor"
    }
    engine_events = [event for event in events if event["type"] == "inference_engine_weight_checksum"
                     and event["rollout_id"] == 2 and event["weight_version"] == 3]
    check(len(engine_events) == 1, "Final v3 engine event missing or duplicated")
    expected_engines = json.loads((root / "engine-reference.json").read_text())
    check(len(expected_engines) == 8 and all(state == expected_engines[0] for state in expected_engines),
          "Reference engine content agreement")
    check(all(len(value) == 64 and set(value) <= set("0123456789abcdef")
              for state in expected_engines for value in state.values()),
          "Benchmark reference must contain full SHA256 values")
    if len(engine_events) == 1:
        final_event = engine_events[0]
        snapshots = final_event["engine_snapshots"]
        final = [snapshot["tensor_checksums"] for snapshot in snapshots]
        # Miles action=checksum records tensor_hash; action=raw_checksum records SHA256.
        # Different algorithms cannot be compared as if they were equal digest strings.
        check(len(final) == 8 and all(state == final[0] for state in final),
              "Final publication engine tensor_hash agreement")
        check(all(set(state) == set(expected_engines[0]) for state in final),
              "Final publication differs from benchmark tensor-name coverage")
        publications = [event for event in events if event["type"] == "weight_update_result"
                        and event["rollout_id"] == 2 and event["published_version"] == 3]
        check(len(publications) == 1, "Final v3 publication missing or duplicated")
        if len(publications) == 1:
            publication = publications[0]
            check(not publication["failed_cell_ids"]
                  and len({snapshot["cell_id"] for snapshot in snapshots}) == 8
                  and {snapshot["cell_id"] for snapshot in snapshots}
                      == set(publication["updated_cell_ids"])
                  and final_event["debug_weight_update_id"] == publication["debug_weight_update_id"],
                  "Final event and publication engine/update identities differ")
    reports = [json.loads((root / f"rank-{rank}.json").read_text()) for rank in range(16)]
    iterations = reports[0]["iterations"]
    check(len(iterations) == len(schedule()), "Benchmark iteration coverage")
    layouts = {}
    for rank, report in enumerate(reports):
        check(report["passed"] and report["rank"] == rank and report["world_size"] == 16
              and report["weight_version"] == 3 and report["engine_count"] == 8,
              f"Rank report contract {rank}")
        check(report["iterations"] == iterations, f"Ranks disagree about timing observations {rank}")
        reference = json.loads((root / f"local-reference-rank-{rank}.json").read_text())
        if rank in learner_states:
            check(reference == learner_states[rank]
                  and model_hashes(reference) == model_hashes(learner_states[rank])
                  and adam_moments(reference) == adam_moments(learner_states[rank]),
                  f"Benchmark changed final model/Adam state {rank}")
        else:
            check(False, f"No terminal learner state for rank {rank}")
        layout = json.loads((root / f"sender-layout-rank-{rank}.json").read_text())
        layouts[rank] = layout
        for item in iterations:
            own = next(m for m in item["rank_measurements"] if m["rank"] == rank)
            check(own["local_state_sha256"] == fingerprint(reference)
                  and own["engine_state_sha256"] == fingerprint(expected_engines if rank == 0 else None),
                  f"State content changed in iteration {item['number']}, rank {rank}")
            check(own["layout_sha256"] == fingerprint(layout)
                  and own["bytes"] == sum(t["bytes"] for bucket in layout for t in bucket)
                  and own["tensors"] == sum(len(bucket) for bucket in layout)
                  and own["buckets"] == len(layout),
                  f"Unequal publication payload in iteration {item['number']}, rank {rank}")
    check(len([rank for rank, layout in layouts.items() if layout]) == 1,
          "Expected one materialized base-weight sender")
    measurements = defaultdict(list)
    transfer = defaultdict(list)
    for number, (warmup, mode, checksums) in enumerate(schedule()):
        item = iterations[number]
        check((item["number"], item["warmup"], item["mode"], item["checksums"])
              == (number, warmup, mode, checksums), f"Frozen case order {number}")
        ranks = item["rank_measurements"]
        check(len(ranks) == 16 and {m["rank"] for m in ranks} == set(range(16))
              and all(math.isfinite(m["measured_seconds"]) and m["measured_seconds"] > 0
                      and math.isfinite(m["transfer_seconds"]) and m["transfer_seconds"] > 0 for m in ranks),
              f"Finite all-rank timing {number}")
        check(item["verified"] and item["measured_seconds"] == max(m["measured_seconds"] for m in ranks)
              and item["transfer_seconds"] == max(m["transfer_seconds"] for m in ranks),
              f"Recomputed critical-rank timing {number}")
        if not warmup:
            key = f"{mode}/checksum-{int(checksums)}"
            measurements[key].append(item["measured_seconds"])
            transfer[key].append(item["transfer_seconds"])
    summary = {
        key: {"samples": len(values), "wall_seconds": values,
              "median_seconds": statistics.median(values),
              "min_seconds": min(values), "max_seconds": max(values),
              "transfer_median_seconds": statistics.median(transfer[key])}
        for key, values in measurements.items()
    }
    speedups = {
        f"packed_speedup_checksum_{int(checks)}":
            summary[f"broadcast/checksum-{int(checks)}"]["median_seconds"]
            / summary[f"broadcast_packed/checksum-{int(checks)}"]["median_seconds"]
        for checks in (False, True)
    }
    overhead = {
        mode: summary[f"{mode}/checksum-1"]["median_seconds"]
        - summary[f"{mode}/checksum-0"]["median_seconds"]
        for mode in ("broadcast", "broadcast_packed")
    }
    return {
        "passed": not errors, "errors": errors, "comparisons": summary,
        "speedup_ratios": speedups, "checksum_added_median_seconds": overhead,
        "scope": "Same final model, optimizer, sender layouts and eight engines; publication only",
        "end_to_end_training_speedup_claim": False, "quality_improvement_claim": False,
        "engine_hash_binding": {
            "benchmark_algorithm": "raw tensor SHA256",
            "benchmark_reference_and_all_iterations_equal": not errors,
            "final_event_algorithm": "SGLang tensor_hash (16 hex digits)",
            "cross_algorithm_digest_equality_claim": False,
            "binding": "Exact final learner state; same eight publication cells/update identity and tensor-name coverage; each algorithm verifies replica agreement separately",
        },
        "sender_bytes": sum(t["bytes"] for layout in layouts.values() for bucket in layout for t in bucket),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = audit(args.results)
    except Exception as exc:
        report = {"passed": False, "errors": [f"{type(exc).__name__}: {exc}"],
                  "end_to_end_training_speedup_claim": False}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report))
    raise SystemExit(0 if report["passed"] else 1)
