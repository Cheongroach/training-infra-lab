"""Audit completed live updates without treating reward or truncation as learning."""

import argparse
import json
import math
from pathlib import Path

from .contracts import UpdateBatch
from .rewards import LEGACY_RULE, STRICT_RULE, score_answer


def load(path):
    return json.loads(path.read_text())


def analyze(root):
    environment = load(root / "environment-rank0.json")
    world, updates = environment["world"], environment["updates"]
    rule = environment.get("reward_rule", LEGACY_RULE)
    answers = {}
    if rule == STRICT_RULE:
        data = root.parents[1] / "code/data.jsonl"
        answers = {r["source_id"]: r["answer"] for r in
                   (json.loads(line) for line in data.read_text().splitlines())}
    histories = [load(root / f"history-rank{rank}.json") for rank in range(world)]
    if any([h["update"] for h in history] != list(range(updates)) for history in histories):
        raise ValueError("Incomplete live update history")
    reports, observed_ids = [], set()
    for update in range(updates):
        batch = UpdateBatch.from_dict(load(root / f"batch-{update}.json"))
        if batch.update_id != update or batch.policy_version != update:
            raise ValueError("Incorrect live policy version")
        trajectories = {t.trajectory_id: t for t in batch.trajectories}
        if observed_ids.intersection(trajectories):
            raise ValueError("Trajectory reused across live updates")
        observed_ids.update(trajectories)
        details, phase_seconds = {}, []
        for rank in range(world):
            groups = load(root / f"rollout-{update}-rank{rank}.json")
            if len(groups) != environment["groups_per_rank"]:
                raise ValueError("Missing rollout groups")
            phase_seconds.append(sum(g["phase_seconds"] for g in groups))
            for group in groups:
                if environment["resident_rollout"] and not group["residency"]["admitted"]:
                    raise ValueError("Live residency was not admitted")
                for row in group["trajectories"]:
                    key = row["trajectory_id"]
                    if key in details:
                        raise ValueError("Duplicate rollout evidence")
                    details[key] = row
        if details.keys() != trajectories.keys():
            raise ValueError("Rollout and update trajectory identities differ")
        counts = {"eos_correct": 0, "eos_wrong": 0, "length_correct": 0, "length_wrong": 0}
        outcomes = {key: 0 for key in ("correct", "incorrect", "missing_final_answer", "truncated")}
        for key, trajectory in trajectories.items():
            row = details[key]
            if rule == STRICT_RULE:
                scored = score_answer(row["text"], answers[trajectory.prompt_group_id],
                                      row["finish_reason"], rule)
                if any(row[key] != value for key, value in scored.items()):
                    raise ValueError("Saved reward differs from independently recomputed outcome")
                outcomes[row["reward_outcome"]] += 1
            if (row["reward"] != trajectory.reward or
                    row["response_tokens"] != sum(trajectory.response_mask) or
                    row["response_tokens"] + row["prompt_tokens"] != len(trajectory.input_ids)):
                raise ValueError("Rollout and update workload differs")
            if row["reward"] not in (0, 1) or row["finish_reason"] not in ("eos", "length"):
                raise ValueError("Unexpected reward or termination")
            counts[row["finish_reason"] + ("_correct" if row["reward"] else "_wrong")] += 1
        advantages = batch.advantages()
        signal_groups = {t.prompt_group_id for t, a in zip(batch.trajectories, advantages) if a}
        completed_wrong_signal_groups = {
            t.prompt_group_id for t, a in zip(batch.trajectories, advantages)
            if a and details[t.trajectory_id]["finish_reason"] == "eos" and not t.reward
            and (rule != STRICT_RULE or details[t.trajectory_id]["reward_outcome"] == "incorrect")
        }
        rank_steps = []
        for rank in range(world):
            history = histories[rank][update]
            diagnostics = load(root / f"batch-diagnostics-{update}-rank{rank}.json")
            if (history["batch_sha256"] != batch.sha256 or
                    diagnostics["batch_sha256"] != batch.sha256 or
                    diagnostics["signal_groups"] != len(signal_groups)):
                raise ValueError("Live update identity or signal diagnostic differs")
            probabilities = diagnostics["probability_replay"]
            for key, limit in (("rms", .05), ("p99_abs", .2), ("outside_clip_fraction", .01)):
                if not math.isfinite(probabilities[key]) or not 0 <= probabilities[key] <= limit:
                    raise ValueError("Behavior probability gate failed")
            result = history["update_result"]
            if (result["rank"] != rank or result["update_id"] != update or
                    result["batch_sha256"] != batch.sha256):
                raise ValueError("Optimizer update identity differs")
            if not math.isfinite(result["grad_norm"]) or result["grad_norm"] < 0:
                raise ValueError("Invalid live gradient norm")
            rank_steps.append(result)
        moments = [history[update].get("optimizer_nonzero_counts") for history in histories]
        if any(m != moments[0] for m in moments):
            raise ValueError("Ranks disagree on global Adam moment counts")
        reports.append({
            "update": update, "policy_version": batch.policy_version, "batch_sha256": batch.sha256,
            "trajectories": len(trajectories), "groups": len(batch.trajectories) // batch.group_size,
            "response_tokens": sum(sum(t.response_mask) for t in batch.trajectories),
            "mean_reward": sum(t.reward for t in batch.trajectories) / len(trajectories),
            "signal_groups": len(signal_groups),
            "signal_groups_with_completed_wrong_answer": len(completed_wrong_signal_groups),
            "termination_and_reward": counts,
            "reward_outcomes": outcomes if rule == STRICT_RULE else None,
            "optimizer_nonzero_counts": moments[0],
            "gradient_norm_range": [min(r["grad_norm"] for r in rank_steps),
                                    max(r["grad_norm"] for r in rank_steps)],
            "nonzero_live_gradient_all_ranks": all(r["grad_norm"] > 0 for r in rank_steps),
            "generation_phase_seconds_max_rank": max(phase_seconds),
            "optimizer_update_seconds_max_rank": max(r["total_seconds"] for r in rank_steps),
            "peak_update_allocated_gib": max(r["peak_allocated_bytes"] for r in rank_steps) / 2**30,
        })
    return {
        "world": world, "completed_updates": updates, "updates": reports,
        "reward_rule": rule,
        "direct_adam_evidence_saved": all(r["optimizer_nonzero_counts"] is not None for r in reports),
        "all_updates_have_nonzero_gradients": all(r["nonzero_live_gradient_all_ranks"] for r in reports),
        "all_updates_have_completed_wrong_signal": all(
            r["signal_groups_with_completed_wrong_answer"] > 0 for r in reports),
        "scope": "Short synchronous execution audit; different prompts at each update",
        "limitations": [
            "Reward uses the declared numeric rule, not full mathematical equivalence",
            "Reported phases exclude probability replay and diagnostic experiments; not full RL throughput",
            "Adam counts, when present, measure nonzero elements; absolute moment norms were not saved",
            "No held-out evaluation or convergence claim",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.directory)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
