"""Reject unusable GRPO batches before learner conversion; preserve all samples."""
from collections import defaultdict
import json
import math
from pathlib import Path


def assess_batch(samples, *, prompts=4, answers=4, max_response=4096):
    groups = defaultdict(list)
    errors = []
    indices = set()
    completed = extracted = 0
    for sample in samples:
        groups[sample["group_index"]].append(sample)
        indices.add(sample["index"])
        length = sample["response_length"]
        status = sample["status"]
        if status == "completed":
            completed += 1
        if status not in ("completed", "truncated") or not 0 < length <= max_response:
            errors.append(f'invalid sample {sample["index"]}')
        probability = sample.get("rollout_log_probs")
        if probability is None or len(probability) != length or not all(
            math.isfinite(p) and p <= 1e-5 for p in probability
        ):
            errors.append(f'behavior probabilities {sample["index"]}')
        reward = sample.get("reward")
        if not isinstance(reward, dict) or reward.get("score") not in (-1.0, 1.0):
            errors.append(f'DAPO score {sample["index"]}')
        elif not isinstance(reward.get("pred"), str):
            errors.append(f'DAPO prediction {sample["index"]}')
        elif reward["pred"] != "[INVALID]":
            extracted += 1
    if len(samples) != prompts * answers or len(indices) != len(samples):
        errors.append("sample count or duplicate index")
    if len(groups) != prompts or any(len(g) != answers for g in groups.values()):
        errors.append("incomplete groups")
    mixed = sum(
        len({s["reward"]["score"] for s in group if isinstance(s.get("reward"), dict)
             and s["reward"].get("score") in (-1.0, 1.0)}) > 1
        for group in groups.values()
    )
    fatal_errors = list(errors)
    # At least half the batch must finish and expose a parseable answer.
    if completed < prompts * answers // 2:
        errors.append("fewer than half the answers completed")
    if extracted < prompts * answers // 2:
        errors.append("fewer than half the answers extracted")
    if mixed == 0:
        errors.append("no within-group reward variation")
    return {
        "passed": not errors, "errors": errors, "fatal_errors": fatal_errors,
        "samples": len(samples),
        "groups": len(groups), "completed": completed, "extracted": extracted,
        "mixed_reward_groups": mixed, "max_response_tokens": max_response,
    }


def check_before_train(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    """Miles invokes this after dumping rollout and before learner conversion."""
    report = assess_batch(
        [s.to_dict() for s in samples], prompts=args.rollout_batch_size,
        answers=args.n_samples_per_prompt, max_response=args.rollout_max_response_len,
    )
    report.update(rollout_id=rollout_id, rollout_seconds=rollout_time)
    root = Path(args.save_debug_rollout_data).parent
    (root / f"rollout-gate-{rollout_id}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"event": "rollout_gate", **report}), flush=True)
    if not report["passed"]:
        raise RuntimeError("GRPO_ROLLOUT_GATE_REJECTED: " + "; ".join(report["errors"]))
    return False
