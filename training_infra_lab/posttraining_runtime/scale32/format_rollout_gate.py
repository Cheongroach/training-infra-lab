"""Audit strict answer formatting separately from the complete-group training contract."""

import json
from pathlib import Path

from original_rollout_gate import assess_batch as original_assess_batch
from format_contract import FINAL_LINE


def answer_text(sample):
    """Strip only the decoded terminator for an actually completed Qwen response."""
    text = sample.get("response", "").strip()
    if sample.get("status") == "completed" and text.endswith("<|im_end|>"):
        text = text[:-len("<|im_end|>")].rstrip()
    return text


def assess_batch(samples, *, prompts=16, answers=4, max_response=4096):
    report = original_assess_batch(samples, prompts=prompts, answers=answers, max_response=max_response)
    invalid = []
    for sample in samples:
        if sample["status"] != "completed":
            continue
        lines = answer_text(sample).splitlines()
        match = FINAL_LINE.fullmatch(lines[-1]) if lines else None
        reward = sample.get("reward")
        if not match or not isinstance(reward, dict) or reward.get("pred") != match[1]:
            invalid.append(sample["index"])
    # Keep original DAPO rewards and responses for every admitted batch member.
    valid = report["completed"] - len(invalid)
    report["extracted"] = valid
    report["strict_extraction_threshold"] = prompts * answers // 2
    report["strict_extraction_passed"] = valid >= prompts * answers // 2
    report["format_contract_passed"] = valid == prompts * answers
    report["format_is_training_admission_gate"] = False
    report["invalid_completed_answer_indices"] = invalid
    report["passed"] = not report["errors"]
    return report


def check_before_train(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    report = assess_batch([sample.to_dict() for sample in samples],
                          prompts=args.rollout_batch_size, answers=args.n_samples_per_prompt,
                          max_response=args.rollout_max_response_len)
    report.update(rollout_id=rollout_id, rollout_seconds=rollout_time)
    output = Path(args.save_debug_rollout_data).parent / f"rollout-gate-{rollout_id}.json"
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"event": "rollout_gate", **report}), flush=True)
    if not report["passed"]:
        raise RuntimeError("GRPO_FORMAT_ROLLOUT_GATE_REJECTED: " + "; ".join(report["errors"]))
    return False
