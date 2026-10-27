"""Bounded whole-batch qualification with complete discarded-sample evidence."""
import hashlib
import json

from rollout_gate import assess_batch

MAX_CANDIDATES = 4


def assess_candidate(samples, issued, seen, *, rollout_id, weight_version,
                     prompts=4, answers=4, max_response=4096):
    report = assess_batch(samples, prompts=prompts, answers=answers,
                          max_response=max_response)
    fatal = report["fatal_errors"]
    ids = [s["index"] for s in samples]
    issued_ids = [s["index"] for s in issued]
    if (len(issued_ids) != prompts * answers or len(set(issued_ids)) != len(issued_ids)
            or set(issued_ids) != set(ids) or set(ids) & seen
            or any(type(i) is not int or i < 0 for i in ids)):
        fatal.append("issued/returned/previous sample identity mismatch")
    issued_by_id = {s["index"]: s for s in issued}
    if weight_version != rollout_id:
        fatal.append("unexpected synchronous weight version")
    for sample in samples:
        index = sample["index"]
        if sample["group_index"] != issued_by_id.get(index, {}).get("group_index"):
            fatal.append(f"issued group mismatch {index}")
        lineage = sample.get("lineage")
        if lineage != {"source_sample_index": index, "output_index": 0, "output_count": 1}:
            fatal.append(f"single-output lineage mismatch {index}")
        length = sample["response_length"]
        tokens = sample.get("tokens", [])
        if len(tokens) <= length:
            fatal.append(f"token length mismatch {index}")
        cursor = len(tokens) - length
        spans = [span for call in sample.get("weight_versions", []) for span in call.get("spans", [])]
        if not spans:
            fatal.append(f"missing behavior version {index}")
        for span in spans:
            if (str(span.get("version")) != str(weight_version)
                    or span.get("abs_start") != cursor
                    or not isinstance(span.get("abs_end"), int)
                    or span["abs_end"] <= cursor):
                fatal.append(f"behavior version/coverage mismatch {index}")
                break
            cursor = span["abs_end"]
        if cursor != len(tokens):
            fatal.append(f"incomplete behavior coverage {index}")
    report["errors"] = list(dict.fromkeys(report["errors"] + fatal))
    report["passed"] = not report["errors"]
    report["retryable"] = bool(report["errors"]) and not fatal
    return report


async def qualify(next_batch, *, root, rollout_id, weight_version, seen,
                  prompts=4, answers=4, max_response=4096, on_drop=None):
    """next_batch returns (unmodified Miles output, serialized samples, issued IDs)."""
    directory = root / "qualification"
    directory.mkdir(exist_ok=True)
    manifest_path = directory / f"step-{rollout_id}.json"
    if manifest_path.exists():
        raise RuntimeError("qualification step already exists")
    manifest = {"rollout_id": rollout_id, "weight_version": weight_version,
                "max_candidates": MAX_CANDIDATES, "passed": False, "attempts": []}
    for attempt in range(MAX_CANDIDATES):
        output, samples, issued = await next_batch()
        path = directory / f"candidate-{rollout_id}-{attempt}.json"
        blob = json.dumps({"samples": samples, "issued": issued}).encode()
        # Persist the entire candidate before deciding whether it may train.
        path.write_bytes(blob)
        report = assess_candidate(
            samples, issued, seen, rollout_id=rollout_id, weight_version=weight_version,
            prompts=prompts, answers=answers, max_response=max_response,
        )
        seen.update(s["index"] for s in samples)
        record = {"attempt": attempt, "file": path.name,
                  "sha256": hashlib.sha256(blob).hexdigest(),
                  "sample_indices": [s["index"] for s in samples], "assessment": report,
                  "decision": "admitted" if report["passed"] else
                              "rejected" if report["retryable"] else "fatal"}
        manifest["attempts"].append(record)
        manifest["passed"] = report["passed"]
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(json.dumps({"event": "qualification", "rollout_id": rollout_id,
                          "attempt": attempt, "decision": record["decision"],
                          **report}), flush=True)
        if report["passed"]:
            return output
        if not report["retryable"]:
            raise RuntimeError("GRPO_QUALIFICATION_FATAL: " + "; ".join(report["errors"]))
        if on_drop is not None:
            on_drop(output)
    raise RuntimeError("GRPO_QUALIFICATION_EXHAUSTED: four candidate batches rejected")
