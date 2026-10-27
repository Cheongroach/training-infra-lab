"""Independent artifact audit of the frozen TP2/DP8 diagnostic; exit is insufficient."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import torch


def model_hashes(state):
    return {k: v for k, v in state["param_hashes"].items()
            if "companion" not in k and "witness" not in k}


def adam_moments(state):
    """Ignore scalar step counters: require changes to actual first/second moments."""
    return {
        (optimizer_index, optimizer["param_names"].get(str(index), str(index)), key): value
        for optimizer_index, optimizer in enumerate(state["optimizer_hashes"])
        for index, item in optimizer["state_dict"].get("state", {}).items()
        for key, value in item.items() if key in ("exp_avg", "exp_avg_sq")
    }


def moved(before, after):
    return sorted(k for k in before.keys() & after.keys() if before[k] != after[k])


def audit_qualification(results, rid, admitted, seen, check):
    from qualification import MAX_CANDIDATES, assess_candidate

    directory = results / "qualification"
    manifest = json.loads((directory / f"step-{rid}.json").read_text())
    attempts = manifest["attempts"]
    check(manifest["passed"] and manifest["rollout_id"] == rid
          and manifest["weight_version"] == rid and manifest["max_candidates"] == MAX_CANDIDATES
          and 1 <= len(attempts) <= MAX_CANDIDATES, f"qualification contract {rid}")
    dropped = []
    for number, record in enumerate(attempts):
        name = f"candidate-{rid}-{number}.json"
        blob = (directory / name).read_bytes()
        check(record["file"] == name and record["attempt"] == number
              and hashlib.sha256(blob).hexdigest() == record["sha256"],
              f"candidate provenance {rid}/{number}")
        candidate = json.loads(blob)
        report = assess_candidate(
            candidate["samples"], candidate["issued"], seen,
            rollout_id=rid, weight_version=rid, prompts=16, answers=4, max_response=4096)
        check(report == record["assessment"], f"candidate recomputation {rid}/{number}")
        ids = [s["index"] for s in candidate["samples"]]
        check(ids == record["sample_indices"], f"candidate identities {rid}/{number}")
        seen.update(ids)
        last = number == len(attempts) - 1
        check(record["decision"] == ("admitted" if last else "rejected")
              and (report["passed"] if last else report["retryable"]),
              f"candidate decision {rid}/{number}")
        if last:
            check(candidate["samples"] == admitted, f"admitted batch changed {rid}")
        else:
            dropped.extend(ids)
    check({p.name for p in directory.glob(f"candidate-{rid}-*.json")}
          == {f"candidate-{rid}-{i}.json" for i in range(len(attempts))},
          f"unaccounted qualification candidates {rid}")
    return {"candidate_batches": len(attempts), "rejected_batches": len(attempts) - 1,
            "rejected_sample_indices": dropped}


def audit(results, code, source_manifest):
    errors = []
    result = {"passed": False, "performance_claim": False, "capability_claim": False,
              "updates": [], "publications": []}

    def check(value, description):
        if not value:
            errors.append(description)

    for name, expected in source_manifest.items():
        path = code / name
        check(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected,
              f"frozen source {name}")
    if errors:
        return {**result, "errors": errors}
    # Use the independently checked submitted gate; never import the unverified archive.
    sys.path.insert(0, str(code.resolve()))
    from rollout_gate import assess_batch
    from math_dapo_utils import compute_score

    contract = json.loads((results / "train32-contract.json").read_text())
    check((contract["learner_gpus"], contract["rollout_gpus"], contract["engine_count"],
           contract["tp"], contract["dp"], contract["batch_samples"]) == (16, 16, 8, 2, 8, 64),
          "allocation contract")
    check((results / "driver-completed.json").is_file(), "driver did not complete")
    cluster = json.loads((results / "ray-cluster-verified.json").read_text())
    check(cluster["verified"] and cluster["gpu_resources"] == 32
          and {n["arnold_id"] for n in cluster["nodes"]} == set(range(4))
          and all(n["bf16_passed"] for n in cluster["nodes"]), "four-node Ray GPU probe")
    events = [json.loads(line) for p in sorted((results / "events").rglob("*.jsonl"))
              for line in p.read_text().splitlines() if line.strip()]
    result["event_types"] = dict(Counter(e["type"] for e in events))
    by_type = defaultdict(list)
    for event in events:
        by_type[event["type"]].append(event)
    for name in ("weight_transfer_failed", "inference_engine_checksum_collection_failed"):
        check(not by_type[name], f"failure event {name}")
    publications = {}
    engines = {}
    for event in by_type["weight_update_result"]:
        rid = event["rollout_id"] if event["rollout_id"] is not None else -1
        check(rid not in publications, f"duplicate publication {rid}")
        publications[rid] = event
    for event in by_type["inference_engine_weight_checksum"]:
        rid = event["rollout_id"]
        check(rid not in engines, f"duplicate engine checksum {rid}")
        engines[rid] = event
    previous_engine = None
    for rid in (-1, 0, 1, 2):
        pub, engine = publications.get(rid), engines.get(rid)
        check(pub is not None and engine is not None, f"missing publication/checksum {rid}")
        if not pub or not engine:
            continue
        check(pub["published_version"] == rid + 1 and not pub["failed_cell_ids"]
              and len(set(pub["updated_cell_ids"])) == 8, f"incomplete eight-engine publication {rid}")
        check(engine["weight_version"] == rid + 1
              and engine["debug_weight_update_id"] == pub["debug_weight_update_id"],
              f"publication identity {rid}")
        snapshots = engine["engine_snapshots"]
        check(len(snapshots) == 8, f"eight-engine checksum count {rid}")
        if not snapshots:
            continue
        hashes = snapshots[0]["tensor_checksums"]
        check(bool(hashes) and all(s["tensor_checksums"] == hashes for s in snapshots),
              f"eight-engine content agreement {rid}")
        check(all(any(k.startswith(f"rank{r}/") for k in hashes) for r in range(2)),
              f"engine TP-rank checksum coverage {rid}")
        changed = moved(previous_engine, hashes) if previous_engine is not None else []
        if previous_engine is not None:
            check(bool([k for k in changed if "companion" not in k and "witness" not in k]),
                  f"no real engine weight movement {rid}")
        previous_engine = hashes
        result["publications"].append({"rollout_id": rid, "version": rid + 1,
                                       "engine_count": len(snapshots), "tensors_per_engine": len(hashes),
                                       "changed_tensors": len(changed)})
    qualification_seen = set()
    rejected = []
    for rid in (0, 1, 2):
        summary = {"rollout_id": rid, "learner_ranks": []}
        samples = torch.load(results / f"rollout-{rid}.pt",
                             map_location="cpu", weights_only=False)["samples"]
        qualification = audit_qualification(results, rid, samples, qualification_seen, check)
        rejected.extend(qualification["rejected_sample_indices"])
        summary.update({k: v for k, v in qualification.items() if k != "rejected_sample_indices"})
        report = assess_batch(samples, prompts=16, answers=4, max_response=4096)
        saved = json.loads((results / f"rollout-gate-{rid}.json").read_text())
        check(report["passed"] and all(saved[k] == v for k, v in report.items()),
              f"strict rollout gate {rid}")
        ids = [s["index"] for s in samples]
        groups = defaultdict(list)
        eos = contract["eos_token_ids"]
        eos = set(eos if isinstance(eos, list) else [eos])
        for sample in samples:
            index, n, tokens = sample["index"], sample["response_length"], sample["tokens"]
            groups[sample["group_index"]].append(sample)
            check(len(tokens) > n > 0, f"response tokens {rid}/{index}")
            check(compute_score(sample["response"], str(sample["label"])) == sample["reward"],
                  f"original scorer reward recomputation {rid}/{index}")
            if sample["status"] == "completed":
                check(tokens[-1] in eos, f"completed response lacks natural EOS {rid}/{index}")
            spans = [span for call in sample["weight_versions"] for span in call["spans"]]
            cursor = len(tokens) - n
            for span in spans:
                check(str(span["version"]) == str(rid) and span["abs_start"] == cursor
                      and span["abs_end"] > cursor, f"policy coverage {rid}/{index}")
                cursor = span["abs_end"]
            check(bool(spans) and cursor == len(tokens), f"complete policy coverage {rid}/{index}")
        check(len(samples) == len(set(ids)) == 64 and len(groups) == 16
              and all(len(g) == 4 for g in groups.values()), f"sample/group coverage {rid}")
        by_index = {s["index"]: s for s in samples}
        local = [e for e in by_type["train_engine_local_weight_checksum"]
                 if e["rollout_id"] == rid and e["source"]["component"] == "actor"]
        ranks = [e["source"]["rank_within_cell"] for e in local]
        check(len(local) == 16 and set(ranks) == set(range(16))
              and len({e["source"]["cell_index"] for e in local}) == 1,
              f"all learner state snapshots {rid}")
        by_rank = {e["source"]["rank_within_cell"]: e["state"] for e in local}
        topology = {}
        for rank in range(16):
            paths = list(results.glob(f"pre-update-{rid}-*-rank-{rank}.json"))
            check(len(paths) == 1, f"pre-update snapshot count {rid}/{rank}")
            if len(paths) != 1 or rank not in by_rank:
                continue
            before = json.loads(paths[0].read_text())
            parallel = before["parallel"]
            topology[rank] = parallel
            check(before["rank"] == rank and before["world_size"] == 16
                  and (parallel["tp_size"], parallel["dp_size"],
                       parallel["pp_size"], parallel["cp_size"]) == (2, 8, 1, 1),
                  f"actual topology {rid}/{rank}")
            state = by_rank[rank]
            changed = moved(model_hashes(before["state"]), model_hashes(state))
            moments = adam_moments(state)
            old_moments = adam_moments(before["state"])
            check(bool(changed), f"no parameter movement {rid}/{rank}")
            check(bool(moments) and all(isinstance(v, str) and len(v) == 64 for v in moments.values()),
                  f"missing hashed Adam moments {rid}/{rank}")
            if old_moments:
                check(bool(moved(old_moments, moments)), f"Adam moments did not move {rid}/{rank}")
            summary["learner_ranks"].append({"rank": rank, "changed_model_tensors": len(changed),
                                            "adam_moment_tensors": len(moments)})
        check(len(topology) == 16
              and {(p["dp_rank"], p["tp_rank"]) for p in topology.values()}
                  == {(d, t) for d in range(8) for t in range(2)}
              and len({p["arnold_id"] for p in topology.values()}) == 2,
              f"TP2/DP8 on two physical nodes {rid}")
        # Compare model shards by recorded TP rank, not by assumed rank numbering.
        for tp_rank in range(2):
            states = [model_hashes(by_rank[r]) for r, p in topology.items()
                      if p["tp_rank"] == tp_rank]
            check(len(states) == 8 and all(s == states[0] for s in states),
                  f"DP model replica agreement {rid}/{tp_rank}")
        consumed = []
        nonzero = 0
        dump_ranks = []
        for path in sorted(results.glob(f"train-{rid}-rank-*.pt")):
            dump = torch.load(path, map_location="cpu", weights_only=False)
            rank = dump["rank"]
            dump_ranks.append(rank)
            check(rank in topology and topology[rank]["tp_rank"] == 0
                  and dump["rollout_id"] == rid and dump.get("cp_size") == 1,
                  f"train dump topology {rid}/{rank}")
            td = dump["rollout_data"]
            owned = td["sample_indices"]
            consumed.extend(owned)
            for field in ("tokens", "response_lengths", "loss_masks", "advantages",
                          "log_probs", "rollout_log_probs"):
                check(len(td.get(field, [])) == len(owned), f"train field {rid}/{rank}/{field}")
            for i, index in enumerate(owned):
                check(index in by_index, f"unknown trainer sample {rid}/{rank}/{index}")
                if index not in by_index:
                    continue
                sample = by_index[index]
                n = sample["response_length"]
                check(torch.as_tensor(td["tokens"][i]).tolist() == sample["tokens"]
                      and td["response_lengths"][i] == n, f"trainer token mapping {rid}/{index}")
                check(torch.equal(torch.as_tensor(td["rollout_log_probs"][i], dtype=torch.float32),
                                  torch.as_tensor(sample["rollout_log_probs"], dtype=torch.float32)),
                      f"behavior probability mapping {rid}/{index}")
                mask = torch.as_tensor(td["loss_masks"][i]).bool().flatten()
                for field in ("advantages", "log_probs", "rollout_log_probs"):
                    value = torch.as_tensor(td[field][i]).float().flatten()
                    check(value.numel() == mask.numel() == n and bool(torch.isfinite(value).all()),
                          f"finite aligned train tensor {rid}/{index}/{field}")
                    if field == "advantages" and value.numel() == mask.numel():
                        nonzero += int((value[mask] != 0).sum())
        check(len(dump_ranks) == 8
              and {topology[r]["dp_rank"] for r in dump_ranks if r in topology} == set(range(8)),
              f"all eight DP training shards {rid}")
        check(Counter(consumed) == Counter(ids), f"exactly once DP sample ownership {rid}")
        check(nonzero > 0, f"zero unmasked advantage {rid}")
        ended = [e for e in by_type["train_group_step_end"]
                 if e["rollout_id"] == rid and e["role"] == "actor"]
        check(len(ended) == 1 and len(ended[0]["cell_outcomes"]) == 1
              and all(v == ["normal"] * 16 for v in ended[0]["cell_outcomes"].values()),
              f"all sixteen optimizer outcomes {rid}")
        grad = [e["metrics"]["train/grad_norm"] for e in by_type["metric"]
                if e.get("rollout_id") == rid and "train/grad_norm" in e["metrics"]]
        check(bool(grad) and all(math.isfinite(g) and g > 0 for g in grad),
              f"finite positive real gradient {rid}")
        summary.update(nonzero_advantage_tokens=nonzero, grad_norm=grad,
                       admitted_samples=len(samples), mixed_reward_groups=report["mixed_reward_groups"],
                       completed_samples=report["completed"],
                       strict_extracted_samples=report.get("extracted"),
                       strict_extraction_passed=report.get("strict_extraction_passed"),
                       format_contract_passed=report.get("format_contract_passed"),
                       format_is_training_admission_gate=report.get("format_is_training_admission_gate"),
                       response_tokens=sum(s["response_length"] for s in samples))
        result["updates"].append(summary)
    dropped = [i for e in by_type["explicitly_dropped_samples"] for i in e["source_sample_indices"]]
    check(Counter(dropped) == Counter(rejected)
          and all(e["reason"] == "bounded_qualification" for e in by_type["explicitly_dropped_samples"]),
          "all dropped samples accounted by frozen bounded qualification")
    # The last publication must also be consumed by actual generated tokens.
    # These prompts were exposed during training, so this is protocol evidence.
    final = torch.load(results / "rollout-eval_2.pt", map_location="cpu", weights_only=False)
    samples = final["samples"]
    expected = [json.loads(line) for line in (code / "explicit-prompts.jsonl").read_text().splitlines()[:16]]
    # Reconstruct the frozen Qwen single-user template observed in this runtime.
    expected_prompts = [
        ("<|im_start|>user\n" + row["prompt"][0]["content"]
         + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", str(row["label"]))
        for row in expected
    ]
    check(final["rollout_id"] == 2 and len(samples) == 16
          and len({s["index"] for s in samples}) == 16, "final evaluation sample identity")
    check(Counter((s["prompt"], str(s["label"])) for s in samples) == Counter(expected_prompts),
          "final evaluation frozen exposed prompts")
    completed = correct = strict = response_tokens = 0
    from format_contract import FINAL_LINE
    from rollout_gate import answer_text
    for sample in samples:
        index, n, tokens = sample["index"], sample["response_length"], sample["tokens"]
        check(len(tokens) > n > 0 and n <= 4096
              and sample["status"] in ("completed", "truncated"), f"final response {index}")
        check(compute_score(sample["response"], str(sample["label"])) == sample["reward"],
              f"final original scorer {index}")
        probability = sample.get("rollout_log_probs", [])
        check(len(probability) == n and all(math.isfinite(p) and p <= 1e-5 for p in probability),
              f"final behavior probabilities {index}")
        if sample["status"] == "completed":
            completed += 1
            check(tokens[-1] in eos, f"final natural EOS {index}")
            lines = answer_text(sample).splitlines()
            match = FINAL_LINE.fullmatch(lines[-1]) if lines else None
            strict += int(bool(match and sample["reward"]["pred"] == match[1]))
        correct += int(sample["reward"]["score"] == 1.0)
        response_tokens += n
        cursor = len(tokens) - n
        spans = [span for call in sample["weight_versions"] for span in call["spans"]]
        for span in spans:
            check(str(span["version"]) == "3" and span["abs_start"] == cursor
                  and span["abs_end"] > cursor, f"final v3 token consumption {index}")
            cursor = span["abs_end"]
        check(bool(spans) and cursor == len(tokens), f"final complete version coverage {index}")
    result["final_evaluation"] = {
        "samples": len(samples), "completed": completed, "strict_extracted": strict,
        "original_scorer_correct": correct, "response_tokens": response_tokens,
        "consumed_weight_version": 3, "scope": "Exposed prompts; protocol only",
        "quality_improvement_claim": False,
    }
    result.update(errors=errors, passed=not errors)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = audit(args.results, args.code, json.loads(args.source_manifest.read_text()))
    except Exception as exc:
        report = {"passed": False, "performance_claim": False, "capability_claim": False,
                  "errors": [f"{type(exc).__name__}: {exc}"]}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
