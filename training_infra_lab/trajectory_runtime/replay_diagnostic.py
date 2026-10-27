"""Frozen 9B trajectory diagnostics; never an on-policy training or speedup claim."""
import argparse
import json
import os
import shutil
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from .contracts import UpdateBatch
from .distributed import fail_worker
from .grpo import local, measured_update, scratch_save, scratch_restore, tensors
from .model import Actor, make_lm
from .planner import plan_update
from .probe import fsdp_api, emit
from .profiling import profile_call
from .rl import collate


@torch.no_grad()
def probabilities(actor, batch, device, pad_id, strategy, micro):
    plan = plan_update(batch, dist.get_world_size(), strategy=strategy, micro_batch=micro,
                       max_padded_tokens=2048, activation_checkpoint=True)
    values = {}
    for slot in plan.slots:
        indices = slot[dist.get_rank()]
        if not indices:
            raise ValueError("This diagnostic requires nonempty aligned slots")
        data = collate(batch, indices, device, pad_id)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            out = actor(**{k: data[k] for k in ("input_ids", "attention_mask", "response_mask")})
        for row, index in enumerate(indices):
            values[index] = out[row][data["response_mask"][row].bool()].cpu().tolist()
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, values)
    combined = {i: v for rank in gathered for i, v in rank.items()}
    if len(combined) != len(batch.trajectories):
        raise ValueError("Incomplete probability replay")
    return combined


def compare_probabilities(actual, reference):
    differences = torch.tensor([a - b for i in sorted(actual)
                                for a, b in zip(actual[i], reference[i], strict=True)])
    return {"rms": differences.square().mean().sqrt().item(),
            "max_abs": differences.abs().max().item(),
            "p99_abs": differences.abs().quantile(0.99).item()}


def compare_tensors(path, actor, optimizer):
    totals = defaultdict(lambda: [0.0, 0.0, 0.0])
    groups = defaultdict(lambda: [0.0, 0.0, 0.0])
    names = [n for n, _ in actor.named_parameters()]
    gradients = []
    for name, value in tensors(actor, optimizer, gradients=True):
        index, field = name.split("-", 1)
        parameter_name = names[int(index)]
        expected = torch.load(path / (name + ".pt"), weights_only=True, map_location="cpu").reshape(-1)
        actual = value.detach().cpu().reshape(-1)
        row = [0.0, 0.0, 0.0]
        for begin in range(0, actual.numel(), 2**20):
            a, b = actual[begin:begin + 2**20].double(), expected[begin:begin + 2**20].double()
            row[0] += (a - b).square().sum().item()
            row[1] += b.square().sum().item()
            row[2] = max(row[2], (a - b).abs().max().item())
        for target in (totals[field], groups[(parameter_name.rsplit(".", 1)[0], field)]):
            target[0] += row[0]
            target[1] += row[1]
            target[2] = max(target[2], row[2])
        if field == "gradient":
            gradients.append({"name": parameter_name, "difference_l2": row[0]**0.5,
                              "reference_l2": row[1]**0.5, "max_abs": row[2]})
    def metrics(v):
        return {"relative_l2": (v[0] / max(v[1], 1e-30))**0.5, "max_abs": v[2],
                "difference_squared": v[0], "reference_squared": v[1]}
    return {"totals": {k: metrics(v) for k, v in totals.items()},
            "groups": {"/".join(k): metrics(v) for k, v in groups.items()},
            "largest_gradient_differences": sorted(gradients, key=lambda v: v["difference_l2"], reverse=True)[:20]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--batches", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--adam-eps", type=float, default=1e-4)
    args = p.parse_args()
    torch.set_num_threads(1)
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=600))
    args.output.mkdir(parents=True, exist_ok=True)
    scratch = args.scratch / f"rank-{rank}"
    scratch.mkdir(parents=True, exist_ok=False)
    try:
        from transformers import AutoTokenizer
        warm, batch = [UpdateBatch.from_dict(json.loads((args.batches / f"batch-{i}.json").read_text()))
                       for i in (0, 1)]
        manifest = json.loads((args.batches / "batches-manifest.json").read_text())
        if [warm.sha256, batch.sha256] != manifest["batch_sha256"]:
            raise ValueError("Frozen trajectory identity mismatch")
        if any(warm.advantages()):
            raise ValueError("Expected the archived zero-advantage warm step")
        torch.manual_seed(71)
        lm = make_lm(args.model, kernel="fla", dtype=torch.bfloat16)
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        actor = Actor(lm).to(device=device, dtype=torch.float32)
        del lm
        shard, policy = fsdp_api()
        mp = policy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for layer in actor.backbone.layers:
            shard(layer, mp_policy=mp, reshard_after_forward=True)
        shard(actor, mp_policy=mp, reshard_after_forward=True)
        optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-6, eps=args.adam_eps, foreach=False)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        result, _ = measured_update(actor, optimizer, warm, device, strategy="verl", micro=1,
                                    budget=2048, pad_id=pad_id)
        if result["grad_norm"] != 0:
            raise RuntimeError("Warm-step gradient is not zero")
        base = scratch / "base"
        scratch_save(base, actor, optimizer)
        records = []
        choices = [("verl", 1), ("verl", 1), ("slot_sorted", 1), ("verl", 2), ("slot_sorted", 2)]
        for fp32_head in (False, True):
            actor.fp32_head = fp32_head
            expected = scratch / f"expected-{fp32_head}"
            reference_probs = None
            for case, (strategy, micro) in enumerate(choices):
                scratch_restore(base, actor, optimizer)
                probs = probabilities(actor, batch, device, pad_id, strategy, micro)
                result, _ = measured_update(actor, optimizer, batch, device, strategy=strategy,
                                            micro=micro, budget=2048, pad_id=pad_id)
                record = {"case": case, "fp32_head": fp32_head, "strategy": strategy,
                          "micro_batch": micro, "result": result, "batch_sha256": batch.sha256,
                          "scope": "frozen off-policy diagnostic; timings are not performance samples"}
                if case == 0:
                    scratch_save(expected, actor, optimizer, gradients=True)
                    reference_probs = probs
                else:
                    record["probability_difference"] = compare_probabilities(probs, reference_probs)
                    comparison = compare_tensors(expected, actor, optimizer)
                    (args.output / f"tensors-head{fp32_head}-case{case}-rank{rank}.json").write_text(
                        json.dumps(comparison, indent=2))
                    record["errors"] = comparison["totals"]
                    e = record["errors"]
                    valid = all(e[k]["relative_l2"] <= 0.08 for k in ("gradient", "exp_avg", "exp_avg_sq"))
                    valid &= e["parameter"]["max_abs"] <= 3e-6
                    vote = torch.tensor(int(valid), device=device)
                    dist.all_reduce(vote, op=dist.ReduceOp.MIN)
                    record["accepted"] = bool(vote.item())
                records.append(record)
                (args.output / f"diagnostic-rank{rank}.json").write_text(json.dumps(records, indent=2))
                emit("frozen_replay_diagnostic", **record)
            scratch_restore(base, actor, optimizer)
            profile_call(lambda: measured_update(actor, optimizer, batch, device, strategy="verl", micro=2,
                                                  budget=2048, pad_id=pad_id)[0],
                         args.output / f"profile-head{fp32_head}-rank{rank}", device,
                         {"batch_sha256": batch.sha256, "fp32_head": fp32_head, "strategy": "verl", "micro": 2})
            shutil.rmtree(expected)
        emit("frozen_diagnostic_complete", rank=rank, cases=len(records),
             note="Completion is evidence collection, not numerical acceptance")
    except BaseException:
        fail_worker()
    else:
        shutil.rmtree(scratch)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
