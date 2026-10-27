"""Compare explicit precision contracts; measurements never bypass device gates."""

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.tensor import distribute_tensor

from .device_gates import error_metrics, tensor_states
from .distributed import fail_worker
from .executor import execute_update
from .model import Actor, make_lm
from .planner import plan_update
from .probe import emit, fsdp_api
from .rl import collate, surrogate_sum
from .test_runtime import example_batch


def reference_update(actor, optimizer, batch, device, parameter_dtype):
    data = collate(batch, range(len(batch.trajectories)), device)
    inputs = {k: data[k] for k in ("input_ids", "attention_mask", "response_mask")}
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if parameter_dtype is None:
            out = actor(**inputs)
        else:
            # Differentiable views of the FP32 masters, including embedding,
            # residual path, and non-matmul parameters. FSDP param_dtype applies
            # to those too; plain autocast does not have the same semantics.
            views = {n: p.to(parameter_dtype) for n, p in actor.named_parameters()}
            out = torch.func.functional_call(actor, views, (), inputs)
        loss = surrogate_sum(out, data, batch.reduction) / batch.denominator
    loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0, error_if_nonfinite=True, foreach=False)
    optimizer.step()
    return loss.item()


def worst_parameters(names, actual, expected, warm):
    worst = []
    for i, name in enumerate(names):
        difference = (actual["parameters"][i] - expected["parameters"][i]).abs().reshape(-1)
        index = difference.argmax().item()
        worst.append({
            "name": name, "index": index, "parameter_max_abs": difference[index].item(),
            "before": warm[name].reshape(-1)[index].item(),
            **{side + "_" + field: state[field][i].reshape(-1)[index].item()
               for side, state in (("actual", actual), ("expected", expected))
               for field in ("parameters", "gradient", "exp_avg", "exp_avg_sq")},
        })
    return sorted(worst, key=lambda x: x["parameter_max_abs"], reverse=True)[:12]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.manual_seed(23)
        initial = Actor(make_lm(tiny=True, kernel="fla")).state_dict()
        batch = example_batch((17, 65, 33, 25, 49, 9, 13, 41))
        reports = []
        for label, reference_dtype, fsdp_dtype, epsilon in (
            ("autocast_vs_bf16_fsdp", None, torch.bfloat16, 1e-8),
            ("bf16_views_vs_bf16_fsdp", torch.bfloat16, torch.bfloat16, 1e-8),
            ("autocast_vs_fp32_fsdp", None, None, 1e-8),
            ("bf16_views_eps_1e6", torch.bfloat16, torch.bfloat16, 1e-6),
            ("bf16_views_eps_1e5", torch.bfloat16, torch.bfloat16, 1e-5),
        ):
            payload = [None]
            if rank == 0:
                reference = Actor(make_lm(tiny=True, kernel="fla")).to(device)
                reference.load_state_dict(initial)
                optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-4, eps=epsilon, foreach=False)
                reference_update(reference, optimizer, batch, device, reference_dtype)
                warm = {n: p.detach().cpu().clone() for n, p in reference.named_parameters()}
                moments = {n: {k: v.detach().cpu().clone() for k, v in optimizer.state[p].items()}
                           for n, p in reference.named_parameters()}
                loss = reference_update(reference, optimizer, batch, device, reference_dtype)
                expected = tensor_states(reference, optimizer)
                payload[0] = (warm, moments, loss, expected)
                del reference, optimizer
            dist.broadcast_object_list(payload, src=0)
            warm, moments, reference_loss, expected = payload[0]
            actor = Actor(make_lm(tiny=True, kernel="fla")).to(device)
            actor.load_state_dict(warm)
            names = [n for n, _ in actor.named_parameters()]
            shard, policy = fsdp_api()
            mp = policy(param_dtype=fsdp_dtype, reduce_dtype=torch.float32)
            for layer in actor.backbone.layers:
                shard(layer, mp_policy=mp)
            shard(actor, mp_policy=mp)
            optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-4, eps=epsilon, foreach=False)
            mismatches = []
            for n, p in actor.named_parameters():
                for k, v in moments[n].items():
                    optimizer.state[p][k] = (v.clone() if k == "step" else
                        distribute_tensor(v.to(device), p.device_mesh, p.placements))
                if not torch.equal(p.full_tensor().cpu(), warm[n]):
                    mismatches.append((n, "parameter"))
                for k in ("exp_avg", "exp_avg_sq"):
                    if not torch.equal(optimizer.state[p][k].full_tensor().cpu(), moments[n][k]):
                        mismatches.append((n, k))
            errors_by_rank = [None] * world
            dist.all_gather_object(errors_by_rank, mismatches)
            if any(errors_by_rank):
                raise AssertionError(errors_by_rank)
            plan = plan_update(batch, world, strategy="fixed", micro_batch=2, max_padded_tokens=256)
            result = execute_update(actor, optimizer, batch, plan, device, autocast_dtype=torch.bfloat16)
            actual = tensor_states(actor, optimizer)
            errors = {k: error_metrics(actual[k], expected[k]) for k in expected}
            accepted = (abs(result["loss"] - reference_loss) <= 0.015
                        and errors["parameters"]["max_abs"] <= 3e-5
                        and all(errors[k]["relative_l2"] <= 0.06
                                for k in ("gradient", "exp_avg", "exp_avg_sq")))
            report = {
                "case": label, "rank": rank, "world": world, "errors": errors, "adam_eps": epsilon,
                "loss": result["loss"], "reference_loss": reference_loss,
                "satisfies_existing_thresholds": accepted,
                "initial_parameters_and_adam": "exact_shared_rank0_state",
                "worst_parameters": worst_parameters(names, actual, expected, warm),
                "scope": "diagnostic tiny Qwen3.5; not real training or a replacement gate",
            }
            reports.append(report)
            (args.output / f"rank-{rank}.json").write_text(json.dumps(reports, indent=2))
            emit("precision_diagnostic", **{k: v for k, v in report.items() if k != "worst_parameters"})
            del actor, optimizer
        emit("precision_diagnostic_complete", rank=rank)
    except BaseException:
        fail_worker()
    else:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
