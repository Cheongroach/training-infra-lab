"""Predeclared numeric gates; failures block the real workload, never retune tolerances."""

import argparse
import json
import os
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from .executor import execute_update
from .distributed import fail_worker
from .model import Actor, make_lm
from .planner import Plan, plan_update
from .probe import fsdp_api, emit
from .rl import collate, surrogate_sum
from .test_runtime import example_batch


def error_metrics(actual, expected):
    """Aggregate all tensors, including small/zero gradients without unstable division."""
    difference = signal = dot = norm = maximum = 0.0
    for a, b in zip(actual, expected, strict=True):
        a, b = a.detach().double(), b.detach().double()
        difference += (a - b).square().sum().item()
        signal += b.square().sum().item()
        norm += a.square().sum().item()
        dot += (a * b).sum().item()
        maximum = max(maximum, (a - b).abs().max().item())
    return {"relative_l2": (difference / max(signal, 1e-30)) ** 0.5,
            "cosine": dot / max((signal * norm) ** 0.5, 1e-30), "max_abs": maximum}


def tensor_states(actor, optimizer):
    def materialize(tensor):
        if hasattr(tensor, "full_tensor"):
            tensor = tensor.full_tensor()
        return tensor.detach().cpu().clone()
    return {
        "parameters": [materialize(p) for p in actor.parameters()],
        "gradient": [materialize(p.grad) for p in actor.parameters()],
        "exp_avg": [materialize(optimizer.state[p]["exp_avg"]) for p in actor.parameters()],
        "exp_avg_sq": [materialize(optimizer.state[p]["exp_avg_sq"]) for p in actor.parameters()],
    }


def direct_update(actor, optimizer, batch, device, dtype):
    data = collate(batch, range(len(batch.trajectories)), device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
        inputs = {k: data[k] for k in ("input_ids", "attention_mask", "response_mask")}
        if dtype is None:
            out = actor(**inputs)
        else:
            # Match FSDP param_dtype, including embeddings and non-matmul ops.
            # Views preserve FP32 master parameters and the independent global batch.
            views = {name: value.to(dtype) for name, value in actor.named_parameters()}
            out = torch.func.functional_call(actor, views, (), inputs)
        loss = surrogate_sum(out, data, batch.reduction) / batch.denominator
    loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0, error_if_nonfinite=True, foreach=False)
    optimizer.step()
    return loss.item()


def kernel_gate(device):
    torch.manual_seed(19)
    baseline = Actor(make_lm(tiny=True, kernel="torch")).to(device)
    candidate = Actor(make_lm(tiny=True, kernel="fla")).to(device)
    candidate.load_state_dict(baseline.state_dict())
    batch = example_batch((17, 257, 65, 129))
    data = collate(batch, range(4), device)
    states, outputs = [], []
    for model in (baseline, candidate):
        with torch.autocast(device.type, dtype=torch.bfloat16):
            out = model(**{k: data[k] for k in ("input_ids", "attention_mask", "response_mask")})
            loss = out[data["response_mask"].bool()].mean()
        loss.backward()
        outputs.append(out.detach())
        states.append([p.grad.detach().clone() for p in model.parameters()])
    log_error = error_metrics([outputs[1]], [outputs[0]])
    grad_error = error_metrics(states[1], states[0])
    assert log_error["max_abs"] <= 0.08, log_error
    assert grad_error["relative_l2"] <= 0.06 and grad_error["cosine"] >= 0.998, grad_error
    candidate.eval()
    ids = data["input_ids"][:1, :17]
    cache_errors = []
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        _, cache = candidate(ids[:, :12], torch.ones_like(ids[:, :12]), generation=True)
        for end in range(13, 18):
            cached, cache = candidate(ids[:, end - 1:end], torch.ones_like(ids[:, :end]),
                                      generation=True, past_key_values=cache)
            full, _ = candidate(ids[:, :end], torch.ones_like(ids[:, :end]), generation=True)
            error = error_metrics([cached.float().log_softmax(-1)], [full.float().log_softmax(-1)])
            assert error["max_abs"] <= 0.08, error
            cache_errors.append(error)
    # Changing one padded batch row must not affect another row's GDN state.
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        before = candidate(**{k: data[k] for k in ("input_ids", "attention_mask", "response_mask")})
        data["input_ids"][0] = 12
        after = candidate(**{k: data[k] for k in ("input_ids", "attention_mask", "response_mask")})
    torch.testing.assert_close(before[1:], after[1:], rtol=0, atol=0)
    return {"gate": "fla_vs_torch_bf16", "log_probs": log_error, "gradient": grad_error,
            "cached_vs_full": cache_errors, "cross_row_isolation": "exact",
            "scope": "tiny 4-layer Qwen3.5; lengths 17/65/129/257"}


def distributed_gate(device, kernel, dtype, adam_eps=1e-8, output=None):
    from torch.distributed.tensor import distribute_tensor

    world = dist.get_world_size()
    torch.manual_seed(23)
    initial = Actor(make_lm(tiny=True, kernel=kernel)).state_dict()
    reports = []
    for reduction in ("sequence_mean", "token_mean"):
        batch = replace(example_batch((17, 65, 33, 25, 49, 9, 13, 41)), reduction=reduction)
        snapshot = [None]
        if dist.get_rank() == 0:
            reference = Actor(make_lm(tiny=True, kernel=kernel)).to(device)
            reference.load_state_dict(initial)
            reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-4, eps=adam_eps, foreach=False)
            # Exactly one reference globally. Even identical BF16 warmups on
            # different GPUs can round differently, so do not warm once per rank.
            direct_update(reference, reference_optimizer, batch, device, dtype)
            warm_parameters = {k: v.detach().cpu().clone() for k, v in reference.state_dict().items()}
            warm_adam = {name: {k: v.detach().cpu().clone()
                               for k, v in reference_optimizer.state[p].items()}
                         for name, p in reference.named_parameters()}
            expected_loss = direct_update(reference, reference_optimizer, batch, device, dtype)
            expected = tensor_states(reference, reference_optimizer)
            snapshot[0] = (warm_parameters, warm_adam, expected_loss, expected)
            del reference, reference_optimizer
        dist.broadcast_object_list(snapshot, src=0)
        warm_parameters, warm_adam, expected_loss, expected = snapshot[0]
        variants = [("fixed", 2, False), ("verl", 1, False),
                    ("slot_sorted", 2, True), ("empty", 1, True)]
        for strategy, micro, deferred in variants:
            actor = Actor(make_lm(tiny=True, kernel=kernel)).to(device)
            actor.load_state_dict(warm_parameters)
            if strategy == "slot_sorted":
                actor.backbone.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
            shard, policy = fsdp_api()
            mp = policy(param_dtype=dtype, reduce_dtype=torch.float32)
            for layer in actor.backbone.layers:
                shard(layer, mp_policy=mp)
            shard(actor, mp_policy=mp)
            optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-4, eps=adam_eps, foreach=False)
            mismatches = []
            for name, parameter in actor.named_parameters():
                for key, value in warm_adam[name].items():
                    optimizer.state[parameter][key] = (
                        value.clone() if key == "step" else
                        distribute_tensor(value.to(device), parameter.device_mesh, parameter.placements)
                    )
                if not torch.equal(parameter.full_tensor().cpu(), warm_parameters[name]):
                    mismatches.append((name, "parameter"))
                for key in ("exp_avg", "exp_avg_sq"):
                    if not torch.equal(optimizer.state[parameter][key].full_tensor().cpu(),
                                       warm_adam[name][key]):
                        mismatches.append((name, key))
            gathered_mismatches = [None] * world
            dist.all_gather_object(gathered_mismatches, mismatches)
            if any(gathered_mismatches):
                raise AssertionError(f"Initial state mismatch across ranks: {gathered_mismatches}")
            if strategy == "empty":
                # Rank ownership rotates; every slot has world-1 dummy traversals.
                slots = tuple(tuple((i,) if r == i % world else () for r in range(world))
                              for i in range(len(batch.trajectories)))
                plan = Plan(batch.sha256, slots, strategy, 128, defer_sync=deferred)
            else:
                plan = plan_update(batch, world, strategy=strategy, micro_batch=micro,
                                   max_padded_tokens=256, defer_sync=deferred)
            result = execute_update(actor, optimizer, batch, plan, device, autocast_dtype=dtype)
            actual = tensor_states(actor, optimizer)
            errors = {key: error_metrics(actual[key], expected[key]) for key in expected}
            report = {"gate": "fsdp_vs_independent_global_batch", "kernel": kernel,
                      "dtype": str(dtype), "adam_eps": adam_eps, "strategy": strategy, "defer_sync": deferred,
                      "reduction": reduction, "world": world, "errors": errors,
                      "initial_parameters_and_adam": "byte_exact_full_tensor_checked",
                      "reference_state_owner": "rank0_broadcast_to_all_ranks",
                      "reference_compute": "matching_parameter_views_independent_global_batch",
                      "activation_checkpoint": strategy == "slot_sorted",
                      "loss": result["loss"], "reference_loss": expected_loss}
            tolerance = 0.06 if dtype is not None else 0.0003
            report["accepted"] = (
                abs(result["loss"] - expected_loss) <= (0.015 if dtype else 2e-5)
                and all(errors[field]["relative_l2"] <= tolerance
                        for field in ("gradient", "exp_avg", "exp_avg_sq"))
                and errors["parameters"]["max_abs"] <= (3e-5 if dtype else 2e-6)
            )
            reports.append(report)
            if output is not None:
                (output / f"{kernel}-rank-{dist.get_rank()}.json").write_text(
                    json.dumps(reports, indent=2))
            emit("device_gate_pass" if report["accepted"] else "device_gate_failed", **report)
            del actor, optimizer
    failures = [(r["reduction"], r["strategy"], r["errors"])
                for r in reports if not r["accepted"]]
    if failures:
        raise AssertionError(f"Completed numerical matrix has failures: {failures}")
    return reports



def residency_gate(device):
    from .grpo import tensors
    from .residency import rollout_residency
    from .rollout import sample_group
    from .test_grpo import FakeTokenizer

    torch.manual_seed(91)
    actor = Actor(make_lm(tiny=True, kernel="fla"), fp32_head=True).to(device)
    actor.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    shard, policy = fsdp_api()
    mp = policy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in actor.backbone.layers:
        shard(layer, mp_policy=mp, reshard_after_forward=True)
    shard(actor, mp_policy=mp, reshard_after_forward=True)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-6, eps=1e-4, foreach=False)
    batch = example_batch((17, 65, 33, 25, 49, 9, 13, 41))
    plan = plan_update(batch, dist.get_world_size(), strategy="verl", micro_batch=1)
    execute_update(actor, optimizer, batch, plan, device, autocast_dtype=torch.bfloat16)
    base = {name: t.detach().clone() for name, t in tensors(actor, optimizer, gradients=True)}
    outputs, expected = [], None
    for resident in (False, True):
        with torch.no_grad():
            for name, t in tensors(actor, optimizer):
                t.copy_(base[name])
            torch.cuda.manual_seed(2021 + dist.get_rank())
            with rollout_residency(actor, device, enabled=resident, reserve_bytes=0) as decision:
                trajectories, _ = sample_group(
                    actor, FakeTokenizer(), {"question": "fixture", "answer": "42"},
                    prompt_id="gate", policy_version=1, group_size=2,
                    device=device, max_new_tokens=8)
            assert not resident or decision["admitted"]
            outputs.append(trajectories)
            for name, t in tensors(actor, optimizer):
                torch.testing.assert_close(t, base[name], rtol=0, atol=0)
        # Training immediately after leaving residency must use restored shards.
        execute_update(actor, optimizer, batch, plan, device, autocast_dtype=torch.bfloat16)
        actual = {name: t.detach().clone() for name, t in tensors(actor, optimizer, gradients=True)}
        if expected is None:
            expected = actual
        else:
            errors = {}
            for field in ("parameter", "gradient", "exp_avg", "exp_avg_sq"):
                names = [n for n in actual if n.endswith("-" + field)]
                errors[field] = error_metrics([actual[n] for n in names], [expected[n] for n in names])
            assert errors["parameter"]["max_abs"] <= 3e-6, errors
            assert all(errors[f]["relative_l2"] <= 0.08 for f in ("gradient", "exp_avg", "exp_avg_sq")), errors
    vote = torch.tensor(int(outputs[0] == outputs[1]), device=device)
    dist.all_reduce(vote, op=dist.ReduceOp.MIN)
    assert vote.item(), "Residency changed generation"
    with torch.no_grad():
        with rollout_residency(actor, device, enabled=True, reserve_bytes=2**60) as decision:
            assert not decision["admitted"]
        try:
            with rollout_residency(actor, device, enabled=True, reserve_bytes=0):
                raise RuntimeError("injected phase exception")
        except RuntimeError as exc:
            assert str(exc) == "injected phase exception"
    execute_update(actor, optimizer, batch, plan, device, autocast_dtype=torch.bfloat16)
    return {"gate": "fsdp_rollout_residency", "exact_sampling_and_inference_state": True,
            "post_generation_update_errors": errors, "collective_memory_rejection": True,
            "phase_exception_then_training": "passed", "scope": "tiny Qwen3.5 CUDA"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    args = parser.parse_args()
    if not 0 < args.adam_eps < 1:
        raise ValueError("Invalid Adam epsilon")
    torch.set_num_threads(1)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=600))
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    try:
        reports.append(kernel_gate(device))
        emit("device_gate_pass", **reports[-1])
        for kernel, dtype in (("torch", None), ("fla", torch.bfloat16)):
            reports.extend(distributed_gate(
                device, kernel, dtype, adam_eps=args.adam_eps if dtype else 1e-8,
                output=args.output))
        reports.append(residency_gate(device))
        emit("device_gate_pass", **reports[-1])
        (args.output / f"rank-{dist.get_rank()}.json").write_text(json.dumps(reports, indent=2))
        emit("all_device_gates_pass", rank=dist.get_rank(), count=len(reports))
    except BaseException:
        fail_worker()
    else:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
