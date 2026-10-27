"""Small, explicit full-parameter SFT runner for matched hardware experiments."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from correctness import fsdp_api
from loss import SupervisedModel


def emit(event, **values):
    print(json.dumps({"event": event, **values}), flush=True)


def load_data(path, length):
    records = []
    for line in path.read_text().splitlines():
        item = json.loads(line)
        ids, labels = item["input_ids"][:length], item["labels"][:length]
        if any(t != -100 for t in labels[1:]):
            records.append((ids, labels))
    if not records:
        raise ValueError(f"No supervised examples: {path}")
    return records


def batch_at(records, indices, length, pad, device, padding="fixed"):
    indices = list(indices)
    if padding == "dynamic":
        longest = max(len(records[i % len(records)][0]) for i in indices)
        # Bound the number of accelerator shapes while avoiding mostly empty
        # 4096-token batches. Input/labels are unchanged relative to fixed padding.
        length = min(length, ((longest + 127) // 128) * 128)
    ids = torch.full((len(indices), length), pad, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    attention = torch.zeros_like(ids)
    for row, idx in enumerate(indices):
        x, y = records[idx % len(records)]
        ids[row, :len(x)] = torch.tensor(x)
        labels[row, :len(y)] = torch.tensor(y)
        attention[row, :len(x)] = 1
    return {k: v.to(device) for k, v in {
        "input_ids": ids, "labels": labels, "attention_mask": attention,
    }.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("full", "selected", "chunk"), default="full")
    p.add_argument("--activation-checkpoint", action="store_true")
    p.add_argument("--seq-length", type=int, default=1024)
    p.add_argument("--padding", choices=("fixed", "dynamic"), default="fixed")
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--global-batch", type=int, default=8)
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--chunk-tokens", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=20261019)
    p.add_argument("--eval-examples", type=int, default=16)
    p.add_argument("--save-checkpoint", action="store_true")
    p.add_argument("--require-hosts", type=int, default=1)
    p.add_argument("--checkpoint-dir", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--stop-after", type=int)
    p.add_argument("--fingerprint", action="store_true")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--defer-gradient-sync", action="store_true")
    p.add_argument("--profile-step", type=int, default=-1)
    args = p.parse_args()
    backend = os.environ.get("LAB_DEVICE", "cuda")
    if backend == "npu":
        import torch_npu  # noqa: F401
    rank, local_rank, world = (int(os.environ.get(k, default)) for k, default in (
        ("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1"),
    ))
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    accelerator = getattr(torch, backend)
    accelerator.set_device(local_rank)
    device = torch.device(backend, local_rank)
    torch.manual_seed(args.seed)
    accelerator.manual_seed_all(args.seed)
    torch.set_num_threads(1)
    dist.init_process_group("hccl" if backend == "npu" else "nccl", timeout=timedelta(seconds=300))
    try:
        hosts = [None] * world
        dist.all_gather_object(hosts, platform.node())
        assert len(set(hosts)) >= args.require_hosts, f"Expected physical hosts: {hosts}"
        assert args.global_batch % (world * args.micro_batch) == 0
        assert args.steps > args.warmup >= 0
        accumulation = args.global_batch // (world * args.micro_batch)
        args.output.mkdir(parents=True, exist_ok=True)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        load_start = time.monotonic()
        lm = AutoModelForCausalLM.from_pretrained(
            args.assets / "model", local_files_only=True,
            torch_dtype=torch.float32, attn_implementation="sdpa",
        )
        parameter_count = sum(p.numel() for p in lm.parameters())
        trainable_count = sum(p.numel() for p in lm.parameters() if p.requires_grad)
        assert trainable_count == parameter_count
        lm.config.use_cache = False
        if args.activation_checkpoint:
            lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model = SupervisedModel(lm, args.mode, args.chunk_tokens).to(device)
        del lm
        shard, policy = fsdp_api()
        mp = policy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for layer in model.backbone.layers:
            shard(layer, mp_policy=mp, reshard_after_forward=True)
        # Keep tied input/output embedding under the same root FSDP owner.
        shard(model, mp_policy=mp, reshard_after_forward=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, foreach=False)
        tokenizer = AutoTokenizer.from_pretrained(args.assets / "model", local_files_only=True)
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        records = load_data(args.assets / "train.jsonl", args.seq_length)
        validation = load_data(args.assets / "test.jsonl", args.seq_length)
        contract = {
            key: getattr(args, key) for key in (
                "mode", "activation_checkpoint", "seq_length", "padding",
                "micro_batch", "global_batch", "chunk_tokens", "lr", "seed", "deterministic",
                "defer_gradient_sync",
            )
        }
        contract["assets_manifest_sha256"] = hashlib.sha256((args.assets / "manifest.json").read_bytes()).hexdigest()
        start_step = 0
        if args.resume:
            from checkpointing import load_checkpoint
            start_step = load_checkpoint(args.resume, model, optimizer, contract, accelerator)
            emit("checkpoint_restored", rank=rank, next_step=start_step)
        end_step = min(args.steps, args.stop_after) if args.stop_after is not None else args.steps
        assert end_step - start_step > args.warmup
        emit(
            "training_environment", rank=rank, host=platform.node(), world_size=world,
            device=backend, device_name=accelerator.get_device_name(local_rank),
            torch=torch.__version__, transformers=importlib.metadata.version("transformers"),
            parameters=parameter_count, trainable_parameters=trainable_count,
            load_seconds=time.monotonic() - load_start,
            args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            assets_manifest_sha256=hashlib.sha256((args.assets / "manifest.json").read_bytes()).hexdigest(),
        )

        @torch.no_grad()
        def evaluate():
            model.eval()
            totals = torch.zeros(2, device=device, dtype=torch.float32)
            # Equal forwards per rank; rounding is explicit in reported count.
            count = max(world, args.eval_examples // world * world)
            for i in range(rank, count, world):
                batch = batch_at(validation, [i], args.seq_length, pad, device, args.padding)
                with torch.autocast(backend, dtype=torch.bfloat16):
                    value = model(**batch)
                totals[0] += value.float()
                totals[1] += batch["labels"][:, 1:].ne(-100).sum()
            dist.all_reduce(totals)
            model.train()
            return {"loss": (totals[0] / totals[1]).item(), "tokens": int(totals[1].item()), "examples": count}

        start_fingerprints = None
        if args.fingerprint:
            from checkpointing import state_fingerprints
            start_fingerprints = state_fingerprints(model, optimizer)
            emit("training_start_fingerprints", rank=rank, state=start_fingerprints)
        initial_eval = evaluate()
        emit("eval_initial", rank=rank, **initial_eval)
        steps = []
        for local_step, step in enumerate(range(start_step, end_step)):
            # Preload outside timed region. This experiment measures the training
            # step with resident tokenized batches, not storage/network throughput.
            batches = []
            for micro in range(accumulation):
                start = step * args.global_batch + (micro * world + rank) * args.micro_batch
                batches.append(batch_at(
                    records, range(start, start + args.micro_batch),
                    args.seq_length, pad, device, args.padding,
                ))
            counts = torch.stack([
                sum((b["labels"][:, 1:].ne(-100).sum() for b in batches)),
                sum((b["attention_mask"].sum() for b in batches)),
                torch.tensor(sum(b["input_ids"].numel() for b in batches), device=device),
            ])
            dist.all_reduce(counts)
            supervised_tokens, nonpad_tokens, padded_tokens = [int(x.item()) for x in counts]
            assert supervised_tokens > 0
            dist.barrier()
            accelerator.synchronize()
            if local_step == args.warmup:
                accelerator.reset_peak_memory_stats()
            begin = time.monotonic()
            profiler = None
            if step == args.profile_step:
                if backend == "npu":
                    import torch_npu.profiler as profiler_api
                    activity = profiler_api.ProfilerActivity.NPU
                else:
                    profiler_api = torch.profiler
                    activity = profiler_api.ProfilerActivity.CUDA
                profiler = profiler_api.profile(
                    activities=[profiler_api.ProfilerActivity.CPU, activity],
                    record_shapes=True, profile_memory=True,
                )
            with profiler if profiler else nullcontext():
                optimizer.zero_grad(set_to_none=True)
                local_sum = torch.zeros((), device=device)
                for micro, batch in enumerate(batches):
                    if args.defer_gradient_sync:
                        # Keep full gradients until the last microbatch. This
                        # trades memory for fewer reduce-scatter collectives.
                        model.set_requires_gradient_sync(micro == accumulation - 1)
                    with torch.profiler.record_function("lab_forward_loss"):
                        with torch.autocast(backend, dtype=torch.bfloat16):
                            summed_loss = model(**batch)
                            # FSDP averages rank gradients. Compensate that average;
                            # never average the microbatch means when masks differ.
                            normalized = summed_loss * world / supervised_tokens
                    with torch.profiler.record_function("lab_backward"):
                        normalized.backward()
                    local_sum += summed_loss.detach()
                # DTensor norms require a global norm; no rank-local clipping.
                with torch.profiler.record_function("lab_optimizer"):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
                    optimizer.step()
                accelerator.synchronize()
            elapsed = torch.tensor(time.monotonic() - begin, device=device)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            dist.all_reduce(local_sum)
            if profiler:
                profiler.export_chrome_trace(str(args.output / f"trace-rank-{rank}.json"))
                (args.output / f"operators-rank-{rank}.txt").write_text(
                    profiler.key_averages().table(sort_by="self_device_time_total", row_limit=40),
                )
            loss_value = (local_sum / supervised_tokens).item()
            if not torch.isfinite(local_sum).item():
                raise FloatingPointError(f"Non-finite loss at step {step}")
            item = {
                "step": step, "measured": local_step >= args.warmup and profiler is None,
                "profiled": profiler is not None,
                "seconds": elapsed.item(), "loss": loss_value,
                "supervised_tokens": supervised_tokens, "nonpad_tokens": nonpad_tokens,
                "padded_tokens": padded_tokens,
            }
            steps.append(item)
            if rank == 0:
                emit("step", **item)
                (args.output / "steps.json").write_text(json.dumps(steps, indent=2))
        peak = torch.tensor([
            accelerator.max_memory_allocated(), accelerator.max_memory_reserved(),
        ], device=device, dtype=torch.int64)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        final_eval = evaluate()
        measured = [s for s in steps if s["measured"]]
        duration = sum(s["seconds"] for s in measured)
        result = {
            "world_size": world, "device": backend, "mode": args.mode,
            "hosts": sorted(set(hosts)), "padding": args.padding,
            "activation_checkpoint": args.activation_checkpoint,
            "deterministic": args.deterministic,
            "defer_gradient_sync": args.defer_gradient_sync,
            "seq_length": args.seq_length, "micro_batch": args.micro_batch,
            "global_batch": args.global_batch, "parameters": parameter_count,
            "measured_steps": len(measured), "training_seconds": duration,
            "supervised_tokens_per_second": sum(s["supervised_tokens"] for s in measured) / duration,
            "nonpad_tokens_per_second": sum(s["nonpad_tokens"] for s in measured) / duration,
            "padded_tokens_per_second": sum(s["padded_tokens"] for s in measured) / duration,
            "peak_allocated_gib": peak[0].item() / 2**30,
            "peak_reserved_gib": peak[1].item() / 2**30,
            "eval_initial": initial_eval, "eval_final": final_eval,
            "start_step": start_step, "next_step": end_step,
            "steps": steps,
        }
        if args.fingerprint:
            from checkpointing import state_fingerprints
            result["state_fingerprints"] = state_fingerprints(model, optimizer)
            result["start_fingerprints"] = start_fingerprints
        if args.save_checkpoint:
            from checkpointing import save_checkpoint
            result["checkpoint_seconds"] = save_checkpoint(
                args.checkpoint_dir or args.output / "checkpoint",
                model, optimizer, end_step, contract, accelerator,
            )
        if rank == 0:
            (args.output / "result.json").write_text(json.dumps(result, indent=2))
            emit("experiment_complete", **{k: v for k, v in result.items() if k != "steps"})
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
