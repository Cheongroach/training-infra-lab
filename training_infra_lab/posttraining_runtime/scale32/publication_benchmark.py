"""Equal-content weight publication A/B inside a paused, real 32-GPU job."""

import hashlib
import json
from pathlib import Path
import time


def schedule():
    """Two warmups, then four balanced cycles; checksum is a separate factor."""
    cases = [(mode, checks) for checks in (False, True)
             for mode in ("broadcast", "broadcast_packed")]
    return [(True, mode, False) for mode in ("broadcast", "broadcast_packed")] + [
        (False, mode, checks)
        for cycle in range(4)
        for mode, checks in (cases if cycle % 2 == 0 else list(reversed(cases)))
    ]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def run(actor, info, rollout_id):
    import torch
    import torch.distributed as dist
    from miles.backends.megatron_utils.local_weight_checksum import _compute_weight_checksum_state
    from miles.utils import async_utils
    from miles.utils.audit_utils.checksum_utils import merge_inference_engine_ranks
    from miles.utils.distributed_utils import get_gloo_group

    if rollout_id != 2:
        return
    rank = dist.get_rank()
    group = get_gloo_group()
    updater = actor.weight_updater
    protocol = updater.protocol
    if actor.args.colocate or updater.is_lora or len(info.rollout_engines) != 8:
        raise ValueError("Benchmark requires non-colocated base weights and eight TP2 engines")
    root = Path(actor.args.save_debug_rollout_data).parent / "publication-benchmark"
    root.mkdir(exist_ok=True)
    path = root / f"rank-{rank}.json"
    if path.exists():
        raise ValueError("Refuse a second benchmark at the same frozen model state")
    version = actor._get_actor_weight_version()
    if version != 3:
        raise ValueError("Benchmark must follow the third real learner update")

    def local_state():
        return _compute_weight_checksum_state(
            model=actor.model, optimizer=actor.optimizer).model_dump(mode="json")

    def engine_state():
        if rank:
            return None
        futures = [async_utils.submit(client.check_weights(action="raw_checksum"))
                   for client in info.rollout_engines]
        bodies = [future.result(timeout=120) for future in futures]
        states = [merge_inference_engine_ranks(body) for body in bodies]
        if not states or any(state != states[0] for state in states):
            raise ValueError("Engine contents disagree during publication benchmark")
        if not all(any(name.startswith(f"rank{r}/") for name in states[0]) for r in (0, 1)):
            raise ValueError("Missing TP shard in engine checksum")
        return states

    expected_local = local_state()
    expected_engines = engine_state()
    (root / f"local-reference-rank-{rank}.json").write_text(json.dumps(expected_local, sort_keys=True))
    if not rank:
        (root / "engine-reference.json").write_text(json.dumps(expected_engines, sort_keys=True))
    original_send = protocol.send_bucket
    original_mode = actor.args.update_weight_transfer_mode
    observed = []
    report = {
        "passed": False, "rank": rank, "world_size": dist.get_world_size(),
        "weight_version": version, "engine_count": 8, "rollout_id": rollout_id,
        "scope": "Repeated publication of identical post-update weights; no end-to-end training speedup",
        "profiler_enabled": False, "checksum_scope": "Full learner model/Adam plus all engine TP-shard SHA256",
        "verification_outside_timer_for_checksum_off": True, "iterations": [],
    }

    def send(bucket):
        observed.append([
            {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape),
             "bytes": tensor.numel() * tensor.element_size()}
            for name, tensor in bucket
        ])
        original_send(bucket)

    protocol.send_bucket = send
    try:
        for number, (warmup, mode, checks) in enumerate(schedule()):
            actor.args.update_weight_transfer_mode = mode
            observed.clear()
            dist.barrier(group=group)
            torch.cuda.synchronize()
            start = time.perf_counter()
            sent = updater.update_weights(weight_version=version)
            torch.cuda.synchronize()
            transfer_seconds = time.perf_counter() - start
            local = local_state() if checks else None
            engines = engine_state() if checks else None
            dist.barrier(group=group)
            elapsed = time.perf_counter() - start
            # Correctness work is outside the measured interval in checksum-off cases.
            if not checks:
                local, engines = local_state(), engine_state()
            if not sent or local != expected_local or engines != expected_engines:
                raise ValueError(f"Publication changed learner/optimizer/engine contents in case {number}")
            if any(cell.is_errored for cell in protocol.cell_updaters_of_cell_id.values()):
                raise ValueError("A receiving engine failed")
            if number == 0:
                (root / f"sender-layout-rank-{rank}.json").write_text(json.dumps(observed, sort_keys=True))
            own = {
                "rank": rank, "transfer_seconds": transfer_seconds, "measured_seconds": elapsed,
                "sender": protocol.is_sender, "buckets": len(observed),
                "tensors": sum(len(bucket) for bucket in observed),
                "bytes": sum(tensor["bytes"] for bucket in observed for tensor in bucket),
                "layout_sha256": fingerprint(observed),
                "local_state_sha256": fingerprint(local),
                "engine_state_sha256": fingerprint(engines),
            }
            ranks = [None] * dist.get_world_size(group=group)
            dist.all_gather_object(ranks, own, group=group)
            report["iterations"].append({
                "number": number, "warmup": warmup, "mode": mode, "checksums": checks,
                "measured_seconds": max(item["measured_seconds"] for item in ranks),
                "transfer_seconds": max(item["transfer_seconds"] for item in ranks),
                "rank_measurements": ranks, "verified": True,
            })
        report["passed"] = True
    finally:
        protocol.send_bucket = original_send
        actor.args.update_weight_transfer_mode = original_mode
        path.write_text(json.dumps(report, indent=2, allow_nan=False))


def install(repo, report_path):
    """Patch only an exact method boundary in the disposable Miles checkout."""
    import ast

    target = Path(repo) / "miles/backends/megatron_utils/actor.py"
    before = target.read_bytes()
    old = '            print_memory("after update_weights")\n'
    new = old + (
        '            if __import__("os").environ.get("GRPO_PUBLICATION_BENCHMARK") == "1":\n'
        '                from publication_benchmark import run as benchmark_publication\n'
        '                benchmark_publication(self, info, rollout_id)\n'
    )
    source = before.decode()
    if source.count(old) != 1:
        raise ValueError("Unexpected Miles actor boundary")
    after = source.replace(old, new).encode()
    ast.parse(after)
    target.write_bytes(after)
    Path(report_path).write_text(json.dumps({
        "path": str(target), "before_sha256": hashlib.sha256(before).hexdigest(),
        "after_sha256": hashlib.sha256(after).hexdigest(), "gpu_verified": False,
    }, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/root/miles"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    install(args.repo, args.report)
