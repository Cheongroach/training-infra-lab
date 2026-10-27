"""Loss/gradient/update gates, including unequal masks across ranks and micros."""

from __future__ import annotations

import copy
import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist

from loss import full_logits_sum, supervised_chunk_sum


def fsdp_api():
    try:
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    except ImportError:
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
    return fully_shard, MixedPrecisionPolicy


class ToyLM(torch.nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.embedding = torch.nn.Embedding(97, 32)
        self.block = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.GELU())
        self.head = torch.nn.Linear(32, 97, bias=False)
        self.head.weight = self.embedding.weight
        self.mode = mode

    def forward(self, ids, labels):
        hidden = self.block(self.embedding(ids))
        if self.mode == "full":
            return full_logits_sum(hidden, self.head, labels)
        return supervised_chunk_sum(hidden, self.head, labels, 7)


def main():
    backend = os.environ.get("LAB_DEVICE", "cpu")
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if backend == "npu":
        import torch_npu  # noqa: F401
    if backend != "cpu":
        getattr(torch, backend).set_device(int(os.environ.get("LOCAL_RANK", 0)))
    device = torch.device(backend, int(os.environ.get("LOCAL_RANK", 0))) if backend != "cpu" else torch.device("cpu")
    torch.manual_seed(73)
    torch.set_num_threads(1)
    if world > 1:
        dist.init_process_group("hccl" if backend == "npu" else "nccl", timeout=timedelta(seconds=180))
    try:
        # Unequal token counts, ignored prompt/padding, and an entirely ignored row.
        ids = torch.randint(0, 97, (4 * world, 13), device=device)
        labels = ids.clone()
        for row in range(labels.size(0)):
            labels[row, : 1 + row % 9] = -100
            labels[row, -2:] = -100
        labels[0] = -100
        denom = labels[:, 1:].ne(-100).sum()
        ref = ToyLM("full").to(device)
        opt = copy.deepcopy(ref)
        opt.mode = "chunk"
        ref_loss = ref(ids, labels) / denom
        ref_loss.backward()
        # Accumulate unevenly sized micros with the same global denominator.
        opt_loss = torch.zeros((), device=device)
        for i in range(0, len(ids), 2):
            value = opt(ids[i : i + 2], labels[i : i + 2]) / denom
            value.backward()
            opt_loss += value.detach()
        torch.testing.assert_close(ref_loss.detach(), opt_loss, rtol=2e-5, atol=2e-5)
        for a, b in zip(ref.parameters(), opt.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-4, atol=2e-6)
        zero = copy.deepcopy(ref)
        zero.zero_grad(set_to_none=True)
        zero.mode = "chunk"
        zero_loss = zero(ids[:1], torch.full_like(labels[:1], -100))
        zero_loss.backward()
        assert zero_loss.item() == 0
        assert all(p.grad is not None and p.grad.count_nonzero().item() == 0 for p in zero.parameters())
        print(json.dumps({"event": "loss_gradient_gate_passed", "rank": rank, "loss": ref_loss.item(), "tokens": denom.item()}), flush=True)

        if backend != "cpu":
            # BF16 changes reduction order; report relative L2 gradient errors.
            a, b = copy.deepcopy(ref), copy.deepcopy(ref)
            a.zero_grad(set_to_none=True)
            b.zero_grad(set_to_none=True)
            b.mode = "chunk"
            with torch.autocast(backend, dtype=torch.bfloat16):
                la, lb = a(ids, labels) / denom, b(ids, labels) / denom
            la.backward()
            lb.backward()
            errors = [(x.grad - y.grad).float().norm() / x.grad.float().norm().clamp_min(1e-10) for x, y in zip(a.parameters(), b.parameters())]
            worst = max(x.item() for x in errors)
            assert worst < 0.02, worst
            torch.testing.assert_close(la.detach(), lb.detach(), rtol=0.003, atol=0.003)
            print(json.dumps({"event": "bf16_loss_gate_passed", "rank": rank, "max_relative_l2_grad_error": worst}), flush=True)

        if world > 1:
            for deferred in (False, True):
                sharded = copy.deepcopy(ref)
                sharded.zero_grad(set_to_none=True)
                sharded.mode = "chunk"
                shard, policy = fsdp_api()
                shard(sharded.block)
                shard(sharded)
                optimizer = torch.optim.SGD(sharded.parameters(), lr=0.01)
                local = list(range(rank, len(ids), world))
                for micro, idx in enumerate(local):
                    if deferred:
                        sharded.set_requires_gradient_sync(micro == len(local) - 1)
                    (sharded(ids[idx : idx + 1], labels[idx : idx + 1]) * world / denom).backward()
                for actual, expected in zip(sharded.parameters(), ref.parameters()):
                    torch.testing.assert_close(actual.grad.full_tensor(), expected.grad, rtol=2e-4, atol=3e-6)
                optimizer.step()
                for actual, expected in zip(sharded.parameters(), ref.parameters()):
                    torch.testing.assert_close(actual.full_tensor(), expected - 0.01 * expected.grad, rtol=2e-4, atol=3e-6)
                print(json.dumps({"event": "fsdp2_global_token_update_gate_passed", "rank": rank,
                                  "world_size": world, "defer_gradient_sync": deferred}), flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
