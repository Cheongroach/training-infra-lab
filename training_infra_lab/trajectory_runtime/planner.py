"""All-rank slot schedules. Fixed topology; no sequence truncation or packing."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .baselines import karmarkar_karp
from .contracts import UpdateBatch, digest


class CapacityError(ValueError):
    pass


@dataclass(frozen=True)
class Plan:
    batch_sha256: str
    # slots[slot][rank] contains zero or more original batch indices.
    slots: tuple[tuple[tuple[int, ...], ...], ...]
    strategy: str
    max_padded_tokens: int
    defer_sync: bool = False
    activation_checkpoint: bool = False
    schema: int = 1

    @property
    def world_size(self):
        return len(self.slots[0]) if self.slots else 0

    @property
    def sha256(self):
        return digest(asdict(self))

    def validate(self, batch):
        if self.batch_sha256 != batch.sha256 or not self.slots:
            raise ValueError("Wrong batch or empty plan")
        seen = []
        for slot in self.slots:
            if len(slot) != self.world_size or not any(slot):
                raise ValueError("Invalid aligned slot")
            for indices in slot:
                seen.extend(indices)
                if any(type(i) is not int or not 0 <= i < len(batch.trajectories) for i in indices):
                    raise ValueError("Invalid trajectory index")
                if indices:
                    padded = len(indices) * max(len(batch.trajectories[i].input_ids) for i in indices)
                    if padded > self.max_padded_tokens:
                        raise CapacityError("Padded microbatch exceeds admission budget")
        if sorted(seen) != list(range(len(batch.trajectories))):
            raise ValueError("Missing or duplicated trajectory")


def plan_update(batch: UpdateBatch, world_size: int, *, strategy="fixed",
                max_padded_tokens=4096, micro_batch=1, defer_sync=False,
                activation_checkpoint=False) -> Plan:
    """Baselines plus slot sorting; profiled planning is a separate opt-in.

    Token/verl policies retain equal sample counts and use KK partitioning.
    slot_sorted groups similar lengths across ranks, reducing each slot's
    straggler for micro_batch=1. It is a simple existing sorting heuristic,
    not an architecture-aware learned planner.
    """
    n = len(batch.trajectories)
    lengths = [len(t.input_ids) for t in batch.trajectories]
    if world_size < 1 or micro_batch < 1 or max_padded_tokens < 2:
        raise ValueError("Invalid execution dimensions")
    if max(lengths) > max_padded_tokens:
        raise CapacityError("A single trajectory exceeds capacity; do not silently truncate")
    if strategy == "fixed":
        partitions = [list(range(r, n, world_size)) for r in range(world_size)]
    elif strategy in ("token", "verl"):
        weights = lengths if strategy == "token" else [24576 * x + x * x for x in lengths]
        partitions = karmarkar_karp(weights, world_size, equal_size=True)
        # Tune the baseline too: consistent local descending order aligns heavy slots.
        partitions = [sorted(p, key=lambda i: (-lengths[i], i)) for p in partitions]
    elif strategy == "slot_sorted":
        order = sorted(range(n), key=lambda i: (-lengths[i], i))
        partitions = [order[r::world_size] for r in range(world_size)]
    else:
        raise ValueError(strategy)
    micros = []
    for partition in partitions:
        local, current = [], []
        for i in partition:
            trial = current + [i]
            if len(trial) > micro_batch or len(trial) * max(lengths[j] for j in trial) > max_padded_tokens:
                local.append(tuple(current))
                current = []
            current.append(i)
        if current:
            local.append(tuple(current))
        micros.append(local)
    count = max(map(len, micros))
    slots = tuple(tuple(local[s] if s < len(local) else () for local in micros) for s in range(count))
    plan = Plan(batch.sha256, slots, strategy, max_padded_tokens, defer_sync, activation_checkpoint)
    plan.validate(batch)
    return plan


def estimated_slot_cost(plan, lengths, cost):
    """Dummy forward costs included. Not a measurement or exact FSDP model."""
    return math.fsum(max(cost(tuple(lengths[i] for i in ids) or (2,)) for ids in slot)
                     for slot in plan.slots)


def choose_profiled_plan(batch, world_size, profile, context_sha256, *,
                         memory_budget_bytes, safe_token_budget, micro_batches=(1, 2, 4),
                         activation_checkpoint=False, minimum_predicted_gain=0.10):
    """Bounded plan search with explicit baseline fallback and admission reasons.

    safe_token_budget must have independently passed the device capacity gate.
    No candidate changes recomputation/synchronization or the update data.
    """
    baseline = plan_update(
        batch, world_size, strategy="verl", max_padded_tokens=safe_token_budget,
        activation_checkpoint=activation_checkpoint,
    )
    lengths = [len(t.input_ids) for t in batch.trajectories]
    diagnostics = []

    def score(plan):
        slot_costs = []
        for slot in plan.slots:
            predictions = [profile.predict(tuple(lengths[i] for i in ids) or (2,), context_sha256)
                           for ids in slot]
            if any(p is None for p in predictions):
                return None, "outside_validated_profile"
            if any(p[1] > memory_budget_bytes for p in predictions):
                return None, "memory_guard_rejected"
            slot_costs.append(max(p[0] for p in predictions))
        return math.fsum(slot_costs), "predicted"

    baseline_score, reason = score(baseline)
    if baseline_score is None:
        if reason == "memory_guard_rejected":
            raise CapacityError("Even the baseline violates the measured memory guard")
        return baseline, {"selected": "baseline_fallback", "reason": reason}
    candidates = [(baseline_score, baseline)]
    for strategy in ("token", "verl", "slot_sorted"):
        for micro in micro_batches:
            candidate = plan_update(
                batch, world_size, strategy=strategy, max_padded_tokens=safe_token_budget,
                micro_batch=micro, activation_checkpoint=activation_checkpoint,
            )
            predicted, reason = score(candidate)
            diagnostics.append({"strategy": strategy, "micro_batch": micro,
                                "predicted_seconds": predicted, "reason": reason})
            if predicted is not None:
                candidates.append((predicted, candidate))
    best_score, best = min(candidates, key=lambda item: item[0])
    if best_score > baseline_score * (1 - minimum_predicted_gain):
        best_score, best = baseline_score, baseline
    return best, {
        "selected": best.strategy, "baseline_predicted_seconds": baseline_score,
        "selected_predicted_seconds": best_score, "candidates": diagnostics,
        "note": "Profile prediction; actual wall time must include planning overhead",
    }
