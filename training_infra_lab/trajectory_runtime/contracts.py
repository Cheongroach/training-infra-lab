"""Immutable actor-update inputs. Masks/log-probs index the NEXT token."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass


def digest(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


@dataclass(frozen=True)
class Trajectory:
    trajectory_id: str
    prompt_group_id: str
    policy_version: int
    input_ids: tuple[int, ...]
    response_mask: tuple[int, ...]
    old_log_probs: tuple[float, ...]
    reward: float
    reference_log_probs: tuple[float, ...] = ()

    def __post_init__(self):
        if not self.trajectory_id or not self.prompt_group_id:
            raise ValueError("Trajectory/group identity is required")
        if self.policy_version < 0 or len(self.input_ids) < 2:
            raise ValueError("Invalid policy version or empty sequence")
        if any(type(x) is not int or x < 0 for x in self.input_ids):
            raise ValueError("Token IDs must be nonnegative integers")
        n = len(self.input_ids) - 1
        if len(self.response_mask) != n or len(self.old_log_probs) != n:
            raise ValueError("Mask/log-probs must describe input_ids[1:]")
        if set(self.response_mask) - {0, 1} or not sum(self.response_mask):
            raise ValueError("Every trajectory needs a nonempty binary response mask")
        if self.reference_log_probs and len(self.reference_log_probs) != n:
            raise ValueError("Reference log-probs shape mismatch")
        if not all(math.isfinite(x) for x in (
            self.reward, *self.old_log_probs, *self.reference_log_probs,
        )):
            raise ValueError("Nonfinite rollout input")

    @classmethod
    def from_dict(cls, item):
        item = dict(item)
        for key in ("input_ids", "response_mask", "old_log_probs", "reference_log_probs"):
            if key in item:
                item[key] = tuple(item[key])
        return cls(**item)


@dataclass(frozen=True)
class UpdateBatch:
    trajectories: tuple[Trajectory, ...]
    update_id: int
    policy_version: int
    group_size: int
    reduction: str = "sequence_mean"

    def __post_init__(self):
        if self.update_id < 0 or self.group_size < 2 or not self.trajectories:
            raise ValueError("Invalid update identity or GRPO group size")
        if self.reduction not in ("sequence_mean", "token_mean"):
            raise ValueError("Unsupported reduction")
        ids, groups = set(), defaultdict(list)
        for t in self.trajectories:
            if t.trajectory_id in ids or t.policy_version != self.policy_version:
                raise ValueError("Duplicate trajectory or stale policy")
            ids.add(t.trajectory_id)
            groups[t.prompt_group_id].append(t)
        if any(len(g) != self.group_size for g in groups.values()):
            raise ValueError("Incomplete prompt group; never normalize a local subgroup")

    @property
    def sha256(self):
        return digest(asdict(self))

    @property
    def denominator(self):
        if self.reduction == "sequence_mean":
            return len(self.trajectories)
        return sum(sum(t.response_mask) for t in self.trajectories)

    def advantages(self, epsilon=1e-6) -> tuple[float, ...]:
        """Whole-group sample standard deviation, matching verl GRPO."""
        groups = defaultdict(list)
        for t in self.trajectories:
            groups[t.prompt_group_id].append(t.reward)
        stats = {}
        for key, rewards in groups.items():
            mean = sum(rewards) / len(rewards)
            std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / (len(rewards) - 1))
            stats[key] = mean, std
        return tuple((t.reward - stats[t.prompt_group_id][0]) /
                     (stats[t.prompt_group_id][1] + epsilon) for t in self.trajectories)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["trajectories"] = tuple(Trajectory.from_dict(t) for t in value["trajectories"])
        return cls(**value)
