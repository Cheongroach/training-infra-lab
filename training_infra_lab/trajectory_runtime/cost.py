"""Auditable profile fits. Validation residuals are guards, not OOM guarantees."""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass

import numpy as np

from .contracts import digest


def features(lengths, layout):
    if not lengths or any(type(x) is not int or x < 2 for x in lengths):
        raise ValueError("Expected nonempty valid sequence lengths")
    if layout == "padded":
        linear = len(lengths) * max(lengths)
        quadratic = len(lengths) * max(lengths) ** 2
    elif layout == "isolated_packed":
        linear, quadratic = sum(lengths), sum(x * x for x in lengths)
    else:
        raise ValueError("Unknown layout")
    return np.asarray([linear / 1024, quadratic / 1024**2, len(lengths), 1.0])


@dataclass(frozen=True)
class Observation:
    sample_id: str
    split: str
    context_sha256: str
    lengths: tuple[int, ...]
    seconds: float
    peak_bytes: int

    def __post_init__(self):
        if self.split not in ("profile", "validation", "evaluation"):
            raise ValueError("Explicit disjoint profile/validation/evaluation split required")
        if not self.sample_id or not math.isfinite(self.seconds) or self.seconds <= 0 or self.peak_bytes <= 0:
            raise ValueError("Invalid observation")


def nnls_small(matrix, targets):
    """Exact active-set enumeration for four nonnegative coefficients."""
    best = None
    for count in range(1, matrix.shape[1] + 1):
        for columns in itertools.combinations(range(matrix.shape[1]), count):
            weights = np.linalg.lstsq(matrix[:, columns], targets, rcond=None)[0]
            if min(weights) < -1e-9:
                continue
            coefficient = np.zeros(matrix.shape[1])
            coefficient[list(columns)] = np.maximum(weights, 0)
            error = float(np.square(matrix @ coefficient - targets).sum())
            if best is None or error < best[0]:
                best = error, coefficient
    if best is None:
        raise ValueError("No finite nonnegative fit")
    return best[1]


@dataclass(frozen=True)
class ProfileModel:
    context_sha256: str
    layout: str
    time_coefficients: tuple[float, ...]
    memory_coefficients: tuple[float, ...]
    min_length: int
    max_length: int
    max_sequences: int
    max_processed_tokens: int
    memory_residual_bytes: float
    max_relative_time_error: float
    observations_sha256: str
    admitted: bool

    def predict(self, lengths, context_sha256):
        if context_sha256 != self.context_sha256:
            raise ValueError("Profile does not match model/kernel/device/execution context")
        f = features(lengths, self.layout)
        if (not self.admitted or min(lengths) < self.min_length or max(lengths) > self.max_length
                or len(lengths) > self.max_sequences or f[0] * 1024 > self.max_processed_tokens):
            return None
        seconds = float(f @ self.time_coefficients)
        memory = (float(f @ self.memory_coefficients) + self.memory_residual_bytes) * 1.10
        return seconds, memory


def fit_profile(observations, context, *, layout="padded", max_time_error=0.25):
    identity = digest(context)
    if context.get("defer_sync"):
        raise ValueError("Accumulated unsharded gradients need a separately validated model")
    if any(o.context_sha256 != identity for o in observations):
        raise ValueError("Cannot mix runtime contexts")
    if len({o.sample_id for o in observations}) != len(observations):
        raise ValueError("Duplicate or leaking sample IDs")
    if any(o.split == "evaluation" for o in observations):
        raise ValueError("Evaluation data must not influence fitting/admission")
    train = [o for o in observations if o.split == "profile"]
    validation = [o for o in observations if o.split == "validation"]
    if len(train) < 4 or len(validation) < 2:
        raise ValueError("Need at least four profile and two held-out observations")
    matrix = np.stack([features(o.lengths, layout) for o in train])
    times = nnls_small(matrix, np.asarray([o.seconds for o in train]))
    memory = nnls_small(matrix, np.asarray([o.peak_bytes for o in train], dtype=np.float64))
    relative_error, residual = 0.0, 0.0
    for o in validation:
        f = features(o.lengths, layout)
        relative_error = max(relative_error, abs(float(f @ times) - o.seconds) / o.seconds)
        residual = max(residual, o.peak_bytes - float(f @ memory))
    lengths = [n for o in train for n in o.lengths]
    return ProfileModel(
        identity, layout, tuple(times), tuple(memory), min(lengths), max(lengths),
        max(len(o.lengths) for o in train), int(max(matrix[:, 0]) * 1024),
        residual, relative_error, digest([asdict(o) for o in observations]),
        relative_error <= max_time_error,
    )
