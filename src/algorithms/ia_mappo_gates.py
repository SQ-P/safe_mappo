"""Preregistered paired-bootstrap decision gates for IA-MAPPO R5."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PairedMetricIntervalV1:
    estimate: float
    lower_95: float
    upper_95: float


@dataclass(frozen=True)
class FormalR5GateResultV1:
    passed: bool
    metrics: Mapping[str, PairedMetricIntervalV1]
    pair_collision_count: int
    boundary_collision_count: int
    failures: tuple[str, ...]
    schema_version: str = "FormalR5GateResultV1"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def two_level_paired_bootstrap(
    values: Sequence[float],
    train_seeds: Sequence[int],
    layout_ids: Sequence[int],
    *,
    replicates: int = 5000,
    seed: int = 0,
) -> PairedMetricIntervalV1:
    """Resample training seeds, then paired layout rows within each seed."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    train_seeds = np.asarray(train_seeds).reshape(-1)
    layout_ids = np.asarray(layout_ids).reshape(-1)
    if not (values.shape == train_seeds.shape == layout_ids.shape):
        raise ValueError("paired bootstrap arrays must align")
    if not np.isfinite(values).all() or replicates <= 0:
        raise ValueError("paired bootstrap values/replicates are invalid")
    unique_seeds = np.unique(train_seeds)
    if len(unique_seeds) < 2:
        raise ValueError("two-level bootstrap requires at least two training seeds")
    rows = {value: np.flatnonzero(train_seeds == value) for value in unique_seeds}
    for value, indices in rows.items():
        if len(np.unique(layout_ids[indices])) != len(indices):
            raise ValueError(f"duplicate layout within training seed {value}")
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(replicates), dtype=np.float64)
    for index in range(int(replicates)):
        sampled_seeds = rng.choice(unique_seeds, len(unique_seeds), replace=True)
        seed_means = []
        for sampled_seed in sampled_seeds:
            indices = rows[sampled_seed]
            sampled_rows = rng.choice(indices, len(indices), replace=True)
            seed_means.append(float(np.mean(values[sampled_rows])))
        samples[index] = float(np.mean(seed_means))
    return PairedMetricIntervalV1(
        estimate=float(np.mean(values)),
        lower_95=float(np.quantile(samples, 0.025)),
        upper_95=float(np.quantile(samples, 0.975)),
    )


def evaluate_formal_r5_gate(
    *,
    deltas: Mapping[str, Sequence[float]],
    train_seeds: Sequence[int],
    layout_ids: Sequence[int],
    pair_collision_count: int,
    boundary_collision_count: int,
    replicates: int = 5000,
    seed: int = 0,
) -> FormalR5GateResultV1:
    """Apply the fixed R5 noninferiority and modification-reduction rules."""

    required = {
        "material_modification", "cfms", "progress", "collision", "timeout",
    }
    missing = required - set(deltas)
    if missing:
        raise ValueError(f"missing formal R5 deltas: {sorted(missing)}")
    intervals = {
        name: two_level_paired_bootstrap(
            deltas[name], train_seeds, layout_ids,
            replicates=replicates, seed=int(seed) + index,
        )
        for index, name in enumerate(sorted(required))
    }
    failures = []
    modification = intervals["material_modification"]
    if modification.estimate > -0.05 or modification.upper_95 >= 0.0:
        failures.append("material_modification_gate_failed")
    if intervals["cfms"].lower_95 < -0.025:
        failures.append("cfms_noninferiority_failed")
    if intervals["progress"].lower_95 < -0.02:
        failures.append("progress_noninferiority_failed")
    if intervals["collision"].upper_95 > 0.025:
        failures.append("collision_guardrail_failed")
    if intervals["timeout"].upper_95 > 0.025:
        failures.append("timeout_guardrail_failed")
    if int(pair_collision_count) != 0:
        failures.append("pair_collision_nonzero")
    if int(boundary_collision_count) != 0:
        failures.append("boundary_collision_nonzero")
    return FormalR5GateResultV1(
        passed=not failures, metrics=intervals,
        pair_collision_count=int(pair_collision_count),
        boundary_collision_count=int(boundary_collision_count),
        failures=tuple(failures),
    )
