"""Preregistered paired analysis for the CSA-MAPPO paper matrix."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


METRICS = (
    "avoidable_intervention_rate", "cfms", "goal_progress_ratio",
    "collision_rate", "timeout_rate", "agent_collision_rate",
    "boundary_collision_rate",
)


def _finite_rate(value, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite rate in [0,1]")
    return result


def _row_metrics(row: Mapping) -> dict[str, float]:
    required = {
        "avoidable_intervention_rate", "goal_progress_ratio", "safe_success",
        "collision_count", "truncated", "inter_agent_collision_count",
        "boundary_collision_count",
    }
    missing = required - set(row)
    if missing:
        raise ValueError(f"missing CSA outcome fields: {sorted(missing)}")
    if type(row["safe_success"]) is not bool or type(row["truncated"]) is not bool:
        raise ValueError("safe_success and truncated must be boolean")
    return {
        "avoidable_intervention_rate": _finite_rate(row["avoidable_intervention_rate"], "avoidable_intervention_rate"),
        "cfms": float(row["safe_success"]),
        "goal_progress_ratio": _finite_rate(row["goal_progress_ratio"], "goal_progress_ratio"),
        "collision_rate": float(float(row["collision_count"]) > 0.0),
        "timeout_rate": float(row["truncated"]),
        "agent_collision_rate": float(float(row["inter_agent_collision_count"]) > 0.0),
        "boundary_collision_rate": float(float(row["boundary_collision_count"]) > 0.0),
    }


def _group(
    rows: Iterable[Mapping], *, expected_train_seeds: Sequence[int],
    expected_layout_count: int,
) -> dict[int, dict[int, Mapping]]:
    grouped: dict[int, dict[int, Mapping]] = defaultdict(dict)
    for row in rows:
        for key in ("train_seed", "seed", "task_config_fingerprint", "method"):
            if key not in row:
                raise ValueError(f"missing CSA provenance field: {key}")
        train_seed, layout = row["train_seed"], row["seed"]
        if type(train_seed) is not int or type(layout) is not int:
            raise ValueError("train_seed and layout seed must be integers")
        if layout in grouped[train_seed]:
            raise ValueError("duplicate (train_seed, layout_seed)")
        if not isinstance(row["task_config_fingerprint"], str) or not row["task_config_fingerprint"]:
            raise ValueError("task_config_fingerprint must be non-empty")
        _row_metrics(row)
        grouped[train_seed][layout] = row
    if set(grouped) != set(map(int, expected_train_seeds)):
        raise ValueError("formal training seed set is incomplete or unexpected")
    for seed, panel in grouped.items():
        if len(panel) != int(expected_layout_count):
            raise ValueError(f"training seed {seed} does not contain the frozen layout count")
    flat = [row for panel in grouped.values() for row in panel.values()]
    if len({row["task_config_fingerprint"] for row in flat}) != 1:
        raise ValueError("mixed task fingerprints within one method arm")
    if len({row["method"] for row in flat}) != 1:
        raise ValueError("mixed methods within one arm")
    return dict(grouped)


def compare_csa_formal(
    baseline_rows: Iterable[Mapping], candidate_rows: Iterable[Mapping], *,
    expected_train_seeds: Sequence[int] = (23, 41, 67),
    expected_layout_count: int = 200,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20261116,
) -> dict:
    """Apply two-level paired bootstrap and every preregistered success gate."""

    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    baseline = _group(
        baseline_rows, expected_train_seeds=expected_train_seeds,
        expected_layout_count=expected_layout_count,
    )
    candidate = _group(
        candidate_rows, expected_train_seeds=expected_train_seeds,
        expected_layout_count=expected_layout_count,
    )
    if next(iter(next(iter(baseline.values())).values()))["method"] != "standard_mappo":
        raise ValueError("formal baseline method must be standard_mappo")
    fingerprints = {
        next(iter(next(iter(arm.values())).values()))["task_config_fingerprint"]
        for arm in (baseline, candidate)
    }
    if len(fingerprints) != 1:
        raise ValueError("formal arms must share the exact task fingerprint")

    seeds = list(map(int, expected_train_seeds))
    paired_arrays: dict[str, list[np.ndarray]] = {name: [] for name in METRICS}
    seed_results = []
    for seed in seeds:
        if set(baseline[seed]) != set(candidate[seed]):
            raise ValueError("paired layout panels differ; no silent dropping")
        layouts = sorted(baseline[seed])
        left = [_row_metrics(baseline[seed][layout]) for layout in layouts]
        right = [_row_metrics(candidate[seed][layout]) for layout in layouts]
        differences = {}
        for metric in METRICS:
            values = np.asarray([r[metric] - l[metric] for l, r in zip(left, right)])
            paired_arrays[metric].append(values)
            differences[metric] = float(values.mean())
        seed_results.append({
            "train_seed": seed,
            "layout_count": len(layouts),
            "candidate_minus_standard": differences,
        })

    rng = np.random.default_rng(int(bootstrap_seed))
    sampled = {metric: np.empty(bootstrap_samples, dtype=np.float64) for metric in METRICS}
    for draw in range(bootstrap_samples):
        seed_indices = rng.integers(0, len(seeds), size=len(seeds))
        for metric in METRICS:
            seed_means = []
            for seed_index in seed_indices:
                values = paired_arrays[metric][int(seed_index)]
                layout_indices = rng.integers(0, len(values), size=len(values))
                seed_means.append(float(values[layout_indices].mean()))
            sampled[metric][draw] = float(np.mean(seed_means))

    estimates = {}
    for metric in METRICS:
        differences = np.asarray([values.mean() for values in paired_arrays[metric]])
        baseline_mean = float(np.mean([
            np.mean([_row_metrics(row)[metric] for row in baseline[seed].values()])
            for seed in seeds
        ]))
        candidate_mean = baseline_mean + float(differences.mean())
        interval = np.quantile(sampled[metric], (0.025, 0.975)).tolist()
        estimates[metric] = {
            "standard_seed_mean": baseline_mean,
            "candidate_seed_mean": candidate_mean,
            "paired_mean_difference": float(differences.mean()),
            "hierarchical_paired_bootstrap_95_ci": [float(item) for item in interval],
            "per_seed_differences": [float(item) for item in differences],
        }

    avoidable = estimates["avoidable_intervention_rate"]
    relative_reduction = (
        -avoidable["paired_mean_difference"] / avoidable["standard_seed_mean"]
        if avoidable["standard_seed_mean"] > 0.0 else float("-inf")
    )
    gates = {
        "avoidable_ci_upper_below_zero": avoidable["hierarchical_paired_bootstrap_95_ci"][1] < 0.0,
        "avoidable_relative_reduction_at_least_15pct": relative_reduction >= 0.15,
        "avoidable_improves_all_training_seeds": all(
            item < 0.0 for item in avoidable["per_seed_differences"]
        ),
        "cfms_noninferior": estimates["cfms"]["hierarchical_paired_bootstrap_95_ci"][0] > -0.05,
        "goal_progress_noninferior": estimates["goal_progress_ratio"]["hierarchical_paired_bootstrap_95_ci"][0] > -0.03,
        "collision_noninferior": estimates["collision_rate"]["hierarchical_paired_bootstrap_95_ci"][1] < 0.02,
        "timeout_noninferior": estimates["timeout_rate"]["hierarchical_paired_bootstrap_95_ci"][1] < 0.05,
        "agent_collision_not_increased": estimates["agent_collision_rate"]["paired_mean_difference"] <= 0.0,
        "boundary_collision_not_increased": estimates["boundary_collision_rate"]["paired_mean_difference"] <= 0.0,
    }
    return {
        "schema_version": "CSAMAPPOFormalComparisonV1",
        "unit_of_replication": "training_seed",
        "paired_layout_unit": "layout_seed",
        "training_seeds": seeds,
        "layouts_per_seed": int(expected_layout_count),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "seed_results": seed_results,
        "estimates": estimates,
        "avoidable_relative_reduction": float(relative_reduction),
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "claim_authorized": all(gates.values()),
    }
