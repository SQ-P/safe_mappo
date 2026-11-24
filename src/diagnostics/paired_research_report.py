"""Descriptive paired estimates; training seeds, not episodes, are replicates."""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np


def _group(rows):
    grouped = defaultdict(dict)
    for row in rows:
        for key in ("train_seed", "seed", "update", "algorithm", "collision_mode", "task_config_fingerprint",
                    "safe_success", "reach_success", "collision_count", "terminated", "truncated"):
            if key not in row:
                raise ValueError(f"missing episode provenance/outcome: {key}")
        if any(type(row[key]) is not int for key in ("train_seed", "seed", "update")):
            raise ValueError("training seed, evaluation seed and update must be integers")
        train_seed, eval_seed = row["train_seed"], row["seed"]
        if eval_seed in grouped[train_seed]:
            raise ValueError("duplicate (train_seed, evaluation_seed); do not mix checkpoints")
        if not isinstance(row["task_config_fingerprint"], str) or not row["task_config_fingerprint"]:
            raise ValueError("empty task fingerprint")
        if row["algorithm"] != "ppo" or row["collision_mode"] != "terminal":
            raise ValueError("unsupported algorithm/collision mode provenance")
        if any(type(row[key]) is not bool for key in ("safe_success", "reach_success", "terminated", "truncated")):
            raise ValueError("episode outcomes must be boolean")
        if row["terminated"] == row["truncated"]:
            raise ValueError("final episode must be exactly terminated or truncated")
        if type(row["collision_count"]) not in {int, float}:
            raise ValueError("collision count must be numeric")
        count = float(row["collision_count"])
        if not math.isfinite(count) or count < 0 or not count.is_integer():
            raise ValueError("invalid collision count")
        if row["safe_success"] and (not row["reach_success"] or count > 0):
            raise ValueError("safe success contradicts reach/collision evidence")
        if row["reach_success"] and not row["terminated"]:
            raise ValueError("success cannot be a timeout")
        if row["collision_mode"] == "terminal":
            if row["safe_success"] != row["reach_success"] or (count > 0 and not row["terminated"]):
                raise ValueError("terminal-mode outcome is inconsistent")
        grouped[train_seed][eval_seed] = row
    if not grouped:
        raise ValueError("no evaluation episodes")
    # The fixed-update research protocol cannot pool different checkpoints/tasks.
    flat = [row for episodes in grouped.values() for row in episodes.values()]
    for key in ("update", "algorithm", "collision_mode", "task_config_fingerprint"):
        if len({row[key] for row in flat}) != 1:
            raise ValueError(f"mixed {key} within one research arm")
    return grouped


def _metrics(rows):
    rows = list(rows)
    return {
        "cfms": float(np.mean([row["safe_success"] for row in rows])),
        "reach_success_rate": float(np.mean([row["reach_success"] for row in rows])),
        "collision_rate": float(np.mean([row["collision_count"] > 0 for row in rows])),
        "obstacle_collision_rate": float(np.mean([
            float(row.get("obstacle_collision_count", 0)) > 0 for row in rows
        ])),
        "agent_collision_rate": float(np.mean([
            float(row.get("inter_agent_collision_count", 0)) > 0 for row in rows
        ])),
        "boundary_collision_rate": float(np.mean([
            float(row.get("boundary_collision_count", 0)) > 0 for row in rows
        ])),
        "timeout_rate": float(np.mean([row["truncated"] for row in rows])),
    }


def compare_paired_episodes(baseline_rows, candidate_rows, *, allow_task_difference=False,
                            bootstrap_samples=10000, bootstrap_seed=20260921):
    baseline, candidate = _group(baseline_rows), _group(candidate_rows)
    if set(baseline) != set(candidate):
        raise ValueError("training seed sets must match exactly; no silent dropping")
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    seeds = sorted(baseline)
    paired = []
    task_differences = []
    for seed in seeds:
        left, right = baseline[seed], candidate[seed]
        if set(left) != set(right):
            raise ValueError("evaluation seeds must match within each training-seed pair")
        left_row, right_row = next(iter(left.values())), next(iter(right.values()))
        if left_row["update"] != right_row["update"]:
            raise ValueError("paired checkpoints must have the same update")
        task_diff = left_row["task_config_fingerprint"] != right_row["task_config_fingerprint"]
        if task_diff and not allow_task_difference:
            raise ValueError("task fingerprints differ; explicitly declare the intended intervention")
        if task_diff:
            task_differences.append(seed)
        lm, rm = _metrics(left.values()), _metrics(right.values())
        paired.append({"train_seed": seed, "evaluation_episodes_per_arm": len(left),
                       "baseline": lm, "candidate": rm,
                       "candidate_minus_baseline": {key: rm[key] - lm[key] for key in lm}})
    rng = np.random.default_rng(bootstrap_seed)
    samples = rng.integers(0, len(seeds), size=(bootstrap_samples, len(seeds)))
    estimates = {}
    for metric in paired[0]["baseline"]:
        differences = np.array([item["candidate_minus_baseline"][metric] for item in paired])
        interval = (np.quantile(differences[samples].mean(axis=1), [.025, .975]).tolist()
                    if len(seeds) > 1 else None)
        estimates[metric] = {
            "baseline_seed_mean": float(np.mean([item["baseline"][metric] for item in paired])),
            "candidate_seed_mean": float(np.mean([item["candidate"][metric] for item in paired])),
            "paired_mean_difference": float(differences.mean()),
            "paired_median_difference": float(np.median(differences)),
            "positive_seed_count": int(np.count_nonzero(differences > 0.0)),
            "negative_seed_count": int(np.count_nonzero(differences < 0.0)),
            "zero_seed_count": int(np.count_nonzero(differences == 0.0)),
            "seed_bootstrap_percentile_95_interval": interval,
            "beneficial_direction": (
                "negative"
                if metric in {
                    "collision_rate", "obstacle_collision_rate",
                    "agent_collision_rate", "boundary_collision_rate",
                    "timeout_rate",
                }
                else "positive"
            ),
        }
    warnings = [
        "Descriptive estimates, not a COMMIT gate or proof of causality.",
        "Equal update does not prove equal environment-step/search budgets; verify configs and resource ledger.",
        "Heldout separation, preregistration and no checkpoint selection leakage must be verified externally.",
        "Intervals resample paired training seeds and are conditional on the evaluated episode batches; no multiplicity correction.",
    ]
    if len(seeds) < 5:
        warnings.append("Fewer than five independent training seeds: screening only; small-sample bootstrap coverage is unreliable.")
    return {"unit_of_replication": "training_seed", "training_seed_count": len(seeds),
            "seed_pairs": paired, "estimates": estimates, "warnings": warnings,
            "declared_task_difference_seeds": task_differences,
            "bootstrap_samples": bootstrap_samples, "bootstrap_seed": bootstrap_seed}
