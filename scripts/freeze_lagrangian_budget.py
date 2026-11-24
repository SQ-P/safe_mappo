"""Freeze the 90% MAPPO-Lagrangian budget from parent validation rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.utils.io_utils import save_json


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-validation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    source = Path(args.parent_validation).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError(f"cost budget output already exists: {output}")
    rows = [
        json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or {int(row["train_seed"]) for row in rows} != {23}:
        raise ValueError("budget must use only the registered seed-23 parent validation")
    if len({int(row["seed"]) for row in rows}) != 40:
        raise ValueError("budget requires exactly 40 unique parent-validation layouts")
    task_fingerprints = {str(row["task_config_fingerprint"]) for row in rows}
    checkpoint_hashes = {str(row["checkpoint_sha256"]) for row in rows}
    if len(task_fingerprints) != 1 or len(checkpoint_hashes) != 1:
        raise ValueError("budget rows mix task contracts or parent checkpoints")
    rates = [float(row["material_intervention_rate"]) for row in rows]
    if any(not 0.0 <= value <= 1.0 for value in rates):
        raise ValueError("material intervention rates must be in [0,1]")
    parent_mean = sum(rates) / len(rates)
    result = {
        "schema_version": "MAPPOLagrangianCostBudgetV1",
        "train_seed": 23,
        "validation_layout_count": 40,
        "parent_mean_material_intervention_rate": parent_mean,
        "budget_fraction": 0.90,
        "cost_limit": 0.90 * parent_mean,
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "frozen_for_all_training_seeds": True,
        "task_config_fingerprint": task_fingerprints.pop(),
        "seed23_parent_checkpoint_sha256": checkpoint_hashes.pop(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(result, output)
    return result


if __name__ == "__main__":
    main()
