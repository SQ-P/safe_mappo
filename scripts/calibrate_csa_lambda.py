"""Freeze CSA lambda from seed-23 fixed-rollout gradient diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.algorithms.safety_alignment import select_alignment_lambda
from src.utils.io_utils import save_json


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    source = Path(args.diagnostics).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError(f"lambda calibration output already exists: {output}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if int(payload.get("train_seed", -1)) != 23 or not bool(payload.get("fixed_rollout", False)):
        raise ValueError("lambda calibration requires the registered seed-23 fixed rollout")
    ratio = float(payload["unscaled_risk_to_ppo_gradient_ratio"])
    provenance_fields = (
        "parent_checkpoint_sha256", "critic_artifact_sha256",
        "task_config_fingerprint", "rollout_sha256",
    )
    if any(
        not isinstance(payload.get(key), str) or not payload.get(key)
        for key in provenance_fields
    ):
        raise ValueError("fixed-rollout diagnostics lack CSA artifact provenance")
    selected = select_alignment_lambda(ratio)
    result = {
        "schema_version": "CSALambdaCalibrationV1",
        "train_seed": 23,
        "fixed_rollout": True,
        "unscaled_risk_to_ppo_gradient_ratio": ratio,
        "candidate_lambdas": [0.01, 0.03, 0.10, 0.30],
        "target_scaled_ratio": [0.05, 0.10],
        "selected_lambda_a": selected,
        "frozen_for_all_training_seeds": True,
        "source": str(source),
        **{key: payload[key] for key in provenance_fields},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(result, output)
    return result


if __name__ == "__main__":
    main()
