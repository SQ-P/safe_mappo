"""Train absolute/proposed R4.4 critics and freeze the protocol on PASS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.intervention_advantage_certification import (
    certify_advantage_critic, freeze_protocol, train_probability_twin,
)
from src.utils.io_utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--alpha-pair", type=float, default=0.25)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261116)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"R4.4A output already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    absolute = train_probability_twin(
        dataset_dir=args.dataset_dir, output_dir=output / "absolute_probability",
        alpha_pair=0.0, device=args.device, epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.learning_rate, seed=args.seed,
    )
    proposed = train_probability_twin(
        dataset_dir=args.dataset_dir, output_dir=output / "advantage_pairwise",
        alpha_pair=args.alpha_pair, device=args.device, epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.learning_rate, seed=args.seed + 17,
    )
    absolute_report = certify_advantage_critic(
        dataset_dir=args.dataset_dir, artifact_path=absolute,
        output_path=output / "absolute_probability" / "development_certification.json",
        partition="test", device=args.device, batch_size=args.batch_size,
        bootstrap_replicates=args.bootstrap_replicates, seed=args.seed,
    )
    proposed_report = certify_advantage_critic(
        dataset_dir=args.dataset_dir, artifact_path=proposed,
        output_path=output / "advantage_pairwise" / "development_certification.json",
        partition="test", device=args.device, batch_size=args.batch_size,
        bootstrap_replicates=args.bootstrap_replicates, seed=args.seed + 1,
    )
    absolute_pair = float(absolute_report["pairwise"]["pooled"]["point"])
    proposed_pair = float(proposed_report["pairwise"]["pooled"]["point"])
    noninferior = proposed_pair >= absolute_pair - 0.02
    passed = bool(proposed_report["certified"] and noninferior)
    decision = {
        "schema_version": "IAMAPPOR4_4ADevelopmentDecisionV1",
        "status": "PASS" if passed else "FAIL",
        "ready_for_fresh_certification": passed,
        "proposed_all_gates_passed": bool(proposed_report["certified"]),
        "pairwise_noninferiority_margin": 0.02,
        "absolute_pairwise_accuracy": absolute_pair,
        "proposed_pairwise_accuracy": proposed_pair,
        "proposed_pairwise_noninferior": noninferior,
        "actor_training_authorized": False,
        "lambda_calibration_authorized": False,
    }
    save_json(decision, output / "development_decision.json")
    if passed:
        freeze_protocol(
            development_dataset=args.dataset_dir, proposed_artifact=proposed,
            development_report=output / "advantage_pairwise" / "development_certification.json",
            output_path=output / "frozen_protocol.json", alpha_pair=args.alpha_pair,
            bootstrap_replicates=args.bootstrap_replicates, bootstrap_seed=args.seed + 1,
        )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit("R4.4A development gate failed; fresh panel remains closed")


if __name__ == "__main__":
    main()
