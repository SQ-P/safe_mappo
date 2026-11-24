"""Run the frozen R4.4B protocol exactly once on the fresh panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.intervention_advantage_certification import (
    PROTOCOL_SCHEMA, certify_advantage_critic, file_sha256, validate_advantage_dataset,
)
from src.utils.io_utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--development-dataset", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"R4.4B certification output exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != PROTOCOL_SCHEMA or protocol.get("status") != "FROZEN":
        raise ValueError("R4.4B requires a frozen R4.4 protocol")
    if file_sha256(Path(args.artifact).resolve()) != protocol["proposed_artifact_sha256"]:
        raise ValueError("R4.4B artifact differs from frozen protocol")
    if file_sha256(Path(args.development_dataset).resolve() / "manifest.json") != protocol["development_dataset_manifest_sha256"]:
        raise ValueError("R4.4B development provenance differs from frozen protocol")
    fresh_manifest = validate_advantage_dataset(args.dataset_dir)
    if fresh_manifest.get("role") != "fresh_certification":
        raise ValueError("R4.4B requires fresh-certification data")
    dev_sources = json.loads((Path(args.development_dataset).resolve() / "sources.json").read_text(encoding="utf-8"))["sources"]
    fresh_sources = json.loads((Path(args.dataset_dir).resolve() / "sources.json").read_text(encoding="utf-8"))["sources"]
    dev_checkpoints = {row["name"]: row["checkpoint_sha256"] for row in dev_sources}
    fresh_checkpoints = {row["name"]: row["checkpoint_sha256"] for row in fresh_sources}
    if fresh_checkpoints != dev_checkpoints:
        raise ValueError("R4.4B policy anchors differ from development")
    report = certify_advantage_critic(
        dataset_dir=args.dataset_dir, artifact_path=args.artifact,
        output_path=output / "fresh_certification.json", partition="test",
        device=args.device, batch_size=args.batch_size,
        bootstrap_replicates=int(protocol["bootstrap"]["replicates"]),
        seed=int(protocol["bootstrap"]["seed"]),
    )
    decision = {
        "schema_version": "IAMAPPOR4_4BDecisionV1",
        "status": "PASS" if report["certified"] else "FAIL",
        "ready_for_r4_5": bool(report["certified"]),
        "actor_training_authorized": False,
        "lambda_calibration_authorized": bool(report["certified"]),
        "no_second_fresh_panel": True,
        "protocol_sha256": file_sha256(protocol_path),
    }
    save_json(decision, output / "decision.json")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    if not report["certified"]:
        raise SystemExit("R4.4B fresh certification failed; advantage route stops")


if __name__ == "__main__":
    main()
