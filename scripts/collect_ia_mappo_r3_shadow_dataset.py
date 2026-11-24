"""Collect the immutable fresh-layout IA-MAPPO R3.2 shadow dataset."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.factory import create_trainer
from src.algorithms.intervention_r3 import (
    validate_r3_entry_evidence,
    validate_r3_shadow_dataset,
)
from src.safety.config_guard import (
    validate_active_config_file,
    validate_active_research_config,
    validate_checkpoint_research_line,
)
from src.utils.io_utils import load_yaml, save_json


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--layout-seeds-file", required=True)
    parser.add_argument("--r2r-gate-report", required=True)
    parser.add_argument("--r2r-primary-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-update", type=int, default=200)
    args = parser.parse_args(argv)

    config_path = _resolve(args.config).resolve()
    checkpoint_path = _resolve(args.checkpoint).resolve()
    layout_path = _resolve(args.layout_seeds_file).resolve()
    gate_path = _resolve(args.r2r_gate_report).resolve()
    primary_report_path = _resolve(args.r2r_primary_report).resolve()
    output = _resolve(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"R3 collection output already exists: {output}")
    for path, name in (
        (config_path, "config"), (checkpoint_path, "checkpoint"),
        (layout_path, "layout panel"), (gate_path, "R2R gate report"),
        (primary_report_path, "R2R primary report"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    validate_active_config_file(config_path)
    config = load_yaml(config_path)
    validate_active_research_config(config, source_path=config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_checkpoint_research_line(config, checkpoint.get("config", {}))
    saved = checkpoint.get("config", {})
    for key in (
        "algorithm", "seed", "model", "train", "env", "reward",
        "multi_agent", "oracle_subgoal", "observation_normalization",
        "safety_execution", "subgoal_selector", "llm_subgoal_selector",
    ):
        if key in saved:
            config[key] = copy.deepcopy(saved[key])
    if int(config.get("seed", -1)) != 41:
        raise ValueError("R3 collection is frozen to the primary seed-41 policy")
    update = int(checkpoint.get("update", checkpoint.get("current_update", -1)))
    if update != int(args.expected_update):
        raise ValueError(f"checkpoint update {update} != expected {args.expected_update}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if (
        gate.get("schema_version") != "IAMAPPOR2RCombinedDecisionV2"
        or gate.get("primary_decision") != "advance_r3_flat_and_agent"
    ):
        raise RuntimeError("R3 requires the passed primary Flat+Agent GateFix decision")
    entry_evidence = validate_r3_entry_evidence(
        combined_gate_report=gate_path,
        primary_r2r_report=primary_report_path,
        checkpoint=checkpoint_path,
    )
    panel = json.loads(layout_path.read_text(encoding="utf-8"))
    if (
        not isinstance(panel, dict)
        or panel.get("schema_version") != "IAMAPPOR3LayoutPanelV3"
    ):
        raise ValueError("R3.2 requires the fresh IAMAPPOR3LayoutPanelV3 panel")
    layout_seeds = panel.get("layout_seeds") if isinstance(panel, dict) else panel
    if not isinstance(layout_seeds, list) or len(layout_seeds) != 200:
        raise ValueError("R3 layout panel must contain exactly 200 seeds")
    layout_seeds = [int(value) for value in layout_seeds]
    if len(set(layout_seeds)) != 200:
        raise ValueError("R3 layout panel contains duplicates")
    r1_panel = json.loads(
        _resolve("configs/evaluation/ia_mappo_v1_r1_40_layouts.json")
        .read_text(encoding="utf-8")
    )["layout_seeds"]
    if set(layout_seeds) & set(map(int, r1_panel)):
        raise ValueError("R3 layout panel overlaps the R1/R2 feasibility panel")
    opened_r3_panel = json.loads(
        _resolve("configs/evaluation/ia_mappo_v1_r3_200_layouts.json")
        .read_text(encoding="utf-8")
    )["layout_seeds"]
    if set(layout_seeds) & set(map(int, opened_r3_panel)):
        raise ValueError("R3.2 panel overlaps the already-opened R3 V1 test panel")
    opened_r3_1_panel = json.loads(
        _resolve("configs/evaluation/ia_mappo_v1_r3_1_200_layouts.json")
        .read_text(encoding="utf-8")
    )["layout_seeds"]
    if set(layout_seeds) & set(map(int, opened_r3_1_panel)):
        raise ValueError("R3.2 panel overlaps the already-opened R3.1 test panel")
    config["device"] = args.device
    config["output"] = {
        "output_dir": str(output / ".runtime"),
        "checkpoint_dir": str(output / ".runtime" / "checkpoints"),
        "reject_nonempty": True,
    }
    validate_active_research_config(config, source_path=config_path)
    trainer = create_trainer(config, project_root=PROJECT_ROOT)
    trainer.load_checkpoint(
        checkpoint_path,
        allow_task_config_mismatch=False,
        allow_selector_config_mismatch=False,
    )
    checkpoint_hash_before = _sha256(checkpoint_path)
    evaluation = trainer.evaluate(
        layout_seeds=layout_seeds,
        episode_records_path=output / "evaluation_episodes.jsonl",
        summary_output_path=output / "evaluation_summary.json",
        counterfactual_dataset_dir=output / "dataset",
        r2r_gate_report_path=gate_path,
    )
    if _sha256(checkpoint_path) != checkpoint_hash_before:
        raise RuntimeError("R3 read-only collection changed the Standard checkpoint")
    manifest = validate_r3_shadow_dataset(output / "dataset")
    provenance = {
        "schema_version": "IAMAPPOR3CollectionProvenanceV3",
        "policy_seed": 41,
        "checkpoint_update": int(args.expected_update),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash_before,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "layout_panel": str(layout_path),
        "layout_panel_sha256": _sha256(layout_path),
        "r2r_gate_report": str(gate_path),
        "r2r_gate_report_sha256": _sha256(gate_path),
        "r2r_primary_report": str(primary_report_path),
        "r2r_primary_report_sha256": _sha256(primary_report_path),
        "entry_evidence": entry_evidence,
        "task_config_fingerprint": evaluation["task_config_fingerprint"],
        "dataset_manifest": manifest,
    }
    save_json(provenance, output / "provenance.json")
    print(json.dumps({
        "status": "PASS",
        "layouts": manifest["layout_count"],
        "team_steps": manifest["team_step_count"],
        "supported_heads": manifest["supported_heads"],
        "output": str(output),
    }, ensure_ascii=False))
    return provenance


if __name__ == "__main__":
    main()
