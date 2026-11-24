"""Collect one fresh-layout sampled-action shadow dataset for R4.3.2+ repairs."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.factory import create_trainer
from src.algorithms.intervention_r3 import validate_r3_shadow_dataset
from src.algorithms.intervention_r4_3_2 import file_sha256
from src.safety.config_guard import (
    validate_active_config_file,
    validate_active_research_config,
    validate_checkpoint_research_line,
)
from src.utils.io_utils import load_yaml, save_json


OPENED_PANELS = (
    "configs/evaluation/ia_mappo_v1_r1_40_layouts.json",
    "configs/evaluation/ia_mappo_v1_r3_200_layouts.json",
    "configs/evaluation/ia_mappo_v1_r3_1_200_layouts.json",
    "configs/evaluation/ia_mappo_v1_r3_2_200_layouts.json",
)

PANEL_CONTRACTS = {
    "IAMAPPOR4_3_2LayoutPanelV1": ("R4.3.2", 200),
    "IAMAPPOR4_3_4FreshTestPanelV1": ("R4.3.4", 40),
    "IAMAPPOR4_3_5FreshTestPanelV1": ("R4.3.5", 40),
    "IAMAPPOR4_4FreshCertificationPanelV1": ("R4.4B", 200),
}

ADDITIONAL_OPENED_PANELS = {
    "IAMAPPOR4_3_2LayoutPanelV1": (),
    "IAMAPPOR4_3_4FreshTestPanelV1": (
        "configs/evaluation/ia_mappo_v1_r4_3_2_200_layouts.json",
    ),
    "IAMAPPOR4_3_5FreshTestPanelV1": (
        "configs/evaluation/ia_mappo_v1_r4_3_2_200_layouts.json",
        "configs/evaluation/ia_mappo_v1_r4_3_4_fresh_test_40_layouts.json",
    ),
    "IAMAPPOR4_4FreshCertificationPanelV1": (
        "configs/evaluation/ia_mappo_v1_r4_3_2_200_layouts.json",
        "configs/evaluation/ia_mappo_v1_r4_3_4_fresh_test_40_layouts.json",
        "configs/evaluation/ia_mappo_v1_r4_3_5_fresh_test_40_layouts.json",
    ),
}


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate_layout_panel(panel_path: str | Path) -> tuple[str, int, list[int]]:
    """Validate schema, cardinality, uniqueness, and prior-panel isolation."""

    path = Path(panel_path).resolve()
    panel = json.loads(path.read_text(encoding="utf-8"))
    panel_schema = str(panel.get("schema_version"))
    if panel_schema not in PANEL_CONTRACTS:
        raise ValueError("unsupported sampled-action layout panel")
    phase_name, expected_layout_count = PANEL_CONTRACTS[panel_schema]
    layout_seeds = [int(value) for value in panel.get("layout_seeds", [])]
    if (
        len(layout_seeds) != expected_layout_count
        or len(set(layout_seeds)) != expected_layout_count
    ):
        raise ValueError(
            f"{phase_name} panel must contain {expected_layout_count} unique layouts"
        )
    opened_panels = list(OPENED_PANELS)
    opened_panels.extend(ADDITIONAL_OPENED_PANELS[panel_schema])
    for opened in opened_panels:
        opened_seeds = json.loads(resolve(opened).read_text(encoding="utf-8"))[
            "layout_seeds"
        ]
        if set(layout_seeds) & set(map(int, opened_seeds)):
            raise ValueError(f"{phase_name} panel overlaps opened panel: {opened}")
    return phase_name, expected_layout_count, layout_seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--layout-seeds-file", required=True)
    parser.add_argument("--r2r-gate-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-update", type=int, required=True)
    parser.add_argument("--sampling-seed", type=int, default=20261112)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    config_path = resolve(args.config).resolve()
    checkpoint_path = resolve(args.checkpoint).resolve()
    panel_path = resolve(args.layout_seeds_file).resolve()
    gate_path = resolve(args.r2r_gate_report).resolve()
    output = resolve(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"sampled-action collection output exists: {output}")
    for path in (config_path, checkpoint_path, panel_path, gate_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    phase_name, expected_layout_count, layout_seeds = validate_layout_panel(
        panel_path
    )

    validate_active_config_file(config_path)
    config = copy.deepcopy(load_yaml(config_path))
    validate_active_research_config(config, source_path=config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_checkpoint_research_line(config, checkpoint.get("config", {}))
    update = int(checkpoint.get("update", checkpoint.get("current_update", -1)))
    if update != int(args.expected_update):
        raise ValueError(f"checkpoint update {update} != expected {args.expected_update}")

    config["device"] = args.device
    config["output"] = {
        "output_dir": str(output / ".runtime"),
        "checkpoint_dir": str(output / ".runtime" / "checkpoints"),
        "reject_nonempty": True,
    }
    trainer = create_trainer(config, project_root=PROJECT_ROOT)
    trainer.load_checkpoint(
        checkpoint_path,
        allow_task_config_mismatch=False,
        allow_selector_config_mismatch=False,
    )
    checkpoint_hash = file_sha256(checkpoint_path)
    local_seed = (
        int(args.sampling_seed) + int(config.get("seed", 0)) * 10_003 + update
    )
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(local_seed)
    dataset = output / "shadow_dataset"
    evaluation = trainer.evaluate(
        layout_seeds=layout_seeds,
        episode_records_path=output / "episodes.jsonl",
        summary_output_path=output / "evaluation_summary.json",
        counterfactual_dataset_dir=dataset,
        r2r_gate_report_path=gate_path,
        counterfactual_expected_layout_count=expected_layout_count,
        policy_deterministic=False,
    )
    if file_sha256(checkpoint_path) != checkpoint_hash:
        raise RuntimeError(f"{phase_name} collection changed the policy checkpoint")
    manifest = validate_r3_shadow_dataset(dataset)
    if manifest.get("action_query_kind") != "sampled_bounded_rollout_action":
        raise RuntimeError(f"{phase_name} collection did not record sampled actions")
    if not all(bool(value) for value in manifest.get("invariance", {}).values()):
        raise RuntimeError(f"{phase_name} shadow invariance failed")
    provenance = {
        "schema_version": f"IAMAPPO{phase_name.replace('.', '_')}CollectionProvenanceV1",
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_update": update,
        "policy_seed": int(config.get("seed", -1)),
        "layout_panel": str(panel_path),
        "layout_panel_sha256": file_sha256(panel_path),
        "r2r_gate_report": str(gate_path),
        "r2r_gate_report_sha256": file_sha256(gate_path),
        "sampling_seed": local_seed,
        "task_config_fingerprint": evaluation["task_config_fingerprint"],
        "dataset_manifest_sha256": file_sha256(dataset / "manifest.json"),
    }
    save_json(provenance, output / "provenance.json")
    print(json.dumps({
        "status": "PASS",
        "policy_seed": provenance["policy_seed"],
        "checkpoint_update": update,
        "layouts": manifest["layout_count"],
        "team_steps": manifest["team_step_count"],
        "output": str(output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
