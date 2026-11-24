"""Evaluate a PPO, MAPPO or HAPPO checkpoint."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.factory import create_trainer
from src.safety.config_guard import (
    validate_active_config_file,
    validate_active_research_config,
    validate_checkpoint_research_line,
)
from src.utils.io_utils import load_yaml, save_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ppo.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument(
        "--layout-seeds-file",
        help="JSON list or {'layout_seeds': [...]} for a fixed non-contiguous panel",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-task-config-override", action="store_true")
    parser.add_argument(
        "--allow-safety-ablation",
        action="store_true",
        help=(
            "evaluation-only: allow only registered Guard, TTC, final-Guard "
            "and recovery Boolean switches to differ from the checkpoint"
        ),
    )
    parser.add_argument(
        "--allow-ood-eval-intervention",
        action="store_true",
        help=(
            "evaluation-only: permit only env.static_obstacle_num to differ "
            "from the checkpoint and record the sanctioned OOD intervention"
        ),
    )
    parser.add_argument("--episode-records", default=None)
    parser.add_argument(
        "--reward-audit-dir", default=None,
        help="non-overwriting per-episode and per-step reward/control audit directory",
    )
    parser.add_argument("--summary-output", default=None)
    parser.add_argument(
        "--run-output",
        help="evaluation runtime directory for logs and audit records",
    )
    parser.add_argument(
        "--provenance-output",
        help="write a non-overwriting evaluation checkpoint/config provenance JSON",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    override_modes = sum(bool(value) for value in (
        args.allow_task_config_override,
        args.allow_ood_eval_intervention,
        args.allow_safety_ablation,
    ))
    if override_modes > 1:
        raise ValueError(
            "evaluation override modes are mutually exclusive"
        )
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    validate_active_config_file(config_path)
    config = load_yaml(config_path)
    validate_active_research_config(config, source_path=config_path)
    requested_ood_obstacle_count = None
    if args.allow_ood_eval_intervention:
        requested_ood_obstacle_count = config.get("env", {}).get(
            "static_obstacle_num"
        )
        if requested_ood_obstacle_count is None:
            raise ValueError(
                "OOD evaluation config must declare env.static_obstacle_num"
            )
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"evaluation checkpoint does not exist: {checkpoint_path}"
        )
    checkpoint_payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    validate_checkpoint_research_line(
        config, checkpoint_payload.get("config", {})
    )
    provenance_path = Path(args.provenance_output) if args.provenance_output else None
    if provenance_path is not None:
        if not provenance_path.is_absolute():
            provenance_path = PROJECT_ROOT / provenance_path
        if provenance_path.exists():
            raise FileExistsError("evaluation provenance output already exists")
        checkpoint_hash_before = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if not args.allow_task_config_override:
        saved_config = checkpoint_payload.get("config", {})
        for key in (
            "algorithm", "seed", "model", "train", "env", "reward",
            "multi_agent",
            "oracle_subgoal", "observation_normalization",
        ):
            if key in saved_config:
                config[key] = copy.deepcopy(saved_config[key])
        for key in ("subgoal_selector", "llm_subgoal_selector"):
            if key in saved_config:
                config[key] = copy.deepcopy(saved_config[key])
        if args.allow_ood_eval_intervention:
            saved_count = config.setdefault("env", {}).get(
                "static_obstacle_num"
            )
            requested_ood_obstacle_count = int(requested_ood_obstacle_count)
            if requested_ood_obstacle_count == int(saved_count):
                raise ValueError(
                    "OOD intervention must change env.static_obstacle_num"
                )
            config["env"]["static_obstacle_num"] = requested_ood_obstacle_count
    if args.device is not None:
        config["device"] = args.device
    if args.run_output is not None:
        run_output = Path(args.run_output)
        if not run_output.is_absolute():
            run_output = PROJECT_ROOT / run_output
        config["output"] = {
            "output_dir": str(run_output),
            "checkpoint_dir": str(run_output / "checkpoints"),
            "reject_nonempty": False,
        }
    validate_active_research_config(config, source_path=config_path)
    trainer = create_trainer(config, project_root=PROJECT_ROOT)
    trainer.load_checkpoint(
        checkpoint_path,
        allow_task_config_mismatch=args.allow_task_config_override,
        allow_selector_config_mismatch=False,
        allow_ood_evaluation_config_mismatch=(
            args.allow_ood_eval_intervention
        ),
        allow_safety_ablation_config_mismatch=args.allow_safety_ablation,
    )
    evaluation = dict(config.get("evaluation", {}))
    summary_path = Path(args.summary_output) if args.summary_output else None
    if summary_path is not None and not summary_path.is_absolute():
        summary_path = PROJECT_ROOT / summary_path
    records_path = Path(args.episode_records) if args.episode_records else None
    if records_path is not None and not records_path.is_absolute():
        records_path = PROJECT_ROOT / records_path
    reward_audit_dir = Path(args.reward_audit_dir) if args.reward_audit_dir else None
    if reward_audit_dir is not None and not reward_audit_dir.is_absolute():
        reward_audit_dir = PROJECT_ROOT / reward_audit_dir
    evaluated_episodes = args.episodes if args.episodes is not None else int(evaluation.get("episodes", 80))
    evaluated_seed_base = args.seed_base if args.seed_base is not None else int(evaluation.get("seed_base", 31_000))
    layout_seeds = None
    if args.layout_seeds_file:
        layout_path = Path(args.layout_seeds_file)
        if not layout_path.is_absolute():
            layout_path = PROJECT_ROOT / layout_path
        payload = json.loads(layout_path.read_text(encoding="utf-8"))
        layout_seeds = payload.get("layout_seeds") if isinstance(payload, dict) else payload
        if not isinstance(layout_seeds, list):
            raise ValueError("layout seed file must contain a JSON list")
        evaluated_episodes = len(layout_seeds)
    result = trainer.evaluate(
        episodes=evaluated_episodes,
        seed_base=evaluated_seed_base,
        layout_seeds=layout_seeds,
        episode_records_path=records_path,
        summary_output_path=summary_path,
        reward_audit_dir=reward_audit_dir,
    )
    if provenance_path is not None:
        checkpoint_hash_after = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if checkpoint_hash_after != checkpoint_hash_before:
            raise RuntimeError("checkpoint changed during evaluation")
        save_json({
            "schema_version": "ppo_evaluation_provenance_v1",
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_hash_before,
            "config": str(config_path.resolve()),
            "config_sha256": config_hash,
            "episodes": evaluated_episodes,
            "evaluation_seed_base": evaluated_seed_base,
            "layout_seeds": layout_seeds,
            "task_config_fingerprint": result["task_config_fingerprint"],
            "parent_task_config_fingerprint": result.get("parent_task_config_fingerprint"),
            "selector_intervention_used": result["selector_intervention_used"],
            "ood_evaluation_intervention_used": result["ood_evaluation_intervention_used"],
            "task_config_override_used": result["task_config_override_used"],
            "safety_ablation_intervention_used": result.get(
                "safety_ablation_intervention_used", False
            ),
            "safety_ablation_changes": result.get(
                "safety_ablation_changes", {}
            ),
            "summary_output": str(summary_path.resolve()) if summary_path else None,
            "episode_records": str(records_path.resolve()) if records_path else None,
            "reward_audit_dir": str(reward_audit_dir.resolve()) if reward_audit_dir else None,
        }, provenance_path)
    headline = {
        key: result[key]
        for key in ("cfms", "collision_rate", "timeout_rate", "mean_goal_progress_ratio")
    }
    print(f"PPO evaluation complete: {headline}")
    return result


if __name__ == "__main__":
    main()
