"""Run the frozen Seed-23 A0-A4 safety-stack ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_ppo import main as evaluate_ppo
from src.utils.io_utils import save_json


ARMS = ("A0", "A1", "A2", "A3", "A4")
PAIRWISE = (("A1", "A0"), ("A2", "A0"), ("A3", "A1"),
            ("A3", "A2"), ("A4", "A3"))
CHECKPOINT = (
    "results/paper/standard_mappo_3u_8s2d_legacy_execution_v1/"
    "seed_23/checkpoints/update_0800.pt"
)
FORMAL_PANEL = "configs/evaluation/safety_ablation_seed23_200.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--layout-seeds-file", default=FORMAL_PANEL)
    parser.add_argument(
        "--output-dir", default="results/ablations/safety_seed23"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args(argv)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _metric_rows(rows: list[dict]) -> dict[str, np.ndarray]:
    return {
        "safe_success": np.asarray([float(row["safe_success"]) for row in rows]),
        "collision": np.asarray([float(row["collision_count"] > 0) for row in rows]),
        "static_collision_count": np.asarray([row["static_obstacle_collision_count"] for row in rows], dtype=float),
        "dynamic_collision_count": np.asarray([row["dynamic_obstacle_collision_count"] for row in rows], dtype=float),
        "uav_collision_count": np.asarray([row["inter_agent_collision_count"] for row in rows], dtype=float),
        "boundary_collision_count": np.asarray([row["boundary_collision_count"] for row in rows], dtype=float),
        "timeout": np.asarray([float(row["reason"] == "timeout") for row in rows]),
        "goal_progress_ratio": np.asarray([row["goal_progress_ratio"] for row in rows], dtype=float),
        "episode_length": np.asarray([row["episode_length"] for row in rows], dtype=float),
        "guard_interventions": np.asarray([row["joint_guard_intervention_count"] for row in rows], dtype=float),
        "ttc_interventions": np.asarray([row["obstacle_ttc_projection_count"] for row in rows], dtype=float),
        "ttc_fallbacks": np.asarray([row["obstacle_ttc_fallback_count"] for row in rows], dtype=float),
        "recovery_entries": np.asarray([row["safe_recover_entry_count"] for row in rows], dtype=float),
    }


def _paired_bootstrap(
    left: np.ndarray, right: np.ndarray, *, samples: int, rng: np.random.Generator
) -> dict[str, float]:
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("paired ablation metrics must be aligned one-dimensional arrays")
    delta = left - right
    indices = rng.integers(0, len(delta), size=(samples, len(delta)))
    draws = delta[indices].mean(axis=1)
    return {
        "mean_delta": float(delta.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
    }


def _write_panel_analysis(root: Path, samples: int) -> None:
    records = {arm: _read_jsonl(root / arm / "episodes.jsonl") for arm in ARMS}
    seed_orders = [[int(row["seed"]) for row in records[arm]] for arm in ARMS]
    if any(order != seed_orders[0] for order in seed_orders[1:]):
        raise RuntimeError("ablation episode panels are not paired in the same order")
    metrics = {arm: _metric_rows(rows) for arm, rows in records.items()}
    rng = np.random.default_rng(20261119)
    comparisons = {}
    for left, right in PAIRWISE:
        comparisons[f"{left}-{right}"] = {
            name: _paired_bootstrap(
                metrics[left][name], metrics[right][name],
                samples=samples, rng=rng,
            )
            for name in metrics[left]
        }
    interaction = {}
    for name in metrics["A0"]:
        values = (
            metrics["A3"][name] - metrics["A2"][name]
            - metrics["A1"][name] + metrics["A0"][name]
        )
        zeros = np.zeros_like(values)
        interaction[name] = _paired_bootstrap(
            values, zeros, samples=samples, rng=rng
        )
    save_json({
        "schema_version": "SafetyAblationPairedAnalysisV1",
        "layout_seeds": seed_orders[0],
        "bootstrap_samples": int(samples),
        "comparisons": comparisons,
        "guard_ttc_interaction_A0_A3": interaction,
    }, root / "paired_analysis.json")


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.output_dir)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    for arm in args.arm:
        arm_dir = root / arm
        if arm_dir.exists():
            raise FileExistsError(f"ablation output already exists: {arm_dir}")
        started = perf_counter()
        result = evaluate_ppo([
            "--config", f"configs/ablations/safety_seed23/{arm}.yaml",
            "--checkpoint", args.checkpoint,
            "--layout-seeds-file", args.layout_seeds_file,
            "--device", args.device,
            "--allow-safety-ablation",
            "--episode-records", str(arm_dir / "episodes.jsonl"),
            "--summary-output", str(arm_dir / "summary.json"),
            "--provenance-output", str(arm_dir / "provenance.json"),
            "--run-output", str(arm_dir / "runtime"),
        ])
        elapsed = perf_counter() - started
        episode_steps = sum(
            int(row["episode_length"])
            for row in _read_jsonl(arm_dir / "episodes.jsonl")
        )
        save_json(result.get("action_stage_attribution", {}),
                  arm_dir / "stage_metrics.json")
        save_json({
            "schema_version": "SafetyAblationTimingV1",
            "wall_seconds": elapsed,
            "episode_steps": episode_steps,
            "wall_seconds_per_step": elapsed / max(episode_steps, 1),
        }, arm_dir / "timing.json")
    if all((root / arm / "episodes.jsonl").is_file() for arm in ARMS):
        _write_panel_analysis(root, args.bootstrap_samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
