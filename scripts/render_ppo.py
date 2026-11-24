"""Render deterministic or stochastic PPO/MAPPO/HAPPO trajectories."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.factory import create_trainer
from src.utils.io_utils import ensure_dir, load_yaml
from src.visualization.mp4_renderer import render_episode_mp4
from src.visualization.replay import collect_replay_episode


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Render a PPO checkpoint to MP4")
    parser.add_argument("--config", default="configs/ppo.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--output-dir")
    parser.add_argument("--seed-base", type=int, default=40_000)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--allow-task-config-override", action="store_true")
    parser.add_argument("--allow-selector-intervention", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = load_yaml(config_path)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    if not args.allow_task_config_override:
        payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        saved_config = payload.get("config", {})
        for key in (
            "algorithm", "seed", "model", "train", "multi_agent",
            "env",
            "reward",
            "oracle_subgoal",
            "observation_normalization",
        ):
            if key in saved_config:
                config[key] = copy.deepcopy(saved_config[key])
        if not args.allow_selector_intervention:
            for key in ("subgoal_selector", "llm_subgoal_selector"):
                if key in saved_config:
                    config[key] = copy.deepcopy(saved_config[key])
    config["device"] = args.device
    trainer = create_trainer(config, project_root=PROJECT_ROOT)
    trainer.load_checkpoint(
        checkpoint,
        allow_task_config_mismatch=args.allow_task_config_override,
        allow_selector_config_mismatch=args.allow_selector_intervention,
    )
    trainer.actor.eval()

    configured_output = Path(config.get("output", {}).get("output_dir", "results/ppo")) / "replays"
    output_dir = Path(args.output_dir) if args.output_dir else configured_output
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    ensure_dir(output_dir)
    outputs = []
    for episode_index in range(args.episodes):
        replay = collect_replay_episode(
            trainer.env,
            trainer.actor,
            seed=args.seed_base + episode_index,
            deterministic=not args.stochastic,
            device=trainer.device,
            observation_transform=trainer.policy_observations,
        )
        output = output_dir / f"episode_{episode_index + 1:03d}.mp4"
        outputs.append(
            render_episode_mp4(
                replay,
                output,
                trainer.env.map_size,
                trainer.env.goal_radius,
                trainer.env.agent_radius,
                fps=args.fps,
            )
        )
        print(f"Replay written: {output} | reason={replay.final_reason} | return={replay.total_return:.3f}")
    return outputs


if __name__ == "__main__":
    main()
