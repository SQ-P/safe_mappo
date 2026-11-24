"""Render one MP4 replay from the current Standard MAPPO Seed 23 run.

This script intentionally defaults to the paper run's best checkpoint.  It
loads the task configuration embedded in that checkpoint, so later edits to an
active YAML file cannot silently change the replay environment.
"""

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
from src.visualization.mp4_renderer import render_episode_mp4
from src.visualization.replay import collect_replay_episode


RUN_DIR = PROJECT_ROOT / "results/paper/standard_mappo_3u_8s2d/seed_23"
DEFAULT_CHECKPOINT = RUN_DIR / "checkpoints/best.pt"
DEFAULT_OUTPUT = RUN_DIR / "replays/seed23_best_episode_seed23.mp4"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Render one MP4 from a Standard MAPPO Seed 23 checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Seed 23 checkpoint to replay (default: current paper run best.pt).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination MP4 path.",
    )
    parser.add_argument(
        "--episode-seed",
        type=int,
        default=23,
        help="Environment layout seed used for this replay (default: 23).",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample policy actions instead of using the deterministic mean action.",
    )
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main(argv=None) -> Path:
    args = parse_args(argv)
    checkpoint = _resolve(args.checkpoint)
    output = _resolve(args.output)

    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if output.suffix.lower() != ".mp4":
        raise ValueError("--output must end with .mp4")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_config = payload.get("config")
    if not isinstance(saved_config, dict):
        raise ValueError("Checkpoint does not contain a saved configuration")
    if int(saved_config.get("seed", -1)) != 23:
        raise ValueError(
            f"Expected a Seed 23 checkpoint, got seed={saved_config.get('seed')!r}"
        )

    # Replay the checkpoint's own environment contract, not a mutable YAML.
    config = copy.deepcopy(saved_config)
    config["device"] = args.device
    trainer = create_trainer(config, project_root=PROJECT_ROOT)
    trainer.load_checkpoint(checkpoint)
    trainer.actor.eval()

    replay = collect_replay_episode(
        trainer.env,
        trainer.actor,
        seed=args.episode_seed,
        deterministic=not args.stochastic,
        device=trainer.device,
        observation_transform=trainer.policy_observations,
    )
    rendered = render_episode_mp4(
        replay,
        output,
        trainer.env.map_size,
        trainer.env.goal_radius,
        trainer.env.agent_radius,
        fps=args.fps,
    )
    print(
        f"MP4 written: {rendered}\n"
        f"checkpoint_update={payload.get('update', 'unknown')} | "
        f"episode_seed={replay.seed} | deterministic={replay.deterministic} | "
        f"reason={replay.final_reason} | return={replay.total_return:.3f}"
    )
    return rendered


if __name__ == "__main__":
    main()
