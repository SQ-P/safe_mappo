"""Train PPO, MAPPO or HAPPO from a YAML configuration."""

from __future__ import annotations

import argparse
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
from src.utils.io_utils import load_yaml


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ppo.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--total-updates", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--allow-task-config-override", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    validate_active_config_file(config_path)
    config = load_yaml(config_path)
    validate_active_research_config(config, source_path=config_path)
    if args.seed is not None:
        if args.seed < 0:
            raise ValueError("seed must be nonnegative")
        config["seed"] = args.seed
    if args.total_updates is not None:
        if args.total_updates <= 0:
            raise ValueError("total-updates must be positive")
        config.setdefault("train", {})["total_updates"] = args.total_updates
    if args.output_dir is not None:
        output = config.setdefault("output", {})
        output["output_dir"] = args.output_dir
        output["checkpoint_dir"] = str(Path(args.output_dir) / "checkpoints")
    if args.device is not None:
        config["device"] = args.device
    trainer = create_trainer(config, project_root=PROJECT_ROOT)
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = PROJECT_ROOT / resume_path
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        validate_checkpoint_research_line(config, payload.get("config", {}))
        trainer.load_checkpoint(
            resume_path,
            allow_task_config_mismatch=args.allow_task_config_override,
            allow_selector_config_mismatch=False,
        )
    return trainer.train()


if __name__ == "__main__":
    main()
