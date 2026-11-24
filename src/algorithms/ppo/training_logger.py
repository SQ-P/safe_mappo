"""File logging helpers for PPO training."""

from __future__ import annotations

import logging
from pathlib import Path

from src.utils.io_utils import ensure_dir


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def configure_training_logger(output_dir: Path, algorithm: str = "ppo") -> logging.Logger:
    directory = ensure_dir(output_dir)
    logger = logging.getLogger(str(algorithm).upper())
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in tuple(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(directory / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
