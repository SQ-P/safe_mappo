"""Render one MP4 from the legacy-execution Standard MAPPO Seed 23 run."""

from __future__ import annotations

import sys
from pathlib import Path

from render_current_seed23 import main


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = (
    PROJECT_ROOT
    / "results/paper/standard_mappo_3u_8s2d_legacy_execution_v1/seed_23"
)
DEFAULT_CHECKPOINT = RUN_DIR / "checkpoints/best.pt"
DEFAULT_OUTPUT = RUN_DIR / "replays/legacy_seed23_best_episode_seed23.mp4"


if __name__ == "__main__":
    main(
        [
            "--checkpoint",
            str(DEFAULT_CHECKPOINT),
            "--output",
            str(DEFAULT_OUTPUT),
            *sys.argv[1:],
        ]
    )
