"""Low-level deterministic actions shared by all research groups."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def deterministic_hold_actions(
    velocities: Sequence[Sequence[float]], *, max_acceleration: float, dt: float
) -> np.ndarray:
    """Return normalized braking requests for the current velocities."""

    if max_acceleration <= 0.0 or dt <= 0.0:
        raise ValueError("max_acceleration and dt must be positive")
    values = np.asarray(velocities, dtype=np.float32)
    return np.clip(-values / (max_acceleration * dt), -1.0, 1.0).astype(np.float32)
