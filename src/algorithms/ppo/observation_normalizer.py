"""PPO 使用的无状态、可复现 observation 缩放。"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


class UAVObservationNormalizer:
    """Versioned, stateless scaling for legacy and tracked observations."""

    VERSION = "fixed_scale_v1"
    TRACKED_VERSION = "fixed_scale_v2_tracked_obstacles"
    SUPPORTED_VERSIONS = {VERSION, TRACKED_VERSION}

    def __init__(
        self,
        env_config: Mapping[str, Any],
        enabled: bool = True,
        version: str = VERSION,
    ):
        cfg = dict(env_config)
        self.enabled = bool(enabled)
        self.version = str(version)
        if self.version not in self.SUPPORTED_VERSIONS:
            raise ValueError("unsupported observation normalization version")
        self.schema_version = str(
            cfg.get("observation_schema_version", "legacy_obstacle_v1")
        )
        self.obstacle_slots = int(cfg.get("obstacle_slots", 4))
        self.obstacle_feature_dim = 7 if self.schema_version == "tracked_obstacle_v2" else 4
        self.local_observation_dim = 10 + self.obstacle_slots * self.obstacle_feature_dim
        if self.schema_version == "legacy_obstacle_v1" and self.version != self.VERSION:
            raise ValueError("legacy observations require fixed_scale_v1")
        if self.schema_version == "tracked_obstacle_v2" and self.version != self.TRACKED_VERSION:
            raise ValueError("tracked observations require fixed_scale_v2_tracked_obstacles")
        self.map_size = np.asarray(cfg.get("map_size", [100.0, 100.0]), dtype=np.float32)
        self.max_speed = float(cfg.get("agent_max_speed", 2.0))
        self.sensor_range = float(cfg.get("detection_radius", 25.0))
        radius_range = np.asarray(cfg.get("obstacle_radius_range", [2.0, 5.0]), dtype=np.float32)
        self.max_obstacle_radius = float(np.max(radius_range))
        if (
            self.map_size.shape != (2,)
            or np.any(self.map_size <= 0.0)
            or min(self.max_speed, self.sensor_range, self.max_obstacle_radius) <= 0.0
        ):
            raise ValueError("observation normalization scales must be positive")

    def transform(self, observations: np.ndarray) -> np.ndarray:
        values = np.asarray(observations, dtype=np.float32)
        if values.shape[-1] != self.local_observation_dim:
            raise ValueError(
                f"local observations must end with {self.local_observation_dim} values"
            )
        if not self.enabled:
            return values.copy()
        result = values.copy()
        # 绝对坐标居中到 [-1, 1]；相对量按物理上界缩放，padding 仍保持为 0。
        for start in (0, 4, 6):
            result[..., start : start + 2] = (
                2.0 * result[..., start : start + 2] / self.map_size - 1.0
            )
        result[..., 2:4] /= self.max_speed
        result[..., 8:10] /= self.map_size
        for slot in range(self.obstacle_slots):
            start = 10 + slot * self.obstacle_feature_dim
            result[..., start : start + 2] /= self.sensor_range
            if self.schema_version == "tracked_obstacle_v2":
                # Relative velocity is a finite-difference measurement of
                # obstacle minus UAV motion, hence the conservative 2*vmax scale.
                result[..., start + 2 : start + 4] /= max(2.0 * self.max_speed, 1e-6)
                result[..., start + 4] /= self.max_obstacle_radius
                # slot-valid and velocity-valid are already in {0, 1}.
            else:
                result[..., start + 2] /= self.max_obstacle_radius
                result[..., start + 3] /= self.sensor_range
        if not np.isfinite(result).all():
            raise ValueError("normalized observations must be finite")
        return result.astype(np.float32)
