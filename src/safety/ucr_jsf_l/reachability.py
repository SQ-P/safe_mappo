"""Observable obstacle reachability for UCR-JSF-L V1.

The model uses only causal track observations.  Invalid velocities and recently
lost tracks are expanded by the registered physical speed bound instead of
being silently ignored or automatically rejecting every joint action.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import JointSafetyInputV1


@dataclass(frozen=True)
class ReachabilityConfigV1:
    maximum_obstacle_speed_mps: float = 2.0
    lost_track_ttl_steps: int = 2

    @classmethod
    def from_mapping(cls, payload: dict) -> "ReachabilityConfigV1":
        return cls(
            maximum_obstacle_speed_mps=float(
                payload.get("maximum_obstacle_speed_mps", 2.0)
            ),
            lost_track_ttl_steps=int(payload.get("lost_track_ttl_steps", 2)),
        )


@dataclass(frozen=True)
class ReachableObstacleV1:
    observer_id: int
    track_id: int
    center: tuple[float, float]
    velocity: tuple[float, float]
    radius: float
    velocity_valid: bool
    age_steps: int

    def envelope(
        self,
        time_s: float,
        *,
        observation_dt: float,
        map_size: tuple[float, float],
        maximum_speed_mps: float,
    ) -> tuple[np.ndarray, float]:
        """Return a causal nominal center and radial uncertainty bound."""

        elapsed = max(float(time_s), 0.0)
        center = np.asarray(self.center, dtype=np.float64)
        velocity = np.asarray(self.velocity, dtype=np.float64)
        if self.velocity_valid and self.age_steps == 0:
            projected = center + velocity * elapsed
            low = np.full(2, self.radius, dtype=np.float64)
            high = np.asarray(map_size, dtype=np.float64) - self.radius
            if np.all(projected >= low) and np.all(projected <= high):
                return projected, 0.0
            # A boundary hit makes the post-reflection direction ambiguous to
            # this observation-only model.  Bound it by the physical speed.
            return np.clip(projected, low, high), maximum_speed_mps * elapsed
        age_s = float(self.age_steps) * float(observation_dt)
        uncertainty = maximum_speed_mps * (elapsed + age_s)
        return center, uncertainty


class ReachabilityMemoryV1:
    """Retain only short-lived, previously observed track centers."""

    def __init__(self, config: ReachabilityConfigV1) -> None:
        if config.maximum_obstacle_speed_mps <= 0.0 or config.lost_track_ttl_steps < 0:
            raise ValueError("invalid reachability configuration")
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._tracks: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, float, bool, int]] = {}
        self._unresolved: set[tuple[int, int]] = set()

    @property
    def unresolved_evidence(self) -> tuple[str, ...]:
        return tuple(
            f"track:{observer_id}:{track_id}:lost_beyond_ttl"
            for observer_id, track_id in sorted(self._unresolved)
        )

    def update(
        self, state: JointSafetyInputV1
    ) -> tuple[tuple[ReachableObstacleV1, ...], ...]:
        positions = np.asarray(state.positions, dtype=np.float64)
        current_keys: set[tuple[int, int]] = set()
        for observer_id, tracks in enumerate(state.visible_obstacles):
            for track in tracks:
                key = (observer_id, track.track_id)
                current_keys.add(key)
                self._unresolved.discard(key)
                center = positions[observer_id] + np.asarray(
                    track.relative_position, dtype=np.float64
                )
                absolute_velocity = np.asarray(state.velocities[observer_id]) + np.asarray(
                    track.relative_velocity, dtype=np.float64
                )
                self._tracks[key] = (
                    center.copy(), absolute_velocity.copy(), float(track.radius),
                    bool(track.velocity_valid), int(state.step),
                )

        expired = [
            key for key, row in self._tracks.items()
            if key not in current_keys
            and state.step - row[4] > self.config.lost_track_ttl_steps
        ]
        for key in expired:
            del self._tracks[key]
            self._unresolved.add(key)

        rows: list[list[ReachableObstacleV1]] = [
            [] for _ in range(state.num_agents)
        ]
        for (observer_id, track_id), (center, velocity, radius, valid, last_step) in sorted(
            self._tracks.items()
        ):
            age_steps = max(int(state.step - last_step), 0)
            rows[observer_id].append(ReachableObstacleV1(
                observer_id=observer_id,
                track_id=track_id,
                center=(float(center[0]), float(center[1])),
                velocity=(float(velocity[0]), float(velocity[1])),
                radius=radius,
                velocity_valid=bool(valid and age_steps == 0),
                age_steps=age_steps,
            ))
        return tuple(tuple(row) for row in rows)


def current_reachability(
    state: JointSafetyInputV1,
) -> tuple[tuple[ReachableObstacleV1, ...], ...]:
    """Stateless reachability view used by direct constraint unit tests."""

    positions = np.asarray(state.positions, dtype=np.float64)
    rows: list[list[ReachableObstacleV1]] = [[] for _ in range(state.num_agents)]
    for observer_id, tracks in enumerate(state.visible_obstacles):
        for track in tracks:
            center = positions[observer_id] + np.asarray(track.relative_position)
            velocity = np.asarray(state.velocities[observer_id]) + np.asarray(
                track.relative_velocity
            )
            rows[observer_id].append(ReachableObstacleV1(
                observer_id=observer_id,
                track_id=track.track_id,
                center=(float(center[0]), float(center[1])),
                velocity=(float(velocity[0]), float(velocity[1])),
                radius=float(track.radius),
                velocity_valid=bool(track.velocity_valid),
                age_steps=int(track.observation_age_steps),
            ))
    return tuple(tuple(row) for row in rows)
