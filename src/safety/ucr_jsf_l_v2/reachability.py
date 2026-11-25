"""Permutation-aware observable-track memory for UCR-JSF-L V2."""

from __future__ import annotations

import numpy as np

from src.safety.ucr_jsf_l.reachability import (
    ReachabilityConfigV1,
    ReachableObstacleV1,
)

from .contracts import JointSafetyInputV2

ReachabilityConfigV2 = ReachabilityConfigV1
ReachableObstacleV2 = ReachableObstacleV1


class ReachabilityMemoryV2:
    """V1 reachability semantics plus explicit memory permutation support."""

    def __init__(self, config: ReachabilityConfigV2) -> None:
        if config.maximum_obstacle_speed_mps <= 0.0 or config.lost_track_ttl_steps < 0:
            raise ValueError("invalid reachability configuration")
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._tracks: dict[
            tuple[int, int], tuple[np.ndarray, np.ndarray, float, bool, int]
        ] = {}
        self._unresolved: set[tuple[int, int]] = set()

    @property
    def unresolved_evidence(self) -> tuple[str, ...]:
        return tuple(
            f"track:{observer_id}:{track_id}:lost_beyond_ttl"
            for observer_id, track_id in sorted(self._unresolved)
        )

    def memory_signature(self, agent_id: int) -> tuple:
        tracks = tuple(sorted(
            (
                int(track_id),
                tuple(float(value) for value in center),
                tuple(float(value) for value in velocity),
                float(radius),
                bool(valid),
                int(last_step),
            )
            for (observer_id, track_id), (
                center, velocity, radius, valid, last_step,
            ) in self._tracks.items() if observer_id == agent_id
        ))
        unresolved = tuple(sorted(
            int(track_id)
            for observer_id, track_id in self._unresolved
            if observer_id == agent_id
        ))
        return tracks, unresolved

    def permuted_copy(self, permutation: tuple[int, ...]) -> "ReachabilityMemoryV2":
        """Return memory aligned to ``new_rows = old_rows[permutation]``."""

        if sorted(permutation) != list(range(len(permutation))):
            raise ValueError("permutation must contain every agent exactly once")
        old_to_new = {old: new for new, old in enumerate(permutation)}
        result = ReachabilityMemoryV2(self.config)
        result._tracks = {
            (old_to_new[observer_id], track_id): (
                center.copy(), velocity.copy(), radius, valid, last_step,
            )
            for (observer_id, track_id), (
                center, velocity, radius, valid, last_step,
            ) in self._tracks.items()
        }
        result._unresolved = {
            (old_to_new[observer_id], track_id)
            for observer_id, track_id in self._unresolved
        }
        return result

    def update(
        self, state: JointSafetyInputV2
    ) -> tuple[tuple[ReachableObstacleV2, ...], ...]:
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
                absolute_velocity = (
                    np.asarray(state.velocities[observer_id], dtype=np.float64)
                    + np.asarray(track.relative_velocity, dtype=np.float64)
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

        rows: list[list[ReachableObstacleV2]] = [
            [] for _ in range(state.num_agents)
        ]
        for (observer_id, track_id), (
            center, velocity, radius, valid, last_step,
        ) in sorted(self._tracks.items()):
            age_steps = max(int(state.step - last_step), 0)
            rows[observer_id].append(ReachableObstacleV2(
                observer_id=observer_id,
                track_id=track_id,
                center=(float(center[0]), float(center[1])),
                velocity=(float(velocity[0]), float(velocity[1])),
                radius=radius,
                velocity_valid=bool(valid and age_steps == 0),
                age_steps=age_steps,
            ))
        return tuple(tuple(row) for row in rows)
