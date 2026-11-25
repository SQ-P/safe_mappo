"""Stateful progress, passing-side and arrival-hold support."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import JointSafetyInputV1
from .dynamics import velocity_first_step


@dataclass(frozen=True)
class LivenessConfigV1:
    stall_patience_steps: int = 12
    minimum_progress_m: float = 0.02
    passing_side_ttl_steps: int = 20
    straggler_weight: float = 1.0
    stall_weight: float = 0.25
    arrival_hold_weight: float = 2.0
    actor_deviation_weight: float = 0.05
    blockage_distance_m: float = 12.0
    blockage_safety_margin_m: float = 1.5
    tangential_progress_weight: float = 0.75
    passing_side_flip_penalty: float = 1.0
    goal_exit_penalty: float = 20.0
    passing_side_release_stall_multiplier: float = 2.0

    @classmethod
    def from_mapping(cls, payload: dict) -> "LivenessConfigV1":
        return cls(
            stall_patience_steps=int(payload.get("stall_patience_steps", 12)),
            minimum_progress_m=float(payload.get("minimum_progress_m", 0.02)),
            passing_side_ttl_steps=int(payload.get("passing_side_ttl_steps", 20)),
            straggler_weight=float(payload.get("straggler_weight", 1.0)),
            stall_weight=float(payload.get("stall_weight", 0.25)),
            arrival_hold_weight=float(payload.get("arrival_hold_weight", 2.0)),
            actor_deviation_weight=float(payload.get("actor_deviation_weight", 0.05)),
            blockage_distance_m=float(payload.get("blockage_distance_m", 12.0)),
            blockage_safety_margin_m=float(
                payload.get("blockage_safety_margin_m", 1.5)
            ),
            tangential_progress_weight=float(
                payload.get("tangential_progress_weight", 0.75)
            ),
            passing_side_flip_penalty=float(
                payload.get("passing_side_flip_penalty", 1.0)
            ),
            goal_exit_penalty=float(payload.get("goal_exit_penalty", 20.0)),
            passing_side_release_stall_multiplier=float(
                payload.get("passing_side_release_stall_multiplier", 2.0)
            ),
        )


class LivenessMemoryV1:
    def __init__(self, num_agents: int, config: LivenessConfigV1) -> None:
        self.num_agents = int(num_agents)
        self.config = config
        if self.num_agents < 2 or config.stall_patience_steps < 1:
            raise ValueError("invalid liveness configuration")
        self.reset()

    def reset(self) -> None:
        self.previous_distances: np.ndarray | None = None
        self.stall_counts = np.zeros(self.num_agents, dtype=np.int64)
        self.passing_sides: dict[tuple[int, int], tuple[int, int]] = {}
        self.passing_side_flip_count = 0
        self.last_score_goal_exit_count = 0

    def observe(self, state: JointSafetyInputV1) -> tuple[np.ndarray, np.ndarray]:
        positions = np.asarray(state.positions)
        goals = np.asarray(state.final_goals)
        distances = np.linalg.norm(goals - positions, axis=1)
        arrived = distances <= state.goal_radius
        if self.previous_distances is not None:
            progress = self.previous_distances - distances
            stalled = (progress < self.config.minimum_progress_m) & ~arrived
            self.stall_counts = np.where(stalled, self.stall_counts + 1, 0)
        self.previous_distances = distances.copy()
        visible = {
            (observer_id, track.track_id)
            for observer_id, row in enumerate(state.visible_obstacles)
            for track in row
        }
        self.passing_sides = {
            key: value for key, value in self.passing_sides.items()
            if key in visible and state.step - value[1] <= self.config.passing_side_ttl_steps
        }
        return distances, arrived

    def preferred_passing_side(
        self,
        state: JointSafetyInputV1,
        agent_id: int,
        track_id: int,
        relative_position: np.ndarray,
        *,
        commit: bool = False,
    ) -> int:
        key = (int(agent_id), int(track_id))
        existing = self.passing_sides.get(key)
        if existing is not None:
            return int(existing[0])
        goal_delta = (
            np.asarray(state.final_goals[agent_id])
            - np.asarray(state.positions[agent_id])
        )
        cross = float(goal_delta[0] * relative_position[1] - goal_delta[1] * relative_position[0])
        side = 1 if cross > 1e-12 else -1 if cross < -1e-12 else (1 if agent_id % 2 == 0 else -1)
        if commit:
            self.passing_sides[key] = (side, state.step)
        return side

    def passing_side(
        self,
        state: JointSafetyInputV1,
        agent_id: int,
        track_id: int,
        relative_position: np.ndarray,
    ) -> int:
        """Compatibility helper used by direct memory tests."""

        return self.preferred_passing_side(
            state, agent_id, track_id, relative_position, commit=True
        )

    def blocking_track(self, state: JointSafetyInputV1, agent_id: int):
        """Return the first track intersecting the local final-goal corridor."""

        goal_delta = (
            np.asarray(state.final_goals[agent_id], dtype=float)
            - np.asarray(state.positions[agent_id], dtype=float)
        )
        goal_distance = float(np.linalg.norm(goal_delta))
        if goal_distance <= 1e-12:
            return None
        direction = goal_delta / goal_distance
        corridor_length = min(goal_distance, self.config.blockage_distance_m)
        blockers = []
        for track in state.visible_obstacles[agent_id]:
            relative = np.asarray(track.relative_position, dtype=float)
            along = float(np.dot(relative, direction))
            if along <= 0.0 or along > corridor_length:
                continue
            lateral = float(np.linalg.norm(relative - along * direction))
            protected = (
                state.agent_radius + float(track.radius)
                + self.config.blockage_safety_margin_m
            )
            if lateral <= protected:
                blockers.append((along, lateral, int(track.track_id), track))
        return min(blockers)[3] if blockers else None

    def commit_selected(
        self,
        state: JointSafetyInputV1,
        source: str,
        actions: np.ndarray,
    ) -> int:
        """Commit passing direction only for the behavior actually executed."""

        labels = source.split("+")
        positions = np.asarray(state.positions, dtype=float)
        next_positions, _ = velocity_first_step(
            positions,
            np.asarray(state.velocities, dtype=float),
            np.asarray(actions, dtype=float),
            dt=state.dt,
            max_acceleration=state.max_acceleration,
            max_speed=state.max_speed,
        )
        displacements = next_positions - positions
        flips = 0
        for agent_id, label in enumerate(labels[: self.num_agents]):
            blocker = self.blocking_track(state, agent_id)
            if blocker is None:
                continue
            if label not in {
                "committed_tangent", "tangent_goal_blend", "stall_escape",
                "opposite_tangent_probe", "emergency_tangent",
            } and (agent_id, blocker.track_id) not in self.passing_sides:
                continue
            radial = np.asarray(blocker.relative_position, dtype=float)
            norm = float(np.linalg.norm(radial))
            if norm <= 1e-12:
                continue
            base = np.asarray([-radial[1], radial[0]], dtype=float) / norm
            lateral = float(np.dot(displacements[agent_id], base))
            if abs(lateral) <= 1e-7:
                continue
            selected_side = 1 if lateral > 0.0 else -1
            key = (agent_id, blocker.track_id)
            previous = self.passing_sides.get(key)
            if previous is not None and previous[0] != selected_side:
                flips += 1
            self.passing_sides[key] = (selected_side, state.step)
        self.passing_side_flip_count += flips
        return flips

    @property
    def stalled(self) -> np.ndarray:
        return self.stall_counts >= self.config.stall_patience_steps

    def score(
        self,
        state: JointSafetyInputV1,
        actions: np.ndarray,
        requested_actions: np.ndarray,
        arrived: np.ndarray,
    ) -> tuple[float, float]:
        positions = np.asarray(state.positions, dtype=float)
        velocities = np.asarray(state.velocities, dtype=float)
        goals = np.asarray(state.final_goals, dtype=float)
        next_positions, _ = velocity_first_step(
            positions, velocities, actions, dt=state.dt,
            max_acceleration=state.max_acceleration, max_speed=state.max_speed,
        )
        current_distance = np.linalg.norm(goals - positions, axis=1)
        next_distance = np.linalg.norm(goals - next_positions, axis=1)
        progress = current_distance - next_distance
        tangential_progress = np.zeros(self.num_agents, dtype=float)
        flip_count = 0
        for agent_id in range(self.num_agents):
            blocker = self.blocking_track(state, agent_id)
            if blocker is None:
                continue
            radial = np.asarray(blocker.relative_position, dtype=float)
            radial_norm = float(np.linalg.norm(radial))
            if radial_norm <= 1e-12:
                continue
            side = self.preferred_passing_side(
                state, agent_id, blocker.track_id, radial, commit=False
            )
            tangent = side * np.asarray([-radial[1], radial[0]]) / radial_norm
            displacement = next_positions[agent_id] - positions[agent_id]
            tangential_progress[agent_id] = max(
                float(np.dot(displacement, tangent)), 0.0
            )
            existing = self.passing_sides.get((agent_id, blocker.track_id))
            releasing = self.stall_counts[agent_id] >= int(
                np.ceil(
                    self.config.passing_side_release_stall_multiplier
                    * self.config.stall_patience_steps
                )
            )
            if (
                existing is not None and not releasing
                and float(np.dot(displacement, tangent)) < -1e-7
            ):
                flip_count += 1
        scale = max(float(np.max(current_distance)), 1e-6)
        straggler = current_distance / scale
        weights = (
            1.0 + self.config.straggler_weight * straggler
            + self.config.stall_weight * self.stall_counts
        )
        weights[arrived] = 0.0
        hold_penalty = self.config.arrival_hold_weight * float(
            np.sum(np.linalg.norm(next_positions[arrived] - positions[arrived], axis=1))
        ) if np.any(arrived) else 0.0
        goal_exit_count = int(np.sum(arrived & (next_distance > state.goal_radius)))
        self.last_score_goal_exit_count = goal_exit_count
        actor_deviation = float(np.linalg.norm(actions - requested_actions))
        score = (
            float(np.dot(weights, progress))
            + self.config.tangential_progress_weight
            * float(np.dot(weights, tangential_progress))
            - hold_penalty
            - self.config.passing_side_flip_penalty * float(flip_count)
            - self.config.goal_exit_penalty * float(goal_exit_count)
            - self.config.actor_deviation_weight * actor_deviation
        )
        return score, actor_deviation

    def predicted_goal_exit_count(
        self,
        state: JointSafetyInputV1,
        actions: np.ndarray,
        arrived: np.ndarray,
    ) -> int:
        positions = np.asarray(state.positions, dtype=float)
        next_positions, _ = velocity_first_step(
            positions,
            np.asarray(state.velocities, dtype=float),
            np.asarray(actions, dtype=float),
            dt=state.dt,
            max_acceleration=state.max_acceleration,
            max_speed=state.max_speed,
        )
        distances = np.linalg.norm(
            np.asarray(state.final_goals, dtype=float) - next_positions, axis=1
        )
        return int(np.sum(np.asarray(arrived, dtype=bool) & (distances > state.goal_radius)))

    def side_snapshot(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (f"{agent_id}:{track_id}", int(side))
            for (agent_id, track_id), (side, _step) in sorted(self.passing_sides.items())
        )
