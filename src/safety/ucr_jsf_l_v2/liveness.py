"""Permutation-equivariant progress and passing-side memory for V2."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, fsum

import numpy as np

from src.safety.ucr_jsf_l.dynamics import velocity_first_step

from .contracts import JointSafetyInputV2
from .ordering import (
    CanonicalOrderingV2,
    canonical_fsum,
    deterministic_actor_deviation,
)


@dataclass(frozen=True)
class LivenessConfigV2:
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
    def from_mapping(cls, payload: dict) -> "LivenessConfigV2":
        return cls(**{
            field: type(default)(payload.get(field, default))
            for field, default in (
                ("stall_patience_steps", 12),
                ("minimum_progress_m", 0.02),
                ("passing_side_ttl_steps", 20),
                ("straggler_weight", 1.0),
                ("stall_weight", 0.25),
                ("arrival_hold_weight", 2.0),
                ("actor_deviation_weight", 0.05),
                ("blockage_distance_m", 12.0),
                ("blockage_safety_margin_m", 1.5),
                ("tangential_progress_weight", 0.75),
                ("passing_side_flip_penalty", 1.0),
                ("goal_exit_penalty", 20.0),
                ("passing_side_release_stall_multiplier", 2.0),
            )
        })


class LivenessMemoryV2:
    def __init__(self, num_agents: int, config: LivenessConfigV2) -> None:
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

    def memory_signature(self, agent_id: int) -> tuple:
        previous = (
            None if self.previous_distances is None
            else float(self.previous_distances[agent_id])
        )
        sides = tuple(sorted(
            (int(track_id), int(side), int(step))
            for (observer_id, track_id), (side, step) in self.passing_sides.items()
            if observer_id == agent_id
        ))
        return previous, int(self.stall_counts[agent_id]), sides

    def permuted_copy(self, permutation: tuple[int, ...]) -> "LivenessMemoryV2":
        if sorted(permutation) != list(range(self.num_agents)):
            raise ValueError("permutation must contain every agent exactly once")
        old_to_new = {old: new for new, old in enumerate(permutation)}
        result = LivenessMemoryV2(self.num_agents, self.config)
        if self.previous_distances is not None:
            result.previous_distances = self.previous_distances[list(permutation)].copy()
        result.stall_counts = self.stall_counts[list(permutation)].copy()
        result.passing_sides = {
            (old_to_new[observer_id], track_id): value
            for (observer_id, track_id), value in self.passing_sides.items()
        }
        result.passing_side_flip_count = self.passing_side_flip_count
        result.last_score_goal_exit_count = self.last_score_goal_exit_count
        return result

    def observe(self, state: JointSafetyInputV2) -> tuple[np.ndarray, np.ndarray]:
        positions = np.asarray(state.positions, dtype=np.float64)
        goals = np.asarray(state.final_goals, dtype=np.float64)
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
            if key in visible
            and state.step - value[1] <= self.config.passing_side_ttl_steps
        }
        return distances, arrived

    def blocking_track(self, state: JointSafetyInputV2, agent_id: int):
        goal_delta = (
            np.asarray(state.final_goals[agent_id], dtype=np.float64)
            - np.asarray(state.positions[agent_id], dtype=np.float64)
        )
        goal_distance = float(np.linalg.norm(goal_delta))
        if goal_distance <= 1e-12:
            return None
        direction = goal_delta / goal_distance
        corridor_length = min(goal_distance, self.config.blockage_distance_m)
        blockers = []
        for track in state.visible_obstacles[agent_id]:
            relative = np.asarray(track.relative_position, dtype=np.float64)
            along = float(np.dot(relative, direction))
            if not 0.0 < along <= corridor_length:
                continue
            lateral = float(np.linalg.norm(relative - along * direction))
            protected = (
                state.agent_radius + float(track.radius)
                + self.config.blockage_safety_margin_m
            )
            if lateral <= protected:
                blockers.append((along, lateral, int(track.track_id), track))
        return min(blockers)[3] if blockers else None

    @staticmethod
    def _choice(left: float, right: float, tolerance: float = 1e-12) -> int | None:
        if left > right + tolerance:
            return 1
        if right > left + tolerance:
            return -1
        return None

    def _geometric_fallback_side(
        self,
        state: JointSafetyInputV2,
        agent_id: int,
        track,
        radial: np.ndarray,
    ) -> int:
        norm = float(np.linalg.norm(radial))
        if norm <= 1e-12:
            return 1
        base = np.asarray([-radial[1], radial[0]], dtype=np.float64) / norm
        position = np.asarray(state.positions[agent_id], dtype=np.float64)

        if track.velocity_valid:
            predicted_relative = radial + np.asarray(
                track.relative_velocity, dtype=np.float64
            ) * state.dt
            clearances = []
            for side in (1, -1):
                agent_shift = side * base * state.max_speed * state.dt
                clearances.append(float(np.linalg.norm(predicted_relative - agent_shift)))
            choice = self._choice(clearances[0], clearances[1])
            if choice is not None:
                return choice

        boundary_spaces = []
        for side in (1, -1):
            point = position + side * base * state.max_speed * state.dt
            boundary_spaces.append(min(
                float(point[0] - state.agent_radius),
                float(point[1] - state.agent_radius),
                float(state.map_size[0] - state.agent_radius - point[0]),
                float(state.map_size[1] - state.agent_radius - point[1]),
            ))
        choice = self._choice(boundary_spaces[0], boundary_spaces[1])
        if choice is not None:
            return choice

        positions = np.asarray(state.positions, dtype=np.float64)
        velocities = np.asarray(state.velocities, dtype=np.float64)
        teammate_spaces = []
        for side in (1, -1):
            next_self = position + side * base * state.max_speed * state.dt
            teammate_spaces.append(min(
                (
                    float(np.linalg.norm(
                        next_self - (positions[other] + velocities[other] * state.dt)
                    ))
                    for other in range(state.num_agents) if other != agent_id
                ),
                default=float("inf"),
            ))
        choice = self._choice(teammate_spaces[0], teammate_spaces[1])
        if choice is not None:
            return choice

        lateral_velocity = float(np.dot(velocities[agent_id], base))
        if abs(lateral_velocity) > 1e-12:
            return 1 if lateral_velocity > 0.0 else -1
        return 1

    def preferred_passing_side(
        self,
        state: JointSafetyInputV2,
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
            np.asarray(state.final_goals[agent_id], dtype=np.float64)
            - np.asarray(state.positions[agent_id], dtype=np.float64)
        )
        radial = np.asarray(relative_position, dtype=np.float64)
        cross = float(goal_delta[0] * radial[1] - goal_delta[1] * radial[0])
        if abs(cross) > 1e-12:
            side = 1 if cross > 0.0 else -1
        else:
            track = next(
                (item for item in state.visible_obstacles[agent_id]
                 if item.track_id == track_id),
                None,
            )
            side = 1 if track is None else self._geometric_fallback_side(
                state, agent_id, track, radial
            )
        if commit:
            self.passing_sides[key] = (side, state.step)
        return side

    def passing_side(
        self,
        state: JointSafetyInputV2,
        agent_id: int,
        track_id: int,
        relative_position: np.ndarray,
    ) -> int:
        return self.preferred_passing_side(
            state, agent_id, track_id, relative_position, commit=True
        )

    def commit_selected(
        self,
        state: JointSafetyInputV2,
        source: str,
        actions: np.ndarray,
    ) -> int:
        labels = source.split("+")
        positions = np.asarray(state.positions, dtype=np.float64)
        next_positions, _ = velocity_first_step(
            positions,
            np.asarray(state.velocities, dtype=np.float64),
            np.asarray(actions, dtype=np.float64),
            dt=state.dt,
            max_acceleration=state.max_acceleration,
            max_speed=state.max_speed,
        )
        displacements = next_positions - positions
        flips = 0
        tangent_labels = {
            "committed_tangent", "tangent_goal_blend", "stall_escape",
            "opposite_tangent_probe", "emergency_tangent",
        }
        for agent_id, label in enumerate(labels[:self.num_agents]):
            blocker = self.blocking_track(state, agent_id)
            if blocker is None:
                continue
            key = (agent_id, blocker.track_id)
            if label not in tangent_labels and key not in self.passing_sides:
                continue
            radial = np.asarray(blocker.relative_position, dtype=np.float64)
            norm = float(np.linalg.norm(radial))
            if norm <= 1e-12:
                continue
            base = np.asarray([-radial[1], radial[0]], dtype=np.float64) / norm
            lateral = float(np.dot(displacements[agent_id], base))
            if abs(lateral) <= 1e-7:
                continue
            selected_side = 1 if lateral > 0.0 else -1
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
        state: JointSafetyInputV2,
        actions: np.ndarray,
        requested_actions: np.ndarray,
        arrived: np.ndarray,
        ordering: CanonicalOrderingV2,
    ) -> tuple[float, float]:
        positions = np.asarray(state.positions, dtype=np.float64)
        velocities = np.asarray(state.velocities, dtype=np.float64)
        goals = np.asarray(state.final_goals, dtype=np.float64)
        next_positions, _ = velocity_first_step(
            positions, velocities, actions, dt=state.dt,
            max_acceleration=state.max_acceleration, max_speed=state.max_speed,
        )
        current_distance = np.linalg.norm(goals - positions, axis=1)
        next_distance = np.linalg.norm(goals - next_positions, axis=1)
        progress = current_distance - next_distance
        tangential_progress = np.zeros(self.num_agents, dtype=np.float64)
        flip_count = 0
        for agent_id in range(self.num_agents):
            blocker = self.blocking_track(state, agent_id)
            if blocker is None:
                continue
            radial = np.asarray(blocker.relative_position, dtype=np.float64)
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
            releasing = self.stall_counts[agent_id] >= int(ceil(
                self.config.passing_side_release_stall_multiplier
                * self.config.stall_patience_steps
            ))
            if (
                existing is not None and not releasing
                and float(np.dot(displacement, tangent)) < -1e-7
            ):
                flip_count += 1
        scale = max(float(np.max(current_distance)), 1e-6)
        weights = (
            1.0
            + self.config.straggler_weight * (current_distance / scale)
            + self.config.stall_weight * self.stall_counts
        )
        weights[np.asarray(arrived, dtype=bool)] = 0.0
        hold_rows = [
            self.config.arrival_hold_weight
            * float(np.linalg.norm(next_positions[index] - positions[index]))
            if bool(arrived[index]) else 0.0
            for index in range(self.num_agents)
        ]
        hold_penalty = canonical_fsum(hold_rows, ordering)
        goal_exit_count = int(fsum(
            1.0 if bool(arrived[index]) and next_distance[index] > state.goal_radius else 0.0
            for index in ordering.flat
        ))
        self.last_score_goal_exit_count = goal_exit_count
        actor_deviation = deterministic_actor_deviation(
            actions, requested_actions, ordering
        )
        weighted_progress = canonical_fsum(weights * progress, ordering)
        weighted_tangent = canonical_fsum(weights * tangential_progress, ordering)
        score = float(fsum((
            weighted_progress,
            self.config.tangential_progress_weight * weighted_tangent,
            -hold_penalty,
            -self.config.passing_side_flip_penalty * float(flip_count),
            -self.config.goal_exit_penalty * float(goal_exit_count),
            -self.config.actor_deviation_weight * actor_deviation,
        )))
        return score, actor_deviation

    def predicted_goal_exit_count(
        self,
        state: JointSafetyInputV2,
        actions: np.ndarray,
        arrived: np.ndarray,
    ) -> int:
        positions = np.asarray(state.positions, dtype=np.float64)
        next_positions, _ = velocity_first_step(
            positions,
            np.asarray(state.velocities, dtype=np.float64),
            np.asarray(actions, dtype=np.float64),
            dt=state.dt,
            max_acceleration=state.max_acceleration,
            max_speed=state.max_speed,
        )
        distances = np.linalg.norm(
            np.asarray(state.final_goals, dtype=np.float64) - next_positions,
            axis=1,
        )
        return int(np.sum(np.asarray(arrived, dtype=bool) & (distances > state.goal_radius)))

    def side_snapshot(
        self, ordering: CanonicalOrderingV2
    ) -> tuple[tuple[str, int], ...]:
        group_by_agent = {
            agent_id: group_rank
            for group_rank, group in enumerate(ordering.groups)
            for agent_id in group
        }
        return tuple(sorted(
            (f"g{group_by_agent[agent_id]}:{track_id}", int(side))
            for (agent_id, track_id), (side, _step) in self.passing_sides.items()
        ))
