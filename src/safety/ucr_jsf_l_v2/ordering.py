"""Physical canonical ordering and deterministic aggregation for V2."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, sqrt
from typing import Any, Iterable

import numpy as np

from src.safety.ucr_jsf_l.contracts import JointSafetyInputV1


def _clean(value: float) -> float:
    value = float(value)
    return 0.0 if value == 0.0 else value


def _row(values: Iterable[float]) -> tuple[float, ...]:
    return tuple(_clean(value) for value in values)


def _collision_ttc(
    relative_position: np.ndarray,
    relative_velocity: np.ndarray,
    protected_radius: float,
) -> float:
    """First non-negative circle-contact time for obstacle-minus-agent motion."""

    r = np.asarray(relative_position, dtype=np.float64)
    v = np.asarray(relative_velocity, dtype=np.float64)
    c = float(np.dot(r, r) - protected_radius * protected_radius)
    if c <= 0.0:
        return 0.0
    a = float(np.dot(v, v))
    if a <= 1e-15:
        return float("inf")
    b = 2.0 * float(np.dot(r, v))
    discriminant = b * b - 4.0 * a * c
    if b >= 0.0 or discriminant < 0.0:
        return float("inf")
    root = (-b - float(np.sqrt(discriminant))) / (2.0 * a)
    return max(root, 0.0) if root >= 0.0 else float("inf")


def _obstacle_features(state: JointSafetyInputV1, agent_id: int) -> tuple:
    return tuple(sorted(
        (
            _row(track.relative_position),
            _row(track.relative_velocity),
            _clean(track.radius),
            bool(track.velocity_valid),
            int(track.observation_age_steps),
            int(track.track_id),
        )
        for track in state.visible_obstacles[agent_id]
    ))


def _memory_signature(memory: Any, agent_id: int) -> tuple:
    if memory is None or not hasattr(memory, "memory_signature"):
        return ()
    return tuple(memory.memory_signature(agent_id))


def canonical_agent_key(
    state: JointSafetyInputV1,
    requested_actions: np.ndarray,
    agent_id: int,
    *,
    liveness_memory: Any = None,
    reachability_memory: Any = None,
) -> tuple:
    """Return an agent-id-free key headed by physical urgency."""

    positions = np.asarray(state.positions, dtype=np.float64)
    velocities = np.asarray(state.velocities, dtype=np.float64)
    goals = np.asarray(state.final_goals, dtype=np.float64)
    position = positions[agent_id]
    velocity = velocities[agent_id]
    goal = goals[agent_id]
    tracks = state.visible_obstacles[agent_id]
    stall_count = int(
        getattr(liveness_memory, "stall_counts", np.zeros(state.num_agents))[agent_id]
    )
    ttc_values = [
        _collision_ttc(
            np.asarray(track.relative_position),
            np.asarray(track.relative_velocity),
            state.agent_radius + float(track.radius),
        )
        for track in tracks if track.velocity_valid
    ]
    minimum_ttc = min(ttc_values, default=float("inf"))
    obstacle_clearance = min(
        (
            float(np.linalg.norm(track.relative_position))
            - state.agent_radius - float(track.radius)
            for track in tracks
        ),
        default=float("inf"),
    )
    boundary_clearance = min(
        float(position[0] - state.agent_radius),
        float(position[1] - state.agent_radius),
        float(state.map_size[0] - state.agent_radius - position[0]),
        float(state.map_size[1] - state.agent_radius - position[1]),
    )
    predicted = positions + velocities * state.dt
    teammate_clearance = min(
        (
            float(np.linalg.norm(predicted[agent_id] - predicted[other]))
            - 2.0 * state.agent_radius
            for other in range(state.num_agents) if other != agent_id
        ),
        default=float("inf"),
    )
    goal_distance = float(np.linalg.norm(goal - position))
    urgency = (
        -stall_count,
        _clean(minimum_ttc),
        _clean(obstacle_clearance),
        _clean(boundary_clearance),
        _clean(teammate_clearance),
        -_clean(goal_distance),
    )
    return (
        urgency,
        _row(position),
        _row(velocity),
        _row(goal),
        _row(requested_actions[agent_id]),
        _obstacle_features(state, agent_id),
        _memory_signature(liveness_memory, agent_id),
        _memory_signature(reachability_memory, agent_id),
    )


@dataclass(frozen=True)
class CanonicalOrderingV2:
    """Ordered physical groups; equal keys are explicit degeneracy groups."""

    groups: tuple[tuple[int, ...], ...]
    keys: tuple[tuple, ...]

    @property
    def flat(self) -> tuple[int, ...]:
        return tuple(agent_id for group in self.groups for agent_id in group)

    @property
    def rank_by_agent(self) -> tuple[int, ...]:
        result = [-1] * sum(len(group) for group in self.groups)
        for rank, agent_id in enumerate(self.flat):
            result[agent_id] = rank
        return tuple(result)

    @property
    def has_degenerate_group(self) -> bool:
        return any(len(group) > 1 for group in self.groups)

    def ranks_are_symmetric(self, ranks: tuple[int, ...]) -> bool:
        return all(
            len({ranks[agent_id] for agent_id in group}) == 1
            for group in self.groups if len(group) > 1
        )


def build_canonical_ordering(
    state: JointSafetyInputV1,
    requested_actions: np.ndarray,
    *,
    liveness_memory: Any = None,
    reachability_memory: Any = None,
) -> CanonicalOrderingV2:
    requested = np.asarray(requested_actions, dtype=np.float64)
    keys = [
        canonical_agent_key(
            state,
            requested,
            agent_id,
            liveness_memory=liveness_memory,
            reachability_memory=reachability_memory,
        )
        for agent_id in range(state.num_agents)
    ]
    distinct = sorted(set(keys))
    groups = tuple(
        tuple(agent_id for agent_id, key in enumerate(keys) if key == distinct_key)
        for distinct_key in distinct
    )
    return CanonicalOrderingV2(groups=groups, keys=tuple(distinct))


def canonical_action_signature(
    actions: np.ndarray,
    ordering: CanonicalOrderingV2,
) -> tuple[tuple[float, float], ...]:
    values = np.asarray(actions, dtype=np.float64)
    rows: list[tuple[float, float]] = []
    for group in ordering.groups:
        group_rows = [
            (_clean(values[agent_id, 0]), _clean(values[agent_id, 1]))
            for agent_id in group
        ]
        rows.extend(sorted(group_rows))
    return tuple(rows)


def canonical_fsum(
    per_agent_values: Iterable[float],
    ordering: CanonicalOrderingV2,
) -> float:
    values = tuple(float(value) for value in per_agent_values)
    ordered: list[float] = []
    for group in ordering.groups:
        ordered.extend(sorted(values[agent_id] for agent_id in group))
    return float(fsum(ordered))


def deterministic_actor_deviation(
    actions: np.ndarray,
    requested_actions: np.ndarray,
    ordering: CanonicalOrderingV2,
) -> float:
    delta = np.asarray(actions, dtype=np.float64) - np.asarray(
        requested_actions, dtype=np.float64
    )
    squared = [float(fsum(float(value * value) for value in row)) for row in delta]
    return float(sqrt(max(canonical_fsum(squared, ordering), 0.0)))


def minimum_effective_margin(report: Any) -> float:
    values = (
        report.minimum_boundary_margin_m,
        report.minimum_pair_margin_m,
        report.minimum_obstacle_margin_m,
    )
    return min((float(value) for value in values if value is not None), default=float("inf"))
