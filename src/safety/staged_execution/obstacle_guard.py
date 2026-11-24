"""Observable-only shared TTC execution guard.

The guard consumes the same causal obstacle tracks exposed by
``tracked_obstacle_v2``.  It never reads simulator obstacle velocity, future
positions, or static/dynamic identity.  Projection is deterministic and
bounded; an unresolved or emergency risk is returned to the recovery state
machine instead of silently stopping the whole team.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .contracts import ObservableObstacleV1


OBSTACLE_EXECUTION_GUARD_V1 = "causal_ttc_projection_v1"


@dataclass(frozen=True)
class ObstacleExecutionRiskV1:
    agent_id: int
    track_id: int
    ttc_seconds: float
    predicted_clearance_m: float
    emergency: bool


@dataclass(frozen=True)
class ObstacleExecutionDecisionV1:
    actions: np.ndarray
    affected_agent_mask: tuple[bool, ...]
    risks: tuple[ObstacleExecutionRiskV1, ...]
    intervention_count: int
    fallback_required: bool
    reason: str


def _closest_approach(
    relative_position: np.ndarray,
    relative_velocity: np.ndarray,
    *,
    horizon_seconds: float,
    safe_distance: float,
) -> tuple[float, float, bool]:
    speed_squared = float(np.dot(relative_velocity, relative_velocity))
    closing = bool(float(np.dot(relative_position, relative_velocity)) < 0.0)
    if speed_squared <= 1e-12:
        return float("inf"), float(np.linalg.norm(relative_position) - safe_distance), False
    t_star = float(np.clip(
        -float(np.dot(relative_position, relative_velocity)) / speed_squared,
        0.0,
        horizon_seconds,
    ))
    clearance = float(
        np.linalg.norm(relative_position + t_star * relative_velocity)
        - safe_distance
    )
    c = float(np.dot(relative_position, relative_position) - safe_distance ** 2)
    if c <= 0.0:
        entry_ttc = 0.0
    else:
        b = 2.0 * float(np.dot(relative_position, relative_velocity))
        discriminant = b * b - 4.0 * speed_squared * c
        roots = () if discriminant < 0.0 else tuple(
            value for value in (
                (-b - np.sqrt(discriminant)) / (2.0 * speed_squared),
                (-b + np.sqrt(discriminant)) / (2.0 * speed_squared),
            ) if value >= 0.0
        )
        entry_ttc = float(min(roots)) if roots else float("inf")
    return entry_ttc, clearance, closing


def assess_obstacle_actions(
    *,
    actions: np.ndarray,
    velocities: np.ndarray,
    visible_obstacles: Sequence[Sequence[ObservableObstacleV1]],
    agent_radius: float,
    obstacle_safety_margin: float,
    invariant_buffer: float,
    max_acceleration: float,
    dt: float,
    horizon_seconds: float,
    warning_ttc_seconds: float,
    emergency_ttc_seconds: float,
) -> tuple[ObstacleExecutionRiskV1, ...]:
    """Return actionable risks produced by the proposed acceleration matrix."""

    values = np.asarray(actions, dtype=np.float64)
    velocity_values = np.asarray(velocities, dtype=np.float64)
    if values.shape != velocity_values.shape or values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("obstacle guard actions and velocities must be matching Nx2 matrices")
    if len(visible_obstacles) != len(values):
        raise ValueError("obstacle guard tracks must contain one row per agent")
    risks: list[ObstacleExecutionRiskV1] = []
    for agent_id, tracks in enumerate(visible_obstacles):
        acceleration_delta = (
            np.clip(values[agent_id], -1.0, 1.0)
            * float(max_acceleration) * float(dt)
        )
        for obstacle in tracks:
            if not obstacle.velocity_valid:
                continue
            relative_position = np.asarray(
                obstacle.relative_position, dtype=np.float64
            )
            # The track stores obstacle-minus-agent relative velocity.  A
            # candidate acceleration changes only the agent term.
            relative_velocity = np.asarray(
                obstacle.relative_velocity, dtype=np.float64
            ) - acceleration_delta
            safe_distance = (
                float(agent_radius) + float(obstacle.radius)
                + float(obstacle_safety_margin) + float(invariant_buffer)
            )
            ttc, clearance, closing = _closest_approach(
                relative_position,
                relative_velocity,
                horizon_seconds=float(horizon_seconds),
                safe_distance=safe_distance,
            )
            if closing and ttc <= float(warning_ttc_seconds) and clearance < 0.0:
                risks.append(ObstacleExecutionRiskV1(
                    agent_id=int(agent_id),
                    track_id=int(obstacle.track_id),
                    ttc_seconds=float(ttc),
                    predicted_clearance_m=float(clearance),
                    emergency=bool(ttc <= float(emergency_ttc_seconds)),
                ))
    return tuple(sorted(
        risks,
        key=lambda item: (
            item.agent_id, item.ttc_seconds, item.predicted_clearance_m,
            item.track_id,
        ),
    ))


def minimum_predicted_obstacle_clearance_m(
    *,
    actions: np.ndarray,
    velocities: np.ndarray,
    visible_obstacles: Sequence[Sequence[ObservableObstacleV1]],
    agent_radius: float,
    obstacle_safety_margin: float,
    invariant_buffer: float,
    max_acceleration: float,
    dt: float,
    horizon_seconds: float,
) -> float:
    """Return observable-only closest-approach clearance for an action.

    Positive values are safety margins and negative values are predicted
    overlap. Callers can pass empty track rows outside the intervention scope.
    """

    values = np.asarray(actions, dtype=np.float64)
    velocity_values = np.asarray(velocities, dtype=np.float64)
    if values.shape != velocity_values.shape or values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("obstacle clearance actions and velocities must be matching Nx2 matrices")
    if len(visible_obstacles) != len(values):
        raise ValueError("obstacle clearance tracks must contain one row per agent")
    clearances: list[float] = []
    for agent_id, tracks in enumerate(visible_obstacles):
        acceleration_delta = (
            np.clip(values[agent_id], -1.0, 1.0)
            * float(max_acceleration) * float(dt)
        )
        for obstacle in tracks:
            if not obstacle.velocity_valid:
                continue
            relative_position = np.asarray(
                obstacle.relative_position, dtype=np.float64
            )
            relative_velocity = np.asarray(
                obstacle.relative_velocity, dtype=np.float64
            ) - acceleration_delta
            safe_distance = (
                float(agent_radius) + float(obstacle.radius)
                + float(obstacle_safety_margin) + float(invariant_buffer)
            )
            _, clearance, _ = _closest_approach(
                relative_position,
                relative_velocity,
                horizon_seconds=float(horizon_seconds),
                safe_distance=safe_distance,
            )
            clearances.append(float(clearance))
    return min(clearances) if clearances else float("inf")


def _candidate_actions(
    raw_action: np.ndarray,
    velocity: np.ndarray,
    risk_rows: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    max_acceleration: float,
    dt: float,
) -> tuple[np.ndarray, ...]:
    """Construct a fixed, deterministic approximation to minimum projection."""

    raw = np.clip(np.asarray(raw_action, dtype=np.float64), -1.0, 1.0)
    candidates = [raw]
    brake = np.clip(
        -np.asarray(velocity, dtype=np.float64)
        / max(float(max_acceleration) * float(dt), 1e-12),
        -1.0,
        1.0,
    )
    candidates.append(brake)
    for relative_position, relative_velocity in risk_rows:
        norm = float(np.linalg.norm(relative_position))
        if norm <= 1e-12:
            continue
        away = -relative_position / norm
        tangent = np.asarray([-away[1], away[0]], dtype=np.float64)
        if float(np.dot(tangent, relative_velocity)) > 0.0:
            tangent = -tangent
        for magnitude in (0.25, 0.50, 0.75, 1.00):
            for direction in (away, tangent, -tangent):
                candidates.append(np.clip(
                    raw + magnitude * direction, -1.0, 1.0
                ))
                candidates.append(np.clip(
                    (1.0 - magnitude) * raw + magnitude * direction,
                    -1.0,
                    1.0,
                ))
    unique: dict[tuple[float, float], np.ndarray] = {}
    for candidate in candidates:
        key = tuple(float(x) for x in np.round(candidate, 10))
        unique.setdefault(key, candidate)
    return tuple(unique[key] for key in sorted(unique))


def project_obstacle_actions(
    *,
    actions: np.ndarray,
    velocities: np.ndarray,
    visible_obstacles: Sequence[Sequence[ObservableObstacleV1]],
    agent_radius: float,
    obstacle_safety_margin: float,
    invariant_buffer: float,
    max_acceleration: float,
    dt: float,
    horizon_seconds: float = 10.0,
    warning_ttc_seconds: float = 5.0,
    emergency_ttc_seconds: float = 2.5,
) -> ObstacleExecutionDecisionV1:
    """Project each implicated action while preserving unrelated actors."""

    values = np.asarray(actions, dtype=np.float64).copy()
    velocity_values = np.asarray(velocities, dtype=np.float64)
    initial = assess_obstacle_actions(
        actions=values,
        velocities=velocity_values,
        visible_obstacles=visible_obstacles,
        agent_radius=agent_radius,
        obstacle_safety_margin=obstacle_safety_margin,
        invariant_buffer=invariant_buffer,
        max_acceleration=max_acceleration,
        dt=dt,
        horizon_seconds=horizon_seconds,
        warning_ttc_seconds=warning_ttc_seconds,
        emergency_ttc_seconds=emergency_ttc_seconds,
    )
    affected = np.zeros(len(values), dtype=bool)
    if not initial:
        return ObstacleExecutionDecisionV1(
            values.astype(np.float32), tuple(bool(x) for x in affected), (),
            0, False, "safe",
        )
    for risk in initial:
        affected[risk.agent_id] = True
    if any(risk.emergency for risk in initial):
        return ObstacleExecutionDecisionV1(
            values.astype(np.float32), tuple(bool(x) for x in affected), initial,
            0, True, "emergency_ttc",
        )

    interventions = 0
    for agent_id in np.flatnonzero(affected):
        tracks = visible_obstacles[int(agent_id)]
        risk_rows = [
            (
                np.asarray(item.relative_position, dtype=np.float64),
                np.asarray(item.relative_velocity, dtype=np.float64),
            )
            for item in tracks if item.velocity_valid
        ]
        candidates = _candidate_actions(
            values[agent_id], velocity_values[agent_id], risk_rows,
            max_acceleration=max_acceleration, dt=dt,
        )
        feasible: list[np.ndarray] = []
        for candidate in candidates:
            trial = values.copy()
            trial[agent_id] = candidate
            scoped_tracks = tuple(
                tuple(tracks) if index == int(agent_id) else ()
                for index in range(len(values))
            )
            remaining = assess_obstacle_actions(
                actions=trial,
                velocities=velocity_values,
                visible_obstacles=scoped_tracks,
                agent_radius=agent_radius,
                obstacle_safety_margin=obstacle_safety_margin,
                invariant_buffer=invariant_buffer,
                max_acceleration=max_acceleration,
                dt=dt,
                horizon_seconds=horizon_seconds,
                warning_ttc_seconds=warning_ttc_seconds,
                emergency_ttc_seconds=emergency_ttc_seconds,
            )
            if not remaining:
                feasible.append(candidate)
        if not feasible:
            return ObstacleExecutionDecisionV1(
                values.astype(np.float32), tuple(bool(x) for x in affected),
                initial, interventions, True, "no_feasible_projection",
            )
        chosen = min(
            feasible,
            key=lambda row: (
                float(np.linalg.norm(row - values[agent_id])),
                float(row[0]), float(row[1]),
            ),
        )
        if np.any(np.abs(chosen - values[agent_id]) > 1e-7):
            interventions += 1
        values[agent_id] = chosen

    remaining = assess_obstacle_actions(
        actions=values,
        velocities=velocity_values,
        visible_obstacles=visible_obstacles,
        agent_radius=agent_radius,
        obstacle_safety_margin=obstacle_safety_margin,
        invariant_buffer=invariant_buffer,
        max_acceleration=max_acceleration,
        dt=dt,
        horizon_seconds=horizon_seconds,
        warning_ttc_seconds=warning_ttc_seconds,
        emergency_ttc_seconds=emergency_ttc_seconds,
    )
    return ObstacleExecutionDecisionV1(
        values.astype(np.float32), tuple(bool(x) for x in affected), initial,
        interventions, bool(remaining),
        "unresolved_after_projection" if remaining else "projected",
    )
