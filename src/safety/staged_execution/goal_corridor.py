"""Causal goal-boundary corridor projection for the shared safety stack.

The projection is deliberately small: when an agent is already inside a
pre-registered corridor next to a boundary goal, remove only the outward
normal acceleration.  Tangential Actor control is preserved verbatim.  The
routine uses current observable kinematics and configuration geometry only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


GOAL_BOUNDARY_CORRIDOR_V1 = "goal_boundary_corridor_v1"


@dataclass(frozen=True)
class GoalBoundaryCorridorDecisionV1:
    """Result of one bounded goal-corridor projection."""

    actions: np.ndarray
    affected_agent_mask: tuple[bool, ...]
    affected_axes: tuple[tuple[bool, bool], ...]
    tangential_axes: tuple[int | None, ...]
    intervention_count: int
    reason: str


def project_goal_boundary_corridor(
    *,
    actions: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    goals: np.ndarray,
    map_size: np.ndarray,
    agent_radius: float,
    goal_radius: float,
    max_acceleration: float,
    dt: float,
    corridor_coordinate: float = 94.0,
    boundary_axis: int = 0,
    boundary_side: str = "high",
) -> GoalBoundaryCorridorDecisionV1:
    """Remove outward normal acceleration in the registered goal corridor.

    V5.1 registers the right-boundary corridor used by the 3U/4S+1D task.
    The axis/side arguments keep the pure logic reusable and testable without
    looking at simulator obstacle truth or static/dynamic identities.
    """

    values = np.asarray(actions, dtype=np.float32).copy()
    points = np.asarray(positions, dtype=np.float32)
    speeds = np.asarray(velocities, dtype=np.float32)
    targets = np.asarray(goals, dtype=np.float32)
    extent = np.asarray(map_size, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("goal corridor actions must have shape (N, 2)")
    if points.shape != values.shape or speeds.shape != values.shape:
        raise ValueError("goal corridor position/velocity shape mismatch")
    if targets.shape != values.shape or extent.shape != (2,):
        raise ValueError("goal corridor goal/map shape mismatch")
    if boundary_axis not in (0, 1) or boundary_side not in {"low", "high"}:
        raise ValueError("unsupported goal corridor boundary")
    if max_acceleration <= 0.0 or dt <= 0.0 or goal_radius <= 0.0:
        raise ValueError("goal corridor dynamics must be positive")

    outward_sign = 1.0 if boundary_side == "high" else -1.0
    boundary = float(extent[boundary_axis]) if boundary_side == "high" else 0.0
    trigger = float(corridor_coordinate)
    if boundary_side == "low":
        trigger = boundary + (float(extent[boundary_axis]) - trigger)
    tangential_axis = 1 - boundary_axis
    affected = np.zeros(values.shape[0], dtype=bool)
    affected_axes = np.zeros_like(values, dtype=bool)

    for agent_id in range(values.shape[0]):
        goal_near_boundary = (
            targets[agent_id, boundary_axis] >= trigger
            if boundary_side == "high"
            else targets[agent_id, boundary_axis] <= trigger
        )
        inside_corridor = (
            points[agent_id, boundary_axis] >= trigger
            if boundary_side == "high"
            else points[agent_id, boundary_axis] <= trigger
        )
        if not goal_near_boundary or not inside_corridor:
            continue
        # Do not turn a completed goal into a recovery target.  The corridor
        # still removes acceleration that would carry it farther outward.
        outward_velocity = outward_sign * float(speeds[agent_id, boundary_axis])
        outward_action = outward_sign * float(values[agent_id, boundary_axis])
        predicted_outward_velocity = (
            outward_velocity + outward_action * max_acceleration * dt
        )
        if predicted_outward_velocity <= 1e-8:
            continue
        braking = float(np.clip(-outward_velocity / (max_acceleration * dt), -1.0, 0.0))
        projected_outward_action = min(outward_action, braking)
        values[agent_id, boundary_axis] = outward_sign * projected_outward_action
        affected[agent_id] = True
        affected_axes[agent_id, boundary_axis] = True

    return GoalBoundaryCorridorDecisionV1(
        actions=values,
        affected_agent_mask=tuple(bool(item) for item in affected),
        affected_axes=tuple(tuple(bool(axis) for axis in row) for row in affected_axes),
        tangential_axes=tuple(
            tangential_axis if item else None for item in affected
        ),
        intervention_count=int(np.count_nonzero(affected)),
        reason=("projected_outward_normal" if np.any(affected) else "no_corridor_risk"),
    )
