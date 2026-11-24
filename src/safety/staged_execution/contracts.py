"""Hash-stable contracts retained by the shared staged safety executor.

New code should prefer ``src.safety.contracts`` for intervention telemetry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PLANNING_SNAPSHOT_SCHEMA = "TeamPlanningSnapshotV1"


@dataclass(frozen=True)
class RobustRecoveryResultV1:
    safe: bool
    robust_margin_m: float | None
    team_progress_m: float
    affected_min_progress_m: float
    actor_deviation: float
    reason: str
    schema_version: str = "RobustRecoveryResultV1"

CANDIDATE_FEASIBILITY_LEGACY = "legacy_soft_v1"
CANDIDATE_FEASIBILITY_V2 = "guard_aligned_observable_v2"
CANDIDATE_FEASIBILITY_V3 = "guard_aligned_recovery_v3"



@dataclass(frozen=True)
class ObservableObstacleV1:
    """One causal obstacle track as seen by one agent."""

    observer_id: int
    track_id: int
    relative_position: tuple[float, float]
    relative_velocity: tuple[float, float]
    radius: float
    velocity_valid: bool

    def __post_init__(self) -> None:
        if self.observer_id < 0 or self.track_id < 0 or self.radius <= 0.0:
            raise ValueError("invalid observable obstacle identity or radius")
        values = (*self.relative_position, *self.relative_velocity, self.radius)
        if not np.isfinite(values).all():
            raise ValueError("observable obstacle values must be finite")


@dataclass(frozen=True)
class TeamPlanningSnapshotV1:
    """Observable-only, dynamically sized team planning state."""

    positions: tuple[tuple[float, float], ...]
    velocities: tuple[tuple[float, float], ...]
    final_goals: tuple[tuple[float, float], ...]
    visible_obstacles: tuple[tuple[ObservableObstacleV1, ...], ...]
    stagnation_counts: tuple[int, ...]
    map_size: tuple[float, float]
    agent_radius: float
    max_speed: float
    max_acceleration: float
    dt: float
    replanning_interval: int
    teammate_safety_distance: float
    obstacle_safety_margin: float
    step: int
    boundary_surface_clearance: float = 0.0
    boundary_invariant_buffer: float = 0.0
    candidate_feasibility_version: str = CANDIDATE_FEASIBILITY_LEGACY
    schema_version: str = PLANNING_SNAPSHOT_SCHEMA

    def __post_init__(self) -> None:
        n = len(self.positions)
        if self.schema_version != PLANNING_SNAPSHOT_SCHEMA or n < 2:
            raise ValueError("unsupported planning snapshot schema or team size")
        if not (len(self.velocities) == len(self.final_goals) == len(self.visible_obstacles)
                == len(self.stagnation_counts) == n):
            raise ValueError("all per-agent snapshot fields must have identical length")
        matrix = np.asarray((*self.positions, *self.velocities, *self.final_goals), dtype=float)
        if matrix.shape != (3 * n, 2) or not np.isfinite(matrix).all():
            raise ValueError("positions, velocities and goals must be finite Nx2 values")
        if (len(self.map_size) != 2 or min(self.map_size) <= 0.0 or self.agent_radius <= 0.0
                or self.max_speed <= 0.0 or self.max_acceleration <= 0.0 or self.dt <= 0.0
                or self.replanning_interval <= 0 or self.teammate_safety_distance <= 0.0):
            raise ValueError("invalid planning dynamics or safety limits")
        if self.step < 0 or any(value < 0 for value in self.stagnation_counts):
            raise ValueError("step and stagnation counts must be nonnegative")
        if self.boundary_surface_clearance < 0.0 or self.boundary_invariant_buffer < 0.0:
            raise ValueError("planning boundary-domain margins must be nonnegative")
        if self.candidate_feasibility_version not in {
            CANDIDATE_FEASIBILITY_LEGACY,
            CANDIDATE_FEASIBILITY_V2,
            CANDIDATE_FEASIBILITY_V3,
        }:
            raise ValueError("unsupported planning candidate feasibility version")
        if np.any(self.admissible_position_high <= self.admissible_position_low):
            raise ValueError("STAR admissible position domain is empty")

    @property
    def num_agents(self) -> int:
        return len(self.positions)

    @property
    def admissible_position_low(self) -> np.ndarray:
        margin = (
            self.agent_radius
            + self.boundary_surface_clearance
            + self.boundary_invariant_buffer
        )
        return np.full(2, margin, dtype=np.float64)

    @property
    def admissible_position_high(self) -> np.ndarray:
        return np.asarray(self.map_size, dtype=np.float64) - self.admissible_position_low

    @classmethod
    def from_arrays(
        cls,
        *,
        positions: np.ndarray,
        velocities: np.ndarray,
        final_goals: np.ndarray,
        visible_obstacles: Sequence[Sequence[ObservableObstacleV1]],
        stagnation_counts: Sequence[int],
        map_size: Sequence[float],
        agent_radius: float,
        max_speed: float,
        max_acceleration: float,
        dt: float,
        replanning_interval: int,
        teammate_safety_distance: float,
        obstacle_safety_margin: float,
        step: int,
        boundary_surface_clearance: float = 0.0,
        boundary_invariant_buffer: float = 0.0,
        candidate_feasibility_version: str = CANDIDATE_FEASIBILITY_LEGACY,
    ) -> "TeamPlanningSnapshotV1":
        def rows(values: np.ndarray) -> tuple[tuple[float, float], ...]:
            array = np.asarray(values, dtype=np.float64)
            return tuple((float(row[0]), float(row[1])) for row in array)

        return cls(
            positions=rows(positions), velocities=rows(velocities),
            final_goals=rows(final_goals),
            visible_obstacles=tuple(tuple(items) for items in visible_obstacles),
            stagnation_counts=tuple(int(value) for value in stagnation_counts),
            map_size=(float(map_size[0]), float(map_size[1])),
            agent_radius=float(agent_radius), max_speed=float(max_speed),
            max_acceleration=float(max_acceleration), dt=float(dt),
            replanning_interval=int(replanning_interval),
            teammate_safety_distance=float(teammate_safety_distance),
            obstacle_safety_margin=float(obstacle_safety_margin), step=int(step),
            boundary_surface_clearance=float(boundary_surface_clearance),
            boundary_invariant_buffer=float(boundary_invariant_buffer),
            candidate_feasibility_version=str(candidate_feasibility_version),
        )

    def permuted(self, order: Sequence[int]) -> "TeamPlanningSnapshotV1":
        order = tuple(int(value) for value in order)
        if sorted(order) != list(range(self.num_agents)):
            raise ValueError("order must be a permutation of all agents")
        remap = {old: new for new, old in enumerate(order)}
        visible = []
        for old in order:
            visible.append(tuple(
                ObservableObstacleV1(
                    observer_id=remap[item.observer_id], track_id=item.track_id,
                    relative_position=item.relative_position,
                    relative_velocity=item.relative_velocity, radius=item.radius,
                    velocity_valid=item.velocity_valid,
                ) for item in self.visible_obstacles[old]
            ))
        values = asdict(self)
        for key in ("positions", "velocities", "final_goals", "stagnation_counts"):
            values[key] = tuple(getattr(self, key)[index] for index in order)
        values["visible_obstacles"] = tuple(visible)
        return TeamPlanningSnapshotV1(**values)
