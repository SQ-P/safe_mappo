"""Data contracts owned exclusively by UCR-JSF-L V1."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

import numpy as np


class SafetyMode(str, Enum):
    NORMAL = "NORMAL"
    STALL_ESCAPE = "STALL_ESCAPE"
    SAFETY_RECOVERY = "SAFETY_RECOVERY"


@dataclass(frozen=True)
class ObstacleTrackV1:
    observer_id: int
    track_id: int
    relative_position: tuple[float, float]
    relative_velocity: tuple[float, float]
    radius: float
    velocity_valid: bool
    observation_age_steps: int = 0

    def __post_init__(self) -> None:
        values = (*self.relative_position, *self.relative_velocity, self.radius)
        if (
            self.observer_id < 0 or self.track_id < 0 or self.radius <= 0.0
            or self.observation_age_steps < 0
        ):
            raise ValueError("invalid observable obstacle track")
        if not np.isfinite(values).all():
            raise ValueError("obstacle track values must be finite")


def _rows(values: np.ndarray, name: str) -> tuple[tuple[float, float], ...]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite Nx2 matrix")
    return tuple((float(row[0]), float(row[1])) for row in array)


@dataclass(frozen=True)
class JointSafetyInputV1:
    positions: tuple[tuple[float, float], ...]
    velocities: tuple[tuple[float, float], ...]
    final_goals: tuple[tuple[float, float], ...]
    visible_obstacles: tuple[tuple[ObstacleTrackV1, ...], ...]
    map_size: tuple[float, float]
    agent_radius: float
    goal_radius: float
    max_speed: float
    max_acceleration: float
    dt: float
    step: int
    obstacle_slot_saturated: tuple[bool, ...] = ()
    schema_version: str = "UCRJSFLJointSafetyInputV1"

    def __post_init__(self) -> None:
        count = len(self.positions)
        if count < 2 or not (
            len(self.velocities) == len(self.final_goals)
            == len(self.visible_obstacles) == count
        ):
            raise ValueError("UCR-JSF-L input requires aligned team fields")
        matrix = np.asarray(
            (*self.positions, *self.velocities, *self.final_goals), dtype=float
        )
        if matrix.shape != (3 * count, 2) or not np.isfinite(matrix).all():
            raise ValueError("invalid UCR-JSF-L team state")
        scalars = (
            *self.map_size, self.agent_radius, self.goal_radius, self.max_speed,
            self.max_acceleration, self.dt,
        )
        if not np.isfinite(scalars).all() or min(scalars) <= 0.0 or self.step < 0:
            raise ValueError("invalid UCR-JSF-L dynamics")
        for observer_id, tracks in enumerate(self.visible_obstacles):
            if any(track.observer_id != observer_id for track in tracks):
                raise ValueError("obstacle tracks must remain observer-scoped")
        if not self.obstacle_slot_saturated:
            object.__setattr__(
                self, "obstacle_slot_saturated", tuple(False for _ in range(count))
            )
        elif len(self.obstacle_slot_saturated) != count:
            raise ValueError("obstacle slot saturation must align with the team")

    @property
    def num_agents(self) -> int:
        return len(self.positions)

    @classmethod
    def from_arrays(
        cls,
        *,
        positions: np.ndarray,
        velocities: np.ndarray,
        final_goals: np.ndarray,
        visible_obstacles: Sequence[Sequence[ObstacleTrackV1]],
        map_size: Sequence[float],
        agent_radius: float,
        goal_radius: float,
        max_speed: float,
        max_acceleration: float,
        dt: float,
        step: int,
        obstacle_slot_saturated: Sequence[bool] | None = None,
    ) -> "JointSafetyInputV1":
        return cls(
            positions=_rows(positions, "positions"),
            velocities=_rows(velocities, "velocities"),
            final_goals=_rows(final_goals, "final_goals"),
            visible_obstacles=tuple(tuple(row) for row in visible_obstacles),
            map_size=(float(map_size[0]), float(map_size[1])),
            agent_radius=float(agent_radius), goal_radius=float(goal_radius),
            max_speed=float(max_speed), max_acceleration=float(max_acceleration),
            dt=float(dt), step=int(step),
            obstacle_slot_saturated=tuple(
                bool(value) for value in (
                    obstacle_slot_saturated
                    if obstacle_slot_saturated is not None
                    else (False for _ in range(len(positions)))
                )
            ),
        )


@dataclass(frozen=True)
class ConstraintReportV1:
    model_admissible: bool
    evidence_status: str = "MODEL_ADMISSIBLE"
    violations: tuple[str, ...] = ()
    unknown_evidence: tuple[str, ...] = ()
    bounded_uncertainty: tuple[str, ...] = ()
    minimum_boundary_margin_m: float | None = None
    minimum_pair_margin_m: float | None = None
    minimum_obstacle_margin_m: float | None = None
    schema_version: str = "UCRJSFLConstraintReportV1"

    @property
    def certified_safe(self) -> bool:
        """Backward-compatible alias; this is not a formal safety certificate."""

        return self.model_admissible


@dataclass(frozen=True)
class CandidateEvaluationV1:
    candidate_id: str
    actions: tuple[tuple[float, float], ...]
    constraints: ConstraintReportV1
    progress_score: float | None
    actor_deviation: float
    source: str
    emergency_risk_score: float | None = None


@dataclass(frozen=True)
class JointSafetyTraceV1:
    step: int
    mode_before: str
    mode_after: str
    requested_actions: tuple[tuple[float, float], ...]
    executed_actions: tuple[tuple[float, float], ...]
    selected_candidate_id: str
    selected_source: str
    selection_reason: str
    generated_candidate_count: int
    evaluated_candidate_count: int
    certified_candidate_count: int
    candidate_budget_exhausted: bool
    candidate_search_exhausted: bool
    search_exhaustion_is_physical_infeasibility: bool
    unknown_evidence: tuple[str, ...]
    violations: tuple[str, ...]
    intervention_mask: tuple[bool, ...]
    passing_side_memory: tuple[tuple[str, int], ...]
    stall_counts: tuple[int, ...]
    arrived_mask: tuple[bool, ...]
    minimum_boundary_margin_m: float | None
    minimum_pair_margin_m: float | None
    minimum_obstacle_margin_m: float | None
    selected_evidence_status: str = "MODEL_ADMISSIBLE"
    observed_constraint_pass_count: int = 0
    model_admissible_candidate_count: int = 0
    evidence_insufficient_candidate_count: int = 0
    rejected_candidate_count: int = 0
    unknown_candidate_count: int = 0
    emergency_action_selected: bool = False
    residual_emergency_risk: float | None = None
    passing_side_flip_count_step: int = 0
    stall_agent_count: int = 0
    predicted_goal_exit_count: int = 0
    liveness_recovery_success_step: bool = False
    terminal_transition: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "JointSafetyTraceV1"


@dataclass(frozen=True)
class JointSafetyOutputV1:
    actions: np.ndarray
    trace: JointSafetyTraceV1

    def __post_init__(self) -> None:
        values = np.asarray(self.actions)
        if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
            raise ValueError("executed UCR-JSF-L actions must be finite Nx2")
