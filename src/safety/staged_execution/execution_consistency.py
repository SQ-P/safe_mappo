"""Execution-consistency contracts and deterministic TTC candidate utilities.

The helpers in this module are intentionally independent from the simulator.
They only operate on observable actions, velocities and goals.  Environment
code supplies the causal three-step safety evaluation; this module owns the
fixed candidate vocabulary, audit schema and V2 lexicographic selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np


TTC_HEADROOM_AUDIT_SCHEMA = "TTCActionCandidateAuditV1"
PROJECTION_TRANSITION_SCHEMA = "ProjectionAwareTransitionV1"
TTC_PROGRESS_PROJECTION_V2 = "causal_ttc_progress_projection_v2"


@dataclass(frozen=True)
class TTCActionCandidateEvaluationV1:
    """One registered candidate after the causal three-step shadow."""

    label: str
    agent_id: int
    action: tuple[float, float]
    safe_three_step: bool
    minimum_safety_margin_m: float
    goal_direction_velocity_mps: float
    actor_deviation_norm: float
    rejection_reason: str

    def __post_init__(self) -> None:
        values = (*self.action, self.minimum_safety_margin_m,
                  self.goal_direction_velocity_mps,
                  self.actor_deviation_norm)
        if self.agent_id < 0 or len(self.action) != 2:
            raise ValueError("invalid TTC candidate identity or action")
        if not all(np.isfinite(value) for value in values):
            raise ValueError("TTC candidate metrics must be finite")
        if self.actor_deviation_norm < 0.0:
            raise ValueError("candidate deviation must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TTCActionCandidateAuditV1:
    """Trace-only record for one V1 TTC intervention event."""

    episode_step: int
    affected_agent_ids: tuple[int, ...]
    v1_selected_labels: tuple[str, ...]
    evaluations: tuple[TTCActionCandidateEvaluationV1, ...]
    safe_progress_headroom: bool
    v1_choice_pareto_dominated: bool
    best_goal_velocity_gain_mps: float
    schema_version: str = TTC_HEADROOM_AUDIT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != TTC_HEADROOM_AUDIT_SCHEMA:
            raise ValueError("unsupported TTC audit schema")
        if self.episode_step < 0:
            raise ValueError("episode step must be nonnegative")
        if tuple(sorted(set(self.affected_agent_ids))) != self.affected_agent_ids:
            raise ValueError("affected agent ids must be unique and sorted")
        if len(self.v1_selected_labels) != len(self.affected_agent_ids):
            raise ValueError("one V1 label is required per affected agent")
        if not np.isfinite(self.best_goal_velocity_gain_mps):
            raise ValueError("goal velocity gain must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "evaluations": [value.to_dict() for value in self.evaluations],
        }


@dataclass(frozen=True)
class ProjectionAwareTransitionV1:
    """Auxiliary-supervision payload that never replaces the PPO action."""

    sampled_action: tuple[float, float]
    sampled_raw_action: tuple[float, float]
    executed_action: tuple[float, float]
    projection_delta_norm: float
    intervention_mask: bool
    execution_target_eligible: bool
    execution_stage: str
    schema_version: str = PROJECTION_TRANSITION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PROJECTION_TRANSITION_SCHEMA:
            raise ValueError("unsupported projection transition schema")
        values = (*self.sampled_action, *self.sampled_raw_action,
                  *self.executed_action, self.projection_delta_norm)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("projection transition values must be finite")
        if self.projection_delta_norm < 0.0:
            raise ValueError("projection delta must be nonnegative")


def _unique_labeled_actions(
    rows: Sequence[tuple[str, np.ndarray]],
) -> tuple[tuple[str, np.ndarray], ...]:
    unique: dict[tuple[float, float], tuple[str, np.ndarray]] = {}
    for label, value in rows:
        action = np.clip(np.asarray(value, dtype=np.float64), -1.0, 1.0)
        key = tuple(float(item) for item in np.round(action, 10))
        unique.setdefault(key, (str(label), action))
    # Preserve registration order.  Selection has an explicit label tie break,
    # so no incidental dictionary or platform ordering affects the result.
    return tuple(unique.values())


def registered_ttc_actions_v1(
    *,
    raw_action: np.ndarray,
    velocity: np.ndarray,
    risk_rows: Sequence[tuple[np.ndarray, np.ndarray]],
    max_acceleration: float,
    dt: float,
) -> tuple[tuple[str, np.ndarray], ...]:
    """Return the exact bounded V1 action vocabulary with stable labels."""

    raw = np.clip(np.asarray(raw_action, dtype=np.float64), -1.0, 1.0)
    brake = np.clip(
        -np.asarray(velocity, dtype=np.float64)
        / max(float(max_acceleration) * float(dt), 1e-12),
        -1.0,
        1.0,
    )
    rows: list[tuple[str, np.ndarray]] = [("raw", raw), ("hold_brake", brake)]
    for risk_index, (relative_position, relative_velocity) in enumerate(risk_rows):
        relative = np.asarray(relative_position, dtype=np.float64)
        norm = float(np.linalg.norm(relative))
        if norm <= 1e-12:
            continue
        away = -relative / norm
        tangent = np.asarray([-away[1], away[0]], dtype=np.float64)
        if float(np.dot(tangent, relative_velocity)) > 0.0:
            tangent = -tangent
        directions = (("away", away), ("left_tangent", tangent),
                      ("right_tangent", -tangent))
        for magnitude in (0.25, 0.50, 0.75, 1.00):
            for direction_name, direction in directions:
                prefix = f"risk{risk_index}_{direction_name}_{magnitude:.2f}"
                rows.append((f"{prefix}_offset", np.clip(
                    raw + magnitude * direction, -1.0, 1.0
                )))
                rows.append((f"{prefix}_blend", np.clip(
                    (1.0 - magnitude) * raw + magnitude * direction,
                    -1.0,
                    1.0,
                )))
    return _unique_labeled_actions(rows)


def registered_ttc_headroom_actions(
    *,
    raw_action: np.ndarray,
    velocity: np.ndarray,
    goal_delta: np.ndarray,
    risk_rows: Sequence[tuple[np.ndarray, np.ndarray]],
    max_acceleration: float,
    dt: float,
    safe_v1_labels: Sequence[str] | None = None,
) -> tuple[tuple[str, np.ndarray], ...]:
    """V1 vocabulary plus the preregistered goal and fixed blends."""

    rows = list(registered_ttc_actions_v1(
        raw_action=raw_action,
        velocity=velocity,
        risk_rows=risk_rows,
        max_acceleration=max_acceleration,
        dt=dt,
    ))
    goal = np.asarray(goal_delta, dtype=np.float64)
    norm = float(np.linalg.norm(goal))
    if norm > 1e-12:
        goal_action = np.clip(goal / norm, -1.0, 1.0)
        rows.append(("goal_aligned", goal_action))
        # The blend is defined against every safe V1 registered action.  It is
        # deterministic and does not inspect performance or simulator truth.
        allowed = (
            None if safe_v1_labels is None
            else {str(value) for value in safe_v1_labels}
        )
        for base_label, base_action in tuple(rows):
            if base_label == "goal_aligned":
                continue
            if allowed is not None and base_label not in allowed:
                continue
            for weight in (0.25, 0.50):
                rows.append((
                    f"{base_label}_goal_blend_{weight:.2f}",
                    np.clip((1.0 - weight) * base_action
                            + weight * goal_action, -1.0, 1.0),
                ))
    return _unique_labeled_actions(rows)


def is_safe_progress_headroom(
    candidate: TTCActionCandidateEvaluationV1,
    baseline: TTCActionCandidateEvaluationV1,
    *,
    minimum_goal_velocity_gain_mps: float = 0.10,
    maximum_extra_actor_deviation: float = 0.25,
) -> bool:
    return bool(
        candidate.safe_three_step
        and candidate.minimum_safety_margin_m
        >= baseline.minimum_safety_margin_m - 1e-9
        and candidate.goal_direction_velocity_mps
        >= baseline.goal_direction_velocity_mps
        + float(minimum_goal_velocity_gain_mps) - 1e-9
        and candidate.actor_deviation_norm
        <= baseline.actor_deviation_norm
        + float(maximum_extra_actor_deviation) + 1e-9
    )


def select_ttc_progress_candidate_v2(
    evaluations: Sequence[TTCActionCandidateEvaluationV1],
) -> TTCActionCandidateEvaluationV1 | None:
    """Apply the preregistered V2 deterministic lexicographic order."""

    feasible = [value for value in evaluations if value.safe_three_step]
    if not feasible:
        return None
    return min(
        feasible,
        key=lambda value: (
            -float(value.goal_direction_velocity_mps),
            -float(value.minimum_safety_margin_m),
            float(value.actor_deviation_norm),
            str(value.label),
            float(value.action[0]),
            float(value.action[1]),
        ),
    )
