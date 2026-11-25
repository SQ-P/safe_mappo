"""Unified full-team constraint evaluation for UCR-JSF-L."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import ConstraintReportV1, JointSafetyInputV1
from .dynamics import rollout_constant_action, rollout_with_stop_hold
from .reachability import (
    ReachabilityConfigV1,
    ReachableObstacleV1,
    current_reachability,
)


def _swept_distance(start: np.ndarray, end: np.ndarray) -> float:
    delta = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    denom = float(np.dot(delta, delta))
    alpha = 0.0 if denom <= 1e-15 else float(
        np.clip(-np.dot(start, delta) / denom, 0.0, 1.0)
    )
    return float(np.linalg.norm(start + alpha * delta))


@dataclass(frozen=True)
class UnifiedConstraintConfigV1:
    horizon_steps: int = 3
    boundary_clearance_m: float = 0.0
    boundary_buffer_m: float = 0.25
    teammate_safety_distance_m: float = 6.0
    obstacle_safety_margin_m: float = 1.5
    numerical_tolerance_m: float = 1e-7
    maximum_obstacle_speed_mps: float = 2.0
    lost_track_ttl_steps: int = 2

    @classmethod
    def from_mapping(cls, payload: dict) -> "UnifiedConstraintConfigV1":
        return cls(
            horizon_steps=int(payload.get("prediction_horizon_steps", 3)),
            boundary_clearance_m=float(payload.get("boundary_clearance_m", 0.0)),
            boundary_buffer_m=float(payload.get("boundary_buffer_m", 0.25)),
            teammate_safety_distance_m=float(
                payload.get("teammate_safety_distance_m", 6.0)
            ),
            obstacle_safety_margin_m=float(
                payload.get("obstacle_safety_margin_m", 1.5)
            ),
            numerical_tolerance_m=float(payload.get("numerical_tolerance_m", 1e-7)),
            maximum_obstacle_speed_mps=float(
                payload.get("maximum_obstacle_speed_mps", 2.0)
            ),
            lost_track_ttl_steps=int(payload.get("lost_track_ttl_steps", 2)),
        )


class UnifiedConstraintEvaluatorV1:
    def __init__(self, config: UnifiedConstraintConfigV1) -> None:
        self.config = config
        if (
            config.horizon_steps < 1
            or min(
                config.boundary_clearance_m,
                config.boundary_buffer_m,
                config.obstacle_safety_margin_m,
                config.numerical_tolerance_m,
            ) < 0.0
            or config.teammate_safety_distance_m <= 0.0
            or config.maximum_obstacle_speed_mps <= 0.0
            or config.lost_track_ttl_steps < 0
        ):
            raise ValueError("invalid unified constraint configuration")

    def evaluate(
        self,
        state: JointSafetyInputV1,
        actions: np.ndarray,
        *,
        reachable_obstacles: tuple[tuple[ReachableObstacleV1, ...], ...] | None = None,
        braking_mask: np.ndarray | None = None,
        unresolved_evidence: tuple[str, ...] = (),
    ) -> ConstraintReportV1:
        values = np.asarray(actions, dtype=np.float64)
        positions = np.asarray(state.positions, dtype=np.float64)
        velocities = np.asarray(state.velocities, dtype=np.float64)
        if values.shape != positions.shape or not np.isfinite(values).all():
            raise ValueError("candidate actions must be a finite team matrix")
        if braking_mask is None:
            paths, _ = rollout_constant_action(
                positions, velocities, values,
                horizon_steps=self.config.horizon_steps, dt=state.dt,
                max_acceleration=state.max_acceleration, max_speed=state.max_speed,
            )
        else:
            paths, _ = rollout_with_stop_hold(
                positions, velocities, values,
                braking_mask=np.asarray(braking_mask, dtype=bool),
                horizon_steps=self.config.horizon_steps, dt=state.dt,
                max_acceleration=state.max_acceleration, max_speed=state.max_speed,
            )
        tol = self.config.numerical_tolerance_m
        violations: list[str] = []
        unknown: list[str] = list(unresolved_evidence)
        bounded: list[str] = []
        boundary_margin = float("inf")
        pair_margin = float("inf")
        obstacle_margin = float("inf")
        domain_margin = (
            state.agent_radius + self.config.boundary_clearance_m
            + self.config.boundary_buffer_m
        )
        high = np.asarray(state.map_size, dtype=float) - domain_margin
        for time_index in range(1, paths.shape[0]):
            current = paths[time_index]
            margins = np.concatenate((current - domain_margin, high - current), axis=1)
            boundary_margin = min(boundary_margin, float(np.min(margins)))
            for agent_id in range(state.num_agents):
                if float(np.min(margins[agent_id])) < -tol:
                    violations.append(f"boundary:{agent_id}:t{time_index}")
            for left in range(state.num_agents):
                for right in range(left + 1, state.num_agents):
                    start = paths[time_index - 1, left] - paths[time_index - 1, right]
                    end = current[left] - current[right]
                    margin = _swept_distance(start, end) - self.config.teammate_safety_distance_m
                    pair_margin = min(pair_margin, margin)
                    if margin < -tol:
                        violations.append(f"pair:{left}:{right}:t{time_index}")

        reachable = reachable_obstacles or current_reachability(state)
        for observer_id, tracks in enumerate(reachable):
            for track in tracks:
                key = f"track:{observer_id}:{track.track_id}"
                if track.age_steps > self.config.lost_track_ttl_steps:
                    unknown.append(f"{key}:lost_beyond_ttl")
                    continue
                if not track.velocity_valid:
                    bounded.append(
                        f"{key}:reachable_envelope_age_{track.age_steps}"
                    )
                protected = (
                    state.agent_radius + track.radius
                    + self.config.obstacle_safety_margin_m
                )
                for time_index in range(1, paths.shape[0]):
                    t0 = (time_index - 1) * state.dt
                    t1 = time_index * state.dt
                    center0, radius0 = track.envelope(
                        t0,
                        observation_dt=state.dt,
                        map_size=state.map_size,
                        maximum_speed_mps=self.config.maximum_obstacle_speed_mps,
                    )
                    center1, radius1 = track.envelope(
                        t1,
                        observation_dt=state.dt,
                        map_size=state.map_size,
                        maximum_speed_mps=self.config.maximum_obstacle_speed_mps,
                    )
                    rel0 = paths[time_index - 1, observer_id] - center0
                    rel1 = paths[time_index, observer_id] - center1
                    margin = _swept_distance(rel0, rel1) - protected - max(
                        radius0, radius1
                    )
                    obstacle_margin = min(obstacle_margin, margin)
                    if margin < -tol:
                        kind = "reachable_obstacle" if (
                            not track.velocity_valid or track.age_steps > 0
                        ) else "obstacle"
                        violations.append(
                            f"{kind}:{observer_id}:{track.track_id}:t{time_index}"
                        )

        def finite_or_none(value: float) -> float | None:
            return float(value) if np.isfinite(value) else None

        violations = sorted(set(violations))
        unknown = sorted(set(unknown))
        bounded = sorted(set(bounded))
        for observer_id, saturated in enumerate(state.obstacle_slot_saturated):
            if saturated:
                unknown.append(f"observer:{observer_id}:slot_saturated_uncovered")
        unknown = sorted(set(unknown))
        evidence_status = (
            "MODEL_REJECTED" if violations else
            "EVIDENCE_INSUFFICIENT" if unknown else
            "MODEL_ADMISSIBLE"
        )
        return ConstraintReportV1(
            model_admissible=not violations,
            evidence_status=evidence_status,
            violations=tuple(violations), unknown_evidence=tuple(unknown),
            bounded_uncertainty=tuple(bounded),
            minimum_boundary_margin_m=finite_or_none(boundary_margin),
            minimum_pair_margin_m=finite_or_none(pair_margin),
            minimum_obstacle_margin_m=finite_or_none(obstacle_margin),
        )
