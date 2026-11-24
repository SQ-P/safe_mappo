"""Versioned, observable-only safety telemetry contracts.

The types here are intentionally independent from the archived FACS planner.
They may be serialized in rollouts and replayed by shadow safety evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from hashlib import sha256
import json
from typing import Any, Mapping, Sequence

import numpy as np


CAUSAL_SAFETY_SNAPSHOT_SCHEMA = "CausalSafetySnapshotV1"
INTERVENTION_TRACE_SCHEMA = "StageInterventionTraceV1"
SAFETY_STAGES = (
    "actor_request",
    "initial_joint_guard",
    "obstacle_ttc",
    "final_joint_guard",
    "executed",
)


class InterventionLevel(IntEnum):
    NONE = 0
    ANY = 1
    MATERIAL = 2
    LARGE = 3
    DIRECTION_REVERSAL = 4


def _finite_tuple(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


@dataclass(frozen=True)
class ObservableObstacleV1:
    """One causal obstacle track as observed by one UAV."""

    observer_id: int
    track_id: int
    relative_position: tuple[float, float]
    relative_velocity: tuple[float, float]
    radius: float
    velocity_valid: bool

    def __post_init__(self) -> None:
        if self.observer_id < 0 or self.track_id < 0 or self.radius <= 0.0:
            raise ValueError("invalid obstacle identity or radius")
        _finite_tuple((*self.relative_position, *self.relative_velocity, self.radius), name="obstacle")


@dataclass(frozen=True)
class CausalSafetySnapshotV1:
    """Immutable state available to the online safety stack at one step.

    ``team_observation`` and ``safety_state`` must be derived from the current
    observation only. Future simulator state is intentionally absent.
    """

    step: int
    team_observation: tuple[tuple[float, ...], ...]
    requested_actions: tuple[tuple[float, ...], ...]
    safety_state: tuple[float, ...]
    obstacle_tracks: tuple[tuple[ObservableObstacleV1, ...], ...]
    rng_state_digest: str
    recovery_state_digest: str
    schema_version: str = CAUSAL_SAFETY_SNAPSHOT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CAUSAL_SAFETY_SNAPSHOT_SCHEMA:
            raise ValueError("unsupported causal safety snapshot schema")
        if self.step < 0 or not self.team_observation:
            raise ValueError("snapshot requires a non-negative step and at least one UAV")
        if len(self.requested_actions) != len(self.team_observation):
            raise ValueError("requested action count must equal UAV count")
        if len(self.obstacle_tracks) != len(self.team_observation):
            raise ValueError("obstacle track rows must equal UAV count")
        for row in (*self.team_observation, *self.requested_actions):
            _finite_tuple(row, name="snapshot row")
        _finite_tuple(self.safety_state, name="safety_state")
        for digest in (self.rng_state_digest, self.recovery_state_digest):
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
                raise ValueError("state digests must be lowercase SHA256 hex")

    def canonical_payload(self) -> dict[str, Any]:
        def obstacle_payload(item: ObservableObstacleV1) -> dict[str, Any]:
            return {
                "observer_id": item.observer_id,
                "track_id": item.track_id,
                "relative_position": list(item.relative_position),
                "relative_velocity": list(item.relative_velocity),
                "radius": item.radius,
                "velocity_valid": item.velocity_valid,
            }

        return {
            "schema_version": self.schema_version,
            "step": self.step,
            "team_observation": [list(row) for row in self.team_observation],
            "requested_actions": [list(row) for row in self.requested_actions],
            "safety_state": list(self.safety_state),
            "obstacle_tracks": [
                [obstacle_payload(item) for item in row]
                for row in self.obstacle_tracks
            ],
            "rng_state_digest": self.rng_state_digest,
            "recovery_state_digest": self.recovery_state_digest,
        }

    @property
    def digest(self) -> str:
        encoded = json.dumps(
            self.canonical_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class EnvironmentShadowSnapshotV1:
    """Exact reversible simulator snapshot used only for shadow execution tests."""

    environment_state: Mapping[str, Any]
    environment_rng_state: Mapping[str, Any]
    actor_gru_state: Any
    state_fingerprint: str
    schema_version: str = "EnvironmentShadowSnapshotV1"

    @staticmethod
    def fingerprint(payload: Mapping[str, Any]) -> str:
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return sha256(encoded).hexdigest()


def action_modification_level(
    before: Sequence[float],
    after: Sequence[float],
    *,
    acceleration_scale: float,
    dt: float,
    any_action_l2: float = 1e-7,
    material_delta_v_mps: float = 0.05,
    large_delta_v_mps: float = 0.25,
) -> tuple[InterventionLevel, float, float]:
    """Classify a stage modification using physical velocity thresholds."""

    before_array = np.asarray(before, dtype=np.float64)
    after_array = np.asarray(after, dtype=np.float64)
    if before_array.shape != after_array.shape or before_array.ndim != 1:
        raise ValueError("before/after actions must be equal one-dimensional shapes")
    if acceleration_scale <= 0.0 or dt <= 0.0:
        raise ValueError("acceleration_scale and dt must be positive")
    action_l2 = float(np.linalg.norm(after_array - before_array))
    delta_v = action_l2 * float(acceleration_scale) * float(dt)
    denominator = float(np.linalg.norm(before_array) * np.linalg.norm(after_array))
    cosine = 1.0 if denominator <= 1e-12 else float(
        np.dot(before_array, after_array) / denominator
    )
    if cosine < 0.0:
        level = InterventionLevel.DIRECTION_REVERSAL
    elif delta_v >= large_delta_v_mps:
        level = InterventionLevel.LARGE
    elif delta_v >= material_delta_v_mps:
        level = InterventionLevel.MATERIAL
    elif action_l2 > any_action_l2:
        level = InterventionLevel.ANY
    else:
        level = InterventionLevel.NONE
    return level, delta_v, cosine


@dataclass(frozen=True)
class StageInterventionRecordV1:
    uav_id: int
    stage: str
    input_action: tuple[float, ...]
    output_action: tuple[float, ...]
    direction_cosine: float
    reason: str
    safety_margin: float | None
    eligible_for_actor_risk: bool
    modification_level: InterventionLevel
    delta_v_mps: float

    def __post_init__(self) -> None:
        if self.uav_id < 0 or self.stage not in SAFETY_STAGES:
            raise ValueError("invalid UAV id or safety stage")
        if self.stage in {"actor_request", "executed"} and self.input_action != self.output_action:
            raise ValueError("boundary trace stages may not modify actions")
        _finite_tuple((*self.input_action, *self.output_action), name="trace actions")
        if not np.isfinite(self.direction_cosine) or not -1.000001 <= self.direction_cosine <= 1.000001:
            raise ValueError("direction cosine must be finite and in [-1, 1]")
        if self.safety_margin is not None and not np.isfinite(self.safety_margin):
            raise ValueError("safety margin must be finite when present")
        if self.delta_v_mps < 0.0 or not np.isfinite(self.delta_v_mps):
            raise ValueError("delta_v_mps must be finite and non-negative")


@dataclass(frozen=True)
class StageInterventionTraceV1:
    step: int
    snapshot_digest: str
    records: tuple[StageInterventionRecordV1, ...]
    recovery_active: bool = False
    safe_brake_active: bool = False
    terminal_transition: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = INTERVENTION_TRACE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != INTERVENTION_TRACE_SCHEMA or self.step < 0:
            raise ValueError("invalid intervention trace schema or step")
        if len(self.snapshot_digest) != 64:
            raise ValueError("snapshot_digest must be SHA256 hex")
        keys = [(record.uav_id, record.stage) for record in self.records]
        if len(keys) != len(set(keys)):
            raise ValueError("trace contains duplicate UAV-stage records")
        if self.recovery_active or self.safe_brake_active or self.terminal_transition:
            if any(record.eligible_for_actor_risk for record in self.records):
                raise ValueError("recovery, safe-brake, and terminal traces are not actor-risk labels")

    def records_for_stage(self, stage: str) -> tuple[StageInterventionRecordV1, ...]:
        if stage not in SAFETY_STAGES:
            raise ValueError("unknown safety stage")
        return tuple(record for record in self.records if record.stage == stage)
