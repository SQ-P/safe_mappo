"""Versioned contracts for balanced permutation-equivariant UCR-JSF-L V2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.safety.ucr_jsf_l.contracts import (
    ConstraintReportV1,
    JointSafetyInputV1,
    ObstacleTrackV1,
    SafetyMode,
)

JointSafetyInputV2 = JointSafetyInputV1
ObstacleTrackV2 = ObstacleTrackV1
ConstraintReportV2 = ConstraintReportV1


@dataclass(frozen=True)
class JointCandidateV2:
    candidate_id: str
    actions: np.ndarray
    source: str
    braking_mask: tuple[bool, ...] = ()
    layer: str = "normal"
    canonical_ranks: tuple[int, ...] = ()


@dataclass(frozen=True)
class CandidateEvaluationV2:
    candidate_id: str
    actions: tuple[tuple[float, float], ...]
    constraints: ConstraintReportV1
    progress_score: float | None
    actor_deviation: float
    source: str
    canonical_action_signature: tuple[tuple[float, float], ...]
    minimum_effective_margin_m: float
    emergency_risk_score: float | None = None


@dataclass(frozen=True)
class JointSafetyTraceV2:
    step: int
    mode_before: str
    mode_after: str
    requested_actions: tuple[tuple[float, float], ...]
    executed_actions: tuple[tuple[float, float], ...]
    selected_candidate_id: str
    selected_source: str
    selection_reason: str
    selected_canonical_action_signature: tuple[tuple[float, float], ...]
    selected_progress_score: float | None
    selected_actor_deviation: float
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
    normal_layer_counts: tuple[tuple[str, int], ...] = ()
    single_agent_coverage: tuple[int, ...] = ()
    pair_coverage: tuple[tuple[str, int], ...] = ()
    triple_candidate_count: int = 0
    emergency_agent_coverage: tuple[int, ...] = ()
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
    schema_version: str = "JointSafetyTraceV2"


@dataclass(frozen=True)
class JointSafetyOutputV2:
    actions: np.ndarray
    trace: JointSafetyTraceV2

    def __post_init__(self) -> None:
        values = np.asarray(self.actions)
        if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
            raise ValueError("executed UCR-JSF-L V2 actions must be finite Nx2")
