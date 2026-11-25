"""One-pass UCR-JSF-L V1 joint action selection."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .candidates import CandidateConfigV1, CandidateGeneratorV1, JointCandidateV1
from .constraints import UnifiedConstraintConfigV1, UnifiedConstraintEvaluatorV1
from .contracts import (
    CandidateEvaluationV1,
    JointSafetyInputV1,
    JointSafetyOutputV1,
    JointSafetyTraceV1,
)
from .liveness import LivenessConfigV1, LivenessMemoryV1
from .recovery import RecoveryConfigV1, RecoveryControllerV1
from .reachability import ReachabilityConfigV1, ReachabilityMemoryV1


class UCRJSFLExecutorV1:
    """Filter every candidate with one constraint contract, then select once."""

    VERSION = "ucr_jsf_l_v1"

    def __init__(self, config: dict, *, num_agents: int) -> None:
        self.config = dict(config)
        self.constraint_config = UnifiedConstraintConfigV1.from_mapping(
            dict(config.get("constraints", {}))
        )
        self.candidate_config = CandidateConfigV1.from_mapping(
            dict(config.get("candidates", {}))
        )
        self.liveness_config = LivenessConfigV1.from_mapping(
            dict(config.get("liveness", {}))
        )
        self.recovery_config = RecoveryConfigV1.from_mapping(
            dict(config.get("recovery", {}))
        )
        self.constraints = UnifiedConstraintEvaluatorV1(self.constraint_config)
        self.generator = CandidateGeneratorV1(self.candidate_config)
        self.liveness = LivenessMemoryV1(num_agents, self.liveness_config)
        self.recovery = RecoveryControllerV1(self.recovery_config)
        self.reachability = ReachabilityMemoryV1(ReachabilityConfigV1(
            maximum_obstacle_speed_mps=(
                self.constraint_config.maximum_obstacle_speed_mps
            ),
            lost_track_ttl_steps=self.constraint_config.lost_track_ttl_steps,
        ))
        self.last_trace: JointSafetyTraceV1 | None = None

    def reset(self) -> None:
        self.liveness.reset()
        self.recovery.reset()
        self.reachability.reset()
        self.last_trace = None

    @staticmethod
    def _emergency_risk(report) -> float:
        margins = (
            report.minimum_boundary_margin_m,
            report.minimum_pair_margin_m,
            report.minimum_obstacle_margin_m,
        )
        negative = sum(max(-float(value), 0.0) for value in margins if value is not None)
        positive = min(
            (float(value) for value in margins if value is not None),
            default=0.0,
        )
        return float(
            1_000.0 * len(report.violations)
            + 100.0 * len(report.unknown_evidence)
            + negative
            - 0.01 * positive
        )

    def execute(
        self, state: JointSafetyInputV1, requested_actions: np.ndarray
    ) -> JointSafetyOutputV1:
        requested = np.clip(np.asarray(requested_actions, dtype=np.float64), -1.0, 1.0)
        if requested.shape != (state.num_agents, 2) or not np.isfinite(requested).all():
            raise ValueError("requested actions must be a finite team matrix")
        _distances, arrived = self.liveness.observe(state)
        mode_before = self.recovery.mode
        candidates, budget_exhausted = self.generator.generate(
            state, requested, self.liveness, mode_before, arrived
        )
        evaluations: list[CandidateEvaluationV1] = []
        admissible: list[CandidateEvaluationV1] = []
        emergency_evaluations: list[CandidateEvaluationV1] = []
        reachable = self.reachability.update(state)

        def evaluate_candidate(candidate: JointCandidateV1) -> CandidateEvaluationV1:
            report = self.constraints.evaluate(
                state,
                candidate.actions,
                reachable_obstacles=reachable,
                unresolved_evidence=self.reachability.unresolved_evidence,
                braking_mask=(
                    np.asarray(candidate.braking_mask, dtype=bool)
                    if candidate.braking_mask else None
                ),
            )
            score = None
            actor_deviation = float(np.linalg.norm(candidate.actions - requested))
            if report.model_admissible:
                score, actor_deviation = self.liveness.score(
                    state, candidate.actions, requested, arrived
                )
            return CandidateEvaluationV1(
                candidate_id=candidate.candidate_id,
                actions=tuple(tuple(float(v) for v in row) for row in candidate.actions),
                constraints=report,
                progress_score=score,
                actor_deviation=actor_deviation,
                source=candidate.source,
                emergency_risk_score=self._emergency_risk(report),
            )

        for candidate in candidates:
            item = evaluate_candidate(candidate)
            evaluations.append(item)
            if item.constraints.model_admissible:
                admissible.append(item)

        emergency_selected = False
        if admissible:
            selected = min(
                admissible,
                key=lambda item: (
                    -float(item.progress_score), item.actor_deviation,
                    item.candidate_id,
                ),
            )
            executed = np.asarray(selected.actions, dtype=np.float64)
            reason = "highest_progress_among_model_admissible_candidates"
            exhausted = False
        else:
            # A finite normal-search failure enters a bounded physical-risk
            # comparison.  Braking remains one option, not a fixed outcome.
            remaining = self.candidate_config.max_joint_candidates - len(evaluations)
            emergency = self.generator.emergency_candidates(
                state, requested, self.liveness
            )[:max(remaining, 0)]
            emergency_evaluations = [evaluate_candidate(row) for row in emergency]
            evaluations.extend(emergency_evaluations)
            if not emergency_evaluations:
                raise RuntimeError("joint candidate budget omitted emergency reserve")
            selected = min(
                emergency_evaluations,
                key=lambda item: (
                    float(item.emergency_risk_score),
                    item.actor_deviation,
                    item.candidate_id,
                ),
            )
            executed = np.asarray(selected.actions, dtype=np.float64)
            reason = "bounded_emergency_minimum_residual_risk"
            exhausted = True
            emergency_selected = True

        selected_margins = (
            selected.constraints.minimum_boundary_margin_m,
            selected.constraints.minimum_pair_margin_m,
            selected.constraints.minimum_obstacle_margin_m,
        )
        clearance_ok = all(
            value is None or float(value) >= self.recovery_config.release_margin_m
            for value in selected_margins
        )
        mode_after = self.recovery.update(
            any_stalled=bool(np.any(self.liveness.stalled)),
            certified_candidate=bool(admissible),
            clearance_ok=clearance_ok,
        )
        side_flips = self.liveness.commit_selected(
            state, selected.source, executed
        )
        report = selected.constraints
        intervention = tuple(
            bool(value) for value in np.any(np.abs(executed - requested) > 1e-7, axis=1)
        )
        observed_constraint_pass_count = sum(
            int(item.constraints.model_admissible) for item in evaluations
        )
        fully_admissible_count = sum(
            int(item.constraints.evidence_status == "MODEL_ADMISSIBLE")
            for item in evaluations
        )
        evidence_insufficient_count = sum(
            int(item.constraints.evidence_status == "EVIDENCE_INSUFFICIENT")
            for item in evaluations
        )
        rejected_count = sum(
            int(item.constraints.evidence_status == "MODEL_REJECTED")
            for item in evaluations
        )
        trace = JointSafetyTraceV1(
            step=state.step,
            mode_before=mode_before.value,
            mode_after=mode_after.value,
            requested_actions=tuple(tuple(float(v) for v in row) for row in requested),
            executed_actions=tuple(tuple(float(v) for v in row) for row in executed),
            selected_candidate_id=selected.candidate_id,
            selected_source=selected.source,
            selection_reason=reason,
            generated_candidate_count=len(candidates) + len(emergency_evaluations),
            evaluated_candidate_count=len(evaluations),
            certified_candidate_count=fully_admissible_count,
            candidate_budget_exhausted=bool(budget_exhausted),
            candidate_search_exhausted=exhausted,
            search_exhaustion_is_physical_infeasibility=False,
            unknown_evidence=tuple(report.unknown_evidence),
            violations=tuple(report.violations),
            intervention_mask=intervention,
            passing_side_memory=self.liveness.side_snapshot(),
            stall_counts=tuple(int(value) for value in self.liveness.stall_counts),
            arrived_mask=tuple(bool(value) for value in arrived),
            minimum_boundary_margin_m=report.minimum_boundary_margin_m,
            minimum_pair_margin_m=report.minimum_pair_margin_m,
            minimum_obstacle_margin_m=report.minimum_obstacle_margin_m,
            selected_evidence_status=report.evidence_status,
            observed_constraint_pass_count=observed_constraint_pass_count,
            model_admissible_candidate_count=fully_admissible_count,
            evidence_insufficient_candidate_count=evidence_insufficient_count,
            rejected_candidate_count=rejected_count,
            unknown_candidate_count=evidence_insufficient_count,
            emergency_action_selected=emergency_selected,
            residual_emergency_risk=(
                float(selected.emergency_risk_score)
                if emergency_selected else None
            ),
            passing_side_flip_count_step=int(side_flips),
            stall_agent_count=int(np.sum(self.liveness.stalled)),
            predicted_goal_exit_count=self.liveness.predicted_goal_exit_count(
                state, executed, arrived
            ),
            liveness_recovery_success_step=bool(
                mode_before.value == "STALL_ESCAPE"
                and mode_after.value == "NORMAL"
            ),
            metadata={
                "safety_execution_version": self.VERSION,
                "constraint_config": asdict(self.constraint_config),
                "candidate_config": asdict(self.candidate_config),
                "single_final_selection": True,
                "legacy_projection_called": False,
                "terminology": "model_admissible_not_formal_certificate",
                "obstacle_slot_saturated": tuple(state.obstacle_slot_saturated),
            },
        )
        self.last_trace = trace
        return JointSafetyOutputV1(actions=executed.astype(np.float32), trace=trace)
