"""Single-selection balanced permutation-equivariant UCR-JSF-L V2 executor."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from math import fsum

import numpy as np

from src.safety.ucr_jsf_l.constraints import (
    UnifiedConstraintConfigV1,
    UnifiedConstraintEvaluatorV1,
)
from src.safety.ucr_jsf_l.recovery import RecoveryConfigV1, RecoveryControllerV1

from .candidates import CandidateConfigV2, CandidateGeneratorV2
from .contracts import (
    CandidateEvaluationV2,
    JointCandidateV2,
    JointSafetyInputV2,
    JointSafetyOutputV2,
    JointSafetyTraceV2,
)
from .liveness import LivenessConfigV2, LivenessMemoryV2
from .ordering import (
    build_canonical_ordering,
    canonical_action_signature,
    deterministic_actor_deviation,
    minimum_effective_margin,
)
from .reachability import ReachabilityConfigV2, ReachabilityMemoryV2


class UCRJSFLExecutorV2:
    VERSION = "ucr_jsf_l_v2"

    def __init__(self, config: dict, *, num_agents: int) -> None:
        self.config = dict(config)
        self.constraint_config = UnifiedConstraintConfigV1.from_mapping(
            dict(config.get("constraints", {}))
        )
        self.candidate_config = CandidateConfigV2.from_mapping(
            dict(config.get("candidates", {}))
        )
        self.liveness_config = LivenessConfigV2.from_mapping(
            dict(config.get("liveness", {}))
        )
        self.recovery_config = RecoveryConfigV1.from_mapping(
            dict(config.get("recovery", {}))
        )
        self.constraints = UnifiedConstraintEvaluatorV1(self.constraint_config)
        self.generator = CandidateGeneratorV2(self.candidate_config)
        self.liveness = LivenessMemoryV2(num_agents, self.liveness_config)
        self.recovery = RecoveryControllerV1(self.recovery_config)
        self.reachability = ReachabilityMemoryV2(ReachabilityConfigV2(
            maximum_obstacle_speed_mps=self.constraint_config.maximum_obstacle_speed_mps,
            lost_track_ttl_steps=self.constraint_config.lost_track_ttl_steps,
        ))
        self.last_trace: JointSafetyTraceV2 | None = None

    def reset(self) -> None:
        self.liveness.reset()
        self.recovery.reset()
        self.reachability.reset()
        self.last_trace = None

    @staticmethod
    def _emergency_risk(report) -> float:
        margins = tuple(
            float(value) for value in (
                report.minimum_boundary_margin_m,
                report.minimum_pair_margin_m,
                report.minimum_obstacle_margin_m,
            ) if value is not None
        )
        negative = fsum(sorted(max(-value, 0.0) for value in margins))
        positive = min(margins, default=0.0)
        return float(fsum((
            1_000.0 * len(report.violations),
            100.0 * len(report.unknown_evidence),
            negative,
            -0.01 * positive,
        )))

    @staticmethod
    def _normal_key(item: CandidateEvaluationV2) -> tuple:
        return (
            -float(item.progress_score),
            float(item.actor_deviation),
            -float(item.minimum_effective_margin_m),
            item.canonical_action_signature,
        )

    @staticmethod
    def _emergency_key(item: CandidateEvaluationV2) -> tuple:
        return (
            float(item.emergency_risk_score),
            float(item.actor_deviation),
            -float(item.minimum_effective_margin_m),
            item.canonical_action_signature,
        )

    def execute(
        self,
        state: JointSafetyInputV2,
        requested_actions: np.ndarray,
    ) -> JointSafetyOutputV2:
        requested = np.clip(
            np.asarray(requested_actions, dtype=np.float64), -1.0, 1.0
        )
        if requested.shape != (state.num_agents, 2) or not np.isfinite(requested).all():
            raise ValueError("requested actions must be a finite team matrix")

        _distances, arrived = self.liveness.observe(state)
        reachable = self.reachability.update(state)
        ordering = build_canonical_ordering(
            state,
            requested,
            liveness_memory=self.liveness,
            reachability_memory=self.reachability,
        )
        mode_before = self.recovery.mode
        candidates, budget_exhausted = self.generator.generate(
            state, requested, self.liveness, mode_before, arrived, ordering
        )
        evaluations: list[CandidateEvaluationV2] = []
        admissible: list[CandidateEvaluationV2] = []
        emergency_evaluations: list[CandidateEvaluationV2] = []

        def evaluate_candidate(candidate: JointCandidateV2) -> CandidateEvaluationV2:
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
            actor_deviation = deterministic_actor_deviation(
                candidate.actions, requested, ordering
            )
            score = None
            if report.model_admissible:
                score, actor_deviation = self.liveness.score(
                    state, candidate.actions, requested, arrived, ordering
                )
            return CandidateEvaluationV2(
                candidate_id=candidate.candidate_id,
                actions=tuple(
                    tuple(float(value) for value in row)
                    for row in candidate.actions
                ),
                constraints=report,
                progress_score=score,
                actor_deviation=actor_deviation,
                source=candidate.source,
                canonical_action_signature=canonical_action_signature(
                    candidate.actions, ordering
                ),
                minimum_effective_margin_m=minimum_effective_margin(report),
                emergency_risk_score=self._emergency_risk(report),
            )

        for candidate in candidates:
            item = evaluate_candidate(candidate)
            evaluations.append(item)
            if item.constraints.model_admissible:
                admissible.append(item)

        emergency_selected = False
        if admissible:
            selected = min(admissible, key=self._normal_key)
            executed = np.asarray(selected.actions, dtype=np.float64)
            reason = "canonical_highest_progress_model_admissible"
            exhausted = False
        else:
            remaining = self.candidate_config.max_joint_candidates - len(evaluations)
            emergency = self.generator.emergency_candidates(
                state, requested, self.liveness, ordering
            )[:max(remaining, 0)]
            emergency_evaluations = [evaluate_candidate(row) for row in emergency]
            evaluations.extend(emergency_evaluations)
            if not emergency_evaluations:
                raise RuntimeError("joint candidate budget omitted emergency reserve")
            selected = min(emergency_evaluations, key=self._emergency_key)
            executed = np.asarray(selected.actions, dtype=np.float64)
            reason = "canonical_bounded_emergency_minimum_residual_risk"
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
            bool(value)
            for value in np.any(np.abs(executed - requested) > 1e-7, axis=1)
        )
        fully_admissible_count = sum(
            item.constraints.evidence_status == "MODEL_ADMISSIBLE"
            for item in evaluations
        )
        evidence_insufficient_count = sum(
            item.constraints.evidence_status == "EVIDENCE_INSUFFICIENT"
            for item in evaluations
        )
        rejected_count = sum(
            item.constraints.evidence_status == "MODEL_REJECTED"
            for item in evaluations
        )
        observed_pass_count = sum(
            item.constraints.model_admissible for item in evaluations
        )
        layer_counts = Counter(candidate.layer for candidate in candidates)
        local_sizes = tuple(
            len(self.generator.last_local_candidates[agent_id])
            for agent_id in ordering.flat
        )
        covered_ranks = [set() for _ in ordering.flat]
        for candidate in candidates:
            for rank, local_rank in enumerate(candidate.canonical_ranks):
                covered_ranks[rank].add(local_rank)
        single_coverage = tuple(
            len(covered_ranks[rank]) for rank in range(len(local_sizes))
        )
        pair_counts: Counter[tuple[int, int]] = Counter()
        triple_count = 0
        for candidate in candidates:
            changed = tuple(
                rank for rank, local_rank in enumerate(candidate.canonical_ranks)
                if local_rank != 0
            )
            if len(changed) == 2:
                pair_counts[changed] += 1
            elif len(changed) == state.num_agents:
                triple_count += 1
        emergency_coverage = tuple(sorted(
            int(candidate.candidate_id.split(":")[1][1:])
            for candidate in self.generator.emergency_candidates(
                state, requested, self.liveness, ordering
            )
            if candidate.candidate_id.endswith(":away")
        ))

        trace = JointSafetyTraceV2(
            step=state.step,
            mode_before=mode_before.value,
            mode_after=mode_after.value,
            requested_actions=tuple(tuple(float(v) for v in row) for row in requested),
            executed_actions=tuple(tuple(float(v) for v in row) for row in executed),
            selected_candidate_id=selected.candidate_id,
            selected_source=selected.source,
            selection_reason=reason,
            selected_canonical_action_signature=selected.canonical_action_signature,
            selected_progress_score=selected.progress_score,
            selected_actor_deviation=float(selected.actor_deviation),
            generated_candidate_count=len(candidates) + len(emergency_evaluations),
            evaluated_candidate_count=len(evaluations),
            certified_candidate_count=int(fully_admissible_count),
            candidate_budget_exhausted=bool(budget_exhausted),
            candidate_search_exhausted=exhausted,
            search_exhaustion_is_physical_infeasibility=False,
            unknown_evidence=tuple(report.unknown_evidence),
            violations=tuple(report.violations),
            intervention_mask=intervention,
            passing_side_memory=self.liveness.side_snapshot(ordering),
            stall_counts=tuple(
                int(self.liveness.stall_counts[agent_id]) for agent_id in ordering.flat
            ),
            arrived_mask=tuple(bool(arrived[agent_id]) for agent_id in ordering.flat),
            minimum_boundary_margin_m=report.minimum_boundary_margin_m,
            minimum_pair_margin_m=report.minimum_pair_margin_m,
            minimum_obstacle_margin_m=report.minimum_obstacle_margin_m,
            normal_layer_counts=tuple(sorted(layer_counts.items())),
            single_agent_coverage=single_coverage,
            pair_coverage=tuple(
                (f"{left}-{right}", count)
                for (left, right), count in sorted(pair_counts.items())
            ),
            triple_candidate_count=triple_count,
            emergency_agent_coverage=emergency_coverage,
            selected_evidence_status=report.evidence_status,
            observed_constraint_pass_count=int(observed_pass_count),
            model_admissible_candidate_count=int(fully_admissible_count),
            evidence_insufficient_candidate_count=int(evidence_insufficient_count),
            rejected_candidate_count=int(rejected_count),
            unknown_candidate_count=int(evidence_insufficient_count),
            emergency_action_selected=emergency_selected,
            residual_emergency_risk=(
                float(selected.emergency_risk_score) if emergency_selected else None
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
                "candidate_id_used_for_decision": False,
                "canonical_groups": tuple(len(group) for group in ordering.groups),
                "local_candidate_counts": local_sizes,
                "obstacle_slot_saturated": tuple(state.obstacle_slot_saturated),
            },
        )
        self.last_trace = trace
        return JointSafetyOutputV2(actions=executed.astype(np.float32), trace=trace)
