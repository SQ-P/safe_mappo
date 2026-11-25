from __future__ import annotations

import numpy as np

from src.safety.ucr_jsf_l.candidates import CandidateConfigV1, CandidateGeneratorV1
from src.safety.ucr_jsf_l.contracts import SafetyMode
from src.safety.ucr_jsf_l.executor import UCRJSFLExecutorV1
from src.safety.ucr_jsf_l.liveness import LivenessConfigV1, LivenessMemoryV1

from .conftest import make_state, make_track


def executor_config():
    return {
        "constraints": {
            "prediction_horizon_steps": 3,
            "boundary_buffer_m": 0.25,
            "teammate_safety_distance_m": 6.0,
            "obstacle_safety_margin_m": 1.5,
        },
        "candidates": {
            "max_local_candidates_per_agent": 8,
            "max_joint_candidates": 64,
        },
        "liveness": {"stall_patience_steps": 2},
        "recovery": {},
    }


def test_candidate_cap_and_order_are_deterministic():
    state = make_state()
    memory = LivenessMemoryV1(3, LivenessConfigV1(stall_patience_steps=2))
    _, arrived = memory.observe(state)
    generator = CandidateGeneratorV1(CandidateConfigV1())
    requested = np.asarray([[0.8, 0.1], [0.7, -0.1], [0.9, 0.0]])
    first, first_exhausted = generator.generate(
        state, requested, memory, SafetyMode.NORMAL, arrived
    )
    second, second_exhausted = generator.generate(
        state, requested, memory, SafetyMode.NORMAL, arrived
    )
    assert len(first) <= 64
    assert first_exhausted == second_exhausted
    assert [item.source for item in first] == [item.source for item in second]
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left.actions, right.actions)
    assert any("brake" in item.source for item in first)


def test_executor_filters_then_selects_once_with_distinct_trace():
    state = make_state()
    executor = UCRJSFLExecutorV1(executor_config(), num_agents=3)
    output = executor.execute(state, np.full((3, 2), [0.5, 0.0]))
    assert output.trace.schema_version == "JointSafetyTraceV1"
    assert output.trace.metadata["single_final_selection"] is True
    assert output.trace.metadata["legacy_projection_called"] is False
    assert output.trace.certified_candidate_count > 0
    assert output.actions.shape == (3, 2)
    assert output.trace.search_exhaustion_is_physical_infeasibility is False


def test_reachable_unknown_obstacle_uses_bounded_emergency_risk_comparison():
    state = make_state(
        tracks=((make_track(relative=(4.5, 0.0), valid=False),), (), ())
    )
    executor = UCRJSFLExecutorV1(executor_config(), num_agents=3)
    output = executor.execute(state, np.zeros((3, 2)))
    assert output.trace.certified_candidate_count == 0
    assert output.trace.candidate_search_exhausted is True
    assert output.trace.mode_after == "SAFETY_RECOVERY"
    assert output.trace.emergency_action_selected
    assert output.trace.residual_emergency_risk is not None
    assert output.trace.evaluated_candidate_count <= 64
    assert "emergency_away" in output.trace.selected_source
    assert not np.allclose(output.actions, 0.0)


def test_evidence_insufficient_counts_are_disjoint_from_model_admissible():
    state = make_state(slot_saturated=(True, False, False))
    executor = UCRJSFLExecutorV1(executor_config(), num_agents=3)
    output = executor.execute(state, np.zeros((3, 2)))
    trace = output.trace
    assert trace.selected_evidence_status == "EVIDENCE_INSUFFICIENT"
    assert trace.observed_constraint_pass_count > 0
    assert trace.evidence_insufficient_candidate_count > 0
    assert trace.model_admissible_candidate_count == 0
    assert trace.certified_candidate_count == 0
    assert (
        trace.observed_constraint_pass_count
        == trace.evidence_insufficient_candidate_count
    )
    assert trace.rejected_candidate_count == 0


def test_arrival_hold_and_stall_escape_are_observable_in_trace():
    executor = UCRJSFLExecutorV1(executor_config(), num_agents=3)
    arrived = make_state(
        positions=[[20, 20], [20, 50], [20, 80]],
        velocities=[[0.5, 0], [0.5, 0], [0.5, 0]],
        goals=[[20, 20], [20, 50], [20, 80]],
    )
    held = executor.execute(arrived, np.zeros((3, 2)))
    assert held.trace.arrived_mask == (True, True, True)
    assert "arrival_hold" in held.trace.selected_source
    np.testing.assert_allclose(held.actions, np.full((3, 2), [-1.0, 0.0]))

    executor.reset()
    stagnant = make_state()
    executor.execute(stagnant, np.zeros((3, 2)))
    executor.execute(make_state(step=1), np.zeros((3, 2)))
    escaped = executor.execute(make_state(step=2), np.zeros((3, 2)))
    assert escaped.trace.mode_after == "STALL_ESCAPE"
    assert escaped.trace.stall_counts == (2, 2, 2)


def test_stall_escape_uses_committed_tangent_not_opposite_probe():
    state = make_state(tracks=((make_track(relative=(5.0, 1.0)),), (), ()))
    memory = LivenessMemoryV1(3, LivenessConfigV1(stall_patience_steps=2))
    _, arrived = memory.observe(state)
    generator = CandidateGeneratorV1(CandidateConfigV1())
    candidates, _ = generator.generate(
        state, np.zeros((3, 2)), memory, SafetyMode.STALL_ESCAPE, arrived
    )
    selected = next(row for row in candidates if "stall_escape" in row.source)
    labels = selected.source.split("+")
    assert labels[0] == "stall_escape"
    assert memory.side_snapshot() == ()


def test_tangent_candidates_exist_only_for_goal_corridor_blockage():
    memory = LivenessMemoryV1(3, LivenessConfigV1(stall_patience_steps=2))
    generator = CandidateGeneratorV1(CandidateConfigV1())
    blocked = make_state(tracks=((make_track(relative=(8.0, 0.0)),), (), ()))
    clear = make_state(tracks=((make_track(relative=(0.0, 8.0)),), (), ()))
    blocked_arrived = memory.observe(blocked)[1]
    blocked_rows = generator._local(
        blocked, np.zeros((3, 2)), 0, memory, SafetyMode.NORMAL, blocked_arrived
    )
    memory.reset()
    clear_arrived = memory.observe(clear)[1]
    clear_rows = generator._local(
        clear, np.zeros((3, 2)), 0, memory, SafetyMode.NORMAL, clear_arrived
    )
    assert "committed_tangent" in {label for label, _ in blocked_rows}
    assert "committed_tangent" not in {label for label, _ in clear_rows}
