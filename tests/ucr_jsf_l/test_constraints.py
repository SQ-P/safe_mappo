from __future__ import annotations

import numpy as np

from src.safety.ucr_jsf_l.constraints import (
    UnifiedConstraintConfigV1,
    UnifiedConstraintEvaluatorV1,
)
from src.safety.ucr_jsf_l.dynamics import velocity_first_step, rollout_with_stop_hold
from src.safety.ucr_jsf_l.reachability import (
    ReachabilityConfigV1,
    ReachabilityMemoryV1,
    current_reachability,
)

from .conftest import make_state, make_track


def evaluator(**overrides):
    payload = {
        "prediction_horizon_steps": 3,
        "boundary_buffer_m": 0.25,
        "teammate_safety_distance_m": 6.0,
        "obstacle_safety_margin_m": 1.5,
    } | overrides
    return UnifiedConstraintEvaluatorV1(
        UnifiedConstraintConfigV1.from_mapping(payload)
    )


def test_velocity_first_dynamics_matches_componentwise_saturation():
    positions = np.asarray([[10.0, 20.0], [30.0, 40.0]])
    velocities = np.asarray([[1.9, -1.9], [-0.5, 0.5]])
    actions = np.asarray([[1.0, -1.0], [0.5, -0.5]])
    next_positions, next_velocities = velocity_first_step(
        positions, velocities, actions, dt=0.5,
        max_acceleration=1.0, max_speed=2.0,
    )
    np.testing.assert_allclose(next_velocities, [[2.0, -2.0], [-0.25, 0.25]])
    np.testing.assert_allclose(next_positions, positions + next_velocities * 0.5)


def test_backup_braking_stops_then_holds_without_reverse_acceleration():
    paths, velocities = rollout_with_stop_hold(
        np.asarray([[10.0, 10.0]]),
        np.asarray([[0.5, 0.0]]),
        np.asarray([[-1.0, 0.0]]),
        braking_mask=np.asarray([True]),
        horizon_steps=3,
        dt=0.5,
        max_acceleration=1.0,
        max_speed=2.0,
    )
    np.testing.assert_allclose(velocities[1:, 0], 0.0)
    np.testing.assert_allclose(paths[:, 0, 0], [10.0, 10.0, 10.0, 10.0])


def test_boundary_and_swept_pair_constraints_are_joint():
    boundary = make_state(
        positions=[[97.9, 20], [20, 50], [20, 80]],
        velocities=[[2, 0], [0, 0], [0, 0]],
    )
    assert any(item.startswith("boundary:0") for item in evaluator().evaluate(
        boundary, np.zeros((3, 2))
    ).violations)

    crossing = make_state(
        positions=[[40, 50], [60, 50], [20, 80]],
        velocities=[[2, 0], [-2, 0], [0, 0]],
    )
    report = evaluator(teammate_safety_distance_m=15.0).evaluate(
        crossing, np.zeros((3, 2))
    )
    assert any(item.startswith("pair:0:1") for item in report.violations)


def test_unknown_velocity_uses_reachable_envelope_without_global_brake():
    invalid = make_track(relative=(20.0, 0.0), valid=False)
    state = make_state(tracks=((invalid,), (), ()))
    report = evaluator().evaluate(state, np.zeros((3, 2)))
    assert report.model_admissible
    assert report.evidence_status == "MODEL_ADMISSIBLE"
    assert report.unknown_evidence == ()
    assert report.bounded_uncertainty == (
        "track:0:1:reachable_envelope_age_0",
    )

    close = make_state(tracks=((make_track(relative=(6.0, 0.0), valid=False),), (), ()))
    close_report = evaluator().evaluate(close, np.zeros((3, 2)))
    assert not close_report.model_admissible
    assert any(item.startswith("reachable_obstacle:0:1") for item in close_report.violations)

    closing = make_track(relative=(6.0, 0.0), velocity=(-2.0, 0.0), valid=True)
    state = make_state(tracks=((closing,), (), ()))
    report = evaluator().evaluate(state, np.zeros((3, 2)))
    assert any(item.startswith("obstacle:0:1") for item in report.violations)


def test_reflection_and_short_lost_track_are_bounded_from_observations():
    state = make_state(
        tracks=((make_track(relative=(77.0, 0.0), velocity=(2.0, 0.0)),), (), ())
    )
    reachable = current_reachability(state)[0][0]
    _center, uncertainty = reachable.envelope(
        1.0, observation_dt=state.dt, map_size=state.map_size,
        maximum_speed_mps=2.0,
    )
    assert uncertainty == 2.0

    memory = ReachabilityMemoryV1(
        ReachabilityConfigV1(maximum_obstacle_speed_mps=2.0, lost_track_ttl_steps=2)
    )
    memory.update(make_state(tracks=((make_track(relative=(20.0, 0.0)),), (), ())))
    lost_state = make_state(step=1)
    lost = memory.update(lost_state)
    assert lost[0][0].age_steps == 1
    report = evaluator().evaluate(
        lost_state, np.zeros((3, 2)), reachable_obstacles=lost
    )
    assert report.model_admissible
    assert report.evidence_status == "MODEL_ADMISSIBLE"
    assert report.bounded_uncertainty == (
        "track:0:1:reachable_envelope_age_1",
    )


def test_unresolved_lost_track_and_slot_saturation_are_unknown_not_global_brake():
    memory = ReachabilityMemoryV1(
        ReachabilityConfigV1(maximum_obstacle_speed_mps=2.0, lost_track_ttl_steps=1)
    )
    memory.update(make_state(tracks=((make_track(relative=(20.0, 0.0)),), (), ())))
    lost_state = make_state(step=2, slot_saturated=(False, True, False))
    reachable = memory.update(lost_state)
    report = evaluator(lost_track_ttl_steps=1).evaluate(
        lost_state,
        np.zeros((3, 2)),
        reachable_obstacles=reachable,
        unresolved_evidence=memory.unresolved_evidence,
    )
    assert report.model_admissible
    assert report.evidence_status == "EVIDENCE_INSUFFICIENT"
    assert report.unknown_evidence == (
        "observer:1:slot_saturated_uncovered",
        "track:0:1:lost_beyond_ttl",
    )


def test_physical_violation_has_model_rejected_status():
    state = make_state(
        tracks=((make_track(relative=(4.0, 0.0), valid=True),), (), ())
    )
    report = evaluator().evaluate(state, np.zeros((3, 2)))
    assert not report.model_admissible
    assert report.evidence_status == "MODEL_REJECTED"
