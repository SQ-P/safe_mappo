from __future__ import annotations

import numpy as np

from src.safety.ucr_jsf_l_v2.liveness import LivenessConfigV2, LivenessMemoryV2

from .conftest import make_state, make_track


def _side(state):
    memory = LivenessMemoryV2(3, LivenessConfigV2())
    track = state.visible_obstacles[0][0]
    return memory.preferred_passing_side(
        state, 0, track.track_id, np.asarray(track.relative_position), commit=False
    )


def test_dynamic_prediction_precedes_boundary_and_chooses_larger_clearance():
    state = make_state(
        positions=[[50, 50], [20, 20], [80, 80]],
        velocities=np.zeros((3, 2)),
        goals=[[90, 50], [90, 20], [90, 80]],
        tracks=((make_track(0, 1, velocity=(0.0, 1.0)),), (), ()),
    )
    assert _side(state) == -1


def test_boundary_precedes_teammates_and_velocity():
    state = make_state(
        positions=[[50, 2], [50, 10], [50, 20]],
        velocities=[[0, -1], [0, 0], [0, 0]],
        goals=[[90, 2], [90, 10], [90, 20]],
        tracks=((make_track(0, 1, valid=False),), (), ()),
    )
    assert _side(state) == 1


def test_teammate_clearance_precedes_lateral_velocity():
    state = make_state(
        positions=[[50, 50], [50, 53], [20, 20]],
        velocities=[[0, 1], [0, 0], [0, 0]],
        goals=[[90, 50], [90, 53], [90, 20]],
        tracks=((make_track(0, 1, valid=False),), (), ()),
    )
    assert _side(state) == -1


def test_velocity_then_global_positive_side_are_deterministic():
    moving = make_state(
        positions=[[50, 50], [20, 50], [80, 50]],
        velocities=[[0, -1], [0, 0], [0, 0]],
        goals=[[90, 50], [90, 50], [90, 50]],
        tracks=((make_track(0, 1, valid=False),), (), ()),
    )
    symmetric = make_state(
        positions=[[50, 50], [20, 50], [80, 50]],
        velocities=np.zeros((3, 2)),
        goals=[[90, 50], [90, 50], [90, 50]],
        tracks=((make_track(0, 1, valid=False),), (), ()),
    )
    assert _side(moving) == -1
    assert _side(symmetric) == 1


def test_commitment_follows_actual_displacement_and_counts_flip():
    state = make_state(
        positions=[[50, 50], [20, 20], [80, 80]],
        velocities=[[0, -1], [0, 0], [0, 0]],
        goals=[[90, 50], [90, 20], [90, 80]],
        tracks=((make_track(0, 1),), (), ()),
        step=4,
    )
    memory = LivenessMemoryV2(3, LivenessConfigV2())
    memory.passing_sides[(0, 1)] = (1, 3)
    actions = np.zeros((3, 2))
    actions[0, 1] = 1.0
    assert memory.commit_selected(state, "opposite_tangent_probe+actor+actor", actions) == 1
    assert memory.passing_sides[(0, 1)][0] == -1
