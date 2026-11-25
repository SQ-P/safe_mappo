from __future__ import annotations

import numpy as np

from src.safety.ucr_jsf_l.liveness import LivenessConfigV1, LivenessMemoryV1

from .conftest import make_state, make_track


def test_passing_side_persists_for_same_observer_track():
    track = make_track(relative=(5.0, 1.0))
    memory = LivenessMemoryV1(3, LivenessConfigV1(stall_patience_steps=2))
    first = make_state(tracks=((track,), (), ()), step=1)
    memory.observe(first)
    side = memory.passing_side(first, 0, 1, np.asarray(track.relative_position))
    reversed_geometry = make_track(relative=(5.0, -1.0))
    second = make_state(tracks=((reversed_geometry,), (), ()), step=2)
    memory.observe(second)
    assert memory.passing_side(
        second, 0, 1, np.asarray(reversed_geometry.relative_position)
    ) == side


def test_stall_and_arrival_state_are_separate():
    memory = LivenessMemoryV1(
        3, LivenessConfigV1(stall_patience_steps=2, minimum_progress_m=0.02)
    )
    state = make_state(goals=[[20, 20], [90, 50], [90, 80]])
    _, arrived = memory.observe(state)
    memory.observe(make_state(goals=[[20, 20], [90, 50], [90, 80]], step=1))
    _, arrived = memory.observe(
        make_state(goals=[[20, 20], [90, 50], [90, 80]], step=2)
    )
    assert arrived.tolist() == [True, False, False]
    assert memory.stalled.tolist() == [False, True, True]


def test_goal_exit_penalty_prefers_stop_hold_near_region_boundary():
    memory = LivenessMemoryV1(3, LivenessConfigV1(stall_patience_steps=2))
    state = make_state(
        positions=[[22.9, 20], [20, 50], [20, 80]],
        velocities=[[0.4, 0], [0, 0], [0, 0]],
        goals=[[20, 20], [90, 50], [90, 80]],
    )
    _, arrived = memory.observe(state)
    moving_score, _ = memory.score(
        state, np.zeros((3, 2)), np.zeros((3, 2)), arrived
    )
    brake = np.zeros((3, 2))
    brake[0, 0] = -0.8
    held_score, _ = memory.score(state, brake, np.zeros((3, 2)), arrived)
    assert held_score > moving_score
    assert memory.predicted_goal_exit_count(state, brake, arrived) == 0


def test_passing_side_commits_only_after_selected_behavior():
    track = make_track(relative=(5.0, 1.0))
    state = make_state(tracks=((track,), (), ()), step=3)
    memory = LivenessMemoryV1(3, LivenessConfigV1(stall_patience_steps=2))
    side = memory.preferred_passing_side(
        state, 0, track.track_id, np.asarray(track.relative_position), commit=False
    )
    assert memory.side_snapshot() == ()
    radial = np.asarray(track.relative_position, dtype=float)
    tangent = side * np.asarray([-radial[1], radial[0]]) / np.linalg.norm(radial)
    actions = np.zeros((3, 2))
    actions[0] = tangent
    flips = memory.commit_selected(
        state, "committed_tangent+actor+actor", actions
    )
    assert flips == 0
    assert memory.side_snapshot() == (("0:1", side),)


def test_blocked_score_prefers_committed_tangential_escape_action():
    track = make_track(relative=(8.0, 0.0))
    state = make_state(tracks=((track,), (), ()), step=4)
    memory = LivenessMemoryV1(3, LivenessConfigV1(stall_patience_steps=2))
    _, arrived = memory.observe(state)
    memory.passing_side(
        state, 0, track.track_id, np.asarray(track.relative_position)
    )
    committed = np.zeros((3, 2))
    committed[0, 1] = 0.8
    opposite = committed.copy()
    opposite[0, 1] = -0.8
    committed_score, _ = memory.score(
        state, committed, np.zeros((3, 2)), arrived
    )
    opposite_score, _ = memory.score(
        state, opposite, np.zeros((3, 2)), arrived
    )
    assert committed_score > opposite_score


def test_only_goal_corridor_intersection_is_treated_as_blockage():
    blocked = make_state(
        tracks=((make_track(relative=(8.0, 0.0)),), (), ()), step=4
    )
    clear = make_state(
        tracks=((make_track(relative=(0.0, 8.0)),), (), ()), step=4
    )
    memory = LivenessMemoryV1(
        3,
        LivenessConfigV1(
            stall_patience_steps=2,
            blockage_distance_m=12.0,
            blockage_safety_margin_m=1.5,
        ),
    )
    assert memory.blocking_track(blocked, 0).track_id == 1
    assert memory.blocking_track(clear, 0) is None
    _, arrived = memory.observe(clear)
    left = np.zeros((3, 2))
    left[0, 1] = 0.8
    right = -left
    left_score, _ = memory.score(clear, left, np.zeros((3, 2)), arrived)
    right_score, _ = memory.score(clear, right, np.zeros((3, 2)), arrived)
    assert left_score == right_score


def test_selected_opposite_probe_updates_committed_side_and_counts_flip():
    track = make_track(relative=(8.0, 0.0))
    state = make_state(tracks=((track,), (), ()), step=5)
    memory = LivenessMemoryV1(3, LivenessConfigV1(stall_patience_steps=2))
    memory.passing_sides[(0, track.track_id)] = (1, 4)
    actions = np.zeros((3, 2))
    actions[0, 1] = -0.8
    flips = memory.commit_selected(
        state, "opposite_tangent_probe+actor+actor", actions
    )
    assert flips == 1
    assert memory.side_snapshot() == (("0:1", -1),)


def test_side_commit_uses_actual_displacement_not_acceleration_direction():
    track = make_track(relative=(8.0, 0.0))
    state = make_state(
        velocities=[[0.0, -1.0], [0.0, 0.0], [0.0, 0.0]],
        tracks=((track,), (), ()),
        step=6,
    )
    memory = LivenessMemoryV1(3, LivenessConfigV1(stall_patience_steps=2))
    actions = np.zeros((3, 2))
    actions[0, 1] = 1.0
    memory.commit_selected(
        state, "opposite_tangent_probe+actor+actor", actions
    )
    # The positive-y acceleration only slows an existing negative-y velocity;
    # the next displacement remains negative-y and must commit that side.
    assert memory.side_snapshot() == (("0:1", -1),)
