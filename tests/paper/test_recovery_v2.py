from dataclasses import replace
import numpy as np
import pytest

from tests.paper.test_robust_recovery import environment
from src.safety.staged_execution.contracts import ObservableObstacleV1
from src.safety.staged_execution.uncertainty import fit_scale_v2, scale, ObservablePredictionHistory
from src.safety.staged_execution.robust_recovery_selector import evaluate_candidate


def test_same_state_disagreement_compares_executed_actions_not_labels():
    from src.safety.staged_execution.robust_recovery_selector import selected_actions_differ
    left = (None, None, None, {"first_executed_actions": np.zeros((3,2))})
    right = (None, None, None, {"first_executed_actions": np.zeros((3,2))})
    assert not selected_actions_differ(left, right)
    right[3]["first_executed_actions"][0,0] = .1
    assert selected_actions_differ(left, right)
    assert selected_actions_differ(left, None)
    assert not selected_actions_differ(None, None)


def test_v2_environment_reuses_shadow_for_same_state_probe(monkeypatch):
    from types import SimpleNamespace
    import src.envs.cooperative_uav_env as module
    import src.safety.staged_execution.robust_recovery_selector as selector
    from src.safety.staged_execution.contracts import RobustRecoveryResultV1
    env = environment()
    env.recovery_study = {"arm": "R3", "protocol": "v2", "artifact": {},
                          "progress_tie_tolerance_m": .05, "obstacle_speed_bound_mps": 2.}
    env.recovery_prediction_history = SimpleNamespace(radius_v2=lambda *args, **kwargs: .1, lost_reachable=lambda snapshot: [])
    env._safe_recovery_entry_sources = ("boundary",)
    env._star_recovery_affected_mask[:] = True
    actions = np.ones((3,2), dtype=np.float32)
    candidate = SimpleNamespace(label="normal", actions=actions, targets=env.positions.copy(),
                                affected_agent_mask=(True,)*3, entry_sources=("boundary",))
    monkeypatch.setattr(module, "build_star_recovery_candidates", lambda **kwargs: [candidate])
    monkeypatch.setattr(env, "_v5_recovery_margin", lambda: 0.)
    shadow_calls, geometry_calls = [], []
    def shadow(*args, **kwargs):
        shadow_calls.append(True)
        return {"final_margin": .1, "minimum_margin": .1, "feasible": True,
                "positions": env.positions.copy(), "position_path": [env.positions.copy()]*3,
                "first_executed_actions": actions, "post_projection_norm": 1.}
    def geometry(*args, **kwargs):
        geometry_calls.append(kwargs)
        return RobustRecoveryResultV1(True, 1., 1., 1., 0., "accepted")
    monkeypatch.setattr(env, "_v5_4_observable_envelope", shadow)
    monkeypatch.setattr(selector, "evaluate_candidate", geometry)
    before = env.positions.copy()
    assert env._v5_4_select_recovery_command(actions) is not None
    assert len(shadow_calls) == 1 and len(geometry_calls) == 2
    assert geometry_calls[0]["obstacle_speed_bound"] == 2.
    event = next(e for e in env.safe_recovery_events if e["event"] == "R2_R3_same_state_choice")
    assert event["different"] is False
    np.testing.assert_array_equal(before, env.positions)


def fit_rows():
    return [{"layout_seed": layout, "prediction_horizon": h, "status": "observed",
             "velocity_valid": True, "recent_residual_scale": x,
             "prediction_error": .1*h + 20*x}
            for layout in range(8) for h in (1,2,3) for x in (0., 1e-6, 1e-4, 1e-3)]


def test_normalized_ridge_fit_is_finite_monotone_and_fit_only():
    model = fit_scale_v2(fit_rows())
    assert model["version"] == "normalized_nonnegative_ridge_v2"
    assert model["feature_scale"] >= .001
    assert np.all(np.isfinite(model["c"])) and min(model["c"]) >= 0
    assert max(model["c"]) < 1.
    assert scale(model, 3, .001) >= scale(model, 3, 0.)
    reordered = fit_scale_v2(list(reversed(fit_rows())))
    np.testing.assert_allclose(model["b"], reordered["b"], atol=1e-12)
    np.testing.assert_allclose(model["c"], reordered["c"], atol=1e-12)


def snapshot_with_track(relative, valid=False):
    env = environment()
    snapshot = env._safety_planning_snapshot()
    track = ObservableObstacleV1(0, 9, relative, (0., 0.), 1., valid)
    return replace(snapshot, visible_obstacles=((track,), (), ()))


def test_unknown_far_track_is_not_blanket_rejected_but_near_track_is():
    for relative, expected in (((20., 0.), True), ((2., 0.), False)):
        snapshot = snapshot_with_track(relative)
        result = evaluate_candidate(snapshot, [np.asarray(snapshot.positions)]*3, True,
            lambda key,h: float("inf"), [True]*3, 0., obstacle_speed_bound=np.sqrt(2.))
        assert result.safe is expected
        if not expected:
            assert result.reason == "reachable_set_intersection"


def test_lost_track_is_retained_and_reachable_radius_grows():
    snapshot = snapshot_with_track((10., 0.), True)
    history = ObservablePredictionHistory(snapshot.dt)
    history.observe(snapshot, 122000)
    lost = replace(snapshot, step=1, visible_obstacles=((), (), ()))
    history.observe(lost, 122000)
    ghosts = history.lost_reachable(lost)
    assert len(ghosts) == 1
    assert ghosts[0][-1] == snapshot.dt
    # An old unseen obstacle can reach the candidate; it is never silently expired.
    result = evaluate_candidate(lost, [np.asarray(snapshot.positions)]*3, True,
        lambda key,h: 0., [True]*3, 0., obstacle_speed_bound=2.,
        lost_tracks=[(0, (10., 0.), 1., 10.)])
    assert not result.safe
    history.observe(replace(snapshot, step=2), 122000)
    assert not history.lost_reachable(replace(snapshot, step=2))
    assert history.recent((0,9)) == 0.


def test_out_of_fit_feature_uses_reachable_fallback_instead_of_extrapolation():
    snapshot = snapshot_with_track((20., 0.), True)
    history = ObservablePredictionHistory(snapshot.dt)
    history.observe(snapshot, 122000)
    model = fit_scale_v2(fit_rows())
    artifact = {"adaptive_model": model, "adaptive_q": 1.,
                "fixed_model": {"b": [1.]*3,"c": [0.]*3}, "fixed_q": 1.}
    history.residuals[(0,9)].extend([.1, .2])
    assert np.isinf(history.radius_v2((0,9), 3, artifact, True))
    assert history.radius_events[-1]["mode"] == "ood_reachable"
    result = evaluate_candidate(snapshot, [np.asarray(snapshot.positions)]*3, True,
        lambda key,h: history.radius_v2(key,h,artifact,True), [True]*3, 0., obstacle_speed_bound=2.)
    assert result.safe
    count = len(history.radius_events)
    history.radius_v2((0,9), 1, artifact, False, record=False)
    assert len(history.radius_events) == count
