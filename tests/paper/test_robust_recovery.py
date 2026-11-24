from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from src.envs.cooperative_uav_env import CooperativeUAVEnv
from src.safety.staged_execution.contracts import RobustRecoveryResultV1, ObservableObstacleV1
from src.safety.staged_execution.robust_recovery_selector import (
    relative_prediction, evaluate_candidate, select_progress_candidate,
)
from src.safety.staged_execution.uncertainty import ObservablePredictionHistory, calibrate
from src.utils.io_utils import load_yaml


def environment():
    config = load_yaml("configs/active/standard_mappo_3u_8s2d.yaml")
    safety = deepcopy(config["safety_execution"])
    safety["safety_state_machine_version"] = "star_persistent_recovery_v5_4"
    safety["goal_boundary_corridor"]["enabled"] = True
    env = CooperativeUAVEnv({**config["env"], "reward": config["reward"],
        "oracle_subgoal": config["oracle_subgoal"], "subgoal_selector": config["subgoal_selector"],
        "safety_execution": safety})
    env.reset(seed=23)
    return env


def test_candidate_conditioned_relative_motion():
    np.testing.assert_allclose(relative_prediction([10., 0.], [2., 0.], [1., 0.], .5), [10., 0.])
    np.testing.assert_allclose(relative_prediction([10., 0.], [2., 0.], [3., 0.], .5), [8., 0.])


@pytest.mark.parametrize("arm", ["R1", "R2", "R3"])
def test_r1_preserves_legacy_admission_and_selected_telemetry(monkeypatch, arm):
    from types import SimpleNamespace
    import src.envs.cooperative_uav_env as module
    import src.safety.staged_execution.robust_recovery_selector as selector
    env = environment()
    env.recovery_study = {"arm": arm, "progress_tie_tolerance_m": .05, "artifact": {}}
    env._safe_recovery_entry_sources = ("boundary",)
    env._star_recovery_affected_mask[:] = True
    actions = np.ones((3, 2), dtype=np.float32)
    candidate = SimpleNamespace(label="normal", actions=actions,
        targets=env.positions.copy(), affected_agent_mask=(True,)*3,
        entry_sources=("boundary",))
    monkeypatch.setattr(module, "build_star_recovery_candidates", lambda **kwargs: [candidate])
    calls = []
    def evaluate(*args):
        calls.append(True)
        return RobustRecoveryResultV1(True, 1., 1., 1., 0., "accepted")
    monkeypatch.setattr(selector, "evaluate_candidate", evaluate)
    monkeypatch.setattr(env, "_v5_recovery_margin", lambda: 0.)
    envelope = {"final_margin": .1, "minimum_margin": .1, "feasible": True,
        "rejection_reason": "ok", "positions": env.positions.copy(),
        "position_path": [env.positions.copy()]*3, "first_executed_actions": actions,
        "post_projection_norm": 0.}
    monkeypatch.setattr(env, "_v5_4_observable_envelope", lambda *args, **kwargs: envelope)
    assert env._v5_4_select_recovery_command(actions) is None
    envelope["post_projection_norm"] = 1.
    assert env._v5_4_select_recovery_command(actions) is not None
    assert env.recovery_candidate_post_projection_norm == 1.
    assert env.recovery_candidate_predicted_margin_gain == pytest.approx(.1)
    assert len(calls) == (0 if arm == "R1" else 2)


def test_safety_dominates_progress_and_ties_are_repeatable():
    def row(safe, margin, progress, label):
        return (RobustRecoveryResultV1(safe, margin, progress, progress, 0., "test"), label)
    rows = [row(False, -1., 100., 0), row(True, 1., 1., 1), row(True, 2., .97, 2)]
    assert select_progress_candidate(rows, .05) == rows[2]
    assert select_progress_candidate(list(reversed(rows)), .05) == rows[2]
    assert select_progress_candidate(rows, 0.) == rows[1]


def test_radius_monotonicity_and_all_tracks_checked():
    env = environment()
    snapshot = env._safety_planning_snapshot()
    visible = tuple((ObservableObstacleV1(i, i, (10., 0.), (0., 0.), 1., True),)
                    for i in range(env.num_agents))
    snapshot = replace(snapshot, visible_obstacles=visible)
    path = [np.asarray(snapshot.positions)] * 3
    nominal = evaluate_candidate(snapshot, path, True, lambda key,h: 0., [True]*3, 0.)
    inflated = evaluate_candidate(snapshot, path, True, lambda key,h: 100., [True]*3, 0.)
    assert nominal.safe
    assert not inflated.safe
    assert inflated.robust_margin_m < nominal.robust_margin_m


def test_missing_velocity_rejects_candidate():
    env = environment()
    snapshot = env._safety_planning_snapshot()
    track = ObservableObstacleV1(0, 1, (10., 0.), (0., 0.), 1., False)
    snapshot = replace(snapshot, visible_obstacles=((track,), (), ()))
    result = evaluate_candidate(snapshot, [np.asarray(snapshot.positions)]*3, True,
                                lambda key,h: 0., [True]*3, 0.)
    assert not result.safe
    assert result.reason == "missing_velocity_or_history"


def test_group_quantile_and_missing_tracks_fail_closed():
    rows = [{"layout_seed": seed, "status": "observed", "velocity_valid": True,
             "prediction_horizon": 1, "recent_residual_scale": 0., "prediction_error": float(seed)}
            for seed in range(1, 21)]
    model = {"b": [1.]*3, "c": [0.]*3}
    assert calibrate(rows, model, .95, adaptive=True) == 20.
    rows += [{**r, "prediction_error": .1} for r in rows]
    assert calibrate(rows, model, .95, adaptive=True) == 20.
    rows[0]["status"] = "track_lost"
    with pytest.raises(ValueError, match="finite coverage"):
        calibrate(rows, model, .95, adaptive=True)


def test_conditional_calibration_is_explicit_and_reports_loss():
    from src.safety.staged_execution.uncertainty import calibration_data_summary
    rows = [{"layout_seed": seed, "status": "observed", "velocity_valid": True,
             "prediction_horizon": 1, "recent_residual_scale": 0., "prediction_error": float(seed)}
            for seed in range(1, 41)]
    rows += [{**row, "status": "track_lost", "prediction_error": None} for row in rows[:37]]
    model = {"b": [1.]*3, "c": [0.]*3}
    with pytest.raises(ValueError, match="finite_groups=3"):
        calibrate(rows, model, .95, adaptive=True)
    assert calibrate(rows, model, .95, adaptive=True, scope="observed_valid") == 39.
    summary = calibration_data_summary(rows)
    assert summary["layouts_with_valid_track_loss"] == 37
    assert summary["status_counts"]["lost_valid"] == 37
    assert summary["valid_track_loss_rate"] == pytest.approx(37 / 77)
    with pytest.raises(ValueError, match="groups=0"):
        calibrate(rows[40:], model, .95, adaptive=False, scope="observed_valid")


def test_conditional_artifact_cannot_launch_original_final_panel(tmp_path, monkeypatch):
    import scripts.evaluate_recovery_study as evaluator
    artifact = tmp_path / "conditional.json"
    artifact.write_text(json.dumps({"calibration_scope": "observed_valid", "strict_gate_passed": False}))
    def forbidden(*args, **kwargs):
        pytest.fail("final evaluation must not start")
    monkeypatch.setattr(evaluator, "run_arm", forbidden)
    with pytest.raises(ValueError, match="original final-test gate"):
        evaluator.main(["--panel", "final", "--output-dir", str(tmp_path / "final"),
                        "--calibration-artifact", str(artifact), "--noninferiority-delta", "0.02"])
    assert not (tmp_path / "final").exists()


def test_history_only_consumes_mature_observations_and_records_loss():
    env = environment()
    snapshot = env._safety_planning_snapshot()
    track = ObservableObstacleV1(0, 7, (10., 0.), (1., 0.), 1., True)
    snapshot = replace(snapshot, visible_obstacles=((track,), (), ()), step=0)
    rows = []
    history = ObservablePredictionHistory(snapshot.dt, rows.append)
    history.observe(snapshot, 122000)
    assert rows == []
    assert history.recent((0, 7)) == 0.
    moved = replace(track, relative_position=(10. + snapshot.dt, 0.))
    history.observe(replace(snapshot, visible_obstacles=((moved,), (), ()), step=1), 122000)
    assert len(rows) == 1
    assert rows[0]["prediction_error"] == pytest.approx(0.)
    history.observe(replace(snapshot, visible_obstacles=((), (), ()), step=2), 122000, terminal=True)
    assert any(row["status"] == "track_lost" for row in rows)
    assert not history.residuals.get((0, 7))


def test_cold_start_requires_valid_residuals_and_reacquisition_clears_history():
    env = environment()
    snapshot = env._safety_planning_snapshot()
    track = ObservableObstacleV1(0, 7, (10., 0.), (0., 0.), 1., False)
    snapshot = replace(snapshot, visible_obstacles=((track,), (), ()), step=0)
    history = ObservablePredictionHistory(snapshot.dt)
    history.observe(snapshot, 122000)
    valid = replace(track, velocity_valid=True)
    history.observe(replace(snapshot, visible_obstacles=((valid,), (), ()), step=1), 122000)
    artifact = {"adaptive_model": {"b": [.1]*3, "c": [1.]*3},
        "fixed_model": {"b": [2.]*3, "c": [0.]*3}, "adaptive_q": 1., "fixed_q": 1.}
    assert history.age[(0, 7)] == 2
    assert history.radius((0, 7), 1, artifact, True) == 2.
    history.residuals[(0, 7)].extend([8., 9.])
    history.observe(replace(snapshot, visible_obstacles=((), (), ()), step=2), 122000)
    history.observe(replace(snapshot, visible_obstacles=((valid,), (), ()), step=3), 122000)
    assert history.recent((0, 7)) == 0.
    assert history.radius((0, 7), 1, artifact, True) == 2.


@pytest.mark.parametrize("bad", ["quantile", "model", "panel"])
def test_invalid_calibration_artifacts_rejected_before_execution(tmp_path, bad):
    import hashlib
    import torch
    from scripts.evaluate_recovery_study import run_arm
    checkpoint = tmp_path / "model.pt"
    config = load_yaml("configs/active/standard_mappo_3u_8s2d.yaml")
    torch.save({"config": config, "update": 800, "obs_dim": 52}, checkpoint)
    panels = load_yaml("configs/experiments/recovery_seed23/panels.json")
    artifact = {"schema_version": "CUPJRCalibrationV1", "adaptive_q": 1., "fixed_q": 1.,
        "fixed_model": {"b": [1.]*3, "c": [0.]*3},
        "adaptive_model": {"b": [1.]*3, "c": [1.]*3},
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "fit_layouts": panels["fit"], "calibration_layouts": panels["calibration"]}
    if bad == "quantile":
        artifact["adaptive_q"] = -1.
    elif bad == "model":
        artifact["adaptive_model"]["b"] = [1.]
    else:
        artifact["fit_layouts"] = [0]
    artifact_path = tmp_path / "bad.json"
    artifact_path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="calibration"):
        run_arm("R3", checkpoint, panels["validation"], tmp_path / "output", "cpu", artifact_path)
    assert not (tmp_path / "output").exists()


def test_r0_selector_and_shadow_do_not_change_dynamics_or_budget():
    original, r0 = environment(), environment()
    r0.recovery_study = {"arm": "R0"}
    for env in (original, r0):
        env._safe_recovery_entry_sources = ("boundary",)
        env._star_recovery_affected_mask[:] = True
    actions = np.zeros((3, 2), dtype=np.float32)
    before = r0.positions.copy()
    rng = json.dumps(r0._rng.bit_generator.state, sort_keys=True)
    budget = r0._recovery_lineage_steps
    a = original._v5_4_select_recovery_command(actions)
    b = r0._v5_4_select_recovery_command(actions)
    assert (a is None) == (b is None)
    if a is not None:
        np.testing.assert_array_equal(a.actions, b.actions)
        assert a.reason == b.reason
    assert original.safe_recovery_events == r0.safe_recovery_events
    np.testing.assert_array_equal(before, r0.positions)
    assert json.dumps(r0._rng.bit_generator.state, sort_keys=True) == rng
    assert r0._recovery_lineage_steps == budget


def test_study_panels_are_disjoint_from_registered_evaluation_panels():
    study = load_yaml("configs/experiments/recovery_seed23/panels.json")
    used = set()
    for split in ("fit", "calibration", "validation", "final"):
        values = set(study[split])
        assert len(values) == len(study[split])
        assert values.isdisjoint(used)
        used.update(values)
    for file in Path("configs/evaluation").glob("*.json"):
        payload = json.loads(file.read_text(encoding="utf-8-sig"))
        seeds = payload.get("layout_seeds", []) if isinstance(payload, dict) else payload
        if isinstance(seeds, list):
            assert used.isdisjoint(seeds), str(file)


def test_recovery_checkpoint_transition_is_restricted_and_evaluation_only(tmp_path):
    from src.algorithms.factory import create_trainer
    config = load_yaml("configs/active/standard_mappo_3u_8s2d.yaml")
    config["device"] = "cpu"
    config["output"] = {"output_dir": str(tmp_path / "parent"), "reject_nonempty": True}
    parent = create_trainer(config, project_root=Path.cwd())
    checkpoint = tmp_path / "parent.pt"
    parent.save_checkpoint(checkpoint)
    child_config = deepcopy(config)
    child_config["output"] = {"output_dir": str(tmp_path / "child"), "reject_nonempty": True}
    child_config["safety_execution"]["safety_state_machine_version"] = "star_persistent_recovery_v5_4"
    child_config["safety_execution"]["goal_boundary_corridor"]["enabled"] = True
    child = create_trainer(child_config, project_root=Path.cwd())
    child.load_checkpoint(checkpoint, allow_recovery_study_evaluation=True)
    with pytest.raises(ValueError, match="evaluation-only"):
        child.train()
    child.config["reward"]["timeout_penalty"] = 99.
    with pytest.raises(ValueError, match="permits only"):
        child.load_checkpoint(checkpoint, allow_recovery_study_evaluation=True)
