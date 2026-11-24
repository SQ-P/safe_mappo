from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.audit_cupjr_gate_a import audit, main as audit_main
from src.safety.staged_execution.contracts import ObservableObstacleV1
from src.safety.staged_execution.robust_recovery_selector import diagnose_candidate, evaluate_candidate
from src.safety.staged_execution.uncertainty import ObservablePredictionHistory
from tests.paper.test_robust_recovery import environment


def snapshot():
    value = environment()._safety_planning_snapshot()
    return replace(value,positions=((10.,10.),(20.,10.),(30.,10.)),velocities=((0.,0.),)*3,
                   visible_obstacles=((),(),()))


def test_initial_expanded_violation_outward_motion_is_evidence_not_permission():
    value = snapshot()
    required = value.agent_radius+1.+value.obstacle_safety_margin+value.boundary_invariant_buffer
    value = replace(value,visible_obstacles=((ObservableObstacleV1(0,7,(required+.5,0.),(0.,0.),1.,True),),(),()))
    positions = np.asarray(value.positions).copy()
    positions[0,0] -= .5
    radius = lambda key,h: 2.
    evidence = diagnose_candidate(value,[positions],radius,obstacle_speed_bound=2.)
    assert evidence["initial_robust_violation"]
    assert evidence["risk_margin_improving"]
    assert evidence["calibration_only_violation"]
    assert evidence["robust_nominal_margin_m"] > 0
    assert not evaluate_candidate(value,[positions],True,radius,[True]*3,0.,obstacle_speed_bound=2.).safe


def test_identical_nominal_clearance_different_uncertainty_and_overlapping_reasons():
    value = snapshot()
    value = replace(value,visible_obstacles=((ObservableObstacleV1(0,7,(5.,0.),(0.,0.),1.,True),),(),()))
    path = [np.asarray(value.positions)]*3
    small = diagnose_candidate(value,path,lambda key,h:.01,obstacle_speed_bound=2.)
    large = diagnose_candidate(value,path,lambda key,h:10.,obstacle_speed_bound=2.)
    assert small["robust_nominal_margin_m"] == large["robust_nominal_margin_m"]
    assert small["robust_calibrated_margin_m"] > large["robust_calibrated_margin_m"]
    value = replace(value,positions=((.1,10.),(.1,10.),(30.,10.)))
    evidence = diagnose_candidate(value,[np.asarray(value.positions)]*3,lambda key,h:10.,
        obstacle_speed_bound=2.,lost_tracks=[(0,(1.,0.),1.,2.)])
    assert set(evidence["violated_constraint_kinds"]) == {"wall","pair","observed_obstacle","lost_track"}
    assert not evidence["calibration_only_violation"]


@pytest.mark.parametrize("arm",["R0","R1","R2","R3"])
@pytest.mark.parametrize("eligible",[True,False])
def test_gate_a_on_off_preserves_actions_events_state_budget_rng_and_radius_stats(monkeypatch,arm,eligible):
    import src.envs.cooperative_uav_env as module
    env = environment()
    env.recovery_study = {"arm":arm,"protocol":"v2","progress_tie_tolerance_m":.05,
        "obstacle_speed_bound_mps":2.,"artifact":{"adaptive_model":{"b":[.1]*3,"c":[0.]*3,"feature_max":1.},
        "fixed_model":{"b":[.1]*3,"c":[0.]*3},"fixed_q":1.,"adaptive_q":1.}}
    history = ObservablePredictionHistory(env.dt)
    env.recovery_prediction_history = history
    history.observe(env._safety_planning_snapshot(),124000)
    env._safe_recovery_entry_sources = ("boundary",)
    env._star_recovery_affected_mask[:] = True
    actions = np.ones((3,2),dtype=np.float32)
    candidates = [SimpleNamespace(label=label,actions=actions*factor,targets=env.positions.copy(),
        affected_agent_mask=(True,)*3,entry_sources=("boundary",))
        for label,factor in (("normal",1.),("left_tangent",.5),("right_tangent",-.5))]
    monkeypatch.setattr(module,"build_star_recovery_candidates",lambda **kwargs:candidates)
    monkeypatch.setattr(env,"_v5_recovery_margin",lambda:0.)
    shadow_calls = []
    def shadow(command,**kwargs):
        shadow_calls.append(True)
        return {"final_margin":.1,"minimum_margin":.1,"feasible":eligible,"rejection_reason":"ok" if eligible else "post_projection_infeasible",
                "positions":env.positions.copy(),"position_path":[env.positions.copy()]*3,"stage_trace":[],
                "first_executed_actions":command,"post_projection_norm":1.}
    monkeypatch.setattr(env,"_v5_4_observable_envelope",shadow)
    rng = json.dumps(env._rng.bit_generator.state,sort_keys=True)
    before_positions, before_velocities, before_step = env.positions.copy(),env.velocities.copy(),env.step_count
    before_budget = env._recovery_lineage_steps
    plain = env._v5_4_select_recovery_command(actions)
    plain_state = env.export_counterfactual_snapshot(None).state_fingerprint
    plain_events = deepcopy(env.safe_recovery_events)
    radius_events = deepcopy(history.radius_events)
    env.safe_recovery_events.clear()
    history.radius_events.clear()
    env.recovery_candidate_rejection_count = 0
    env.recovery_candidate_rejection_reason_counts.clear()
    env.recovery_candidate_selection_count = 0
    traces = []
    env.recovery_gate_a_sink = traces.append
    diagnostic = env._v5_4_select_recovery_command(actions)
    assert (plain is None) == (diagnostic is None)
    if plain is not None:
        np.testing.assert_array_equal(plain.actions,diagnostic.actions)
        np.testing.assert_array_equal(plain.targets,diagnostic.targets)
    assert plain_events == env.safe_recovery_events
    assert radius_events == history.radius_events
    assert len(shadow_calls) == 6  # Three per call; no additional shadow for diagnostics.
    assert json.dumps(env._rng.bit_generator.state,sort_keys=True) == rng
    assert env._recovery_lineage_steps == before_budget
    np.testing.assert_array_equal(before_positions,env.positions)
    np.testing.assert_array_equal(before_velocities,env.velocities)
    assert before_step == env.step_count
    assert plain_state == env.export_counterfactual_snapshot(None).state_fingerprint
    assert len(traces) == 1 and len(traces[0]["candidates"]) == 3
    assert traces[0]["no_feasible_candidate"] == (diagnostic is None)
    assert traces[0]["selected_label"] == traces[0]["same_state_choices"][arm]


def fixture_outputs(root):
    for arm in ("R0","R1","R2","R3"):
        target = root/arm
        target.mkdir(parents=True)
        (target/"episodes.jsonl").write_text(json.dumps({"seed":124000,"safe_success":False,
            "collision_count":1,"goal_progress_ratio":.1,"reason":"collision","safe_recovery_events":[]})+"\n")
        (target/"provenance.json").write_text(json.dumps({"checkpoint_sha256":"test","runtime_task_fingerprint":"same",
            "parent_task_fingerprint":"parent"}))


def test_old_logs_do_not_invent_attribution_or_thresholds_and_cannot_overwrite(tmp_path):
    root = tmp_path/"validation"
    fixture_outputs(root)
    before = {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
    report = audit(root)
    assert report["gate_a_status"] == "PARTIALLY_ATTRIBUTED_REQUIRES_REVIEW"
    assert not report["v3_authorized"] and not report["final_authorized"]
    assert not report["arms"]["R2"]["diagnostic_trace_present"]
    assert all(v is None for v in report["unresolved_scientific_thresholds"].values())
    assert before == {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in before}
    output = tmp_path/"audit.json"
    output.write_text("preserve")
    with pytest.raises(FileExistsError):
        audit_main(["--root",str(root),"--output",str(output)])
    assert output.read_text() == "preserve"


def test_diagnostic_entry_rejects_final_before_running(monkeypatch,tmp_path):
    import scripts.evaluate_recovery_study as evaluator
    monkeypatch.setattr(evaluator,"run_arm",lambda *a,**k:pytest.fail("must not launch evaluation"))
    with pytest.raises(ValueError,match="Final remains sealed"):
        evaluator.main(["--protocol","v2","--panel","final","--gate-a-diagnostics","--output-dir",str(tmp_path/"final")])
    assert not (tmp_path/"final").exists()
