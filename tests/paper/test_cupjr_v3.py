from dataclasses import replace
from types import SimpleNamespace
import json

import numpy as np
import pytest

from src.safety.staged_execution.contracts import ObservableObstacleV1
from src.safety.staged_execution.recovery_v3 import RecoveryResultV3, evaluate_candidate_v3, select_safety_candidate_v3
from tests.paper.test_robust_recovery import environment


def geometry(tracks=None):
    value = environment()._safety_planning_snapshot()
    value = replace(value,positions=((20.,20.),(45.,20.),(70.,20.)),velocities=((0.,0.),)*3,
        final_goals=((90.,70.),)*3,dt=.1,visible_obstacles=((),(),()))
    required = value.agent_radius+1.+value.obstacle_safety_margin+value.boundary_invariant_buffer
    if tracks is None:
        tracks = (ObservableObstacleV1(0,7,(required+.5,0.),(0.,0.),1.,True),)
    value = replace(value,visible_obstacles=(tuple(tracks),(),()))
    path = []
    for h in (1,2,3):
        positions = np.asarray(value.positions).copy()
        positions[0,0] -= .05*h
        path.append(positions)
    return value,path,required


def evaluate(value,path,**kwargs):
    options = dict(legacy_eligible=True,legacy_margin_gain=.1,obstacle_speed_bound=1.)
    options.update(kwargs)
    return evaluate_candidate_v3(value,path,options.pop("stage_feasible",True),options.pop("radius",lambda key,h:2.),[True]*3,0.,**options)


def test_three_modes_and_time_matched_certificate():
    value,path,_ = geometry()
    constrained = evaluate(value,path)
    assert constrained.mode == "constrained_recovery"
    assert constrained.physical_margin_m >= 0 and constrained.risk_improvement_m > 0
    assert constrained.certificate["coverage_claim"] is False
    assert all(c["candidate_margin_m"] >= min(0.,c["reference_margin_m"]) for c in constrained.certificate["risk_checks"])
    assert evaluate(value,path,radius=lambda key,h:.01).mode == "robust_feasible"
    assert evaluate(value,path,stage_feasible=False).reason == "stage_infeasible"
    assert evaluate(value,path,legacy_eligible=False).reason == "legacy_candidate_ineligible"
    assert evaluate(value,[np.asarray(value.positions)]*3).reason == "no_strict_terminal_risk_improvement"


def test_per_constraint_checks_reject_worsening_and_new_violation():
    value,path,required = geometry()
    for distance,reason in ((required+.5,"existing_risk_worsened"),(required+2.+.05,"new_calibrated_violation")):
        other = ObservableObstacleV1(0,8,(-distance,0.),(0.,0.),1.,True)
        snapshot = replace(value,visible_obstacles=((value.visible_obstacles[0][0],other),(),()))
        result = evaluate(snapshot,path)
        assert result.mode == "unavailable" and result.reason == reason


def test_conservative_physical_reachability_rejects_nominally_safe_path():
    value,path,required = geometry()
    assert evaluate(value,path,obstacle_speed_bound=10.).reason == "physical_reachable_intersection"
    lost = [(0,(required+.1,0.),1.,5.)]
    assert evaluate(value,path,lost_tracks=lost).reason == "physical_reachable_intersection"
    unknown = replace(value.visible_obstacles[0][0],relative_position=(required+.1,0.),velocity_valid=False)
    snapshot = replace(value,visible_obstacles=((unknown,),(),()))
    assert evaluate(snapshot,path,radius=lambda key,h:float("inf")).reason == "physical_reachable_intersection"


def test_fail_closed_and_segment_not_endpoint_only():
    value,path,required = geometry()
    for radius in (float("nan"),-1.):
        assert evaluate(value,path,radius=lambda key,h:radius).mode == "unavailable"
    broken = np.asarray(path).copy()
    broken[1,0,0] = np.nan
    assert evaluate(value,broken).reason == "invalid_evidence"
    assert evaluate(value,path,obstacle_speed_bound=float("nan")).reason == "invalid_evidence"
    # Both endpoints have positive nominal clearance but the segment crosses
    # the obstacle. The candidate cannot escape by checking endpoints alone.
    crossing = [p.copy() for p in path]
    crossing[0][0,0] = value.positions[0][0]+2*(required+.5)
    crossing[1:] = [crossing[0].copy(),crossing[0].copy()]
    assert evaluate(value,crossing).mode == "unavailable"


def test_safety_lexicographic_modes_and_exact_ties():
    def row(mode,gain,margin,progress):
        r = RecoveryResultV3(mode,"test",gain,margin,progress,progress,0.,.1,.1)
        return (r,0,None,None)
    robust = row("robust_feasible",.1,.1,-100.)
    assert select_safety_candidate_v3([robust,row("constrained_recovery",100.,10.,100.)]) == robust
    safer = row("legacy_feasible",.2,.1,-100.)
    assert select_safety_candidate_v3([safer,row("legacy_feasible",.1,100.,100.)]) == safer
    larger_margin = row("robust_feasible",.2,.2,-50.)
    assert select_safety_candidate_v3([safer,larger_margin]) == larger_margin
    progress = row("robust_feasible",.2,.2,1.)
    assert select_safety_candidate_v3([larger_margin,progress]) == progress
    assert select_safety_candidate_v3([row("unavailable",0.,0.,100.)]) is None


@pytest.mark.parametrize("arm",["R1","R2"])
@pytest.mark.parametrize("eligible",[True,False])
def test_v3_environment_branch_reuses_candidates_and_budget(monkeypatch,arm,eligible):
    import src.envs.cooperative_uav_env as module
    import src.safety.staged_execution.recovery_v3 as selector
    env = environment()
    env.recovery_study = {"arm":arm,"protocol":"v3","artifact":{},"obstacle_speed_bound_mps":1.}
    env.recovery_prediction_history = SimpleNamespace(radius_v2=lambda *a,**k:.1,lost_reachable=lambda s:[])
    env._safe_recovery_entry_sources = ("boundary",)
    env._star_recovery_affected_mask[:] = True
    actions = np.ones((3,2),dtype=np.float32)
    candidate = SimpleNamespace(label="normal",actions=actions,targets=env.positions.copy(),affected_agent_mask=(True,)*3,entry_sources=("boundary",))
    monkeypatch.setattr(module,"build_star_recovery_candidates",lambda **kwargs:[candidate])
    monkeypatch.setattr(env,"_v5_recovery_margin",lambda:0.)
    calls = []
    def envelope(*a,**k):
        calls.append(1)
        return {"final_margin":.1,"minimum_margin":.1,"feasible":eligible,"positions":env.positions.copy(),
            "position_path":[env.positions.copy()]*3,"first_executed_actions":actions,"post_projection_norm":1.}
    monkeypatch.setattr(env,"_v5_4_observable_envelope",envelope)
    monkeypatch.setattr(selector,"evaluate_candidate_v3",lambda *a,**k:RecoveryResultV3(
        "constrained_recovery" if k["legacy_eligible"] else "unavailable","test",.1,-.1,1.,1.,0.,.1,.1))
    before,rng,budget = env.positions.copy(),json.dumps(env._rng.bit_generator.state,sort_keys=True),env._recovery_lineage_steps
    choice = env._v5_4_select_recovery_command(actions)
    assert (choice is not None) == eligible and len(calls) == 1
    assert json.dumps(env._rng.bit_generator.state,sort_keys=True) == rng
    assert env._recovery_lineage_steps == budget
    np.testing.assert_array_equal(before,env.positions)
    event = "v3_candidate_selected" if eligible else "v3_no_feasible_candidate"
    assert any(e["event"] == event for e in env.safe_recovery_events)


def test_v3_formal_final_r3_and_overlapping_panels_rejected_before_evaluation(monkeypatch,tmp_path):
    import scripts.evaluate_recovery_study as evaluator
    monkeypatch.setattr(evaluator,"run_arm",lambda *a,**k:pytest.fail("no experiment may start"))
    for extra in (["--panel","final"],["--panel","v3_dev_validation"],["--arm","R3","--mechanism-only"]):
        with pytest.raises(ValueError):
            evaluator.main(["--protocol","v3","--output-dir",str(tmp_path/"no_output"),*extra])
    registry = evaluator.v3_registry()
    assert set(registry["v3_dev_validation"]) == set(range(128000,128040))
    original = evaluator.load_yaml
    def invalid(path):
        data = original(path)
        if str(path).endswith("panels_v3_dev.json"):
            data["v3_dev_validation"][0] = 125000
        return data
    monkeypatch.setattr(evaluator,"load_yaml",invalid)
    with pytest.raises(ValueError,match="disjoint"):
        evaluator.v3_registry()
    assert not (tmp_path/"no_output").exists()


@pytest.mark.parametrize("allowed",[True,False])
def test_persistent_target_requires_a_new_certificate_every_step(monkeypatch,allowed):
    import src.safety.staged_execution.recovery_v3 as selector
    env = environment()
    env.recovery_study = {"arm":"R2","protocol":"v3","artifact":{},"obstacle_speed_bound_mps":1.}
    env.recovery_prediction_history = SimpleNamespace(radius_v2=lambda *a,**k:.1,lost_reachable=lambda s:[])
    env._star_recovery_affected_mask[:] = True
    env._safe_recovery_target = env.positions+1.
    env._last_recovery_reason = "arbitrary_previous_reason"
    monkeypatch.setattr(env,"_v5_recovery_margin",lambda:0.)
    calls = []
    def envelope(command,**kwargs):
        calls.append(env.step_count)
        return {"final_margin":.1,"minimum_margin":.1,"feasible":True,"positions":env.positions.copy(),
            "position_path":[env.positions.copy()]*3,"first_executed_actions":command,"post_projection_norm":1.}
    monkeypatch.setattr(env,"_v5_4_observable_envelope",envelope)
    monkeypatch.setattr(selector,"evaluate_candidate_v3",lambda *a,**k:RecoveryResultV3(
        "constrained_recovery" if allowed else "unavailable","test",.1,-.1,1.,1.,0.,.1,.1))
    for _ in range(2):
        result = env._star_active_recovery_actions(np.zeros((3,2),dtype=np.float32))
        assert (result is not None) == allowed
        env.step_count += 1
    assert len(calls) == 2
    assert sum(e["event"] == "v3_active_certificate" for e in env.safe_recovery_events) == 2


def test_inconsistent_shadow_cannot_supply_a_certificate():
    from src.safety.staged_execution.recovery_v3 import shadow_consistent_v3
    value,path,_ = geometry()
    envelope = {"position_path":path,"positions":path[-1].copy(),"first_executed_actions":np.zeros((3,2))}
    assert shadow_consistent_v3(value,envelope)
    envelope["positions"][0,0] += 1.
    assert not shadow_consistent_v3(value,envelope)


def test_fit_features_do_not_use_future_observations_or_targets():
    from scripts.audit_cupjr_fit_features import causal_feature
    row = {"layout_seed":122000,"agent_id":0,"obstacle_track_id":7,"step":5,
        "recent_residual_scale":.1,"observed_relative_velocity":[1.,0.],"prediction_error":100.}
    origins = {(122000,0,7,4):{"velocity_valid":True,"observed_relative_velocity":[.5,0.]}}
    observations = {(122000,0,7,4):[9.,10.],(122000,0,7,5):[10.,10.],(122000,0,7,6):[20.,20.]}
    before = causal_feature(row,origins,observations,.5,[100.,100.])
    observations[122000,0,7,6] = [-999.,-999.]
    row["prediction_error"] = -999.
    assert before == causal_feature(row,origins,observations,.5,[100.,100.])


def test_fit_feature_grouped_diagnostic_and_no_online_model():
    from scripts.audit_cupjr_fit_features import grouped_prediction_diagnostic
    records = [{"layout":g,"horizon":h,"features":[x,x*x,.1],"target":.2+h*x}
        for g in range(8) for h in (2,3) for x in (0.,.1,.2,.3)]
    report = grouped_prediction_diagnostic(records)
    for h in ("2","3"):
        assert report["horizons"][h]["recent_residual"]["mse_gain_over_constant"] > 0.
    assert report["online_model_changed"] is False and report["r3_authorized"] is False


def test_fit_diagnostic_cli_on_synthetic_registered_fit_only(tmp_path):
    import hashlib
    import torch
    from scripts.audit_cupjr_fit_features import main
    checkpoint = tmp_path/"synthetic.pt"
    torch.save({"config":{"env":{"dt":.5,"map_size":[100.,100.]}}},checkpoint)
    (tmp_path/"provenance.json").write_text(json.dumps({"checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest()}))
    rows = [{"layout_seed":layout,"agent_id":0,"obstacle_track_id":7,"step":step,
        "prediction_horizon":h,"status":"observed","velocity_valid":True,
        "later_observed_obstacle_position":[10.+.1*(step+h),10.],
        "recent_residual_scale":.01*step,"observed_relative_velocity":[.1*step,0.],
        "prediction_error":.01*step*h}
        for layout in range(122000,122040) for step in range(6) for h in (1,2,3)]
    data = tmp_path/"predictions.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows))
    output = tmp_path/"diagnostic.json"
    main(["--fit-data",str(data),"--checkpoint",str(checkpoint),"--output",str(output)])
    report = json.loads(output.read_text())
    assert report["online_model_changed"] is False and len(report["layouts"]) == 40
    assert not report["r3_authorized"]
    with pytest.raises(FileExistsError):
        main(["--fit-data",str(data),"--output",str(output)])
    rows[0]["layout_seed"] = 123000
    data.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError,match="only the complete registered Fit panel"):
        main(["--fit-data",str(data),"--output",str(tmp_path/"forbidden.json")])
