from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from src.envs.cooperative_uav_env import CooperativeUAVEnv
from src.safety.ppcsa_one_step import evaluate_ppcsa_normal_shadow, factual_parity, integrity_token, capture_packet, restore_packet
from src.utils.io_utils import load_yaml
from src.algorithms.ppcsa_executed_value import ExecutedValueCritic,executed_gap_loss,zero_adapter,bounded_probe,parameter_probe,audit_action_directions


def normal_environment():
    config = load_yaml("configs/active/standard_mappo_3u_8s2d.yaml")
    env = CooperativeUAVEnv({**config["env"],"reward":config["reward"],"oracle_subgoal":config["oracle_subgoal"],
        "subgoal_selector":config["subgoal_selector"],"safety_execution":config["safety_execution"]})
    env.reset(seed=23)
    env.set_runtime_telemetry_level("full")
    return env


def test_v4_shadow_matches_factual_stages_without_mutating_cache_rng_gru():
    env = normal_environment()
    gru = torch.randn(1,3,128)
    action = np.full((3,2),.1,dtype=np.float32)
    token = integrity_token(env,gru)
    shadow = evaluate_ppcsa_normal_shadow(env,action,gru)
    assert shadow.supported
    assert integrity_token(env,gru) == token
    _,_,_,_,info = env.step(action)
    passed,evidence = factual_parity(shadow,env,info)
    assert passed,evidence


def test_unsupported_modes_are_not_safe_teachers_and_packet_restores_cache():
    env = normal_environment()
    env.build_local_observations()
    packet = capture_packet(env,23,torch.ones(1,3,128))
    token = integrity_token(env,packet["snapshot"].actor_gru_state)
    restored = normal_environment()
    gru = restore_packet(restored,packet)
    assert integrity_token(restored,gru) == token
    for mode in ("SAFE_RECOVER","SAFE_BRAKE"):
        restored._safety_mode = mode
        value = evaluate_ppcsa_normal_shadow(restored,np.zeros((3,2)))
        assert not value.supported and not value.hard_safe
    restored.terminated = True
    value = evaluate_ppcsa_normal_shadow(restored,np.zeros((3,2)))
    assert "terminal_state" in value.rejection_reasons


def test_unknown_motion_is_reported_and_never_promoted_to_safe_reference():
    env = normal_environment()
    from src.envs.entities import CircularObstacle
    env.obstacles = [CircularObstacle(env.positions[0]+np.array([10.,0.]),1.,np.zeros(2),True)]
    env._reset_obstacle_tracks()
    value = evaluate_ppcsa_normal_shadow(env,np.zeros((3,2)))
    assert not value.hard_safe
    assert "unknown_obstacle_motion" in value.rejection_reasons


def test_shadow_freezes_future_obstacle_motion_and_preserves_input():
    from src.envs.entities import CircularObstacle
    env = normal_environment()
    env.obstacles = [CircularObstacle(env.positions[0]+np.array([15.,0.]),1.,np.array([1.,0.]),True)]
    env._reset_obstacle_tracks()
    action = np.zeros((3,2),dtype=np.float32)
    before = env.obstacles[0].position.copy()
    shadow = evaluate_ppcsa_normal_shadow(env,action)
    np.testing.assert_array_equal(before,env.obstacles[0].position)
    np.testing.assert_array_equal(action,np.zeros((3,2)))
    _,_,_,_,info = env.step(action)
    assert factual_parity(shadow,env,info)[0]
    assert not np.array_equal(before,env.obstacles[0].position)


def test_shadow_uses_actual_progress_aware_ttc_selection(monkeypatch):
    from src.envs.entities import CircularObstacle
    env = normal_environment()
    for agent,position in zip(env.agents,((20.,20.),(45.,20.),(70.,20.))):
        agent.position = np.asarray(position,dtype=np.float32)
        agent.velocity = np.array([1.,0.],dtype=np.float32)
    env.goals[:] = np.array([90.,20.],dtype=np.float32)
    env.subgoals[:] = env.goals
    env.obstacles = [CircularObstacle(np.array([29.,20.]),1.,np.zeros(2),False)]
    env._reset_obstacle_tracks()
    for i,agent in enumerate(env.agents):
        relative = env.obstacles[0].position-agent.position
        if np.linalg.norm(relative)<=env.detection_radius:
            env._obstacle_track_history[i][0] = (relative+np.array([.5,0.],dtype=np.float32),-1)
    env.build_local_observations()
    calls = []
    original = CooperativeUAVEnv._audit_ttc_headroom_and_select_v2
    def observed(self,*args,**kwargs):
        calls.append(1)
        return original(self,*args,**kwargs)
    monkeypatch.setattr(CooperativeUAVEnv,"_audit_ttc_headroom_and_select_v2",observed)
    action = np.tile(np.array([1.,0.],dtype=np.float32),(3,1))
    shadow = evaluate_ppcsa_normal_shadow(env,action)
    _,_,_,_,info = env.step(action)
    assert len(calls)>=2
    assert factual_parity(shadow,env,info)[0]
    info["executed_actions"] = np.asarray(info["executed_actions"])+.1
    assert not factual_parity(shadow,env,info)[0]


@pytest.mark.parametrize("guard",["initial_guard","final_guard"])
@pytest.mark.parametrize("field",["pair_infeasible","pair_invariant_violations","recovery_required"])
def test_guard_recovery_request_rejects_geometrically_safe_teacher(monkeypatch,guard,field):
    from dataclasses import replace
    import src.safety.ppcsa_one_step as shadow_api
    env = normal_environment()
    env.obstacles = []
    env._reset_obstacle_tracks()
    action = np.zeros((3,2),dtype=np.float32)
    baseline = shadow_api.evaluate_ppcsa_normal_shadow(env,action)
    assert baseline.supported and baseline.hard_safe
    original = shadow_api._capture
    def request_recovery(actions,branch,checks):
        checks[guard][field] = 1
        return original(actions,branch,checks)
    monkeypatch.setattr(shadow_api,"_capture",request_recovery)
    candidate = shadow_api.evaluate_ppcsa_normal_shadow(env,action)
    # Geometry/actions remain those of the independently safe control, but
    # the Guard's Recovery request is authoritative before state transition.
    assert candidate.executed_actions == baseline.executed_actions
    assert candidate.predicted_next_positions == baseline.predicted_next_positions
    assert not candidate.supported and not candidate.hard_safe
    assert f"{guard}_{field}" in candidate.rejection_reasons
    assert not shadow_api.safe_improvement(replace(candidate,executed_progress=baseline.executed_progress+1.),baseline)


def test_gap_sign_stop_gradients_and_matching_direct_value_gradient():
    critic = ExecutedValueCritic(4,agents=1,hidden_dim=8).freeze()
    context = torch.ones(2,4)
    action = torch.zeros(2,1,2,requires_grad=True)
    current = critic(context,action)
    reference = (current.detach()+1.).requires_grad_(True)
    weight = torch.tensor([1.,.5],requires_grad=True)
    loss = executed_gap_loss(current,reference,weight,torch.ones(2,dtype=torch.bool))
    gap_gradient, = torch.autograd.grad(loss,action,retain_graph=True)
    direct = -(weight.detach()*current).sum()/weight.detach().sum()
    direct_gradient, = torch.autograd.grad(direct,action,retain_graph=True)
    torch.testing.assert_close(gap_gradient,direct_gradient)
    loss.backward()
    assert reference.grad is None and weight.grad is None
    assert all(p.grad is None for p in critic.parameters())
    assert torch.isfinite(action.grad).all()
    # Independent exact monotonic example: descending positive gap increases q.
    scalar = torch.tensor([0.],requires_grad=True)
    gap = executed_gap_loss(scalar,torch.ones(1),torch.ones(1),torch.ones(1,dtype=torch.bool))
    grad, = torch.autograd.grad(gap,scalar)
    assert float((scalar-.1*grad).detach()) > float(scalar.detach())


def test_no_valid_reference_means_zero_gap_gradient_even_with_invalid_teacher():
    current = torch.ones(2,requires_grad=True)
    reference = torch.tensor([float("nan"),2.],requires_grad=True)
    loss = executed_gap_loss(current,reference,torch.ones(2),torch.zeros(2,dtype=torch.bool))
    loss.backward()
    assert torch.equal(current.grad,torch.zeros(2)) and reference.grad is None


def test_adapter_envelope_and_virtual_probe_do_not_change_original():
    adapter = zero_adapter()
    features = torch.randn(3,128)
    before = {k:v.detach().clone() for k,v in adapter.state_dict().items()}
    loss = -adapter(features).sum()
    candidate,norm,_ = parameter_probe(adapter,loss,.01)
    assert norm>0 and torch.isfinite(candidate(features)).all()
    assert float(candidate(features).abs().max())<=.15
    assert all(torch.equal(before[k],v) for k,v in adapter.state_dict().items())
    mean = torch.randn(3,2)
    action,delta = bounded_probe(mean,torch.ones_like(mean),.1)
    assert float(delta.abs().max())<=.15
    assert torch.allclose(action,torch.tanh(mean+delta))


def value_result(action,value,*,hard_safe=True):
    from src.safety.ppcsa_one_step import NormalOneStepResult
    return NormalOneStepResult(True,hard_safe,action.tolist(),action.tolist(),{},[],[],float(value),[],{},[],[],"test")


def test_true_shadow_gradient_direction_and_flat_projection_diagnostics():
    mean = torch.zeros(1,2)
    def critic(action):
        return action.sum()
    correct = audit_action_directions(mean,critic,lambda a:value_result(a,a.sum()),np.ones((1,2)),.05)
    assert correct[0]["outcome"] == "improved" and correct[0]["predicted_gain"]>0
    wrong = audit_action_directions(mean,critic,lambda a:value_result(a,-a.sum()),np.ones((1,2)),.05)
    assert wrong[0]["outcome"] == "worsened"
    def projected(action):
        return value_result(np.zeros_like(action),0.)
    flat = audit_action_directions(mean,critic,projected,np.ones((1,2)),.05)
    assert flat[0]["outcome"] == "absorbed" and flat[0]["predicted_gain"]>0
    assert len({round(r["delta_l2"],6) for r in correct}) == 1


def test_parameter_report_distinguishes_safety_risk_absorption_and_progress():
    from dataclasses import replace
    from src.algorithms.ppcsa_executed_value import assess_probe
    from scripts.audit_ppcsa_gate2 import summarize_parameter_outcomes
    base = replace(value_result(np.zeros((1,2)),0.),risk_proxy=[0.])
    changed = np.ones((1,2))*.1
    candidates = [base,replace(base,executed_actions=changed.tolist()),
        replace(base,hard_safe=False,executed_progress=.2),
        replace(base,risk_proxy=[1.],executed_progress=.3),
        replace(base,executed_actions=changed.tolist(),executed_progress=.4),
        replace(base,executed_actions=changed.tolist(),executed_progress=-.1),
        replace(base,risk_proxy=[None],executed_progress=.5)]
    rows = [{"layout":10+i%2,**assess_probe(base,c)} for i,c in enumerate(candidates)]
    report = summarize_parameter_outcomes(rows)
    assert report["hard_safety_regression_count"] == 1
    assert report["risk_proxy_regression_count"] == 1
    assert report["absorbed"] == 1 and report["flat_changed_action"] == 1
    assert report["improved"] == 1 and report["worsened"] == 1
    assert report["uncertifiable"] == 2 and report["risk_proxy_unknown_count"] == 1
    assert report["paired_progress_net"] == pytest.approx(1.3)
    assert report["certified_nonworse_risk_progress_net"] == pytest.approx(.3)
    assert sum(r["states"] for r in report["per_layout"].values()) == len(rows)
    flat = summarize_parameter_outcomes(rows[:2])
    assert not flat["hard_safety_regression"] and not flat["risk_proxy_regression"]
    assert flat["efficacy_unproven"] and flat["paired_progress_net"] == 0.
    assert "virtual_update_rejected" not in flat


def test_registered_data_contracts_and_approvals_fail_closed(tmp_path):
    import json
    from src.algorithms.ppcsa_protocol import validate_panel,require_gate1_review,read_manifest,PARENT_SHA
    panel = validate_panel(Path("configs/evaluation/ppcsa_dev_121000_121039.json"))
    assert set(panel["partitions"]["fit"]).isdisjoint(panel["partitions"]["audit"])
    panel["layout_seeds"][0] = 120000
    file = tmp_path/"bad_panel.json"
    file.write_text(json.dumps(panel))
    with pytest.raises(ValueError,match="sealed"):
        validate_panel(file)
    with pytest.raises(RuntimeError,match="reviewed"):
        require_gate1_review(tmp_path,"x")
    (tmp_path/"gate1_review.json").write_text(json.dumps({"schema_version":"PPCSAGate1ReviewV1","decision":"approve_critic_diagnostic","report_sha256":"wrong","rationale":"test"}))
    with pytest.raises(RuntimeError):
        require_gate1_review(tmp_path,"x")
    from src.safety.ppcsa_one_step import PROTOCOL
    (tmp_path/"manifest.json").write_text(json.dumps({"protocol":PROTOCOL,"checkpoint_sha256":PARENT_SHA,"file_sha256":{"bad_panel.json":"wrong"}}))
    with pytest.raises(ValueError,match="hash"):
        read_manifest(tmp_path)


def test_parent_contract_rejects_executor_and_checkpoint_drift():
    from src.algorithms.ppcsa_protocol import validate_parent,PARENT_SHA
    cfg = load_yaml("configs/active/standard_mappo_3u_8s2d.yaml")
    payload = {"config":cfg,"update":800,"obs_dim":52,"hidden_dim":128,"num_agents":3}
    validate_parent(payload,PARENT_SHA)
    for changes in ("checkpoint","ttc","v5"):
        wrong = deepcopy(payload)
        sha = PARENT_SHA
        if changes == "checkpoint":
            sha = "0"*64
        elif changes == "ttc":
            wrong["config"]["safety_execution"]["obstacle_ttc"]["version"] = "wrong"
        else:
            wrong["config"]["safety_execution"]["safety_state_machine_version"] = "star_persistent_recovery_v5_4"
        with pytest.raises(ValueError):
            validate_parent(wrong,sha)
    env = normal_environment()
    env.star_safety_state_machine_version = "star_persistent_recovery_v5_4"
    with pytest.raises(ValueError,match="V4/TTCv2"):
        evaluate_ppcsa_normal_shadow(env,np.zeros((3,2)))


def test_candidate_bank_has_references_single_and_joint_pre_tanh_bounds():
    from scripts.audit_ppcsa_gate1 import candidate_bank
    mean = torch.zeros(3,2)
    bank = candidate_bank(mean,np.ones((3,2),dtype=np.float32))
    assert sum(name.startswith("reference_") for name,_,_ in bank) == 6
    assert sum(name.startswith("local_joint_") for name,_,_ in bank) == 2
    for name,action,delta in bank:
        if delta is not None:
            assert np.max(np.abs(delta))<=.15000001
            np.testing.assert_allclose(action,np.tanh(delta),atol=1e-7)


def test_synthetic_executed_value_fit_uses_layout_partitions_and_no_actor(tmp_path):
    from scripts.fit_ppcsa_executed_value import fit_models
    examples = [{"layout":g,"context":np.ones(4,dtype=np.float32)*g/100.,"action":np.full((3,2),a,dtype=np.float32),
        "base_action":np.zeros((3,2),dtype=np.float32),"target":a,"base_target":0.,"heuristic_target":a}
        for g in (1,2,3) for a in (0.,.1,.2)]
    models,metrics = fit_models(examples,{"fit":[1,2],"calibration":[3],"audit":[4]},epochs=2)
    assert set(models) == {"ordinary","paired"} and all(not x["actor_training_authorized"] for x in models.values())
    assert all(np.isfinite(x["calibration_layout_balanced_mse"]) for x in metrics.values())
    with pytest.raises(ValueError,match="leakage"):
        fit_models(examples,{"fit":[1,2],"calibration":[2,3],"audit":[4]},epochs=1)
