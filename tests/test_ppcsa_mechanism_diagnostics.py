from copy import deepcopy

import numpy as np
import pytest
import torch

from src.algorithms.ppcsa_mechanism_diagnostics import (kinematic_value,conflict_summary,layout_weights,
    virtual_adapter,match_calibration_budget,tangent_trace,residual_rms)
from src.algorithms.ppcsa_executed_value import zero_adapter,adapter_jacobian


def geometry():
    return {"positions":np.array([[10.,10.],[20.,30.],[30.,50.]]),"velocities":np.zeros((3,2)),
        "goals":np.array([[90.,15.],[90.,35.],[90.,55.]]),"dt":.5,"acceleration":1.,"max_speed":2.,
        "start_max":81.,"start_mean":71.,"bottleneck":.7}


def test_kinematic_derivative_matches_finite_difference_and_clip_semantics():
    args = geometry()
    action = torch.full((3,2),.1,dtype=torch.float64,requires_grad=True)
    value = kinematic_value(action,**args)
    derivative, = torch.autograd.grad(value,action)
    epsilon = 1e-5
    numerical = torch.zeros_like(action)
    for i in range(action.numel()):
        change = torch.zeros_like(action); change.reshape(-1)[i] = epsilon
        numerical.reshape(-1)[i] = (kinematic_value(action+change,**args)-kinematic_value(action-change,**args))/(2*epsilon)
    torch.testing.assert_close(derivative,numerical,atol=1e-10,rtol=1e-5)
    args["velocities"][:] = 3.
    clipped, = torch.autograd.grad(kinematic_value(action,**args),action)
    assert torch.equal(clipped,torch.zeros_like(action))


def test_kinematic_forward_matches_existing_numpy_goal_heuristic():
    from tests.test_ppcsa_contracts import normal_environment
    env = normal_environment()
    a = np.full((3,2),.1,dtype=np.float32)
    future = env.positions+np.clip(env.velocities+a*env.agent_max_acceleration*env.dt,-env.agent_max_speed,env.agent_max_speed)*env.dt
    expected = env._team_goal_potential(np.linalg.norm(future-env.goals,axis=1))-env._team_goal_potential(np.linalg.norm(env.positions-env.goals,axis=1))
    result = kinematic_value(torch.tensor(a),env.positions,env.velocities,env.goals,dt=env.dt,
        acceleration=env.agent_max_acceleration,max_speed=env.agent_max_speed,
        start_max=env._episode_start_max_goal_distance,start_mean=env._episode_start_mean_goal_distance,bottleneck=env.bottleneck_weight)
    assert float(result) == pytest.approx(expected,abs=2e-7)


def test_zero_initialization_pullback_and_first_order_prediction():
    adapter = zero_adapter()
    features = torch.randn(3,128)
    mean = torch.zeros(3,2)
    jacobian = adapter_jacobian(adapter,features)
    trace = tangent_trace(adapter,features,mean,lambda a:a.sum(),jacobian)
    assert trace["chain_rule_max_error"]<1e-7
    assert all(v["nonzero_jacobian_columns"] == 0 for k,v in trace["layers"].items() if not k.startswith("up."))
    direction = trace["parameter_gradient"]/trace["parameter_gradient"].norm()
    copy = virtual_adapter(adapter,direction,1e-3)
    torch.testing.assert_close(copy(features).detach().reshape(-1),1e-3*jacobian@direction,atol=1e-7,rtol=1e-3)
    assert torch.equal(adapter(features),torch.zeros_like(mean))


def test_one_calibration_scalar_matches_budget_without_per_state_rescaling():
    adapter = zero_adapter()
    features = torch.stack([torch.ones(3,128),torch.randn(3,128)])
    weights = layout_weights([1,2],"cpu")
    direction = torch.zeros(sum(p.numel() for p in adapter.parameters()))
    offset = 0
    for name,p in adapter.named_parameters():
        if name == "up.weight":
            direction[offset:offset+p.numel()] = 1.
        offset += p.numel()
    direction /= direction.norm()
    original = deepcopy(adapter.state_dict())
    step,info = match_calibration_budget(adapter,direction,features,weights,.05)
    assert info["status"] == "MATCHED_ON_CALIBRATION"
    copy = virtual_adapter(adapter,direction,step)
    assert residual_rms(copy,features,weights) == pytest.approx(.05,abs=1e-7)
    assert float(copy(features).abs().max())<=.15
    assert all(torch.equal(original[k],v) for k,v in adapter.state_dict().items())
    step,info = match_calibration_budget(adapter,torch.zeros_like(direction),features,weights,.05)
    assert step == 0 and info["status"] == "ZERO_DIRECTION"


def test_state_layout_conflicts_and_zero_gradients_are_distinct():
    values = [torch.tensor([1.,0.]),torch.tensor([-1.,0.]),torch.zeros(2)]
    result = conflict_summary(values,[1,2,3])
    assert result["per_state"]["defined_pairs"] == 1
    assert result["per_state"]["negative_pairs"] == 1
    assert result["per_state"]["zero_vectors"] == 1
    assert result["per_state"]["cancellation_ratio"] == 0.


def test_synthetic_full_diagnostic_never_accesses_audit_or_changes_sources(tmp_path,monkeypatch):
    from scripts.diagnose_ppcsa_shared_update import run_diagnostics
    from src.safety.ppcsa_one_step import capture_packet,evaluate_ppcsa_normal_shadow,integrity_token
    from tests.test_ppcsa_contracts import normal_environment
    class LinearValue(torch.nn.Module):
        def forward(self,context,action):
            return action.sum(dim=(-2,-1))
    env = normal_environment()
    env.obstacles = []; env._reset_obstacle_tracks()
    data = {"states":[],"records":[]}
    for seed in (121000,121024):
        packet = capture_packet(env,seed,torch.zeros(1,3,128))
        packet.update(base_mu=torch.zeros(3,2),hidden_features=torch.randn(3,128),context=np.zeros(4),
            parity_passed=True,factual_supported=True,step=0)
        result = evaluate_ppcsa_normal_shadow(env,np.zeros((3,2))).payload()
        data["records"].append({"layout":seed,"step":0,"state_index":len(data["states"]),
            "best_local_delta":np.ones((3,2),dtype=np.float32)*.05,
            "candidates":[{"kind":"base","result":result}]})
        data["states"].append(packet)
    # A sentinel that would fail if Audit packet content were accessed.
    data["states"].append(None)
    data["records"].append({"layout":121032,"state_index":2})
    before = deepcopy(data)
    model = LinearValue()
    artifact = {"context_mean":torch.zeros(4),"context_scale":torch.ones(4),"target_mean":0.,"target_scale":1.}
    report = run_diagnostics(data,{"partitions":{"fit":[121000],"calibration":[121024],"audit":[121032]}},
        {"ordinary":model,"paired":model},{"ordinary":artifact,"paired":artifact},env,"cpu")
    assert report["analyzed_states"] == 2 and report["audit_used"] is False
    assert report["source_adapter_unchanged"] and len(report["parameter_probes"]) == 12
    assert len(report["action_probes"]) == 2*3*5
    assert {r["direction"] for r in report["action_probes"]} == {"ordinary","paired","kinematic","random","goal"}
    assert all(r["layout"] in (121000,121024) for r in report["action_probes"])
    for model in report["models"].values():
        assert model["calibration_states"] == 1 and model["calibration_layouts"] == [121024]
    assert data["states"][2] is None and before["records"][2] == data["records"][2]
    for i in (0,1):
        torch.testing.assert_close(before["states"][i]["base_mu"],data["states"][i]["base_mu"])
        np.testing.assert_array_equal(before["records"][i]["best_local_delta"],data["records"][i]["best_local_delta"])
    # Exercise the real CLI orchestration with synthetic files and zero models,
    # never loading the registered parent checkpoint or fitting a model.
    import scripts.diagnose_ppcsa_shared_update as cli
    from types import SimpleNamespace
    from src.algorithms.ppcsa_protocol import file_sha
    from src.algorithms.ppcsa_executed_value import ExecutedValueCritic
    g1,crit,out = (tmp_path/name for name in ("gate1","critics","diagnostic"))
    g1.mkdir(); crit.mkdir()
    torch.save(data,g1/"opportunities.pt")
    parent = tmp_path/"parent.stub"; parent.write_bytes(b"synthetic parent; not a checkpoint")
    source = {"partitions":{"fit":[121000],"calibration":[121024],"audit":[121032]},
        "checkpoint":str(parent),"checkpoint_sha256":file_sha(parent),"gate0_dir":str(tmp_path/"gate0")}
    for name in ("ordinary","paired"):
        frozen = ExecutedValueCritic(4)
        with torch.no_grad():
            for p in frozen.parameters(): p.zero_()
        torch.save({**artifact,"context_dim":4,"objective":name,"partitions":source["partitions"],"state_dict":frozen.state_dict()},crit/f"{name}.pt")
    monkeypatch.setattr(cli,"validate_inputs",lambda *_:(source,{}))
    monkeypatch.setattr(cli,"runtime",lambda *_:(_ for _ in ()).throw(AssertionError("Preflight must not instantiate models/env")))
    args = ["--gate1-dir",str(g1),"--critic-dir",str(crit),"--output-dir",str(out)]
    assert cli.main(args+["--preflight-only"])["diagnostic_run"] is False
    assert not out.exists()
    actor = torch.nn.Linear(1,1)
    monkeypatch.setattr(cli,"runtime",lambda *_:(SimpleNamespace(actor=actor),env,{}))
    result = cli.main(args)
    assert result["source_actor_unchanged"] and result["source_artifacts_unchanged"]
    assert not result["audit_used"] and result["used_partitions"] == ["fit","calibration"]
    assert (out/"shared_mechanism_report.json").exists() and not list(out.rglob("*.pt"))
    with pytest.raises(ValueError,match="new and separate"):
        cli.main(args)
