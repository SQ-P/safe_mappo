from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import threading

import numpy as np
import pytest
import torch

from scripts.run_cross8u_160 import (
    ROOT, load_config, environment_config, create_cross_trainer, isolated_environment_injection,
    read_panels, layout_diagnostics, main)
from src.envs.cross8u_160_env import Cross8U160Env, STARTS, GOALS, initial_geometric_routes
from src.envs.cooperative_uav_env import CooperativeUAVEnv
from src.envs.entities import CircularObstacle
from src.utils.io_utils import load_yaml


def config():
    return load_config("configs/verification/cross8u_160/seed23.yaml")


def test_complete_snapshots_keep_original_training_reward_safety_and_independent_panels():
    original = load_yaml(ROOT/"configs/experiments/standard_mappo_3u_8s2d_seed23.yaml")
    new = config()
    assert load_yaml(ROOT/"configs/verification/cross8u_160/base.yaml") == new
    for section in ("train","model","reward","safety_execution","oracle_subgoal","subgoal_selector","multi_agent","observation_normalization"):
        assert new[section] == original[section]
    assert "extends:" not in (ROOT/"configs/verification/cross8u_160/seed23.yaml").read_text()
    dev, formal = read_panels()
    assert len(dev) == 40 and len(formal) == 200 and not set(dev)&set(formal)
    assert new["output"]["output_dir"].startswith("results/verification/cross8u_160/")


def test_explicit_eight_endpoints_have_no_jitter_and_dimensions_match():
    env = Cross8U160Env(environment_config(config()))
    observation, info = env.reset(seed=240000)
    assert np.array_equal(env.positions, STARTS) and np.array_equal(env.goals, GOALS)
    assert observation.shape == (8,80) and env.get_global_state().shape == (640,)
    assert env.action_space.shape == (8,2) and len(env.obstacles) == 10
    assert sum(o.dynamic for o in env.obstacles) == 6
    assert all(45 <= o.position[0] <= 115 and 45 <= o.position[1] <= 115 and 2 <= o.radius <= 5 for o in env.obstacles)
    assert layout_diagnostics(env)["passed"]
    assert info["cross8u_directional"]["vertical_group_mean_fraction"] == 0.
    assert env.initial_joint_guard_enabled and env.obstacle_execution_guard_enabled and env.final_joint_guard_enabled
    assert env.safety_recovery_enabled


def test_seed_reproducibility_and_reset_without_explicit_seed():
    left = Cross8U160Env(environment_config(config())); right = Cross8U160Env(environment_config(config()))
    def snapshot(env, seed=None):
        obs, _ = env.reset(seed=seed)
        return obs, np.array([list(o.position)+list(o.velocity)+[o.radius] for o in env.obstacles])
    a, b = snapshot(left,240003), snapshot(right,240003)
    assert all(np.array_equal(x,y) for x,y in zip(a,b))
    a, b = snapshot(left), snapshot(right)
    assert all(np.array_equal(x,y) for x,y in zip(a,b))
    assert not np.array_equal(snapshot(left,240004)[1],snapshot(left,240005)[1])


def test_routes_do_not_claim_joint_dynamic_safety_and_reject_a_geometric_wall():
    wall = [CircularObstacle(np.array([80.,y]),12.,np.zeros(2),False) for y in range(0,161,15)]
    assert not initial_geometric_routes(wall)["all_routes_exist"]
    assert initial_geometric_routes([])["all_routes_exist"]


def test_both_direction_groups_use_goal_based_telemetry_and_inherited_rewards():
    env = Cross8U160Env(environment_config(config())); env.reset(seed=240000)
    for i, agent in enumerate(env.agents): agent.position = (STARTS[i]+.5*(GOALS[i]-STARTS[i])).copy()
    diag = env.directional_diagnostics()
    np.testing.assert_allclose(diag["goal_axis_fraction"],.5)
    assert diag["horizontal_group_mean_fraction"] == diag["vertical_group_mean_fraction"] == .5
    assert Cross8U160Env._team_goal_potential is CooperativeUAVEnv._team_goal_potential
    assert Cross8U160Env._team_goal_completion is CooperativeUAVEnv._team_goal_completion


@pytest.mark.parametrize("policy", ["random","goal"])
def test_short_transition_preserves_finite_state_observation_and_team_reward(policy):
    env = Cross8U160Env(environment_config(config())); env.reset(seed=240000)
    rng = np.random.default_rng(3)
    for _ in range(3):
        action = rng.uniform(-1,1,(8,2)) if policy == "random" else np.clip(env.goals-env.positions,-1,1)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (8,80) and np.isfinite(obs).all() and np.isfinite(env.get_global_state()).all()
        assert np.isfinite(reward).all() and np.all(reward == reward[0])
        assert len(info["cross8u_directional"]["goal_axis_fraction"]) == 8
        if terminated or truncated: break


def test_injection_restores_original_symbol_on_success_exception_and_rejects_threads(monkeypatch):
    import src.algorithms.ppo.trainer as module
    original = module.CooperativeUAVEnv
    with isolated_environment_injection(): assert module.CooperativeUAVEnv is Cross8U160Env
    assert module.CooperativeUAVEnv is original
    with pytest.raises(RuntimeError, match="synthetic"):
        with isolated_environment_injection(): raise RuntimeError("synthetic")
    assert module.CooperativeUAVEnv is original
    errors = []
    def worker():
        try:
            with isolated_environment_injection(): pass
        except RuntimeError as error: errors.append(str(error))
    thread = threading.Thread(target=worker); thread.start(); thread.join()
    assert errors and module.CooperativeUAVEnv is original


def test_fresh_untrained_mappo_initialization_and_forward_shapes(tmp_path,monkeypatch):
    import src.algorithms.ppo.trainer as module
    original = module.CooperativeUAVEnv
    new = config(); new["device"] = "cpu"
    new["output"] = {"output_dir":str(tmp_path/"fresh"),"checkpoint_dir":str(tmp_path/"fresh/checkpoints"),"reject_nonempty":True}
    monkeypatch.setattr(torch,"load",lambda *a,**k:(_ for _ in ()).throw(AssertionError("Historical checkpoint loaded")))
    trainer = create_cross_trainer(new,ROOT)
    assert module.CooperativeUAVEnv is original and isinstance(trainer._base_env(),Cross8U160Env)
    assert trainer.current_update == 0 and trainer.num_agents == 8 and trainer.obs_dim == 80 and trainer.global_state_dim == 640
    observations, _ = trainer.env.reset(seed=240000)
    hidden = trainer.actor.initial_hidden(8)
    with torch.no_grad():
        output = trainer.actor.forward_step(torch.tensor(trainer.normalize_local_observations(observations)),hidden,deterministic=True)
        values = trainer.critic(torch.tensor(trainer.normalize_global_state(trainer.env.get_global_state())).unsqueeze(0))
    assert output.action.shape == (8,2) and output.hidden.shape == (1,8,128)
    assert torch.isfinite(output.action).all() and torch.isfinite(values).all()
    assert not list((tmp_path/"fresh").rglob("*.pt"))


@pytest.mark.parametrize("mode",["train","evaluate"])
def test_p2_and_p3_cannot_start_from_p0_p1_entry(mode):
    with pytest.raises(RuntimeError,match="P0/P1"): main(["--mode",mode])


def test_changed_scenario_cannot_silently_fall_back_to_old_environment():
    new = environment_config(config()); new["num_agents"] = 3
    with pytest.raises(ValueError): Cross8U160Env(new)
    assert main(["--mode","preflight"])["training_updates"] == 0


def smoke_config():
    return load_config("configs/verification/cross8u_160/smoke_seed23.yaml")


@pytest.mark.parametrize("section,key,value",[
    ("train","total_updates",2),("train","rollout_steps",129),("train","ppo_epochs",2),
    ("evaluation","episodes",2),("evaluation","seed_base",241000),
    ("env","max_episode_steps",501),("verification_experiment","central_probe_seeds",[241000]),
    ("verification_experiment","p3_authorized",True)])
def test_p2_rejects_budget_expansion_and_formal_panel(section,key,value):
    from scripts.run_cross8u_160 import smoke_config_contract
    new = smoke_config(); new[section][key] = value
    with pytest.raises(ValueError,match="exact one-update"): smoke_config_contract(new)


def test_p2_preflight_does_not_construct_trainer_or_run_probes(monkeypatch):
    import scripts.run_cross8u_160 as runner
    def forbidden(*a,**k): raise AssertionError("Preflight executed a model or trajectory")
    monkeypatch.setattr(runner,"create_cross_trainer",forbidden)
    monkeypatch.setattr(runner,"central_probes",forbidden)
    result = main(["--mode","smoke","--config","configs/verification/cross8u_160/smoke_seed23.yaml","--preflight-only"])
    assert result["training_updates"] == 0 and result["limits"]["evaluation_episodes"] == 1
    assert result["limits"]["central_probe_seeds"] == [240000,240001]


def test_p2_gradient_audit_observes_changes_and_restores_on_exception():
    from scripts.run_cross8u_160 import gradient_audit, parameter_delta
    actor, critic = torch.nn.Linear(2,1), torch.nn.Linear(2,1)
    actor_optimizer, critic_optimizer = torch.optim.Adam(actor.parameters()), torch.optim.Adam(critic.parameters())
    trainer = SimpleNamespace(actor_optimizer=SimpleNamespace(optimizers=[actor_optimizer]),critic_optimizer=critic_optimizer)
    originals = [actor_optimizer.step,critic_optimizer.step]; evidence = []
    before = {n:p.detach().clone() for n,p in actor.named_parameters()}
    with pytest.raises(RuntimeError,match="synthetic exit"):
        with gradient_audit(trainer,evidence):
            for module, optimizer in ((actor,actor_optimizer),(critic,critic_optimizer)):
                module(torch.ones(1,2)).sum().backward(); optimizer.step()
            raise RuntimeError("synthetic exit")
    assert [actor_optimizer.step,critic_optimizer.step] == originals
    assert {item["role"] for item in evidence} == {"actor","critic"}
    assert all(item["post_clip_gradient_l2"] > 0 for item in evidence)
    assert parameter_delta(actor,before) > 0


def test_p2_nonfinite_gradient_is_blocked_before_parameter_update():
    from scripts.run_cross8u_160 import gradient_audit
    module = torch.nn.Linear(1,1); optimizer = torch.optim.Adam(module.parameters())
    trainer = SimpleNamespace(actor_optimizer=SimpleNamespace(optimizers=[optimizer]),critic_optimizer=optimizer)
    before = [p.detach().clone() for p in module.parameters()]
    for p in module.parameters(): p.grad = torch.full_like(p,float("nan"))
    with pytest.raises(FloatingPointError,match="nonfinite"):
        with gradient_audit(trainer,[]): optimizer.step()
    assert all(torch.equal(p,b) for p,b in zip(module.parameters(),before))


def test_p2_diagnostic_serialization_preserves_arrays_and_marks_undefined_telemetry():
    from scripts.run_cross8u_160 import diagnostic_json
    row = diagnostic_json({"actions":np.zeros((8,2)),"undefined":float("nan"),"clearance":float("inf")})
    assert len(row["actions"]) == 8 and row["undefined"] == "NaN" and row["clearance"] == "+Infinity"
    json.dumps(row,allow_nan=False)


def test_p2_central_probe_bounds_and_trace_with_synthetic_environment(tmp_path,monkeypatch):
    import scripts.run_cross8u_160 as runner
    class FakeEnv:
        agent_max_acceleration = 1.; dt = .5
        def __init__(self,*args): pass
        def reset(self,seed):
            self.positions = np.full((8,2),50.); self.goals = np.full((8,2),100.)
            self.velocities = np.zeros((8,2)); self.obstacles = [SimpleNamespace(dynamic=True,position=np.array([116.,60.]))]
        def get_global_state(self): return np.zeros(640)
        def step(self,action):
            return np.zeros((8,80)),np.zeros(8),False,False,{
                "cross8u_directional":{},"action_stage_actions":{"executed":action},
                "initial_joint_guard_intervention_steps":[1],"collision_count":0}
    monkeypatch.setattr(runner,"Cross8U160Env",FakeEnv)
    report = runner.central_probes(smoke_config(),tmp_path)
    rows = [json.loads(line) for line in (tmp_path/"central_trajectories.jsonl").read_text().splitlines()]
    assert len(rows) == report["total_steps"] == 400
    assert {row["seed"] for row in rows} == {240000,240001}
    assert max(row["step"] for row in rows) == 200
    assert report["central_interaction_observed"]
    assert all(s["dynamic_left_initial_spawn_region"] for s in report["seeds"])


@pytest.mark.parametrize("second_evaluation",[False,True])
def test_p2_train_orchestration_with_synthetic_trainer_only(tmp_path,monkeypatch,second_evaluation):
    import scripts.run_cross8u_160 as runner
    class FakeTrainer:
        current_update = 0
        def __init__(self):
            self.actor, self.critic = torch.nn.Linear(2,1), torch.nn.Linear(2,1)
            self.actor_optimizer = SimpleNamespace(optimizers=[torch.optim.Adam(self.actor.parameters())])
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
            self.output_dir = tmp_path; self.checkpoint_dir = tmp_path/"checkpoints"
        def evaluate(self,**kwargs):
            (tmp_path/"evaluation_episodes.jsonl").write_text(json.dumps({"seed":240000,"episode_length":500})+"\n")
        def train(self):
            for module,optimizer in ((self.actor,self.actor_optimizer.optimizers[0]),(self.critic,self.critic_optimizer)):
                module(torch.ones(1,2)).sum().backward(); optimizer.step()
            self.current_update = 1
            self.evaluate(episodes=1,seed_base=240000)
            if second_evaluation: self.evaluate(episodes=1,seed_base=240000)
            return {"updates_completed":1,"environment_steps":128,"policy_loss":.1,"value_loss":.2,
                "actor_grad_norm":.3,"critic_grad_norm":.4}
    trainer = FakeTrainer(); original = trainer.evaluate
    monkeypatch.setattr(runner,"create_cross_trainer",lambda *a:trainer)
    reload_calls = []
    monkeypatch.setattr(runner,"strict_smoke_reload",lambda *a:reload_calls.append(a) or {"strict_reload":True})
    if second_evaluation:
        with pytest.raises(RuntimeError,match="one-development-episode"): runner.smoke_train_once(smoke_config(),tmp_path)
        assert not reload_calls
    else:
        report = runner.smoke_train_once(smoke_config(),tmp_path)
        assert report["training_updates"] == 1 and len(report["evaluations"]) == 1 and len(reload_calls) == 1
        assert all(delta > 0 for delta in report["parameter_delta_l2"].values())
    assert trainer.evaluate == original


def test_p2_strict_reload_rejects_wrong_8u_metadata_without_constructing_trainer(tmp_path,monkeypatch):
    import scripts.run_cross8u_160 as runner
    checkpoint = tmp_path/"synthetic.pt"
    torch.save({"algorithm":"mappo","update":1,"num_agents":3},checkpoint)
    monkeypatch.setattr(runner,"create_cross_trainer",lambda *a:pytest.fail("Wrong metadata reached trainer construction"))
    with pytest.raises(RuntimeError,match="num_agents"):
        runner.strict_smoke_reload(None,checkpoint,smoke_config(),tmp_path)
    assert runner.tensor_tree_equal({"a":[torch.ones(2)]},{"a":[torch.ones(2)]})
    assert not runner.tensor_tree_equal({"a":[torch.ones(2)]},{"a":[torch.zeros(2)]})


def test_p2_existing_outputs_cannot_resume_or_overwrite(tmp_path,monkeypatch):
    import scripts.run_cross8u_160 as runner
    new = smoke_config()
    monkeypatch.setattr(runner,"smoke_config_contract",lambda c:{"updates":1})
    monkeypatch.setattr(runner,"OUTPUT_ROOT",tmp_path)
    output = tmp_path/"smoke/previous"; output.mkdir(parents=True)
    with pytest.raises(FileExistsError,match="no automatic overwrite"): runner.run_smoke(new,output)


@pytest.mark.parametrize("historical_drift",[False,True])
def test_p2_failure_stops_without_training_or_retry_and_preserves_history(tmp_path,monkeypatch,historical_drift):
    import scripts.run_cross8u_160 as runner
    manifest_dir = tmp_path/"results/verification/cross8u_160/p2_implementation_v1"
    manifest_dir.mkdir(parents=True)
    original = tmp_path/"old.txt"; original.write_text("historical evidence")
    (manifest_dir/"source_integrity_before.json").write_text(json.dumps({"protected_file_sha256":{
        "old.txt":"wrong" if historical_drift else runner.sha(original)}}))
    new = smoke_config()
    monkeypatch.setattr(runner,"ROOT",tmp_path); monkeypatch.setattr(runner,"OUTPUT_ROOT",tmp_path/"results/verification/cross8u_160")
    monkeypatch.setattr(runner,"smoke_config_contract",lambda c:{"updates":1})
    monkeypatch.setattr(runner,"sha",lambda p:"synthetic" if "scripts" in p.parts or "src" in p.parts else __import__("hashlib").sha256(p.read_bytes()).hexdigest())
    calls = []
    def broken_probe(*args): calls.append("probe"); raise RuntimeError("synthetic probe failure")
    monkeypatch.setattr(runner,"central_probes",broken_probe)
    monkeypatch.setattr(runner,"smoke_train_once",lambda *a:pytest.fail("Training after a failed gate"))
    output = runner.OUTPUT_ROOT/"smoke/new"
    with pytest.raises(RuntimeError,match="Protected" if historical_drift else "synthetic probe failure"):
        runner.run_smoke(new,output)
    assert original.read_text() == "historical evidence"
    if historical_drift:
        assert not calls and not output.exists()
    else:
        failure = json.loads((output/"p2_failure.json").read_text())
        assert calls == ["probe"] and failure["automatic_retry"] is False
