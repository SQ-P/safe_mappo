"""No-planner ablation checks; no real optimizer updates or training runs."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.algorithms.factory import create_trainer
from src.envs.cooperative_uav_env import CooperativeUAVEnv
from src.planning.oracle_subgoal_planner import OracleSubgoalPlanner
from src.utils.io_utils import load_yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/experiments/mappo_safety_no_subgoal_100_seed23_u400.yaml"


def config():
    return load_yaml(ROOT/CONFIG)


def test_original_geometry_rewards_and_safety_preserved_with_planner_off():
    baseline = load_yaml(ROOT/"configs/experiments/standard_mappo_3u_8s2d_seed23.yaml")
    new = config()
    assert new["algorithm"] == "mappo" and new["seed"] == 23
    assert new["env"] == baseline["env"]
    assert new["env"]["map_size"] == [100.,100.] and new["env"]["num_agents"] == 3
    assert new["env"]["static_obstacle_num"] == 8 and new["env"]["dynamic_obstacle_num"] == 2
    assert new["reward"] == baseline["reward"] and new["safety_execution"] == baseline["safety_execution"]
    assert not new["oracle_subgoal"]["enabled"] and not new["oracle_subgoal"]["recovery_enabled"]
    assert baseline["oracle_subgoal"]["enabled"] and baseline["oracle_subgoal"]["recovery_enabled"]
    assert not new["subgoal_selector"]["enabled"] and new["safety_execution"]["recovery_enabled"]
    assert new["train"]["total_updates"] == 400 and new["train"]["checkpoint_updates"] == [100,200,400]
    assert new["evaluation"]["episodes"] == 10 and new["evaluation"]["updates"] == [100,200,400]
    assert new["output"]["reject_nonempty"] and new["output"]["output_dir"] != baseline["output"]["output_dir"]


@pytest.mark.parametrize("seed",[23,111000])
def test_disabled_planner_never_generates_or_replans_and_targets_final_goal(monkeypatch,seed):
    def forbidden(*a,**k): pytest.fail("Disabled Oracle generated an intermediate target")
    monkeypatch.setattr(OracleSubgoalPlanner,"plan",forbidden)
    monkeypatch.setattr(CooperativeUAVEnv,"_generate_valid_subgoals",forbidden)
    new = config(); cfg = deepcopy(new["env"])
    for key in ("reward","oracle_subgoal","subgoal_selector","safety_execution"):
        cfg[key] = deepcopy(new[key])
    env = CooperativeUAVEnv(cfg); obs,info = env.reset(seed=seed)
    assert not env.oracle_enabled and not env.oracle_recovery_enabled
    assert env.initial_joint_guard_enabled and env.obstacle_execution_guard_enabled
    assert env.final_joint_guard_enabled and env.safety_recovery_enabled
    assert len(env.obstacles) == 10 and sum(o.dynamic for o in env.obstacles) == 2
    assert obs.shape == (3,52) and env.get_global_state().shape == (156,)
    np.testing.assert_array_equal(obs[:,4:6],obs[:,6:8])
    for _ in range(22):  # Past the original20-step Oracle replanning interval.
        obs,reward,terminated,truncated,info = env.step(np.zeros((3,2),dtype=np.float32))
        np.testing.assert_array_equal(env.subgoals,env.goals)
        np.testing.assert_array_equal(obs[:,4:6],obs[:,6:8])
        assert np.isfinite(obs).all() and np.isfinite(reward).all()
        assert env.subgoal_plan_attempt_count == env.subgoal_replan_count == 0
        if terminated or truncated: break
    assert env._maybe_replan_subgoals(np.zeros(3)) is False


def test_fresh_mappo_forward_without_any_checkpoint_load(tmp_path,monkeypatch):
    new = config(); new["device"] = "cpu"
    new["output"] = {"output_dir":str(tmp_path/"fresh"),"checkpoint_dir":str(tmp_path/"fresh/checkpoints"),"reject_nonempty":True}
    monkeypatch.setattr(torch,"load",lambda *a,**k:pytest.fail("Fresh training read checkpoint weights"))
    trainer = create_trainer(new,ROOT)
    assert trainer.current_update == 0 and trainer.obs_dim == 52 and trainer.global_state_dim == 156
    assert not trainer.env.oracle_enabled and trainer.env.safety_recovery_enabled
    obs,_ = trainer.env.reset(seed=23)
    with torch.no_grad():
        action = trainer.actor.forward_step(torch.tensor(trainer.normalize_local_observations(obs)),
            trainer.actor.initial_hidden(3),deterministic=True).action
        value = trainer.critic(torch.tensor(trainer.normalize_global_state(trainer.env.get_global_state())).unsqueeze(0))
    assert action.shape == (3,2) and torch.isfinite(action).all() and torch.isfinite(value).all()
    assert not list(tmp_path.rglob("*.pt"))


def test_generic_training_command_dispatch_with_fake_trainer_only(monkeypatch):
    import scripts.train_ppo as entry
    received = []
    def fake_factory(cfg,project_root):
        received.append(cfg)
        return SimpleNamespace(train=lambda:{"fake_dispatch_only":True})
    monkeypatch.setattr(entry,"create_trainer",fake_factory)
    assert entry.main(["--config",CONFIG]) == {"fake_dispatch_only":True}
    assert len(received) == 1 and received[0]["train"]["total_updates"] == 400
    assert not received[0]["oracle_subgoal"]["enabled"]
