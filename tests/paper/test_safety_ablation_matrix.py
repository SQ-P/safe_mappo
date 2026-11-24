from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import src.envs.cooperative_uav_env as env_module
from src.algorithms.ppo.trainer import _registered_safety_ablation_changes
from src.envs.cooperative_uav_env import CooperativeUAVEnv
from src.utils.io_utils import load_yaml


ARMS = {
    "A0": (False, False, False, False),
    "A1": (True, False, True, False),
    "A2": (False, True, False, False),
    "A3": (True, True, True, False),
    "A4": (True, True, True, True),
}


def _config(arm: str) -> dict:
    return load_yaml(Path("configs/ablations/safety_seed23") / f"{arm}.yaml")


def _env(arm: str) -> CooperativeUAVEnv:
    config = _config(arm)
    return CooperativeUAVEnv({
        **config["env"],
        "reward": config["reward"],
        "oracle_subgoal": config["oracle_subgoal"],
        "subgoal_selector": config["subgoal_selector"],
        "safety_execution": config["safety_execution"],
    })


@pytest.mark.parametrize("arm", tuple(ARMS))
def test_safety_ablation_arm_contracts(arm):
    env = _env(arm)
    initial, ttc, final, recovery = ARMS[arm]
    assert env.safety_execution_enabled is True
    assert env.initial_joint_guard_enabled is initial
    assert env.obstacle_execution_guard_enabled is ttc
    assert env.final_joint_guard_enabled is final
    assert env.safety_recovery_enabled is recovery


def test_a2_runs_ttc_without_any_guard_projection(monkeypatch):
    env = _env("A2")
    env.reset(seed=119000)
    calls = {"guard": 0, "ttc": 0}
    original_guard = env.boundary_guard.project_actions
    original_ttc = env_module.project_obstacle_actions

    def counted_guard(*args, **kwargs):
        calls["guard"] += 1
        return original_guard(*args, **kwargs)

    def counted_ttc(*args, **kwargs):
        calls["ttc"] += 1
        return original_ttc(*args, **kwargs)

    monkeypatch.setattr(env.boundary_guard, "project_actions", counted_guard)
    monkeypatch.setattr(env_module, "project_obstacle_actions", counted_ttc)
    env.step(np.zeros((env.num_agents, 2), dtype=np.float32))
    assert calls == {"guard": 0, "ttc": 1}


def test_a0_has_no_safety_projection(monkeypatch):
    env = _env("A0")
    env.reset(seed=119000)

    def forbidden(*args, **kwargs):
        raise AssertionError("disabled safety module was invoked")

    monkeypatch.setattr(env.boundary_guard, "project_actions", forbidden)
    monkeypatch.setattr(env_module, "project_obstacle_actions", forbidden)
    env.step(np.zeros((env.num_agents, 2), dtype=np.float32))


def test_a4_preserves_the_registered_full_safety_contract():
    assert _config("A4")["safety_execution"] == load_yaml(
        "configs/active/standard_mappo_3u_8s2d.yaml"
    )["safety_execution"]


def test_checkpoint_override_accepts_only_registered_safety_switches():
    parent = {"task_contract_version": 4, "env": {"num_agents": 3},
              "safety_execution": _config("A4")["safety_execution"]}
    child = deepcopy(parent)
    child["safety_execution"] = _config("A2")["safety_execution"]
    changes = _registered_safety_ablation_changes(parent, child)
    assert changes is not None
    assert set(changes) == {
        "safety_execution.recovery_enabled",
        "safety_execution.initial_joint_guard.enabled",
        "safety_execution.initial_joint_guard.action_shield_enabled",
        "safety_execution.final_joint_guard.enabled",
    }
    child["env"]["num_agents"] = 4
    assert _registered_safety_ablation_changes(parent, child) is None


def test_formal_panel_is_fresh_and_disjoint():
    formal = load_yaml("configs/evaluation/safety_ablation_seed23_200.json")
    development = load_yaml(
        "configs/evaluation/csa_actor_validation_119000_119039.json"
    )
    prior_formal = load_yaml(
        "configs/evaluation/csa_formal_test_120000_120199.json"
    )
    seeds = formal["layout_seeds"]
    assert seeds == list(range(121000, 121200))
    assert len(seeds) == len(set(seeds)) == 200
    assert set(seeds).isdisjoint(development["layout_seeds"])
    assert set(seeds).isdisjoint(prior_formal["layout_seeds"])
