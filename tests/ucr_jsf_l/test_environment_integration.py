from __future__ import annotations

import numpy as np

from src.envs.cooperative_uav_env import CooperativeUAVEnv
from src.safety.config_guard import validate_active_research_config

from .conftest import load_ucr_config, make_ucr_env


def test_exact_no_oracle_environment_contract_and_one_final_action(monkeypatch):
    config = load_ucr_config()
    validate_active_research_config(config)
    env = make_ucr_env()
    observations, _ = env.reset(seed=120_001)
    assert observations.shape == (3, 52)
    assert env.static_obstacle_num == 8
    assert env.dynamic_obstacle_num == 2
    assert env.obstacle_slots == 6
    assert not env.oracle_enabled
    assert not env.subgoal_generation_enabled
    assert not env.oracle_recovery_enabled
    np.testing.assert_array_equal(env.subgoals, env.goals)

    def legacy_projection_forbidden(*_args, **_kwargs):
        raise AssertionError("legacy joint guard was called by UCR-JSF-L")

    monkeypatch.setattr(
        env.boundary_guard, "project_actions", legacy_projection_forbidden
    )
    _obs, _reward, _terminated, _truncated, info = env.step(
        np.full((3, 2), [0.25, 0.0], dtype=np.float32)
    )
    assert info["safety_execution_version"] == "ucr_jsf_l_v1"
    assert info["joint_safety_trace"].schema_version == "JointSafetyTraceV1"
    assert info["stage_intervention_trace"] is None
    assert sum(info["ucr_jsf_l_mode_step_counts"].values()) == 1
    assert info["ucr_jsf_l_search_exhaustion_count"] in {0, 1}
    assert 0.0 <= info["ucr_jsf_l_stall_step_ratio"] <= 1.0
    assert 0.0 <= info["ucr_jsf_l_candidate_unknown_rate"] <= 1.0
    assert 0.0 <= info["ucr_jsf_l_joint_intervention_rate"] <= 1.0
    assert info["ucr_jsf_l_passing_side_flip_count"] >= 0
    assert set(info["action_stage_actions"]) == {
        "actor_raw", "ucr_jsf_l_final", "executed"
    }


def test_legacy_executor_still_uses_legacy_trace():
    from src.utils.io_utils import load_yaml
    from copy import deepcopy

    config = load_yaml("configs/active/standard_mappo_3u_8s2d.yaml")
    env_config = deepcopy(config["env"])
    for key in ("reward", "oracle_subgoal", "subgoal_selector", "safety_execution"):
        env_config[key] = deepcopy(config.get(key, {}))
    env = CooperativeUAVEnv(env_config)
    env.reset(seed=120_002)
    *_unused, info = env.step(np.zeros((3, 2), dtype=np.float32))
    assert info["safety_execution_version"] == "staged_joint_guard_ttc_v1"
    assert info["joint_safety_trace"] is None
    assert info["stage_intervention_trace"].metadata[
        "safety_execution_version"
    ] == "staged_joint_guard_ttc_v1"
