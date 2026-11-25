from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import torch

from src.algorithms.factory import create_trainer
from src.algorithms.ppo.trainer import PPOTrainer, _registered_safety_executor_transition
from src.safety.config_guard import validate_active_research_config

from .conftest import load_v2_config, make_v2_env


def test_v2_config_and_environment_use_only_v2_single_selection(monkeypatch):
    config = load_v2_config()
    validate_active_research_config(config)
    env = make_v2_env()
    observations, _ = env.reset(seed=120_023)
    assert observations.shape == (3, 52)
    assert (env.static_obstacle_num, env.dynamic_obstacle_num, env.obstacle_slots) == (8, 2, 6)
    assert not env.oracle_enabled
    assert env.ucr_jsf_l_v2_enabled and not env.ucr_jsf_l_v1_enabled

    def legacy_projection_forbidden(*_args, **_kwargs):
        raise AssertionError("legacy projection was called by UCR-JSF-L V2")

    monkeypatch.setattr(env.boundary_guard, "project_actions", legacy_projection_forbidden)
    *_unused, info = env.step(np.full((3, 2), [0.25, 0.0], dtype=np.float32))
    assert info["safety_execution_version"] == "ucr_jsf_l_v2"
    assert info["joint_safety_trace"].schema_version == "JointSafetyTraceV2"
    assert info["joint_safety_trace"].metadata["candidate_id_used_for_decision"] is False
    assert info["stage_intervention_trace"] is None


def test_v2_config_rejects_budget_drift():
    config = load_v2_config()
    bad = deepcopy(config)
    bad["safety_execution"]["candidates"]["pair_candidate_quota"] = 16
    try:
        validate_active_research_config(bad)
    except ValueError as exc:
        assert "candidate budget mismatch" in str(exc)
    else:
        raise AssertionError("V2 budget drift should be rejected")


def test_transition_registry_is_explicit_for_v1_and_v2_only():
    child = {
        "task_contract_version": 4,
        "env": {"num_agents": 3},
        "oracle_subgoal": {"enabled": False},
        "reward": {"step_penalty": 0.002},
        "safety_execution": {
            "enabled": True,
            "version": "ucr_jsf_l_v2",
            "stages": ["joint_safety_selection"],
        },
    }
    parent = deepcopy(child)
    parent["task_contract_version"] = 5
    parent["safety_execution"] = {
        "enabled": True,
        "version": "staged_joint_guard_ttc_v1",
        "stages": ["initial_joint_guard", "obstacle_ttc", "final_joint_guard"],
    }
    assert _registered_safety_executor_transition(parent, child) == {
        "checkpoint": "staged_joint_guard_ttc_v1",
        "runtime": "ucr_jsf_l_v2",
    }
    v1 = deepcopy(child)
    v1["safety_execution"]["version"] = "ucr_jsf_l_v1"
    assert _registered_safety_executor_transition(parent, v1)["runtime"] == "ucr_jsf_l_v1"
    arbitrary = deepcopy(child)
    arbitrary["safety_execution"]["version"] = "ucr_jsf_l_v3"
    assert _registered_safety_executor_transition(parent, arbitrary) is None


def test_transition_cannot_enter_training():
    trainer = object.__new__(PPOTrainer)
    trainer.recovery_study_evaluation_only = False
    trainer.safety_executor_transition_used = True
    try:
        trainer.train()
    except ValueError as exc:
        assert "evaluation-only" in str(exc)
    else:
        raise AssertionError("registered transition must stay evaluation-only")


def test_real_u800_checkpoint_loads_read_only_for_v2():
    checkpoint = Path(
        "results/ablations/mappo_safety_no_subgoal_100/"
        "seed_23_u400_v1/checkpoints/update_0800.pt"
    )
    if not checkpoint.is_file():
        return
    config = load_v2_config()
    config["device"] = "cpu"
    trainer = create_trainer(config, project_root=Path.cwd())
    before_time = checkpoint.stat().st_mtime_ns
    before_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert _registered_safety_executor_transition(
        payload["task_config"], trainer._task_config_payload()
    )["runtime"] == "ucr_jsf_l_v2"
    trainer.load_checkpoint(checkpoint, allow_safety_executor_transition=True)
    assert trainer.safety_executor_transition_used
    assert checkpoint.stat().st_mtime_ns == before_time
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before_hash
