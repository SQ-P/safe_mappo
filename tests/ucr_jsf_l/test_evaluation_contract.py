from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import torch

from src.algorithms.factory import create_trainer
from src.algorithms.ppo.trainer import _registered_safety_executor_transition
from src.algorithms.ppo.trainer import PPOTrainer
from src.safety.config_guard import validate_active_research_config
from scripts.evaluate_ppo import (
    _normalize_disabled_oracle_for_safety_executor_transition,
)

from .conftest import load_ucr_config


def test_config_rejects_legacy_stage_in_new_executor():
    config = load_ucr_config()
    validate_active_research_config(config)
    bad = deepcopy(config)
    bad["safety_execution"]["initial_joint_guard"] = {"enabled": True}
    try:
        validate_active_research_config(bad)
    except ValueError as exc:
        assert "legacy safety stages" in str(exc)
    else:
        raise AssertionError("legacy stage should be rejected")


def test_checkpoint_transition_allows_only_safety_contract_change():
    child = {
        "task_contract_version": 4,
        "env": {"num_agents": 3},
        "oracle_subgoal": {"enabled": False},
        "reward": {"step_penalty": 0.002},
        "safety_execution": {
            "enabled": True,
            "version": "ucr_jsf_l_v1",
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
        "runtime": "ucr_jsf_l_v1",
    }
    changed_env = deepcopy(child)
    changed_env["env"]["num_agents"] = 4
    assert _registered_safety_executor_transition(parent, changed_env) is None
    generic = deepcopy(child)
    generic["safety_execution"]["version"] = "unknown"
    assert _registered_safety_executor_transition(parent, generic) is None
    unsupported = deepcopy(parent)
    unsupported["task_contract_version"] = 6
    assert _registered_safety_executor_transition(unsupported, child) is None


def test_executor_transition_is_rejected_by_training_entry_without_training():
    trainer = object.__new__(PPOTrainer)
    trainer.recovery_study_evaluation_only = False
    trainer.safety_executor_transition_used = True
    try:
        trainer.train()
    except ValueError as exc:
        assert "evaluation-only" in str(exc)
    else:
        raise AssertionError("executor transition must not enter training")


def test_evaluator_normalizes_only_omitted_generation_flag_for_oracle_off():
    historical_off = {"oracle_subgoal": {"enabled": False, "recovery_enabled": False}}
    _normalize_disabled_oracle_for_safety_executor_transition(historical_off)
    assert historical_off["oracle_subgoal"]["generation_enabled"] is False

    explicit_on = {"oracle_subgoal": {"enabled": False, "generation_enabled": True}}
    _normalize_disabled_oracle_for_safety_executor_transition(explicit_on)
    assert explicit_on["oracle_subgoal"]["generation_enabled"] is True

    oracle_on = {"oracle_subgoal": {"enabled": True}}
    _normalize_disabled_oracle_for_safety_executor_transition(oracle_on)
    assert "generation_enabled" not in oracle_on["oracle_subgoal"]


def test_real_off_checkpoint_loads_read_only_with_registered_transition():
    checkpoint = Path(
        "results/ablations/mappo_safety_no_subgoal_100/"
        "seed_23_u400_v1/checkpoints/update_0800.pt"
    )
    if not checkpoint.is_file():
        return
    config = load_ucr_config()
    config["device"] = "cpu"
    trainer = create_trainer(config, project_root=Path.cwd())
    before = checkpoint.stat().st_mtime_ns
    hash_before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert _registered_safety_executor_transition(
        payload["task_config"], trainer._task_config_payload()
    ) == {
        "checkpoint": "staged_joint_guard_ttc_v1",
        "runtime": "ucr_jsf_l_v1",
    }
    trainer.load_checkpoint(
        checkpoint, allow_safety_executor_transition=True
    )
    assert trainer.safety_executor_transition_used
    assert checkpoint.stat().st_mtime_ns == before
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == hash_before
