import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.algorithms.factory import create_trainer
from src.envs.cooperative_uav_env import CooperativeUAVEnv
from src.safety.config_guard import (
    validate_active_research_config,
    validate_checkpoint_research_line,
)
from src.utils.io_utils import load_yaml


def test_active_config_rejects_archived_research_switches():
    base = {
        "oracle_subgoal": {"enabled": True, "frozen": True},
        "subgoal_selector": {"enabled": False},
        "safety_execution": {"enabled": True, "version": "staged_joint_guard_ttc_v1"},
        "research_line": "ia_mappo_v1",
        "train": {},
    }
    validate_active_research_config(base)
    with pytest.raises(ValueError, match="archived"):
        validate_active_research_config({**base, "subgoal_selector": {"enabled": True, "version": "facs_star_v0"}})
    with pytest.raises(ValueError, match="projection_aware_execution"):
        validate_active_research_config({**base, "train": {"projection_aware_execution": {"enabled": True}}})
    with pytest.raises(ValueError, match="selector artifacts"):
        validate_active_research_config({
            **base,
            "subgoal_selector": {
                "enabled": False,
                "version": "disabled_archived_v1",
                "competence_model_artifact": "historical.pt",
            },
        })
    with pytest.raises(ValueError, match="LLM subgoal selector"):
        validate_active_research_config({
            **base, "llm_subgoal_selector": {"enabled": True},
        })
    with pytest.raises(ValueError, match="fresh-init"):
        validate_checkpoint_research_line(base, {
            **base, "research_line": "", "train": {},
        })


def test_clean_standard_environment_emits_all_five_trace_stages():
    path = Path("configs/active/ia_mappo_v1_3u_6s1d_standard.yaml")
    config = load_yaml(path)
    env = CooperativeUAVEnv({
        **config["env"], "reward": config["reward"],
        "oracle_subgoal": config["oracle_subgoal"],
        "subgoal_selector": config["subgoal_selector"],
        "safety_execution": config["safety_execution"],
    })
    env.reset(seed=23)
    env.step(env.action_space.sample())
    trace = env.last_stage_intervention_trace
    assert len(trace.records) == 5 * env.num_agents
    assert {record.stage for record in trace.records} == {
        "actor_request", "initial_joint_guard", "obstacle_ttc",
        "final_joint_guard", "executed",
    }


def test_terminal_intervention_trace_is_ineligible_for_actor_risk():
    config = load_yaml("configs/active/ia_mappo_v1_3u_6s1d_standard.yaml")
    env = CooperativeUAVEnv({
        **config["env"], "max_episode_steps": 1,
        "reward": config["reward"],
        "oracle_subgoal": config["oracle_subgoal"],
        "subgoal_selector": config["subgoal_selector"],
        "safety_execution": config["safety_execution"],
    })
    env.reset(seed=23)
    _, _, terminated, truncated, info = env.step(
        np.zeros((env.num_agents, 2), dtype=np.float32)
    )
    trace = info["stage_intervention_trace"]
    assert terminated or truncated
    assert trace.terminal_transition is True
    assert not any(record.eligible_for_actor_risk for record in trace.records)


def test_causal_safety_context_uses_only_pre_action_snapshot():
    config = load_yaml("configs/active/ia_mappo_v1_3u_6s1d_standard.yaml")
    env = CooperativeUAVEnv({
        **config["env"], "reward": config["reward"],
        "oracle_subgoal": config["oracle_subgoal"],
        "subgoal_selector": config["subgoal_selector"],
        "safety_execution": config["safety_execution"],
    })
    env.reset(seed=23)
    action = np.zeros((env.num_agents, 2), dtype=np.float32)
    env._current_causal_safety_snapshot = env._causal_safety_snapshot(action)
    before_context, before_margin = env._causal_safety_feature_vector()
    for agent in env.agents:
        agent.position = agent.position + 10.0
    after_context, after_margin = env._causal_safety_feature_vector()
    np.testing.assert_array_equal(after_context, before_context)
    assert after_margin == before_margin


def test_clean_standard_trainer_transmits_and_fingerprints_shared_safety(tmp_path):
    config = load_yaml("configs/experiments/ia_mappo_v1_smoke_cpu.yaml")
    output = tmp_path / "trainer_safety_contract"
    config["output"] = {
        "output_dir": str(output),
        "checkpoint_dir": str(output / "checkpoints"),
        "reject_nonempty": True,
    }
    trainer = create_trainer(config, project_root=Path.cwd())
    env = trainer._base_env()

    assert env.safety_execution_enabled is True
    assert env.boundary_guard.enabled is True
    assert env.boundary_guard.action_shield_enabled is True
    assert env.obstacle_execution_guard_enabled is True
    assert env.safety_recovery_enabled is True
    assert trainer.runtime_contract_assertion["passed"] is True
    assert trainer._task_config_payload()["safety_execution"] == (
        config["safety_execution"]
    )
    assert "safety_execution" in trainer._environment_contract_payload()

    checkpoint = tmp_path / "valid.pt"
    trainer.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    legacy_contract = dict(payload["task_config"])
    legacy_contract.pop("safety_execution")
    legacy_contract["task_contract_version"] = 4
    payload["task_config"] = legacy_contract
    payload["task_config_fingerprint"] = hashlib.sha256(json.dumps(
        legacy_contract, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    legacy_checkpoint = tmp_path / "missing_safety_contract.pt"
    torch.save(payload, legacy_checkpoint)

    with pytest.raises(ValueError, match="safety_execution contract"):
        trainer.load_checkpoint(
            legacy_checkpoint,
            allow_task_config_mismatch=True,
        )


def test_environment_shadow_restore_is_exact_for_rng_recovery_and_gru():
    config = load_yaml("configs/active/ia_mappo_v1_3u_6s1d_standard.yaml")
    env = CooperativeUAVEnv({
        **config["env"], "reward": config["reward"],
        "oracle_subgoal": config["oracle_subgoal"],
        "subgoal_selector": config["subgoal_selector"],
        "safety_execution": config["safety_execution"],
    })
    env.reset(seed=41)
    hidden = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    snapshot = env.export_counterfactual_snapshot(hidden)
    env.step(np.full((3, 2), 0.25, dtype=np.float32))
    restored_hidden = env.restore_counterfactual_snapshot(snapshot)
    restored = env.export_counterfactual_snapshot(restored_hidden)
    assert restored.state_fingerprint == snapshot.state_fingerprint
    np.testing.assert_array_equal(restored_hidden, hidden)


def test_real_shared_safety_shadow_is_state_immutable():
    config = load_yaml("configs/active/ia_mappo_v1_3u_6s1d_standard.yaml")
    env = CooperativeUAVEnv({
        **config["env"], "reward": config["reward"],
        "oracle_subgoal": config["oracle_subgoal"],
        "subgoal_selector": config["subgoal_selector"],
        "safety_execution": config["safety_execution"],
    })
    env.reset(seed=23)
    raw = np.full((3, 2), 0.2, dtype=np.float32)
    causal = env._causal_safety_snapshot(raw)
    before = env.export_counterfactual_snapshot(None)
    result = env.evaluate_shadow_action_v1(causal, -raw)
    after = env.export_counterfactual_snapshot(None)
    assert after.state_fingerprint == before.state_fingerprint
    assert len(result.stage_levels) == 3
    assert np.asarray(result.executed_actions).shape == (3, 2)
    detailed = env.evaluate_shadow_action_v2(causal, -raw)
    assert np.asarray(detailed.per_agent_stage_delta_v_mps).shape == (3, 3)
    assert np.asarray(detailed.per_agent_stage_levels).shape == (3, 3)
    assert np.isclose(detailed.total_cost, np.mean(detailed.stage_costs))
