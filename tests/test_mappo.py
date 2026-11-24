from copy import deepcopy

import numpy as np
import pytest
import torch

from src.algorithms.factory import create_trainer
from src.algorithms.mappo import MAPPOTrainer
from src.algorithms.ppo.rollout_buffer import RecurrentRolloutBuffer
from src.utils.io_utils import load_yaml


def comparison_smoke_config(tmp_path, sharing=True):
    config = load_yaml("configs/active/team_v4_rule_tracked.yaml")
    config["algorithm"] = "mappo"
    config["seed"] = 7
    config["device"] = "cpu"
    config["multi_agent"] = {
        "actor_parameter_sharing": bool(sharing),
        "agent_update_order": "simultaneous",
    }
    config["env"].update(
        {"static_obstacle_num": 0, "dynamic_obstacle_num": 0, "max_episode_steps": 4}
    )
    config["model"]["hidden_dim"] = 16
    config["train"].update(
        {
            "total_updates": 1,
            "rollout_steps": 8,
            "ppo_epochs": 1,
            "sequence_length": 4,
            "sequence_minibatch_size": 4,
            "critic_minibatch_size": 8,
            "save_every": 1,
            "save_periodic_checkpoints": False,
            "log_every": 1,
        }
    )
    config["evaluation"].update(
        {"episodes": 1, "interval_updates": 1, "seed_base": 900, "save_episode_records": True}
    )
    config["output"] = {
        "output_dir": str(tmp_path / f"mappo_{sharing}"),
        "checkpoint_dir": str(tmp_path / f"mappo_{sharing}" / "checkpoints"),
        "reject_nonempty": True,
    }
    return config


def test_factory_builds_shared_and_independent_mappo(tmp_path):
    shared = create_trainer(comparison_smoke_config(tmp_path, True), tmp_path)
    independent = create_trainer(comparison_smoke_config(tmp_path, False), tmp_path)
    assert isinstance(shared, MAPPOTrainer)
    assert shared.obs_dim == independent.obs_dim == 52
    assert shared.global_state_dim == 104
    assert len(shared.actor.actor_modules()) == 1
    assert len(independent.actor.actor_modules()) == 2
    left, right = independent.actor.actor_modules()
    assert next(left.parameters()).data_ptr() != next(right.parameters()).data_ptr()
    assert shared.environment_contract_fingerprint() == independent.environment_contract_fingerprint()
    assert shared.learner_contract_fingerprint() != independent.learner_contract_fingerprint()


def test_shared_actor_uses_local_observation(tmp_path):
    trainer = create_trainer(comparison_smoke_config(tmp_path), tmp_path)
    hidden = trainer.actor.initial_hidden(2)
    assert hidden.shape == (1, 2, 16)
    with pytest.raises(ValueError, match="local observation"):
        trainer.actor.forward_step(torch.zeros(2, 104), hidden)


def test_team_reward_and_team_gae_are_strict():
    trainer = object.__new__(MAPPOTrainer)
    trainer.num_agents = 2
    assert np.array_equal(
        trainer.policy_rewards(np.asarray([1.25, 1.25], dtype=np.float32)),
        np.asarray([1.25, 1.25], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="exactly broadcast"):
        trainer.policy_rewards(np.asarray([1.25, 1.0], dtype=np.float32))

    buffer = RecurrentRolloutBuffer(2, team_gae=True)
    for reward, done, value in ((1.0, False, 0.5), (2.0, True, 0.25)):
        buffer.add(
            local_obs=np.zeros((2, 3), dtype=np.float32),
            global_state=np.zeros(6, dtype=np.float32),
            hidden=np.zeros((1, 2, 4), dtype=np.float32),
            actions=np.zeros((2, 2), dtype=np.float32),
            raw_actions=np.zeros((2, 2), dtype=np.float32),
            log_probs=np.zeros(2, dtype=np.float32),
            rewards=np.full(2, reward, dtype=np.float32),
            dones=np.full(2, done),
            values=np.full(2, value, dtype=np.float32),
        )
    buffer.compute_returns_and_advantages(np.zeros(2, dtype=np.float32), 0.99, 0.95)
    assert np.array_equal(buffer.advantages[:, 0], buffer.advantages[:, 1])


def test_mappo_checkpoint_resume(tmp_path):
    config = comparison_smoke_config(tmp_path)
    trainer = create_trainer(config, tmp_path)
    result = trainer.train()
    assert result["updates_completed"] == 1
    checkpoint = trainer.checkpoint_dir / "latest.pt"
    assert checkpoint.is_file()

    restored_config = deepcopy(config)
    restored_config["output"] = {
        "output_dir": str(tmp_path / "restored"),
        "checkpoint_dir": str(tmp_path / "restored" / "checkpoints"),
        "reject_nonempty": True,
    }
    restored = create_trainer(restored_config, tmp_path)
    restored.load_checkpoint(checkpoint)
    assert restored.current_update == trainer.current_update
