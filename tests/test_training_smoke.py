from pathlib import Path

import numpy as np
import pytest

from src.algorithms.ppo.trainer import PPOTrainer


def smoke_config(tmp_path: Path):
    return {
        "algorithm": "ppo",
        "seed": 3,
        "device": "cpu",
        "env": {
            "num_agents": 2,
            "map_size": [240.0, 180.0],
            "obstacle_num": 0,
            "max_episode_steps": 4,
            "agent_radius": 4.0,
            "agent_max_speed": 1.3,
            "initial_spawn_radius": 30.0,
            "initial_min_agent_distance": 14.0,
            "min_start_goal_distance": 60.0,
            "goal_radius": 25.0,
            "detection_radius": 80.0,
            "teammate_detection_radius": 80.0,
            "teammate_slots": 4,
            "obstacle_slots": 4,
            "collision_mode": "terminal",
            "boundary_mode": "collision",
            "reward_mode": "team_v1",
        },
        "reward": {
            "progress_weight": 1.0,
            "step_penalty": 0.1,
            "collision_penalty": 10.0,
            "goal_success_bonus": 10.0,
            "cooperation_bonus": 1.0,
        },
        "model": {
            "hidden_dim": 16,
            "initial_action_log_std": -1.2,
        },
        "train": {
            "total_updates": 1,
            "rollout_steps": 8,
            "ppo_epochs": 1,
            "sequence_length": 4,
            "sequence_minibatch_size": 4,
            "critic_minibatch_size": 8,
            "target_kl": 0.1,
            "gamma": 0.995,
            "gae_lambda": 0.95,
            "clip_coef": 0.2,
            "value_coef": 0.5,
            "action_entropy_coef_start": 0.0001,
            "action_entropy_coef_end": 0.0,
            "action_entropy_anneal_updates": 1,
            "log_std_min": -4.0,
            "log_std_max": -0.3,
            "actor_learning_rate": 0.0003,
            "critic_learning_rate": 0.0001,
            "max_grad_norm": 0.5,
            "save_every": 1,
            "log_every": 1,
        },
        "evaluation": {
            "episodes": 1,
            "interval_updates": 1,
            "seed_base": 100,
        },
        "output": {
            "output_dir": "results",
            "checkpoint_dir": "results/checkpoints",
        },
    }


def test_training_smoke_supports_joint_two_agent_policy(tmp_path):
    trainer = PPOTrainer(smoke_config(tmp_path), project_root=tmp_path)
    result = trainer.train()
    assert result["updates_completed"] == 1
    assert np.isfinite(result["policy_loss"])
    assert np.isfinite(result["value_loss"])
    assert trainer.actor.initial_hidden(1).shape[1] == 1
    assert trainer.obs_dim == 26
    assert (tmp_path / "results" / "checkpoints" / "latest.pt").exists()
    assert (tmp_path / "results" / "run_manifest.json").exists()
