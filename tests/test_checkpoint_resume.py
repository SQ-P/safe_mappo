import json

import pytest
import torch

from src.algorithms.ppo.trainer import PPOTrainer
from tests.test_training_smoke import smoke_config


def test_completed_resume_preserves_training_summary(tmp_path):
    config = smoke_config(tmp_path)
    trainer = PPOTrainer(config, project_root=tmp_path)
    original = trainer.train()
    summary_path = tmp_path / "results" / "train_summary.json"
    original_text = summary_path.read_text(encoding="utf-8")
    checkpoint = tmp_path / "results" / "checkpoints" / "latest.pt"

    resumed = PPOTrainer(config, project_root=tmp_path)
    resumed.load_checkpoint(checkpoint)
    returned = resumed.train()

    assert returned == original
    assert summary_path.read_text(encoding="utf-8") == original_text
    assert len(json.loads(original_text)) > 5


def test_checkpoint_rejects_silent_task_configuration_mismatch(tmp_path):
    config = smoke_config(tmp_path)
    trainer = PPOTrainer(config, project_root=tmp_path)
    checkpoint = tmp_path / "task_config.pt"
    trainer.save_checkpoint(checkpoint)

    changed = smoke_config(tmp_path)
    changed["reward"] = dict(changed["reward"])
    changed["reward"]["step_penalty"] = 0.25
    resumed = PPOTrainer(changed, project_root=tmp_path)
    with pytest.raises(ValueError, match="task configuration"):
        resumed.load_checkpoint(checkpoint)

    # 显式 override 只用于受控消融实验；默认路径禁止静默漂移。
    resumed.load_checkpoint(checkpoint, allow_task_config_mismatch=True)


def test_best_checkpoint_prioritizes_real_success_over_zero_success(tmp_path):
    trainer = PPOTrainer(smoke_config(tmp_path), project_root=tmp_path)
    trainer.current_update = 1
    trainer._save_best_checkpoints(
        {
            "cfms": 0.0,
            "mean_goal_progress_ratio": 0.9,
            "collision_rate": 0.0,
            "timeout_rate": 1.0,
            "subgoal_reach_rate": 0.5,
            "mean_episode_return": -5.0,
        }
    )
    trainer.current_update = 2
    trainer._save_best_checkpoints(
        {
            "cfms": 0.3,
            "mean_goal_progress_ratio": 0.6,
            "collision_rate": 0.6,
            "timeout_rate": 0.1,
            "subgoal_reach_rate": 0.2,
            "mean_episode_return": 20.0,
        }
    )
    payload = torch.load(
        tmp_path / "results" / "checkpoints" / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert payload["update"] == 2
    assert trainer.best_model_key == pytest.approx(
        (1.0, 0.3, 0.3, -0.6, -0.1, 0.6, 20.0)
    )


def test_zero_success_checkpoint_does_not_select_all_collision_policy(tmp_path):
    trainer = PPOTrainer(smoke_config(tmp_path), project_root=tmp_path)
    trainer.current_update = 1
    trainer._save_best_checkpoints(
        {
            "cfms": 0.0,
            "mean_goal_progress_ratio": 0.9,
            "collision_rate": 1.0,
            "timeout_rate": 0.0,
            "subgoal_reach_rate": 0.8,
            "mean_episode_return": -1.0,
        }
    )
    trainer.current_update = 2
    trainer._save_best_checkpoints(
        {
            "cfms": 0.0,
            "mean_goal_progress_ratio": 0.5,
            "collision_rate": 0.2,
            "timeout_rate": 0.8,
            "subgoal_reach_rate": 0.2,
            "mean_episode_return": -3.0,
        }
    )
    payload = torch.load(
        tmp_path / "results" / "checkpoints" / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert payload["update"] == 2
