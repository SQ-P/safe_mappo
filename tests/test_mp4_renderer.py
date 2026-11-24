import pytest

from src.algorithms.ppo.networks import RecurrentActor
from src.envs.cooperative_uav_env import CooperativeUAVEnv
from src.visualization.mp4_renderer import _draw_frame, render_episode_mp4
from src.visualization.replay import collect_replay_episode


def _episode():
    env = CooperativeUAVEnv(
        {
            "num_agents": 3,
            "map_size": [240.0, 180.0],
            "obstacle_num": 0,
            "max_episode_steps": 2,
            "min_start_goal_distance": 60.0,
            "initial_spawn_radius": 30.0,
            "initial_min_agent_distance": 14.0,
            "goal_radius": 25.0,
            "teammate_slots": 5,
            "obstacle_slots": 4,
        }
    )
    actor = RecurrentActor(env.local_observation_dim, hidden_dim=16)
    return env, collect_replay_episode(env, actor, seed=4, deterministic=True)


def test_renderer_rejects_bad_fps(tmp_path):
    env, episode = _episode()
    with pytest.raises(ValueError):
        render_episode_mp4(
            episode,
            tmp_path / "bad.mp4",
            env.map_size,
            env.goal_radius,
            env.agent_radius,
            fps=0,
        )


def test_renderer_draws_subgoal_panel():
    env, episode = _episode()
    image = _draw_frame(
        episode.frames[0], env.map_size, env.goal_radius, env.agent_radius
    )
    assert image.shape == (600, 1100, 3)
    assert image.dtype == "uint8"
