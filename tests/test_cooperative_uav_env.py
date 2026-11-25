import numpy as np
import pytest

from src.envs.cooperative_uav_env import CooperativeUAVEnv
from src.envs.entities import CircularObstacle
from src.utils.io_utils import load_yaml


def base_config(num_agents=2, static_obstacles=0, dynamic_obstacles=0):
    return {
        "num_agents": num_agents,
        "map_size": [100.0, 100.0],
        "agent_radius": 1.5,
        "agent_max_speed": 2.0,
        "dt": 0.5,
        "max_episode_steps": 300,
        "start_x": 5.0,
        "start_y_positions": [35.0, 65.0],
        "start_y_jitter": 2.0,
        "goal_x": 95.0,
        "goal_y_positions": [35.0, 65.0],
        "cross_goals": True,
        "goal_radius": 3.0,
        "static_obstacle_num": static_obstacles,
        "dynamic_obstacle_num": dynamic_obstacles,
        "obstacle_radius_range": [2.0, 5.0],
        "dynamic_obstacle_velocity_range": [-1.0, 1.0],
        "obstacle_spawn_clearance": 2.0,
        "obstacle_spawn_x_range": [15.0, 85.0],
        "obstacle_spawn_y_range": [8.0, 92.0],
        "detection_radius": 25.0,
        "obstacle_slots": 4,
        "reward": {
            "progress_weight": 1.0,
            "step_penalty": 0.1,
            "collision_penalty": 50.0,
            "goal_success_bonus": 100.0,
            "cooperation_bonus": 5.0,
        },
    }


def test_default_two_uav_observation_and_joint_state_dimensions():
    env = CooperativeUAVEnv(base_config())
    obs, _ = env.reset(seed=1)
    assert obs.shape == (2, 26)
    assert env.local_observation_dim == 26
    assert env.get_global_state().shape == (52,)
    assert env.action_space.shape == (2, 2)


def test_tracked_obstacle_v2_uses_six_local_slots_and_causal_velocity():
    config = base_config()
    config.update({
        "observation_schema_version": "tracked_obstacle_v2",
        "obstacle_slots": 6,
        "obstacle_selection_version": "predicted_clearance_v1",
        "obstacle_velocity_estimation_version": "causal_track_difference_v1",
    })
    env = CooperativeUAVEnv(config)
    env.reset(seed=101)
    env.agents[0].position = np.array([20.0, 20.0], dtype=np.float32)
    env.agents[0].velocity[:] = 0.0
    env.agents[1].position = np.array([20.0, 60.0], dtype=np.float32)
    env.obstacles = [
        CircularObstacle([30.0, 20.0], 2.0, [1.0, 0.0], dynamic=True)
    ]
    env._reset_obstacle_tracks()
    first = env.build_local_observations()
    assert first.shape == (2, 52)
    assert env.get_global_state().shape == (104,)
    assert first[0, 10:17] == pytest.approx(
        [10.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0]
    )
    second, _, _, _, info = env.step(np.zeros((2, 2), dtype=np.float32))
    assert second[0, 12:14] == pytest.approx([1.0, 0.0])
    assert second[0, 16] == pytest.approx(1.0)
    assert info["observation_schema_version"] == "tracked_obstacle_v2"
    assert info["selected_velocity_valid_rate"] > 0.0
    assert np.allclose(second[0, 17:52], 0.0)


def test_near_goal_zero_action_is_logged_without_changing_termination():
    config = base_config()
    config.update({
        "max_episode_steps": 25,
        "near_goal_stall_patience_steps": 3,
        "near_goal_stall_distance_margin": 3.0,
    })
    env = CooperativeUAVEnv(config)
    env.reset(seed=102)
    env.obstacles = []
    for agent_id, agent in enumerate(env.agents):
        agent.position = env.goals[agent_id] + np.array([-5.0, 0.0], dtype=np.float32)
        agent.velocity[:] = 0.0
    env._previous_distances = env._goal_distances()
    for _ in range(3):
        _, _, terminated, truncated, info = env.step(
            np.zeros((2, 2), dtype=np.float32)
        )
    assert not terminated and not truncated
    assert info["near_goal_stall_triggered"].tolist() == [True, True]
    assert info["near_goal_stall_first_steps"].tolist() == [3, 3]


def test_start_jitter_and_crossed_goals():
    env = CooperativeUAVEnv(base_config())
    _, info = env.reset(seed=4)
    assert np.allclose(env.positions[:, 0], 5.0)
    assert np.all(np.abs(env.positions[:, 1] - [35.0, 65.0]) <= 2.0)
    assert np.allclose(info["goals"], [[95.0, 65.0], [95.0, 35.0]])
    assert np.array_equal(info["subgoals"], info["goals"])


def test_subgoal_generation_switch_uses_final_goals_without_replanning(monkeypatch):
    config = load_yaml("configs/active/standard_mappo_3u_8s2d.yaml")
    config["oracle_subgoal"]["generation_enabled"] = False
    env = CooperativeUAVEnv({
        **config["env"],
        "reward": config["reward"],
        "oracle_subgoal": config["oracle_subgoal"],
        "subgoal_selector": config["subgoal_selector"],
        "safety_execution": config["safety_execution"],
    })
    monkeypatch.setattr(env, "_generate_valid_subgoals", lambda *args: pytest.fail(
        "subgoal generation ran with the switch disabled"
    ))
    observations, info = env.reset(seed=23)
    assert observations.shape == (3, 52)
    assert info["subgoal_generation_enabled"] is False
    np.testing.assert_array_equal(info["subgoals"], info["goals"])
    env.step(np.zeros((3, 2), dtype=np.float32))
    assert env.subgoal_replan_count == 0
    np.testing.assert_array_equal(env.subgoals, env.goals)


def test_acceleration_kinematics_uses_dt_and_component_clip():
    env = CooperativeUAVEnv(base_config())
    env.reset(seed=2)
    before = env.positions.copy()
    env.step(np.ones((2, 2), dtype=np.float32))
    assert np.allclose(env.velocities, 0.5)
    assert np.allclose(env.positions, before + 0.25)


@pytest.mark.parametrize(
    ("action", "expected_delta"),
    [
        ([1.0, 0.0], [0.25, 0.0]),
        ([-1.0, 0.0], [-0.25, 0.0]),
        ([0.0, 1.0], [0.0, 0.25]),
        ([0.0, -1.0], [0.0, -0.25]),
    ],
)
def test_action_axis_and_sign_mapping(action, expected_delta):
    env = CooperativeUAVEnv(base_config())
    env.reset(seed=20)
    env.obstacles = []
    env.agents[0].position = np.asarray([20.0, 20.0], dtype=np.float32)
    env.agents[0].velocity[:] = 0.0
    env.agents[1].position = np.asarray([50.0, 50.0], dtype=np.float32)
    env._previous_distances = env._goal_distances()
    before = env.agents[0].position.copy()
    actions = np.zeros((2, 2), dtype=np.float32)
    actions[0] = action
    env.step(actions)
    assert np.allclose(env.agents[0].position - before, expected_delta)


def test_obstacle_generation_counts_radius_and_dynamic_velocity_ranges():
    env = CooperativeUAVEnv(base_config(static_obstacles=8, dynamic_obstacles=3))
    env.reset(seed=8)
    assert len(env.obstacles) == 11
    assert sum(item.dynamic for item in env.obstacles) == 3
    assert all(2.0 <= item.radius <= 5.0 for item in env.obstacles)
    dynamic = [item for item in env.obstacles if item.dynamic]
    assert all(np.all((-1.0 <= item.velocity) & (item.velocity <= 1.0)) for item in dynamic)


def test_nearest_obstacle_features_are_sorted_limited_and_zero_padded():
    env = CooperativeUAVEnv(base_config())
    env.reset(seed=3)
    agent = env.agents[0]
    agent.position = np.array([20.0, 20.0], dtype=np.float32)
    env.obstacles = [
        CircularObstacle([23.0, 24.0], 2.0, [0.0, 0.0]),
        CircularObstacle([30.0, 20.0], 3.0, [0.0, 0.0]),
        CircularObstacle([80.0, 80.0], 4.0, [0.0, 0.0]),
    ]
    nearest = env.get_nearest_obstacles(0)
    assert [item[2] for item in nearest] == pytest.approx([5.0, 10.0])
    obs = env.build_local_observations()[0]
    assert obs[10:18] == pytest.approx([3.0, 4.0, 2.0, 5.0, 10.0, 0.0, 3.0, 10.0])
    assert np.allclose(obs[18:26], 0.0)


def test_collision_helpers_use_strict_circle_thresholds():
    env = CooperativeUAVEnv(base_config())
    env.reset(seed=5)
    env.obstacles = [CircularObstacle([20.0, 20.0], 2.0, [0.0, 0.0])]
    env.agents[0].position = np.array([23.4, 20.0], dtype=np.float32)
    assert env.check_collision()["type"] == "obstacle_collision"
    env.agents[0].position = np.array([23.5, 20.0], dtype=np.float32)
    env.agents[1].position = np.array([50.0, 50.0], dtype=np.float32)
    assert env.check_collision() is None
    env.agents[1].position = np.array([26.4, 20.0], dtype=np.float32)
    assert env.check_collision()["type"] == "agent_collision"


def test_success_requires_all_agents_at_their_own_goals():
    env = CooperativeUAVEnv(base_config())
    env.reset(seed=6)
    env.obstacles = []
    env.agents[0].position = env.goals[0].copy()
    env.agents[1].position = env.goals[1] + np.array([0.0, 4.0], dtype=np.float32)
    assert not env.check_success()
    env.agents[1].position = env.goals[1] + np.array([0.0, 2.9], dtype=np.float32)
    assert env.check_success()


def test_reward_success_collision_and_timeout_components():
    success_env = CooperativeUAVEnv(base_config())
    success_env.reset(seed=7)
    success_env.obstacles = []
    for agent_id, agent in enumerate(success_env.agents):
        agent.position = success_env.goals[agent_id].copy()
        agent.velocity[:] = 0.0
    success_env._previous_distances = success_env._goal_distances()
    _, rewards, terminated, truncated, info = success_env.step(np.zeros((2, 2), dtype=np.float32))
    assert terminated and not truncated and info["reason"] == "team_success"
    assert info["reach_success"] is True
    assert info["safe_success"] is True
    assert np.allclose(rewards, 99.9)

    timeout_cfg = base_config()
    timeout_cfg["max_episode_steps"] = 1
    timeout_env = CooperativeUAVEnv(timeout_cfg)
    timeout_env.reset(seed=8)
    timeout_env.obstacles = []
    _, rewards, terminated, truncated, info = timeout_env.step(np.zeros((2, 2), dtype=np.float32))
    assert not terminated and truncated and info["reason"] == "timeout"
    assert np.allclose(rewards, -0.1)


def test_obstacle_collision_rolls_back_without_terminating_and_deduplicates_event():
    env = CooperativeUAVEnv(base_config())
    env.reset(seed=13)
    env.agents[0].position = np.array([10.0, 20.0], dtype=np.float32)
    env.agents[0].velocity = np.array([0.5, 0.0], dtype=np.float32)
    env.agents[1].position = np.array([50.0, 50.0], dtype=np.float32)
    env.agents[1].velocity[:] = 0.0
    env.obstacles = [CircularObstacle([13.6, 20.0], 2.0, [0.0, 0.0])]
    env._previous_distances = env._goal_distances()
    safe_position = env.agents[0].position.copy()

    _, rewards, terminated, truncated, info = env.step(
        np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    )

    assert not terminated and not truncated
    assert np.array_equal(env.agents[0].position, safe_position)
    assert np.array_equal(env.agents[0].velocity, [0.0, 0.0])
    assert env.check_collision() is None
    assert rewards[0] == pytest.approx(-50.1)
    assert info["collision_count"] == 1
    assert info["obstacle_collision_count"] == 1
    assert info["static_obstacle_collision_count"] == 1
    assert info["dynamic_obstacle_collision_count"] == 0
    assert info["first_collision_type"] == "static_obstacle_collision"
    assert info["episode_has_collision"] is True
    assert info["first_collision_step"] == 1
    assert info["first_collision_goal_progress_ratio"] is not None
    assert info["remaining_goal_progress_at_first_collision"] == pytest.approx(
        1.0 - info["first_collision_goal_progress_ratio"]
    )
    assert info["rolled_back_agent_ids"] == [0]

    # 连续顶住同一障碍仍回滚，但不重复计数或处罚。
    _, rewards, terminated, _, info = env.step(
        np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    )
    assert not terminated
    assert np.array_equal(env.agents[0].position, safe_position)
    assert rewards[0] == pytest.approx(-0.1)
    assert info["collision_count"] == 1


def test_dynamic_obstacle_collision_has_independent_telemetry():
    """动障碰撞必须能与静态障碍碰撞区分，但处罚和回滚语义保持不变。"""
    env = CooperativeUAVEnv(base_config(dynamic_obstacles=1))
    env.reset(seed=17)
    env.agents[0].position = np.array([10.0, 20.0], dtype=np.float32)
    env.agents[0].velocity = np.array([0.5, 0.0], dtype=np.float32)
    env.agents[1].position = np.array([50.0, 50.0], dtype=np.float32)
    env.agents[1].velocity[:] = 0.0
    env.obstacles = [
        CircularObstacle([13.6, 20.0], 2.0, [0.0, 0.0], dynamic=True)
    ]
    env._previous_distances = env._goal_distances()

    _, rewards, terminated, truncated, info = env.step(
        np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    )

    assert not terminated and not truncated
    assert rewards[0] == pytest.approx(-50.1)
    assert info["obstacle_collision_count"] == 1
    assert info["static_obstacle_collision_count"] == 0
    assert info["dynamic_obstacle_collision_count"] == 1
    assert info["first_collision_type"] == "dynamic_obstacle_collision"
    assert info["new_collision_events"][0]["obstacle_kind"] == "dynamic"


def test_inter_agent_and_boundary_collisions_restore_only_participants():
    inter_env = CooperativeUAVEnv(base_config())
    inter_env.reset(seed=14)
    inter_env.obstacles = []
    inter_env.agents[0].position = np.array([20.0, 20.0], dtype=np.float32)
    inter_env.agents[1].position = np.array([23.1, 20.0], dtype=np.float32)
    inter_env._previous_distances = inter_env._goal_distances()
    before = inter_env.positions.copy()
    _, _, terminated, _, info = inter_env.step(
        np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    )
    assert not terminated
    assert np.array_equal(inter_env.positions, before)
    assert np.allclose(inter_env.velocities, 0.0)
    assert info["inter_agent_collision_count"] == 1

    boundary_env = CooperativeUAVEnv(base_config())
    boundary_env.reset(seed=15)
    boundary_env.obstacles = []
    boundary_env.agents[0].position = np.array([1.6, 20.0], dtype=np.float32)
    boundary_env.agents[1].position = np.array([50.0, 50.0], dtype=np.float32)
    boundary_env._previous_distances = boundary_env._goal_distances()
    safe_position = boundary_env.agents[0].position.copy()
    _, _, terminated, _, info = boundary_env.step(
        np.array([[-1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    )
    assert not terminated
    # 边界回滚后额外向内退 1m，打破“原地回滚后继续撞墙”的锁死循环。
    assert np.allclose(boundary_env.agents[0].position, [2.5, 20.0])
    assert not np.array_equal(boundary_env.agents[0].position, safe_position)
    assert np.allclose(boundary_env.agents[0].velocity, 0.0)
    assert info["boundary_collision_count"] == 1
    assert info["left_boundary_collision_count"] == 1
    assert info["boundary_contact_step_count"] == 1
    assert info["max_boundary_contact_streak"] == 1

    # 下一步同向动作仍处于合法区域，连续接触 streak 会被清零。
    _, _, _, _, info = boundary_env.step(
        np.array([[-1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    )
    assert info["boundary_contact_step_count"] == 1
    assert info["max_boundary_contact_streak"] == 1


def test_reach_success_after_collision_is_not_safe_success():
    env = CooperativeUAVEnv(base_config())
    env.reset(seed=16)
    env.agents[0].position = np.array([10.0, 20.0], dtype=np.float32)
    env.agents[1].position = np.array([50.0, 50.0], dtype=np.float32)
    env.obstacles = [CircularObstacle([13.6, 20.0], 2.0, [0.0, 0.0])]
    env._previous_distances = env._goal_distances()
    env.step(np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32))

    env.obstacles = []
    for agent_id, agent in enumerate(env.agents):
        agent.position = env.goals[agent_id].copy()
        agent.velocity[:] = 0.0
    env._previous_distances = env._goal_distances()
    _, _, terminated, truncated, info = env.step(np.zeros((2, 2), dtype=np.float32))
    assert terminated and not truncated
    assert info["reach_success"] is True
    assert info["safe_success"] is False
    assert info["collision_count"] == 1


def test_cooperation_bonus_is_dense_and_distinct_from_goal_success_bonus():
    env = CooperativeUAVEnv(base_config())
    env.reset(seed=12)
    env.obstacles = []
    _, rewards, terminated, _, info = env.step(
        np.tile(np.array([1.0, 0.0], dtype=np.float32), (2, 1))
    )
    assert not terminated
    cooperation = np.asarray(info["reward_components"]["cooperation_reward"])
    assert np.all(cooperation > 0.0)
    assert np.allclose(info["reward_components"]["goal_bonus"], 0.0)
    assert np.all(rewards > info["reward_components"]["progress_reward"] - 0.1)


def test_variable_agent_count_keeps_independent_observation_width():
    env = CooperativeUAVEnv(base_config(num_agents=5))
    obs, _ = env.reset(seed=9)
    assert obs.shape == (5, 26)
    assert env.get_global_state().shape == (130,)


def test_action_shape_and_render_interface():
    env = CooperativeUAVEnv(base_config())
    env.reset(seed=10)
    with pytest.raises(ValueError, match="shape"):
        env.step(np.zeros((3, 2), dtype=np.float32))
    frame = env.render()
    assert frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8
