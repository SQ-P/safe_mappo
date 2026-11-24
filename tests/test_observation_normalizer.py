import numpy as np
import pytest

from src.algorithms.ppo.observation_normalizer import UAVObservationNormalizer
from src.envs.cooperative_uav_env import CooperativeUAVEnv


def _config():
    return {
        "num_agents": 2,
        "map_size": [100.0, 100.0],
        "agent_radius": 1.5,
        "agent_max_speed": 2.0,
        "detection_radius": 25.0,
        "obstacle_slots": 4,
        "obstacle_radius_range": [2.0, 5.0],
        "static_obstacle_num": 0,
        "dynamic_obstacle_num": 0,
        "start_x": 5.0,
        "start_y_positions": [35.0, 65.0],
        "start_y_jitter": 0.0,
        "goal_x": 95.0,
        "goal_y_positions": [35.0, 65.0],
        "cross_goals": True,
    }


def test_fixed_scale_normalization_keeps_26d_layout_and_zero_padding():
    config = _config()
    env = CooperativeUAVEnv(config)
    observations, _ = env.reset(seed=4)
    normalized = UAVObservationNormalizer(config).transform(observations)

    assert normalized.shape == (2, 26)
    assert normalized[0, 0:2] == pytest.approx([-0.9, -0.3])
    assert normalized[0, 2:4] == pytest.approx([0.0, 0.0])
    assert normalized[0, 4:6] == pytest.approx([0.9, 0.3])
    assert normalized[0, 8:10] == pytest.approx([0.0, 0.3])
    assert np.allclose(normalized[:, 10:26], 0.0)
    assert np.max(np.abs(normalized)) <= 1.0


def test_disabled_normalization_is_identity_copy():
    observations = np.arange(52, dtype=np.float32).reshape(2, 26)
    result = UAVObservationNormalizer(_config(), enabled=False).transform(observations)
    assert np.array_equal(result, observations)
    assert result is not observations


def test_tracked_obstacle_normalization_supports_52d_local_layout():
    config = _config() | {
        "observation_schema_version": "tracked_obstacle_v2",
        "obstacle_slots": 6,
        "obstacle_selection_version": "predicted_clearance_v1",
        "obstacle_velocity_estimation_version": "causal_track_difference_v1",
    }
    env = CooperativeUAVEnv(config)
    observations, _ = env.reset(seed=8)
    normalizer = UAVObservationNormalizer(
        config, version=UAVObservationNormalizer.TRACKED_VERSION
    )
    normalized = normalizer.transform(observations)
    assert observations.shape == normalized.shape == (2, 52)
    assert np.isfinite(normalized).all()
    assert np.allclose(normalized[:, 10:52], 0.0)


def test_tracked_schema_rejects_legacy_normalizer_version():
    config = _config() | {
        "observation_schema_version": "tracked_obstacle_v2",
        "obstacle_slots": 6,
        "obstacle_velocity_estimation_version": "causal_track_difference_v1",
    }
    with pytest.raises(ValueError, match="tracked observations require"):
        UAVObservationNormalizer(config)
