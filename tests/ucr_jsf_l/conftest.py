from __future__ import annotations

from copy import deepcopy

import numpy as np

from src.envs.cooperative_uav_env import CooperativeUAVEnv
from src.safety.ucr_jsf_l import JointSafetyInputV1, ObstacleTrackV1
from src.utils.io_utils import load_yaml


def make_state(
    *,
    positions=None,
    velocities=None,
    goals=None,
    tracks=None,
    step=0,
    slot_saturated=None,
):
    positions = np.asarray(
        positions if positions is not None else [[20, 20], [20, 50], [20, 80]],
        dtype=np.float64,
    )
    velocities = np.asarray(
        velocities if velocities is not None else np.zeros((3, 2)),
        dtype=np.float64,
    )
    goals = np.asarray(
        goals if goals is not None else [[90, 20], [90, 50], [90, 80]],
        dtype=np.float64,
    )
    tracks = tracks if tracks is not None else ((), (), ())
    return JointSafetyInputV1.from_arrays(
        positions=positions,
        velocities=velocities,
        final_goals=goals,
        visible_obstacles=tracks,
        map_size=(100.0, 100.0),
        agent_radius=1.5,
        goal_radius=3.0,
        max_speed=2.0,
        max_acceleration=1.0,
        dt=0.5,
        step=step,
        obstacle_slot_saturated=slot_saturated,
    )


def make_track(observer_id=0, track_id=1, *, relative=(10.0, 0.0), velocity=(0.0, 0.0), valid=True, radius=2.0, age_steps=0):
    return ObstacleTrackV1(
        observer_id=observer_id,
        track_id=track_id,
        relative_position=tuple(relative),
        relative_velocity=tuple(velocity),
        radius=radius,
        velocity_valid=valid,
        observation_age_steps=age_steps,
    )


def load_ucr_config():
    return load_yaml("configs/experiments/ucr_jsf_l_3u_8s2d_off_seed23.yaml")


def make_ucr_env():
    config = load_ucr_config()
    env_config = deepcopy(config["env"])
    for key in ("reward", "oracle_subgoal", "subgoal_selector", "safety_execution"):
        env_config[key] = deepcopy(config.get(key, {}))
    return CooperativeUAVEnv(env_config)
