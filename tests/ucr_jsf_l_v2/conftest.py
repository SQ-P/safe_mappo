from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np

from src.envs.cooperative_uav_env import CooperativeUAVEnv
from src.safety.ucr_jsf_l_v2 import JointSafetyInputV2, ObstacleTrackV2
from src.utils.io_utils import load_yaml


def make_track(
    observer_id: int,
    track_id: int,
    *,
    relative=(8.0, 0.0),
    velocity=(0.0, 0.0),
    valid=True,
    radius=2.0,
    age_steps=0,
):
    return ObstacleTrackV2(
        observer_id=observer_id,
        track_id=track_id,
        relative_position=tuple(relative),
        relative_velocity=tuple(velocity),
        radius=radius,
        velocity_valid=valid,
        observation_age_steps=age_steps,
    )


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
        velocities if velocities is not None else [[0.2, 0.0], [0.1, 0.0], [0.3, 0.0]],
        dtype=np.float64,
    )
    goals = np.asarray(
        goals if goals is not None else [[90, 20], [90, 50], [90, 80]],
        dtype=np.float64,
    )
    return JointSafetyInputV2.from_arrays(
        positions=positions,
        velocities=velocities,
        final_goals=goals,
        visible_obstacles=tracks if tracks is not None else ((), (), ()),
        map_size=(100.0, 100.0),
        agent_radius=1.5,
        goal_radius=3.0,
        max_speed=2.0,
        max_acceleration=1.0,
        dt=0.5,
        step=step,
        obstacle_slot_saturated=slot_saturated,
    )


def blocked_state(*, step=0):
    tracks = tuple(
        (make_track(index, 101 + index, relative=(8.0, 0.0)),)
        for index in range(3)
    )
    return make_state(tracks=tracks, step=step)


def permute_state(state, permutation):
    permutation = tuple(permutation)
    tracks = tuple(
        tuple(replace(track, observer_id=new_id) for track in state.visible_obstacles[old_id])
        for new_id, old_id in enumerate(permutation)
    )
    return JointSafetyInputV2.from_arrays(
        positions=np.asarray(state.positions)[list(permutation)],
        velocities=np.asarray(state.velocities)[list(permutation)],
        final_goals=np.asarray(state.final_goals)[list(permutation)],
        visible_obstacles=tracks,
        map_size=state.map_size,
        agent_radius=state.agent_radius,
        goal_radius=state.goal_radius,
        max_speed=state.max_speed,
        max_acceleration=state.max_acceleration,
        dt=state.dt,
        step=state.step,
        obstacle_slot_saturated=np.asarray(state.obstacle_slot_saturated)[list(permutation)],
    )


def map_actions_to_original(actions, permutation):
    result = np.empty_like(actions)
    for new_id, old_id in enumerate(permutation):
        result[old_id] = actions[new_id]
    return result


def load_v2_config():
    return load_yaml("configs/experiments/ucr_jsf_l_v2_3u_8s2d_off_seed23.yaml")


def executor_config():
    return deepcopy(load_v2_config()["safety_execution"])


def make_v2_env():
    config = load_v2_config()
    env_config = deepcopy(config["env"])
    for key in ("reward", "oracle_subgoal", "subgoal_selector", "safety_execution"):
        env_config[key] = deepcopy(config.get(key, {}))
    return CooperativeUAVEnv(env_config)
