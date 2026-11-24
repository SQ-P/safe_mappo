"""收集不可变回放快照，避免渲染逻辑与训练过程耦合。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def _readonly(value, dtype=np.float32) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ReplayFrame:
    """单个时间步的环境快照。"""

    step: int
    agent_positions: np.ndarray
    agent_velocities: np.ndarray
    paths: tuple[np.ndarray, ...]
    goal: np.ndarray
    subgoals: np.ndarray
    previous_subgoals: np.ndarray
    subgoal_reach_radius: float
    subgoal_versions: np.ndarray
    last_subgoal_update_steps: np.ndarray
    last_subgoal_reach_steps: np.ndarray
    subgoal_replan_count: int
    subgoal_progress_reward: float
    oracle_subgoal_enabled: bool
    obstacle_positions: np.ndarray
    obstacle_sizes: np.ndarray
    obstacle_velocities: np.ndarray
    team_return: float
    max_goal_distance: float
    required_goal_distance: float
    reason: str
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class ReplayEpisode:
    seed: int
    deterministic: bool
    frames: tuple[ReplayFrame, ...]
    final_reason: str
    total_return: float


def _snapshot(
    env,
    paths,
    step,
    team_return,
    reason,
    terminated,
    truncated,
    subgoal_versions,
    previous_subgoals,
    last_subgoal_update_steps,
    last_subgoal_reach_steps,
    subgoal_progress_reward,
) -> ReplayFrame:
    """把当前环境状态复制为不可修改的帧。"""
    obstacle_positions = (
        np.stack([item.position for item in env.obstacles])
        if env.obstacles
        else np.empty((0, 2), dtype=np.float32)
    )
    obstacle_sizes = (
        np.stack([item.size for item in env.obstacles])
        if env.obstacles
        else np.empty((0, 2), dtype=np.float32)
    )
    obstacle_velocities = (
        np.stack([item.velocity for item in env.obstacles])
        if env.obstacles
        else np.empty((0, 2), dtype=np.float32)
    )

    distances = env._goal_distances()
    # 团队成功要求所有智能体到达，最远智能体就是当前瓶颈。
    required_goal_distance = float(np.max(distances))

    return ReplayFrame(
        step=int(step),
        agent_positions=_readonly(env.positions),
        agent_velocities=_readonly(env.velocities),
        paths=tuple(_readonly(path) for path in paths),
        goal=_readonly(env.goal),
        subgoals=_readonly(env.subgoals),
        previous_subgoals=_readonly(previous_subgoals),
        subgoal_reach_radius=float(env.subgoal_reach_radius),
        subgoal_versions=_readonly(subgoal_versions, dtype=np.int32),
        last_subgoal_update_steps=_readonly(last_subgoal_update_steps, dtype=np.int32),
        last_subgoal_reach_steps=_readonly(last_subgoal_reach_steps, dtype=np.int32),
        subgoal_replan_count=int(env.subgoal_replan_count),
        subgoal_progress_reward=float(subgoal_progress_reward),
        oracle_subgoal_enabled=bool(env.oracle_enabled),
        obstacle_positions=_readonly(obstacle_positions),
        obstacle_sizes=_readonly(obstacle_sizes),
        obstacle_velocities=_readonly(obstacle_velocities),
        team_return=float(team_return),
        max_goal_distance=float(np.max(distances)),
        required_goal_distance=float(required_goal_distance),
        reason=str(reason),
        terminated=bool(terminated),
        truncated=bool(truncated),
    )


def collect_replay_episode(
    env,
    actor,
    seed: int,
    deterministic: bool = True,
    device="cpu",
    observation_transform=None,
) -> ReplayEpisode:
    """运行一轮策略并返回可离线渲染的轨迹快照。"""
    device = torch.device(device)
    obs, _ = env.reset(seed=int(seed))
    policy_rows = 1 if actor.action_dim == env.num_agents * 2 and actor.obs_dim == env.global_state_dim else env.num_agents
    hidden = actor.initial_hidden(policy_rows, device)
    paths = [[position.copy()] for position in env.positions]
    previous_subgoals = env.subgoals.copy()
    subgoal_versions = np.zeros(env.num_agents, dtype=np.int32)
    last_subgoal_update_steps = np.zeros(env.num_agents, dtype=np.int32)
    last_subgoal_reach_steps = np.full(env.num_agents, -1_000, dtype=np.int32)
    frames = [
        _snapshot(
            env, paths, 0, 0.0, "ongoing", False, False,
            subgoal_versions, previous_subgoals,
            last_subgoal_update_steps, last_subgoal_reach_steps, 0.0,
        )
    ]
    team_return = 0.0
    final_reason = "timeout"

    for step in range(1, env.max_episode_steps + 1):
        active_subgoals = env.subgoals.copy()
        previous_positions = env.positions.copy()
        policy_obs = (
            obs
            if observation_transform is None
            else observation_transform(obs)
        )
        if policy_rows == 1:
            policy_obs = np.asarray(policy_obs).reshape(1, -1)
        with torch.no_grad():
            output = actor.forward_step(
                torch.as_tensor(
                    policy_obs, dtype=torch.float32, device=device
                ),
                hidden,
                deterministic=bool(deterministic),
            )

        obs, rewards, terminated, truncated, info = env.step(
            output.action.detach().cpu().numpy().reshape(env.num_agents, 2)
        )
        hidden = output.hidden.detach()
        team_return += float(np.mean(rewards))

        # The transition reward and any reach event refer to the target that
        # was active before env.step(); the returned observation may already
        # contain a freshly replanned target.
        old_distances = np.linalg.norm(previous_positions - active_subgoals, axis=1)
        new_distances = np.linalg.norm(env.positions - active_subgoals, axis=1)
        reached = (old_distances >= env.subgoal_reach_radius) & (
            new_distances < env.subgoal_reach_radius
        )
        last_subgoal_reach_steps[reached] = step
        changed = np.max(np.abs(env.subgoals - active_subgoals), axis=1) > (
            env.subgoal_identity_tolerance
        )
        previous_subgoals[changed] = active_subgoals[changed]
        subgoal_versions[changed] += 1
        last_subgoal_update_steps[changed] = step
        subgoal_reward = float(np.mean(
            info.get("reward_components", {}).get("subgoal_progress_reward", 0.0)
        ))

        for agent_path, position in zip(paths, env.positions):
            agent_path.append(position.copy())

        final_reason = str(info.get("reason", "ongoing"))
        frames.append(
            _snapshot(
                env, paths, step, team_return, final_reason, terminated, truncated,
                subgoal_versions, previous_subgoals,
                last_subgoal_update_steps, last_subgoal_reach_steps,
                subgoal_reward,
            )
        )
        if terminated or truncated:
            break

    return ReplayEpisode(
        seed=int(seed),
        deterministic=bool(deterministic),
        frames=tuple(frames),
        final_reason=final_reason,
        total_return=team_return,
    )
