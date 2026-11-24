"""固定、透明的诊断 Reward 环境 Wrapper。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np

from .reward_api import (
    REWARD_API_VERSION, compile_reward_function, freeze_reward_context,
    reward_source_hash, validate_reward_result,
)


class DiagnosticRewardWrapper:
    """只替换 reward，不改变 observation/action/done 等环境接口。"""

    def __init__(self, env: Any, config: Mapping[str, Any]):
        self.env = env
        reward_cfg = dict(config.get("reward", {}))
        self.manifest = dict(reward_cfg.get("_diagnostic_reward_function", {}))
        self.reward_fn = None
        if self.manifest:
            source = str(self.manifest["source"])
            expected_hash = str(self.manifest.get("source_hash", ""))
            if expected_hash and expected_hash != reward_source_hash(source):
                raise ValueError("diagnostic reward source hash mismatch")
            if str(self.manifest.get("api_version")) != REWARD_API_VERSION:
                raise ValueError("unsupported diagnostic reward API version")
            self.reward_fn = compile_reward_function(
                source, dict(self.manifest.get("parameters", {}))
            )
        self.max_abs_reward = float(
            config.get("diagnostic", {}).get("max_abs_shaped_reward", 250.0)
        )
        self._reward_state: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self, *args: Any, **kwargs: Any):
        self._reward_state = {}
        return self.env.reset(*args, **kwargs)

    def step(self, actions: Any):
        observations, base_rewards, terminated, truncated, info = self.env.step(actions)
        base = np.asarray(base_rewards, dtype=np.float32)
        if base.shape != (self.num_agents,):
            raise ValueError("base reward violates per-agent reward contract")
        shaped = base.copy()
        if self.reward_fn is not None:
            context = freeze_reward_context(
                self._build_context(actions, info, terminated, truncated)
            )
            shaped, new_state = validate_reward_result(
                self.reward_fn(base.copy(), context, deepcopy(self._reward_state)),
                num_agents=self.num_agents,
                max_abs_reward=self.max_abs_reward,
            )
            self._reward_state = new_state

        components = info.setdefault("reward_components", {})
        # reward_components 是纯数值遥测协议，禁止混入 manifest、字典或字符串。
        for key, value in components.items():
            if np.asarray(value).dtype.kind not in "biufc":
                raise TypeError(f"reward component {key!r} must be numeric")
        components["diagnostic_shaping_reward"] = (shaped - base).astype(np.float32)
        components["total_reward"] = shaped.copy()
        info["team_reward"] = float(np.mean(shaped))
        return observations, shaped, terminated, truncated, info

    def _build_context(
        self, actions: Any, info: Mapping[str, Any], terminated: bool, truncated: bool
    ) -> dict[str, Any]:
        positions = np.asarray([agent.position for agent in self.agents], dtype=np.float32)
        velocities = np.asarray([agent.velocity for agent in self.agents], dtype=np.float32)
        obstacle_clearance = np.asarray(
            info.get("nearest_obstacle_distances", np.full(self.num_agents, np.inf)),
            dtype=np.float32,
        )
        teammate_distance = np.full(self.num_agents, np.inf, dtype=np.float32)
        for index in range(self.num_agents):
            others = np.delete(positions, index, axis=0)
            if len(others):
                teammate_distance[index] = float(
                    np.min(np.linalg.norm(others - positions[index], axis=1))
                )
        low = np.full(2, float(self.agent_radius), dtype=np.float32)
        high = np.asarray(self.map_size, dtype=np.float32) - float(self.agent_radius)
        boundary_clearance = np.column_stack(
            (positions[:, 0] - low[0], high[0] - positions[:, 0],
             positions[:, 1] - low[1], high[1] - positions[:, 1])
        ).astype(np.float32)
        collisions = np.zeros(self.num_agents, dtype=bool)
        for event in info.get("new_collision_events", []):
            for agent_id in event.get("agent_ids", []):
                collisions[int(agent_id)] = True
        return {
            "step": int(info.get("step", 0)),
            "positions": positions,
            "velocities": velocities,
            "actions": np.asarray(actions, dtype=np.float32).copy(),
            "final_goals": np.asarray(self.goals, dtype=np.float32).copy(),
            # step 尾部可能重规划；此字段与返回 observation 中的 subgoal 保持一致。
            "subgoals": np.asarray(self.subgoals, dtype=np.float32).copy(),
            "goal_distances": np.asarray(info.get("goal_distances"), dtype=np.float32),
            "subgoal_distances": np.asarray(info.get("subgoal_distances"), dtype=np.float32),
            "current_obstacle_clearance": obstacle_clearance,
            "current_teammate_distance": teammate_distance,
            "current_boundary_clearance": boundary_clearance,
            "new_collision_by_agent": collisions,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }

    def render(self, *args: Any, **kwargs: Any):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()


def wrap_configured_reward(env: Any, config: Mapping[str, Any]) -> Any:
    return env if isinstance(env, DiagnosticRewardWrapper) else DiagnosticRewardWrapper(env, config)
