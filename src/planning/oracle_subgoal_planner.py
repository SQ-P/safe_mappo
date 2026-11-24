"""基于人工势场的 Oracle 局部目标规划器。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


class OracleSubgoalPlanner:
    """用目标吸引力、障碍排斥力和队友排斥力生成局部目标。

    Planner 不持有 episode 状态；重规划节奏、合法性检查和 fallback 均由环境负责，
    因而不会改变 PPO Actor/Critic 的接口或网络结构。
    """

    def __init__(self, config: Mapping[str, Any]):
        cfg = dict(config)
        self.lookahead_distance = float(cfg["lookahead_distance"])
        self.sensor_range = float(cfg["sensor_range"])
        self.attractive_gain = float(cfg["attractive_gain"])
        self.obstacle_repulsive_gain = float(cfg["obstacle_repulsive_gain"])
        self.teammate_repulsive_gain = float(cfg["teammate_repulsive_gain"])
        self.obstacle_safety_margin = float(cfg["obstacle_safety_margin"])
        self.teammate_safety_distance = float(cfg["teammate_safety_distance"])
        if min(
            self.lookahead_distance,
            self.sensor_range,
            self.attractive_gain,
            self.obstacle_repulsive_gain,
            self.teammate_repulsive_gain,
            self.obstacle_safety_margin,
            self.teammate_safety_distance,
        ) <= 0.0:
            raise ValueError("oracle force and distance parameters must be positive")
        self.eps = 1e-6

    @staticmethod
    def _obstacle_state(obstacle: Any) -> tuple[np.ndarray, float, np.ndarray]:
        """同时兼容环境实体和字典形式的障碍状态。"""
        if isinstance(obstacle, Mapping):
            position = obstacle["position"]
            radius = obstacle["radius"]
            velocity = obstacle.get("velocity", [0.0, 0.0])
        else:
            position = obstacle.position
            radius = obstacle.radius
            velocity = obstacle.velocity
        return (
            np.asarray(position, dtype=np.float32),
            float(radius),
            np.asarray(velocity, dtype=np.float32),
        )

    def plan(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        final_goals: np.ndarray,
        obstacles: Sequence[Any],
        teammate_positions: np.ndarray,
        world_boundary: np.ndarray,
    ) -> np.ndarray:
        """返回 ``(num_agents, 2)`` 的候选 subgoal。

        速度参数被纳入接口，便于后续在不改环境调用链的前提下加入速度预测；第一版
        势场严格基于当前几何状态，避免额外引入未配置的预测时间参数。
        """
        positions = np.asarray(positions, dtype=np.float32)
        velocities = np.asarray(velocities, dtype=np.float32)
        final_goals = np.asarray(final_goals, dtype=np.float32)
        teammate_positions = np.asarray(teammate_positions, dtype=np.float32)
        boundary = np.asarray(world_boundary, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("positions must have shape (num_agents, 2)")
        if velocities.shape != positions.shape or final_goals.shape != positions.shape:
            raise ValueError("velocities and final_goals must match positions")
        if teammate_positions.shape != positions.shape or boundary.shape != (2,):
            raise ValueError("teammate_positions or world_boundary has invalid shape")

        obstacle_states = [self._obstacle_state(item) for item in obstacles]
        result = np.empty_like(positions)
        for agent_id, position in enumerate(positions):
            goal_delta = final_goals[agent_id] - position
            goal_distance = float(np.linalg.norm(goal_delta))
            force = self.attractive_gain * goal_delta / max(goal_distance, self.eps)

            # 障碍物排斥力按“到障碍表面的净距离”计算，越靠近安全边界越强。
            for center, radius, _ in obstacle_states:
                away = position - center
                center_distance = float(np.linalg.norm(away))
                surface_distance = center_distance - radius
                if center_distance > self.sensor_range + radius:
                    continue
                away_direction = away / max(center_distance, self.eps)
                effective = max(surface_distance - self.obstacle_safety_margin, self.eps)
                influence = max(self.sensor_range - self.obstacle_safety_margin, self.eps)
                magnitude = self.obstacle_repulsive_gain * max(
                    1.0 / effective - 1.0 / influence, 0.0
                ) / (effective * effective)
                force += magnitude * away_direction

            # 对所有其他 UAV 施加近距离排斥；两 UAV 场景下即为唯一队友。
            for teammate_id, teammate in enumerate(teammate_positions):
                if teammate_id == agent_id:
                    continue
                away = position - teammate
                distance = float(np.linalg.norm(away))
                if distance >= self.teammate_safety_distance:
                    continue
                direction = away / max(distance, self.eps)
                magnitude = self.teammate_repulsive_gain * max(
                    1.0 / max(distance, self.eps)
                    - 1.0 / self.teammate_safety_distance,
                    0.0,
                ) / max(distance * distance, self.eps)
                force += magnitude * direction

            norm = float(np.linalg.norm(force))
            direction = force / max(norm, self.eps)
            # 接近最终目标时不越过目标；环境随后执行完整合法性验证。
            travel = min(self.lookahead_distance, goal_distance)
            result[agent_id] = np.clip(position + travel * direction, 0.0, boundary)
        return result.astype(np.float32)
