"""循环 PPO 的 on-policy 轨迹缓存。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class SequenceBatch:
    """单个智能体的一段连续序列。"""

    local_obs: np.ndarray
    initial_hidden: np.ndarray
    raw_actions: np.ndarray
    executed_actions: np.ndarray
    projection_delta_norm: np.ndarray
    intervention_mask: np.ndarray
    execution_target_eligible: np.ndarray
    execution_stage: np.ndarray
    old_log_probs: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    agent_id: int
    time_indices: np.ndarray


@dataclass
class PaddedSequenceBatch:
    """将多段不同长度 RNN 序列零填充后的 minibatch。"""

    local_obs: np.ndarray
    initial_hidden: np.ndarray
    raw_actions: np.ndarray
    executed_actions: np.ndarray
    projection_delta_norm: np.ndarray
    intervention_mask: np.ndarray
    execution_target_eligible: np.ndarray
    execution_stage: np.ndarray
    old_log_probs: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    mask: np.ndarray
    agent_ids: np.ndarray
    time_indices: np.ndarray


class RecurrentRolloutBuffer:
    """保存 PPO rollout，并计算 GAE/return。

    旧版把智能体数量固定为4，同时保存 message/raw_message。当前版本完全使用
    ``num_agents`` 动态维度，且通信字段已经删除。
    """

    def __init__(self, num_agents: int, *, team_gae: bool = False):
        self.num_agents = int(num_agents)
        if self.num_agents <= 0:
            raise ValueError("num_agents must be positive")
        self.team_gae = bool(team_gae)
        self.clear()

    def clear(self) -> None:
        self.local_obs = []
        self.global_states = []
        self.hidden = []
        self.actions = []
        self.raw_actions = []
        self.executed_actions = []
        self.projection_delta_norm = []
        self.intervention_mask = []
        self.execution_target_eligible = []
        self.actor_risk_eligible = []
        self.execution_stage = []
        self.safety_context = []
        self.safety_margin = []
        self.preaction_team_velocity = []
        self.preaction_goal_directions = []
        self.preaction_subgoal_directions = []
        self.preaction_per_agent_margins = []
        self.preaction_obstacle_tracks = []
        self.reference_actions = []
        self.reference_valid = []
        self.reference_stage_costs = []
        self.raw_stage_costs = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.advantages = None
        self.returns = None
        self._finalized = False

    def add(
        self,
        *,
        local_obs,
        global_state,
        hidden,
        actions,
        raw_actions,
        log_probs,
        rewards,
        dones,
        values,
        executed_actions=None,
        projection_delta_norm=None,
        intervention_mask=None,
        execution_target_eligible=None,
        actor_risk_eligible=None,
        execution_stage=None,
        safety_context=None,
        safety_margin=None,
        preaction_team_velocity=None,
        preaction_goal_directions=None,
        preaction_subgoal_directions=None,
        preaction_per_agent_margins=None,
        preaction_obstacle_tracks=None,
        reference_actions=None,
        reference_valid=None,
        reference_stage_costs=None,
        raw_stage_costs=None,
    ) -> None:
        """保存一个环境时间步的全部智能体数据。"""
        if self._finalized:
            raise RuntimeError("cannot add to a finalized rollout")

        obs = np.asarray(local_obs, dtype=np.float32)
        state = np.asarray(global_state, dtype=np.float32).reshape(-1)
        hidden_value = np.asarray(hidden, dtype=np.float32)
        expected_rows = self.num_agents

        action_values = np.asarray(actions, dtype=np.float32)
        executed_values = (
            action_values.copy() if executed_actions is None
            else np.asarray(executed_actions, dtype=np.float32)
        )
        projection_values = (
            np.linalg.norm(executed_values - action_values, axis=-1).astype(np.float32)
            if projection_delta_norm is None else
            np.asarray(projection_delta_norm, dtype=np.float32)
        )
        intervention_values = (
            projection_values > 1e-7 if intervention_mask is None else
            np.asarray(intervention_mask, dtype=bool)
        )
        eligibility_values = (
            np.zeros(expected_rows, dtype=bool)
            if execution_target_eligible is None else
            np.asarray(execution_target_eligible, dtype=bool)
        )
        actor_risk_eligibility = bool(
            True if actor_risk_eligible is None else actor_risk_eligible
        )
        stage_values = (
            np.zeros(expected_rows, dtype=np.int16)
            if execution_stage is None else
            np.asarray(execution_stage, dtype=np.int16)
        )
        context_values = (
            np.zeros(8, dtype=np.float32)
            if safety_context is None else
            np.asarray(safety_context, dtype=np.float32).reshape(-1)
        )
        margin_value = float(0.0 if safety_margin is None else safety_margin)
        if context_values.shape != (8,) or not np.isfinite(context_values).all():
            raise ValueError("safety_context must be a finite 8-vector")
        if not np.isfinite(margin_value):
            raise ValueError("safety_margin must be finite")
        velocity_values = np.zeros((expected_rows, 2), dtype=np.float32) if preaction_team_velocity is None else np.asarray(preaction_team_velocity, dtype=np.float32)
        goal_direction_values = np.zeros((expected_rows, 2), dtype=np.float32) if preaction_goal_directions is None else np.asarray(preaction_goal_directions, dtype=np.float32)
        subgoal_direction_values = np.zeros((expected_rows, 2), dtype=np.float32) if preaction_subgoal_directions is None else np.asarray(preaction_subgoal_directions, dtype=np.float32)
        margin_values = np.zeros((expected_rows, 3), dtype=np.float32) if preaction_per_agent_margins is None else np.asarray(preaction_per_agent_margins, dtype=np.float32)
        track_values = np.zeros((expected_rows, 0, 6), dtype=np.float32) if preaction_obstacle_tracks is None else np.asarray(preaction_obstacle_tracks, dtype=np.float32)
        reference_action_values = np.zeros((6, expected_rows, 2), dtype=np.float32) if reference_actions is None else np.asarray(reference_actions, dtype=np.float32)
        reference_valid_values = np.zeros(6, dtype=bool) if reference_valid is None else np.asarray(reference_valid, dtype=bool)
        reference_stage_values = np.zeros((6, 3), dtype=np.float32) if reference_stage_costs is None else np.asarray(reference_stage_costs, dtype=np.float32)
        raw_stage_values = np.zeros(3, dtype=np.float32) if raw_stage_costs is None else np.asarray(raw_stage_costs, dtype=np.float32)
        if velocity_values.shape != (expected_rows, 2):
            raise ValueError("preaction_team_velocity must have shape [UAV,2]")
        if goal_direction_values.shape != (expected_rows, 2):
            raise ValueError("preaction_goal_directions must have shape [UAV,2]")
        if subgoal_direction_values.shape != (expected_rows, 2):
            raise ValueError("preaction_subgoal_directions must have shape [UAV,2]")
        if margin_values.shape != (expected_rows, 3):
            raise ValueError("preaction_per_agent_margins must have shape [UAV,3]")
        if track_values.ndim != 3 or track_values.shape[0] != expected_rows or track_values.shape[2] != 6:
            raise ValueError("preaction_obstacle_tracks must have shape [UAV,slots,6]")
        if reference_action_values.shape != (6, expected_rows, 2):
            raise ValueError("reference_actions must have shape [6,UAV,2]")
        if reference_valid_values.shape != (6,):
            raise ValueError("reference_valid must have shape [6]")
        if reference_stage_values.shape != (6, 3) or raw_stage_values.shape != (3,):
            raise ValueError("reference/raw stage costs violate the three-stage contract")
        if not all(np.isfinite(value).all() for value in (
            velocity_values, goal_direction_values, subgoal_direction_values,
            margin_values, track_values, reference_action_values,
            reference_stage_values, raw_stage_values
        )):
            raise ValueError("pre-action critic features must be finite")
        row_values = {
            "local_obs": obs,
            "actions": action_values,
            "raw_actions": np.asarray(raw_actions, dtype=np.float32),
            "executed_actions": executed_values,
            "projection_delta_norm": projection_values,
            "intervention_mask": intervention_values,
            "execution_target_eligible": eligibility_values,
            "execution_stage": stage_values,
            "log_probs": np.asarray(log_probs, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "dones": np.asarray(dones, dtype=bool),
            "values": np.asarray(values, dtype=np.float32),
        }

        if obs.ndim != 2 or obs.shape[0] != expected_rows:
            raise ValueError(
                "local_obs must contain one row per configured agent"
            )
        if any(value.shape[0] != expected_rows for value in row_values.values()):
            raise ValueError(
                "every transition field must contain one row per configured agent"
            )
        if hidden_value.ndim != 3 or hidden_value.shape[:2] != (1, expected_rows):
            raise ValueError(
                "hidden must have shape [1, num_agents, hidden_dim]"
            )
        if state.size == 0:
            raise ValueError("global_state must be non-empty")
        if not all(
            np.isfinite(value).all()
            for key, value in row_values.items()
            if key != "dones"
        ):
            raise ValueError("rollout values must be finite")

        self.local_obs.append(obs.copy())
        self.global_states.append(state.copy())
        self.hidden.append(hidden_value[0].copy())
        self.actions.append(row_values["actions"].copy())
        self.raw_actions.append(row_values["raw_actions"].copy())
        self.executed_actions.append(row_values["executed_actions"].copy())
        self.projection_delta_norm.append(
            row_values["projection_delta_norm"].copy()
        )
        self.intervention_mask.append(row_values["intervention_mask"].copy())
        self.execution_target_eligible.append(
            row_values["execution_target_eligible"].copy()
        )
        self.actor_risk_eligible.append(actor_risk_eligibility)
        self.execution_stage.append(row_values["execution_stage"].copy())
        self.safety_context.append(context_values.copy())
        self.safety_margin.append(np.float32(margin_value))
        self.preaction_team_velocity.append(velocity_values.copy())
        self.preaction_goal_directions.append(goal_direction_values.copy())
        self.preaction_subgoal_directions.append(
            subgoal_direction_values.copy()
        )
        self.preaction_per_agent_margins.append(margin_values.copy())
        self.preaction_obstacle_tracks.append(track_values.copy())
        self.reference_actions.append(reference_action_values.copy())
        self.reference_valid.append(reference_valid_values.copy())
        self.reference_stage_costs.append(reference_stage_values.copy())
        self.raw_stage_costs.append(raw_stage_values.copy())
        self.log_probs.append(row_values["log_probs"].copy())
        self.rewards.append(row_values["rewards"].copy())
        self.dones.append(row_values["dones"].copy())
        self.values.append(row_values["values"].copy())

    def compute_returns_and_advantages(
        self,
        last_value,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """对 rollout 计算 GAE 优势和 return。"""
        if not self.rewards:
            raise ValueError("cannot finalize an empty rollout")

        self.local_obs = np.stack(self.local_obs)
        self.global_states = np.stack(self.global_states)
        self.hidden = np.stack(self.hidden)
        self.actions = np.stack(self.actions)
        self.raw_actions = np.stack(self.raw_actions)
        self.executed_actions = np.stack(self.executed_actions)
        self.projection_delta_norm = np.stack(self.projection_delta_norm)
        self.intervention_mask = np.stack(self.intervention_mask)
        self.execution_target_eligible = np.stack(
            self.execution_target_eligible
        )
        self.actor_risk_eligible = np.asarray(
            self.actor_risk_eligible, dtype=bool
        )
        self.execution_stage = np.stack(self.execution_stage)
        self.safety_context = np.stack(self.safety_context)
        self.safety_margin = np.asarray(self.safety_margin, dtype=np.float32)
        self.preaction_team_velocity = np.stack(self.preaction_team_velocity)
        self.preaction_goal_directions = np.stack(self.preaction_goal_directions)
        self.preaction_subgoal_directions = np.stack(
            self.preaction_subgoal_directions
        )
        self.preaction_per_agent_margins = np.stack(self.preaction_per_agent_margins)
        self.preaction_obstacle_tracks = np.stack(self.preaction_obstacle_tracks)
        self.reference_actions = np.stack(self.reference_actions)
        self.reference_valid = np.stack(self.reference_valid)
        self.reference_stage_costs = np.stack(self.reference_stage_costs)
        self.raw_stage_costs = np.stack(self.raw_stage_costs)
        self.log_probs = np.stack(self.log_probs)
        self.rewards = np.stack(self.rewards)
        self.dones = np.stack(self.dones)
        self.values = np.stack(self.values)

        final_value = np.asarray(last_value, dtype=np.float32).reshape(-1)
        if final_value.shape != (self.num_agents,):
            raise ValueError("last_value must contain one value per agent")

        self.advantages = np.zeros_like(self.rewards)
        if self.team_gae:
            for name, values in (
                ("rewards", self.rewards),
                ("dones", self.dones),
                ("values", self.values),
            ):
                reference = values[:, :1]
                if not np.all(values == reference):
                    raise ValueError(
                        f"team GAE requires identical per-agent {name}"
                    )
            if not np.all(final_value == final_value[:1]):
                raise ValueError(
                    "team GAE requires identical per-agent last values"
                )

            running = np.float32(0.0)
            next_value = np.float32(final_value[0])
            team_advantages = np.zeros(
                len(self.rewards), dtype=np.float32
            )
            for time in range(len(self.rewards) - 1, -1, -1):
                continuation = np.float32(
                    1.0 - float(self.dones[time, 0])
                )
                delta = np.float32(
                    self.rewards[time, 0]
                    + float(gamma) * next_value * continuation
                    - self.values[time, 0]
                )
                running = np.float32(
                    delta
                    + float(gamma)
                    * float(gae_lambda)
                    * continuation
                    * running
                )
                team_advantages[time] = running
                next_value = np.float32(self.values[time, 0])
            self.advantages[:] = team_advantages[:, None]
        else:
            running = np.zeros(self.num_agents, dtype=np.float32)
            next_value = final_value

            for time in range(len(self.rewards) - 1, -1, -1):
                continuation = 1.0 - self.dones[time].astype(np.float32)
                delta = (
                    self.rewards[time]
                    + float(gamma) * next_value * continuation
                    - self.values[time]
                )
                running = (
                    delta
                    + float(gamma)
                    * float(gae_lambda)
                    * continuation
                    * running
                )
                self.advantages[time] = running
                next_value = self.values[time]

        self.returns = self.advantages + self.values
        self._finalized = True

    def sequence_batches(self, sequence_length: int) -> Iterator[SequenceBatch]:
        """按 agent 切分连续 RNN 序列，episode 边界处不会跨越。"""
        if not self._finalized:
            raise RuntimeError("compute returns before requesting sequences")
        sequence_length = int(sequence_length)
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")

        total_steps = len(self.rewards)
        for agent_id in range(self.num_agents):
            start = 0
            while start < total_steps:
                end = min(start + sequence_length, total_steps)
                done_offsets = np.flatnonzero(
                    self.dones[start : max(start, end - 1), agent_id]
                )
                if done_offsets.size:
                    end = start + int(done_offsets[0]) + 1

                indices = np.arange(start, end, dtype=np.int64)
                yield SequenceBatch(
                    local_obs=self.local_obs[indices, agent_id],
                    initial_hidden=self.hidden[start, agent_id][None, None, :],
                    raw_actions=self.raw_actions[indices, agent_id],
                    executed_actions=self.executed_actions[indices, agent_id],
                    projection_delta_norm=self.projection_delta_norm[indices, agent_id],
                    intervention_mask=self.intervention_mask[indices, agent_id],
                    execution_target_eligible=self.execution_target_eligible[indices, agent_id],
                    execution_stage=self.execution_stage[indices, agent_id],
                    old_log_probs=self.log_probs[indices, agent_id],
                    advantages=self.advantages[indices, agent_id],
                    returns=self.returns[indices, agent_id],
                    agent_id=agent_id,
                    time_indices=indices,
                )
                start = end

    def sequence_minibatches(
        self,
        sequence_length: int,
        minibatch_size: int,
        rng: np.random.Generator,
        *,
        agent_id: int | None = None,
    ) -> Iterator[PaddedSequenceBatch]:
        """随机组合若干 RNN 序列，并用 mask 屏蔽零填充部分。"""
        minibatch_size = int(minibatch_size)
        if minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy Generator")

        if agent_id is not None and not 0 <= int(agent_id) < self.num_agents:
            raise ValueError("agent_id is outside the rollout")
        sequences = [
            item
            for item in self.sequence_batches(sequence_length)
            if agent_id is None or item.agent_id == int(agent_id)
        ]
        if not sequences:
            raise ValueError("no recurrent sequences matched the request")
        order = rng.permutation(len(sequences))

        for offset in range(0, len(order), minibatch_size):
            selected = [
                sequences[int(index)]
                for index in order[offset : offset + minibatch_size]
            ]
            max_time = max(len(item.time_indices) for item in selected)
            batch_size = len(selected)

            def padded(
                trailing_shape: tuple[int, ...], dtype
            ) -> np.ndarray:
                return np.zeros(
                    (max_time, batch_size, *trailing_shape),
                    dtype=dtype,
                )

            local_obs = padded(
                selected[0].local_obs.shape[1:], np.float32
            )
            raw_actions = padded(
                selected[0].raw_actions.shape[1:], np.float32
            )
            executed_actions = padded(
                selected[0].executed_actions.shape[1:], np.float32
            )
            projection_delta_norm = padded((), np.float32)
            intervention_mask = padded((), bool)
            execution_target_eligible = padded((), bool)
            execution_stage = padded((), np.int16)
            old_log_probs = padded((), np.float32)
            advantages = padded((), np.float32)
            returns = padded((), np.float32)
            mask = np.zeros((max_time, batch_size), dtype=bool)
            time_indices = np.full(
                (max_time, batch_size), -1, dtype=np.int64
            )
            initial_hidden = np.concatenate(
                [item.initial_hidden for item in selected], axis=1
            )
            agent_ids = np.asarray(
                [item.agent_id for item in selected], dtype=np.int64
            )

            for column, item in enumerate(selected):
                length = len(item.time_indices)
                local_obs[:length, column] = item.local_obs
                raw_actions[:length, column] = item.raw_actions
                executed_actions[:length, column] = item.executed_actions
                projection_delta_norm[:length, column] = item.projection_delta_norm
                intervention_mask[:length, column] = item.intervention_mask
                execution_target_eligible[:length, column] = item.execution_target_eligible
                execution_stage[:length, column] = item.execution_stage
                old_log_probs[:length, column] = item.old_log_probs
                advantages[:length, column] = item.advantages
                returns[:length, column] = item.returns
                mask[:length, column] = True
                time_indices[:length, column] = item.time_indices

            yield PaddedSequenceBatch(
                local_obs=local_obs,
                initial_hidden=initial_hidden,
                raw_actions=raw_actions,
                executed_actions=executed_actions,
                projection_delta_norm=projection_delta_norm,
                intervention_mask=intervention_mask,
                execution_target_eligible=execution_target_eligible,
                execution_stage=execution_stage,
                old_log_probs=old_log_probs,
                advantages=advantages,
                returns=returns,
                mask=mask,
                agent_ids=agent_ids,
                time_indices=time_indices,
            )
