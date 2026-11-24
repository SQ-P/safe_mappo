"""Immutable shadow-action evaluation for avoidable intervention labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import torch

from .contracts import CausalSafetySnapshotV1, InterventionLevel


STAGE_NAMES = ("initial_joint_guard", "obstacle_ttc", "final_joint_guard")


@dataclass(frozen=True)
class ShadowSafetyResultV1:
    requested_actions: tuple[tuple[float, ...], ...]
    stage_delta_v_mps: tuple[float, float, float]
    stage_levels: tuple[InterventionLevel, InterventionLevel, InterventionLevel]
    executed_actions: tuple[tuple[float, ...], ...]

    @property
    def stage_costs(self) -> tuple[float, float, float]:
        return tuple(
            (1.0 + float(np.clip(delta_v / 0.25, 0.0, 1.0)))
            if level >= InterventionLevel.MATERIAL else 0.0
            for delta_v, level in zip(self.stage_delta_v_mps, self.stage_levels)
        )

    @property
    def total_cost(self) -> float:
        return float(np.mean(self.stage_costs))


@dataclass(frozen=True)
class AvoidableInterventionLabelV1:
    raw_cost: float
    best_valid_shadow_cost: float
    avoidable_cost: float
    best_shadow_index: int | None
    valid_shadow_count: int


@dataclass(frozen=True)
class ShadowSafetyResultV2:
    """Per-agent staged shadow result for the team-mean R3 cost contract."""

    requested_actions: tuple[tuple[float, ...], ...]
    per_agent_stage_delta_v_mps: tuple[tuple[float, float, float], ...]
    per_agent_stage_levels: tuple[
        tuple[InterventionLevel, InterventionLevel, InterventionLevel], ...
    ]
    executed_actions: tuple[tuple[float, ...], ...]

    @property
    def per_agent_stage_costs(self) -> tuple[tuple[float, float, float], ...]:
        delta = np.asarray(self.per_agent_stage_delta_v_mps, dtype=np.float64)
        levels = np.asarray(self.per_agent_stage_levels, dtype=np.int8)
        if delta.ndim != 2 or delta.shape[1] != 3 or levels.shape != delta.shape:
            raise ValueError("V2 shadow result must contain [UAV,3] stage values")
        costs = (levels >= int(InterventionLevel.MATERIAL)) * (
            1.0 + np.clip(delta / 0.25, 0.0, 1.0)
        )
        return tuple(tuple(float(value) for value in row) for row in costs)

    @property
    def stage_costs(self) -> tuple[float, float, float]:
        costs = np.asarray(self.per_agent_stage_costs, dtype=np.float64)
        return tuple(float(value) for value in costs.mean(axis=0))

    @property
    def total_cost(self) -> float:
        return float(np.mean(self.stage_costs))


@dataclass(frozen=True)
class AvoidableInterventionLabelV2:
    raw_stage_costs: tuple[float, float, float]
    best_valid_shadow_stage_costs: tuple[float, float, float]
    stage_avoidable_costs: tuple[float, float, float]
    raw_cost: float
    best_valid_shadow_cost: float
    avoidable_cost: float
    best_shadow_index: int | None
    valid_shadow_count: int
    shadow_stage_costs: tuple[tuple[float, float, float], ...] = ()
    shadow_total_costs: tuple[float, ...] = ()
    valid_shadow_mask: tuple[bool, ...] = ()


def registered_shadow_actions(
    raw_actions: Sequence[Sequence[float]],
    goal_direction_actions: Sequence[Sequence[float]],
    *,
    sobol_count: int = 24,
    sobol_seed: int = 0,
) -> tuple[np.ndarray, ...]:
    """Return raw, HOLD, brake, four goal scales, and Sobol joint actions."""

    raw = np.asarray(raw_actions, dtype=np.float32)
    goal = np.asarray(goal_direction_actions, dtype=np.float32)
    if raw.shape != goal.shape or raw.ndim != 2:
        raise ValueError("raw and goal-direction actions must share [UAV, action] shape")
    if sobol_count != 24:
        raise ValueError("IA-MAPPO V1 preregisters exactly 24 Sobol candidates")
    candidates: list[np.ndarray] = [raw.copy()]
    candidates.extend(registered_reference_actions(raw, goal))
    engine = torch.quasirandom.SobolEngine(
        dimension=int(raw.size), scramble=True, seed=int(sobol_seed)
    )
    sobol = engine.draw(sobol_count).cpu().numpy().astype(np.float32)
    sobol = 2.0 * sobol - 1.0
    candidates.extend(row.reshape(raw.shape) for row in sobol)
    return tuple(np.asarray(item, dtype=np.float32) for item in candidates)


def registered_reference_actions(
    raw_actions: Sequence[Sequence[float]],
    goal_direction_actions: Sequence[Sequence[float]],
) -> tuple[np.ndarray, ...]:
    """Return the six deterministic R4.4/CSA online references."""

    raw = np.asarray(raw_actions, dtype=np.float32)
    goal = np.asarray(goal_direction_actions, dtype=np.float32)
    if raw.shape != goal.shape or raw.ndim != 2:
        raise ValueError("raw and goal-direction actions must share [UAV, action] shape")
    references: list[np.ndarray] = [np.zeros_like(raw), -np.sign(raw)]
    references.extend(
        np.clip(goal * scale, -1.0, 1.0)
        for scale in (0.25, 0.50, 0.75, 1.00)
    )
    return tuple(np.asarray(item, dtype=np.float32) for item in references)


def goal_progress_velocity(
    actions: np.ndarray,
    goal_directions: np.ndarray,
    *,
    acceleration_scale: float,
    dt: float,
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float64)
    directions = np.asarray(goal_directions, dtype=np.float64)
    if actions.shape != directions.shape:
        raise ValueError("actions and goal directions must share shape")
    norms = np.linalg.norm(directions, axis=-1, keepdims=True)
    unit = np.divide(directions, norms, out=np.zeros_like(directions), where=norms > 1e-12)
    return np.sum(actions * unit, axis=-1) * float(acceleration_scale) * float(dt)


def next_goal_progress_velocity(
    actions: np.ndarray,
    current_velocities: np.ndarray,
    goal_directions: np.ndarray,
    *,
    acceleration_scale: float,
    dt: float,
    maximum_speed: float,
) -> np.ndarray:
    """Project the clipped candidate next velocity onto each goal direction."""

    actions = np.asarray(actions, dtype=np.float64)
    velocities = np.asarray(current_velocities, dtype=np.float64)
    directions = np.asarray(goal_directions, dtype=np.float64)
    if actions.shape != velocities.shape or actions.shape != directions.shape:
        raise ValueError("actions, velocities and goal directions must share shape")
    if acceleration_scale <= 0.0 or dt <= 0.0 or maximum_speed <= 0.0:
        raise ValueError("dynamics scales must be positive")
    candidate = np.clip(
        velocities + np.clip(actions, -1.0, 1.0)
        * float(acceleration_scale) * float(dt),
        -float(maximum_speed),
        float(maximum_speed),
    )
    direction_norms = np.linalg.norm(directions, axis=-1, keepdims=True)
    unit = np.divide(
        directions, direction_norms, out=np.zeros_like(directions),
        where=direction_norms > 1e-12,
    )
    return np.sum(candidate * unit, axis=-1)


def evaluate_avoidable_intervention(
    snapshot: CausalSafetySnapshotV1,
    shadows: Iterable[np.ndarray],
    safety_stack: Callable[[CausalSafetySnapshotV1, np.ndarray], ShadowSafetyResultV1],
    *,
    goal_directions: np.ndarray,
    reached_mask: np.ndarray,
    acceleration_scale: float,
    dt: float,
    maximum_progress_loss_mps: float = 0.10,
) -> AvoidableInterventionLabelV1:
    """Evaluate shadows while certifying that the causal snapshot is unchanged."""

    digest_before = snapshot.digest
    shadow_list = tuple(np.asarray(item, dtype=np.float32) for item in shadows)
    if not shadow_list:
        raise ValueError("at least the raw action candidate is required")
    results: list[ShadowSafetyResultV1] = []
    for action in shadow_list:
        result = safety_stack(snapshot, action.copy())
        if snapshot.digest != digest_before:
            raise RuntimeError("shadow safety evaluation mutated its causal snapshot")
        results.append(result)

    raw_result = results[0]
    raw_progress = goal_progress_velocity(
        np.asarray(raw_result.executed_actions), goal_directions,
        acceleration_scale=acceleration_scale, dt=dt,
    )
    valid: list[tuple[int, float]] = []
    reached = np.asarray(reached_mask, dtype=bool)
    for index, result in enumerate(results):
        progress = goal_progress_velocity(
            np.asarray(result.executed_actions), goal_directions,
            acceleration_scale=acceleration_scale, dt=dt,
        )
        active = ~reached
        preserves = np.all(progress[active] >= raw_progress[active] - maximum_progress_loss_mps)
        nonnegative = np.all(progress[active] >= 0.0)
        if bool(preserves and nonnegative):
            valid.append((index, result.total_cost))
    if not valid:
        return AvoidableInterventionLabelV1(
            raw_cost=raw_result.total_cost,
            best_valid_shadow_cost=raw_result.total_cost,
            avoidable_cost=0.0,
            best_shadow_index=None,
            valid_shadow_count=0,
        )
    best_index, best_cost = min(valid, key=lambda pair: (pair[1], pair[0]))
    return AvoidableInterventionLabelV1(
        raw_cost=raw_result.total_cost,
        best_valid_shadow_cost=float(best_cost),
        avoidable_cost=max(0.0, raw_result.total_cost - float(best_cost)),
        best_shadow_index=int(best_index),
        valid_shadow_count=len(valid),
    )


def evaluate_avoidable_intervention_v2(
    snapshot: CausalSafetySnapshotV1,
    shadows: Iterable[np.ndarray],
    safety_stack: Callable[[CausalSafetySnapshotV1, np.ndarray], ShadowSafetyResultV2],
    *,
    goal_directions: np.ndarray,
    current_velocities: np.ndarray,
    reached_mask: np.ndarray,
    acceleration_scale: float,
    dt: float,
    maximum_speed: float,
    maximum_progress_loss_mps: float = 0.10,
) -> AvoidableInterventionLabelV2:
    """Evaluate the immutable panel and attribute total avoidability by stage.

    The best candidate minimizes the preregistered equal-weight total cost.
    Positive stage improvements are scaled so their mean equals the exact
    total avoidable cost.  This preserves stage attribution without allowing
    a safety-stage trade-off to inflate the training target.
    """

    digest_before = snapshot.digest
    shadow_list = tuple(np.asarray(item, dtype=np.float32) for item in shadows)
    if not shadow_list:
        raise ValueError("at least the raw action candidate is required")
    results: list[ShadowSafetyResultV2] = []
    for action in shadow_list:
        result = safety_stack(snapshot, action.copy())
        if snapshot.digest != digest_before:
            raise RuntimeError("shadow safety evaluation mutated its causal snapshot")
        results.append(result)

    raw_result = results[0]
    raw_progress = next_goal_progress_velocity(
        np.asarray(raw_result.executed_actions), current_velocities,
        goal_directions, acceleration_scale=acceleration_scale, dt=dt,
        maximum_speed=maximum_speed,
    )
    reached = np.asarray(reached_mask, dtype=bool)
    valid: list[tuple[int, float]] = []
    for index, result in enumerate(results):
        progress = next_goal_progress_velocity(
            np.asarray(result.executed_actions), current_velocities,
            goal_directions, acceleration_scale=acceleration_scale, dt=dt,
            maximum_speed=maximum_speed,
        )
        active = ~reached
        preserves = np.all(
            progress[active] >= raw_progress[active] - maximum_progress_loss_mps
        )
        nonnegative = np.all(progress[active] >= 0.0)
        if bool(preserves and nonnegative):
            valid.append((index, result.total_cost))

    raw_stage = np.asarray(raw_result.stage_costs, dtype=np.float64)
    raw_total = float(raw_result.total_cost)
    if not valid:
        return AvoidableInterventionLabelV2(
            raw_stage_costs=tuple(map(float, raw_stage)),
            best_valid_shadow_stage_costs=tuple(map(float, raw_stage)),
            stage_avoidable_costs=(0.0, 0.0, 0.0),
            raw_cost=raw_total,
            best_valid_shadow_cost=raw_total,
            avoidable_cost=0.0,
            best_shadow_index=None,
            valid_shadow_count=0,
            shadow_stage_costs=tuple(result.stage_costs for result in results),
            shadow_total_costs=tuple(float(result.total_cost) for result in results),
            valid_shadow_mask=tuple(False for _ in results),
        )
    best_index, best_total = min(valid, key=lambda pair: (pair[1], pair[0]))
    best_stage = np.asarray(results[best_index].stage_costs, dtype=np.float64)
    total_avoidable = max(0.0, raw_total - float(best_total))
    positive_improvement = np.clip(raw_stage - best_stage, 0.0, None)
    if total_avoidable > 0.0 and float(positive_improvement.sum()) > 0.0:
        stage_avoidable = positive_improvement * (
            3.0 * total_avoidable / float(positive_improvement.sum())
        )
    else:
        stage_avoidable = np.zeros(3, dtype=np.float64)
    if not np.isclose(float(stage_avoidable.mean()), total_avoidable, atol=1e-7):
        raise RuntimeError("stage attribution does not conserve avoidable total cost")
    return AvoidableInterventionLabelV2(
        raw_stage_costs=tuple(map(float, raw_stage)),
        best_valid_shadow_stage_costs=tuple(map(float, best_stage)),
        stage_avoidable_costs=tuple(map(float, stage_avoidable)),
        raw_cost=raw_total,
        best_valid_shadow_cost=float(best_total),
        avoidable_cost=total_avoidable,
        best_shadow_index=int(best_index),
        valid_shadow_count=len(valid),
        shadow_stage_costs=tuple(result.stage_costs for result in results),
        shadow_total_costs=tuple(float(result.total_cost) for result in results),
        valid_shadow_mask=tuple(
            index in {item[0] for item in valid}
            for index in range(len(results))
        ),
    )
