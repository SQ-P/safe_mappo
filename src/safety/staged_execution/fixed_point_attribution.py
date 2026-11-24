"""Deterministic fixed-point attribution for frozen execution stacks.

The legacy ``action_fixed_point`` metric observes only the final executed
action.  This module keeps that definition intact while separating an Actor
zero-action failure from a non-zero Actor request that is absorbed by the
frozen recovery/guard/TTC stack.  It is telemetry-only and never changes an
action.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


ACTION_STAGE_ORDER = (
    "actor_raw",
    "recovery_overlay",
    "initial_joint_guard",
    "obstacle_ttc",
    "final_joint_guard",
    "executed",
)


def _matrix(
    traces: Mapping[str, Sequence[np.ndarray]], stage: str,
) -> np.ndarray:
    values = np.asarray(traces[stage], dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError(
            f"action stage {stage!r} must have shape (steps, agents, 2)"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"action stage {stage!r} contains non-finite values")
    return values


def classify_fixed_point_attribution(
    *,
    reason: str,
    final_goal_distances: Sequence[float],
    unfinished_distance_threshold: float,
    positions: Sequence[np.ndarray],
    action_stage_traces: Mapping[str, Sequence[np.ndarray]],
    action_threshold: float = 0.001,
    displacement_threshold_m: float = 0.25,
    tail_steps: int = 20,
    modification_tolerance: float = 1e-7,
) -> dict[str, object]:
    """Classify final-tail stalls without changing legacy execution semantics.

    Every execution fixed-point agent belongs to exactly one of
    ``actor_raw``, ``safety_absorbed`` or ``unattributed``.  Reached agents are
    excluded through ``unfinished_distance_threshold``.
    """

    if tail_steps <= 0:
        raise ValueError("tail_steps must be positive")
    missing = [stage for stage in ACTION_STAGE_ORDER if stage not in action_stage_traces]
    if missing:
        raise ValueError(f"missing action stages: {missing}")
    matrices = {
        stage: _matrix(action_stage_traces, stage)
        for stage in ACTION_STAGE_ORDER
    }
    shapes = {values.shape for values in matrices.values()}
    if len(shapes) != 1:
        raise ValueError("all action stages must have identical shapes")
    step_count, agent_count, _ = next(iter(shapes))
    if step_count <= 0:
        raise ValueError("at least one action step is required")

    position_values = np.asarray(positions, dtype=np.float64)
    if position_values.shape != (step_count + 1, agent_count, 2):
        raise ValueError(
            "positions must have shape (action_steps + 1, agents, 2)"
        )
    final_distances = np.asarray(final_goal_distances, dtype=np.float64)
    if final_distances.shape != (agent_count,):
        raise ValueError("final_goal_distances has the wrong shape")

    tail = min(int(tail_steps), step_count)
    stage_tail = {stage: values[-tail:] for stage, values in matrices.items()}
    stage_norm_means = {
        stage: np.mean(np.linalg.norm(values, axis=2), axis=0)
        for stage, values in stage_tail.items()
    }
    displacement = np.linalg.norm(
        position_values[-1] - position_values[-(tail + 1)], axis=1
    )
    timed_out = str(reason) == "timeout"
    unfinished = final_distances > float(unfinished_distance_threshold)
    stationary = displacement <= float(displacement_threshold_m)
    raw_fixed = (
        timed_out
        & unfinished
        & stationary
        & (stage_norm_means["actor_raw"] <= float(action_threshold))
    )
    execution_fixed = (
        timed_out
        & unfinished
        & stationary
        & (stage_norm_means["executed"] <= float(action_threshold))
    )

    stage_pairs = (
        ("recovery_overlay", "actor_raw", "recovery_overlay"),
        ("initial_joint_guard", "recovery_overlay", "initial_joint_guard"),
        ("obstacle_ttc", "initial_joint_guard", "obstacle_ttc"),
        ("final_joint_guard", "obstacle_ttc", "final_joint_guard"),
        ("executed", "final_joint_guard", "executed"),
    )
    stage_delta_means: dict[str, np.ndarray] = {}
    for label, before, after in stage_pairs:
        stage_delta_means[label] = np.mean(
            np.linalg.norm(stage_tail[after] - stage_tail[before], axis=2),
            axis=0,
        )
    total_delta = np.mean(
        np.linalg.norm(
            stage_tail["executed"] - stage_tail["actor_raw"], axis=2
        ),
        axis=0,
    )
    raw_execution_fixed = raw_fixed & execution_fixed
    safety_absorbed = (
        execution_fixed
        & ~raw_fixed
        & (total_delta > float(modification_tolerance))
    )
    unattributed = (
        execution_fixed & ~raw_execution_fixed & ~safety_absorbed
    )

    labels = list(stage_delta_means)
    delta_stack = np.stack([stage_delta_means[label] for label in labels], axis=0)
    primary_per_agent: list[str | None] = []
    for agent_id in range(agent_count):
        if raw_execution_fixed[agent_id]:
            primary_per_agent.append("actor_raw")
        elif safety_absorbed[agent_id]:
            primary_per_agent.append(labels[int(np.argmax(delta_stack[:, agent_id]))])
        elif unattributed[agent_id]:
            primary_per_agent.append("unattributed")
        else:
            primary_per_agent.append(None)

    stage_counts: dict[str, int] = {}
    for label in primary_per_agent:
        if label is not None:
            stage_counts[label] = stage_counts.get(label, 0) + 1
    episode_primary = None
    if stage_counts:
        episode_primary = sorted(
            stage_counts, key=lambda key: (-stage_counts[key], key)
        )[0]

    return {
        "tail_steps": int(tail),
        "actor_raw_fixed_point_agents": raw_fixed.astype(bool).tolist(),
        "actor_raw_fixed_point": bool(np.any(raw_fixed)),
        "execution_fixed_point_agents": execution_fixed.astype(bool).tolist(),
        "execution_fixed_point": bool(np.any(execution_fixed)),
        "safety_absorbed_fixed_point_agents": (
            safety_absorbed.astype(bool).tolist()
        ),
        "safety_absorbed_fixed_point": bool(np.any(safety_absorbed)),
        "fixed_point_unattributed_agents": unattributed.astype(bool).tolist(),
        "fixed_point_unattributed_count": int(np.sum(unattributed)),
        "fixed_point_primary_stage": episode_primary,
        "fixed_point_primary_stage_per_agent": primary_per_agent,
        "fixed_point_stage_counts": stage_counts,
        "last_20_displacement_m": displacement.astype(float).tolist(),
        "last_20_stage_action_norm_mean_per_agent": {
            stage: values.astype(float).tolist()
            for stage, values in stage_norm_means.items()
        },
        "last_20_stage_delta_norm_mean_per_agent": {
            stage: values.astype(float).tolist()
            for stage, values in stage_delta_means.items()
        },
        "last_20_total_action_delta_norm_mean_per_agent": (
            total_delta.astype(float).tolist()
        ),
        "classification_count_conserved": bool(
            np.array_equal(
                execution_fixed,
                raw_execution_fixed | safety_absorbed | unattributed,
            )
            and not np.any(raw_execution_fixed & safety_absorbed)
            and not np.any(raw_execution_fixed & unattributed)
            and not np.any(safety_absorbed & unattributed)
        ),
    }
