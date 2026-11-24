"""Deterministic, actor-independent recovery control.

The controller is deliberately small and observable-only.  It does not replace
the execution guard: every action it returns must still be projected by
``JointSafetyGuardV2`` before it reaches the dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .contracts import ObservableObstacleV1


@dataclass(frozen=True)
class StarRecoveryCommandV1:
    available: bool
    actions: np.ndarray
    targets: np.ndarray
    affected_agent_mask: tuple[bool, ...]
    reason: str
    entry_sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class StarRecoveryCandidateV2:
    """One observable-only V5.4 recovery candidate before execution scoring."""

    label: str
    actions: np.ndarray
    targets: np.ndarray
    affected_agent_mask: tuple[bool, ...]
    entry_sources: tuple[str, ...] = ()


def _unit(values: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    return np.zeros(2, dtype=np.float64) if norm <= 1e-12 else values / norm


def build_star_recovery_command(
    *,
    positions: np.ndarray,
    velocities: np.ndarray,
    final_goals: np.ndarray,
    visible_obstacles: Sequence[Sequence[ObservableObstacleV1]],
    map_size: np.ndarray,
    agent_radius: float,
    surface_clearance: float,
    invariant_buffer: float,
    max_speed: float,
    max_acceleration: float,
    dt: float,
    teammate_safety_distance: float,
    obstacle_safety_margin: float,
    offending_pairs: Sequence[Sequence[int]] = (),
    affected_agent_mask: Sequence[bool] | None = None,
    target_distance: float = 6.0,
    speed_fraction: float = 0.5,
    boundary_trigger_margin: float = 1.0,
    obstacle_repulsion_margin: float = 4.0,
    strict_affected_scope: bool = False,
    attempt_index: int = 0,
    entry_sources: Sequence[str] = (),
    persistent_margin_mode: bool = False,
) -> StarRecoveryCommandV1:
    """Construct a permutation-equivariant recovery command.

    Pair repulsion has priority, wall-normal motion prevents boundary locking,
    and the remaining goal component supplies a deterministic tangent.  Exact
    coincident agents are reported as unavailable because breaking that
    symmetry with an agent id would violate permutation equivariance.
    """

    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    goals = np.asarray(final_goals, dtype=np.float64)
    n = len(positions)
    shape = (n, 2)
    if positions.shape != shape or velocities.shape != shape or goals.shape != shape:
        raise ValueError("recovery positions, velocities and goals must be Nx2")
    if len(visible_obstacles) != n:
        raise ValueError("recovery visible-obstacle rows must match the team size")

    low_value = agent_radius + surface_clearance + invariant_buffer
    low = np.full(2, low_value, dtype=np.float64)
    high = np.asarray(map_size, dtype=np.float64) - low
    if np.any(high <= low):
        raise ValueError("recovery admissible domain is empty")

    explicit = (
        np.zeros(n, dtype=bool)
        if affected_agent_mask is None
        else np.asarray(affected_agent_mask, dtype=bool).copy()
    )
    if explicit.shape != (n,):
        raise ValueError("affected_agent_mask must contain one value per agent")
    normalized_sources = tuple(sorted({str(value) for value in entry_sources}))
    supported_sources = {"boundary", "pair", "obstacle_guard_fallback"}
    if normalized_sources and not set(normalized_sources) <= supported_sources:
        raise ValueError("unsupported STAR recovery entry source")
    source_limited = bool(normalized_sources)
    pair_enabled = not source_limited or "pair" in normalized_sources
    boundary_enabled = not source_limited or "boundary" in normalized_sources
    obstacle_enabled = (
        not source_limited or "obstacle_guard_fallback" in normalized_sources
    )

    pair_force = np.zeros(shape, dtype=np.float64)
    normalized_pairs: list[tuple[int, int]] = []
    for raw_left, raw_right in offending_pairs if pair_enabled else ():
        left, right = sorted((int(raw_left), int(raw_right)))
        if left < 0 or right >= n or left == right:
            raise ValueError("invalid recovery offending pair")
        delta = positions[left] - positions[right]
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-9:
            return StarRecoveryCommandV1(
                False, np.zeros(shape, dtype=np.float32), positions.astype(np.float32),
                tuple(bool(x) for x in explicit), "coincident_pair_symmetry",
                normalized_sources,
            )
        normal = delta / distance
        protected = teammate_safety_distance + invariant_buffer
        strength = max(1.0, (protected + target_distance - distance) / max(target_distance, 1e-6))
        pair_force[left] += strength * normal
        pair_force[right] -= strength * normal
        explicit[left] = True
        explicit[right] = True
        normalized_pairs.append((left, right))

    boundary_force = np.zeros(shape, dtype=np.float64)
    projected_goal = np.zeros(shape, dtype=np.float64)
    boundary_affected = np.zeros(n, dtype=bool)
    for agent_id in range(n):
        if strict_affected_scope and not explicit[agent_id]:
            continue
        goal_direction = _unit(goals[agent_id] - positions[agent_id])
        for axis in range(2):
            if not boundary_enabled:
                continue
            near_low = positions[agent_id, axis] <= low[axis] + boundary_trigger_margin
            near_high = positions[agent_id, axis] >= high[axis] - boundary_trigger_margin
            if near_low:
                boundary_force[agent_id, axis] += 1.0
                goal_direction[axis] = max(goal_direction[axis], 0.0)
                boundary_affected[agent_id] = True
            if near_high:
                boundary_force[agent_id, axis] -= 1.0
                goal_direction[axis] = min(goal_direction[axis], 0.0)
                boundary_affected[agent_id] = True
        projected_goal[agent_id] = goal_direction
    if not strict_affected_scope:
        explicit |= boundary_affected

    obstacle_force = np.zeros(shape, dtype=np.float64)
    obstacle_tangent = np.zeros(shape, dtype=np.float64)
    for agent_id, tracks in enumerate(visible_obstacles):
        if strict_affected_scope and not explicit[agent_id]:
            continue
        for obstacle in tracks if obstacle_enabled else ():
            relative = np.asarray(obstacle.relative_position, dtype=np.float64)
            closest_relative = relative.copy()
            if persistent_margin_mode and bool(obstacle.velocity_valid):
                # The causal track stores d(relative_position)/dt, so this is
                # already obstacle velocity relative to the observing UAV.
                # Subtracting the UAV velocity again would double-count ego
                # motion and distort the predicted closest point.
                relative_velocity = np.asarray(
                    obstacle.relative_velocity, dtype=np.float64
                )
                speed_squared = float(np.dot(relative_velocity, relative_velocity))
                if speed_squared > 1e-10:
                    closest_time = float(np.clip(
                        -np.dot(relative, relative_velocity) / speed_squared,
                        0.0,
                        5.0,
                    ))
                    closest_relative = relative + closest_time * relative_velocity
            distance = float(np.linalg.norm(closest_relative))
            clearance = (
                distance - obstacle.radius - agent_radius - obstacle_safety_margin
            )
            if clearance >= obstacle_repulsion_margin or distance <= 1e-9:
                continue
            strength = max(
                0.0,
                (obstacle_repulsion_margin - clearance)
                / max(obstacle_repulsion_margin, 1e-6),
            )
            away = -closest_relative / distance
            obstacle_force[agent_id] += strength * away
            if persistent_margin_mode:
                tangent_left = np.asarray([-away[1], away[0]], dtype=np.float64)
                tangent_right = -tangent_left
                goal_direction = _unit(goals[agent_id] - positions[agent_id])
                chosen_tangent = (
                    tangent_left
                    if float(np.dot(tangent_left, goal_direction))
                    >= float(np.dot(tangent_right, goal_direction))
                    else tangent_right
                )
                obstacle_tangent[agent_id] += strength * chosen_tangent
            if not strict_affected_scope:
                explicit[agent_id] = True

    actions = np.zeros(shape, dtype=np.float64)
    targets = positions.copy()
    commanded = False
    desired_speed = max_speed * speed_fraction
    for agent_id in range(n):
        if not explicit[agent_id]:
            actions[agent_id] = np.clip(
                -velocities[agent_id] / max(max_acceleration * dt, 1e-12),
                -1.0,
                1.0,
            )
            continue
        base_direction = _unit(
            2.0 * pair_force[agent_id]
            + 1.5 * boundary_force[agent_id]
            + 1.5 * obstacle_force[agent_id]
            + (0.65 * obstacle_tangent[agent_id] if persistent_margin_mode else 0.0)
            + 0.75 * projected_goal[agent_id]
        )
        # V4 retry attempts rotate the same observable recovery geometry instead
        # of replaying an identical failed target.  The global tangent is
        # deterministic and independent of agent identity, so permutation
        # equivariance is preserved.
        if int(attempt_index) > 0 and float(np.linalg.norm(base_direction)) > 1e-9:
            tangent_sign = 1.0 if int(attempt_index) % 2 else -1.0
            tangent = tangent_sign * np.asarray(
                [-base_direction[1], base_direction[0]], dtype=np.float64
            )
            direction = _unit(
                1.25 * boundary_force[agent_id]
                + 1.00 * pair_force[agent_id]
                + 0.75 * obstacle_force[agent_id]
                + 0.60 * base_direction
                + 0.55 * tangent
            )
        else:
            direction = base_direction
        if float(np.linalg.norm(direction)) <= 1e-9:
            continue
        target = np.clip(
            positions[agent_id] + target_distance * direction, low, high
        )
        target_delta = target - positions[agent_id]
        target_direction = _unit(target_delta)
        if float(np.linalg.norm(target_direction)) <= 1e-9:
            continue
        desired_velocity = desired_speed * target_direction
        actions[agent_id] = np.clip(
            (desired_velocity - velocities[agent_id])
            / max(max_acceleration * dt, 1e-12),
            -1.0,
            1.0,
        )
        targets[agent_id] = target
        commanded = True

    reason = (
        "pair_boundary_obstacle_recovery"
        if normalized_pairs else "boundary_or_obstacle_recovery"
    )
    return StarRecoveryCommandV1(
        bool(commanded), actions.astype(np.float32), targets.astype(np.float32),
        tuple(bool(x) for x in explicit), reason if commanded else "no_recovery_direction",
        normalized_sources,
    )


def build_star_recovery_candidates(
    *,
    positions: np.ndarray,
    velocities: np.ndarray,
    final_goals: np.ndarray,
    visible_obstacles: Sequence[Sequence[ObservableObstacleV1]],
    map_size: np.ndarray,
    agent_radius: float,
    surface_clearance: float,
    invariant_buffer: float,
    max_speed: float,
    max_acceleration: float,
    dt: float,
    teammate_safety_distance: float,
    obstacle_safety_margin: float,
    offending_pairs: Sequence[Sequence[int]] = (),
    affected_agent_mask: Sequence[bool] | None = None,
    target_distance: float = 6.0,
    speed_fraction: float = 0.5,
    boundary_trigger_margin: float = 1.0,
    obstacle_repulsion_margin: float = 4.0,
    entry_sources: Sequence[str] = (),
) -> tuple[StarRecoveryCandidateV2, ...]:
    """Generate explicit normal/left/right recovery alternatives for V5.4.

    This function intentionally does not decide which candidate is safe.  The
    environment scores every candidate *after* corridor, joint-guard and TTC
    projection so a direction that is subsequently projected to zero cannot
    win merely because its pre-projection geometry looked attractive.

    Only current observable relative position and causal relative velocity are
    used.  Simulator obstacle velocity, future trajectory and obstacle type are
    deliberately unavailable here.
    """

    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    goals = np.asarray(final_goals, dtype=np.float64)
    n = len(positions)
    shape = (n, 2)
    if positions.shape != shape or velocities.shape != shape or goals.shape != shape:
        raise ValueError("recovery positions, velocities and goals must be Nx2")
    if len(visible_obstacles) != n:
        raise ValueError("recovery visible-obstacle rows must match the team size")

    sources = tuple(sorted({str(value) for value in entry_sources}))
    supported_sources = {"boundary", "pair", "obstacle_guard_fallback"}
    if not sources or not set(sources) <= supported_sources:
        raise ValueError("V5.4 recovery candidates require supported entry sources")

    explicit = (
        np.zeros(n, dtype=bool)
        if affected_agent_mask is None
        else np.asarray(affected_agent_mask, dtype=bool).copy()
    )
    if explicit.shape != (n,):
        raise ValueError("affected_agent_mask must contain one value per agent")

    low_value = agent_radius + surface_clearance + invariant_buffer
    low = np.full(2, low_value, dtype=np.float64)
    high = np.asarray(map_size, dtype=np.float64) - low
    if np.any(high <= low):
        raise ValueError("recovery admissible domain is empty")

    pair_force = np.zeros(shape, dtype=np.float64)
    if "pair" in sources:
        for raw_left, raw_right in offending_pairs:
            left, right = sorted((int(raw_left), int(raw_right)))
            if left < 0 or right >= n or left == right:
                raise ValueError("invalid recovery offending pair")
            delta = positions[left] - positions[right]
            distance = float(np.linalg.norm(delta))
            if distance <= 1e-9:
                return ()
            normal = delta / distance
            protected = teammate_safety_distance + invariant_buffer
            strength = max(
                1.0,
                (protected + target_distance - distance)
                / max(target_distance, 1e-6),
            )
            pair_force[left] += strength * normal
            pair_force[right] -= strength * normal
            explicit[left] = True
            explicit[right] = True

    boundary_force = np.zeros(shape, dtype=np.float64)
    projected_goal = np.zeros(shape, dtype=np.float64)
    for agent_id in range(n):
        goal_direction = _unit(goals[agent_id] - positions[agent_id])
        if "boundary" in sources and explicit[agent_id]:
            for axis in range(2):
                if positions[agent_id, axis] <= low[axis] + boundary_trigger_margin:
                    boundary_force[agent_id, axis] += 1.0
                    goal_direction[axis] = max(goal_direction[axis], 0.0)
                if positions[agent_id, axis] >= high[axis] - boundary_trigger_margin:
                    boundary_force[agent_id, axis] -= 1.0
                    goal_direction[axis] = min(goal_direction[axis], 0.0)
        projected_goal[agent_id] = goal_direction

    obstacle_normal = np.zeros(shape, dtype=np.float64)
    obstacle_left = np.zeros(shape, dtype=np.float64)
    obstacle_right = np.zeros(shape, dtype=np.float64)
    if "obstacle_guard_fallback" in sources:
        for agent_id, tracks in enumerate(visible_obstacles):
            if not explicit[agent_id]:
                continue
            for obstacle in tracks:
                relative = np.asarray(obstacle.relative_position, dtype=np.float64)
                closest_relative = relative.copy()
                if bool(obstacle.velocity_valid):
                    # ObservableObstacleV1 stores d(relative_position)/dt.
                    # Do not subtract ego velocity again and do not access a
                    # non-existent estimated_velocity attribute.
                    relative_velocity = np.asarray(
                        obstacle.relative_velocity, dtype=np.float64
                    )
                    speed_squared = float(np.dot(relative_velocity, relative_velocity))
                    if speed_squared > 1e-10:
                        closest_time = float(np.clip(
                            -np.dot(relative, relative_velocity) / speed_squared,
                            0.0,
                            5.0,
                        ))
                        closest_relative = relative + closest_time * relative_velocity
                distance = float(np.linalg.norm(closest_relative))
                if distance <= 1e-9:
                    continue
                clearance = (
                    distance - obstacle.radius - agent_radius
                    - obstacle_safety_margin - invariant_buffer
                )
                if clearance >= obstacle_repulsion_margin:
                    continue
                strength = max(
                    0.0,
                    (obstacle_repulsion_margin - clearance)
                    / max(obstacle_repulsion_margin, 1e-6),
                )
                away = -closest_relative / distance
                obstacle_normal[agent_id] += strength * away
                obstacle_left[agent_id] += strength * np.asarray(
                    [-away[1], away[0]], dtype=np.float64
                )
                obstacle_right[agent_id] -= strength * np.asarray(
                    [-away[1], away[0]], dtype=np.float64
                )

    source_evidence = bool(
        ("pair" in sources and np.any(np.linalg.norm(pair_force, axis=1) > 1e-9))
        or (
            "boundary" in sources
            and np.any(np.linalg.norm(boundary_force, axis=1) > 1e-9)
        )
        or (
            "obstacle_guard_fallback" in sources
            and np.any(np.linalg.norm(obstacle_normal, axis=1) > 1e-9)
        )
    )
    if not source_evidence:
        return ()

    variants = (
        ("normal", obstacle_normal, 0.0),
        ("left_tangent", obstacle_left, 0.55),
        ("right_tangent", obstacle_right, 0.55),
    )
    candidates: list[StarRecoveryCandidateV2] = []
    seen_targets: set[bytes] = set()
    desired_speed = max_speed * speed_fraction
    for label, obstacle_component, normal_mix in variants:
        actions = np.zeros(shape, dtype=np.float64)
        targets = positions.copy()
        commanded = False
        for agent_id in range(n):
            if not explicit[agent_id]:
                continue
            pair_tangent = np.zeros(2, dtype=np.float64)
            if label != "normal" and float(np.linalg.norm(pair_force[agent_id])) > 1e-9:
                pair_tangent = np.asarray(
                    [-pair_force[agent_id, 1], pair_force[agent_id, 0]],
                    dtype=np.float64,
                )
                if label == "right_tangent":
                    pair_tangent *= -1.0
            direction = _unit(
                2.0 * pair_force[agent_id]
                + 1.5 * boundary_force[agent_id]
                + 1.25 * obstacle_component[agent_id]
                + normal_mix * obstacle_normal[agent_id]
                + 0.45 * pair_tangent
                + 0.65 * projected_goal[agent_id]
            )
            if float(np.linalg.norm(direction)) <= 1e-9:
                continue
            target = np.clip(
                positions[agent_id] + target_distance * direction, low, high
            )
            target_direction = _unit(target - positions[agent_id])
            if float(np.linalg.norm(target_direction)) <= 1e-9:
                continue
            desired_velocity = desired_speed * target_direction
            actions[agent_id] = np.clip(
                (desired_velocity - velocities[agent_id])
                / max(max_acceleration * dt, 1e-12),
                -1.0,
                1.0,
            )
            targets[agent_id] = target
            commanded = True
        if not commanded:
            continue
        fingerprint = np.round(targets, decimals=6).tobytes()
        if fingerprint in seen_targets:
            continue
        seen_targets.add(fingerprint)
        candidates.append(StarRecoveryCandidateV2(
            label=label,
            actions=actions.astype(np.float32),
            targets=targets.astype(np.float32),
            affected_agent_mask=tuple(bool(value) for value in explicit),
            entry_sources=sources,
        ))
    return tuple(candidates)
