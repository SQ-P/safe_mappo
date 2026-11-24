"""Deterministic boundary admission and shared low-level action shielding."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class BoundaryGuardResult:
    requested_actions: np.ndarray
    executed_actions: np.ndarray
    projected_next_velocities: np.ndarray
    surface_clearances: np.ndarray
    shielded_axes: tuple[tuple[bool, bool], ...]
    reasons: tuple[tuple[str | None, str | None], ...]
    invariant_violations: int
    safe_brake: bool

    def summary(self) -> dict[str, Any]:
        result = asdict(self)
        for key in (
            "requested_actions", "executed_actions", "projected_next_velocities",
            "surface_clearances", "reference_safe_actions",
        ):
            if key in result:
                result[key] = np.asarray(result[key]).astype(float).tolist()
        return result


class BoundarySafetyGuardV1:
    """Axis-aligned surface-clearance guard using the environment dynamics."""

    VERSION = "boundary_safety_guard_v1"

    def __init__(self, config: Mapping[str, Any] | None = None):
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.surface_clearance_m = float(cfg.get("surface_clearance_m", 3.0))
        self.invariant_buffer_m = float(cfg.get("invariant_buffer_m", 0.25))
        self.action_shield_enabled = bool(cfg.get("action_shield_enabled", True))
        self.no_safe_candidate_mode = str(
            cfg.get("no_safe_candidate_mode", "safe_brake")
        )
        if self.surface_clearance_m < 0.0 or self.invariant_buffer_m < 0.0:
            raise ValueError("boundary guard clearances must be nonnegative")
        if self.no_safe_candidate_mode != "safe_brake":
            raise ValueError("V3 boundary guard requires no_safe_candidate_mode=safe_brake")

    @staticmethod
    def stopping_distance(speed: float, *, dt: float, max_acceleration: float) -> float:
        speed = max(float(speed), 0.0)
        dt = float(dt)
        maximum = float(max_acceleration)
        if not math.isfinite(speed) or dt <= 0.0 or maximum <= 0.0:
            raise ValueError("stopping-distance inputs must be finite and positive")
        count = int(math.ceil(speed / (maximum * dt)))
        return float(dt * sum(
            max(0.0, speed - step * maximum * dt)
            for step in range(1, count + 1)
        ))

    @staticmethod
    def surface_clearances(
        position: np.ndarray, map_size: np.ndarray, agent_radius: float
    ) -> np.ndarray:
        position = np.asarray(position, dtype=np.float64)
        map_size = np.asarray(map_size, dtype=np.float64)
        if position.shape != (2,) or map_size.shape != (2,):
            raise ValueError("position and map_size must be 2D")
        return np.asarray([
            position[0] - agent_radius,
            map_size[0] - agent_radius - position[0],
            position[1] - agent_radius,
            map_size[1] - agent_radius - position[1],
        ], dtype=np.float64)

    def candidate_is_safe(
        self,
        start: np.ndarray,
        endpoint: np.ndarray,
        *,
        map_size: np.ndarray,
        agent_radius: float,
    ) -> tuple[bool, float, float]:
        start_clearance = self.surface_clearances(start, map_size, agent_radius)
        endpoint_clearance = self.surface_clearances(endpoint, map_size, agent_radius)
        # For an axis-aligned rectangular safe set, a segment between two points
        # stays inside the set and its minimum wall clearance is an endpoint min.
        segment_clearance = float(min(np.min(start_clearance), np.min(endpoint_clearance)))
        endpoint_minimum = float(np.min(endpoint_clearance))
        safe = (
            endpoint_minimum >= self.surface_clearance_m
            and segment_clearance >= self.surface_clearance_m
        )
        return bool(safe), endpoint_minimum, segment_clearance

    def _axis_limit(
        self,
        available_distance: float,
        *,
        dt: float,
        max_acceleration: float,
        max_speed: float,
    ) -> float:
        limit = 0.0
        # Dynamics only admit half-step velocity increments under the frozen
        # contract, but bisection keeps the helper valid for other dt values.
        low, high = 0.0, float(max_speed)
        for _ in range(48):
            middle = (low + high) / 2.0
            required = middle * dt + self.stopping_distance(
                middle, dt=dt, max_acceleration=max_acceleration
            )
            if required <= available_distance:
                limit = middle
                low = middle
            else:
                high = middle
        return float(limit)

    def project_actions(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        requested_actions: np.ndarray,
        *,
        map_size: np.ndarray,
        agent_radius: float,
        dt: float,
        max_acceleration: float,
        max_speed: float,
        safe_brake: bool = False,
    ) -> BoundaryGuardResult:
        positions = np.asarray(positions, dtype=np.float64)
        velocities = np.asarray(velocities, dtype=np.float64)
        requested = np.asarray(requested_actions, dtype=np.float64)
        if positions.shape != velocities.shape or requested.shape != positions.shape:
            raise ValueError("positions, velocities and actions must have equal shape")
        executed = np.clip(requested, -1.0, 1.0).copy()
        projected = np.clip(
            velocities + executed * max_acceleration * dt,
            -max_speed,
            max_speed,
        )
        all_clearances: list[np.ndarray] = []
        all_axes: list[tuple[bool, bool]] = []
        all_reasons: list[tuple[str | None, str | None]] = []
        violations = 0
        for agent_id in range(len(positions)):
            clearances = self.surface_clearances(
                positions[agent_id], map_size, agent_radius
            )
            all_clearances.append(clearances)
            flags: list[bool] = []
            reasons: list[str | None] = []
            for axis in range(2):
                negative_distance = clearances[axis * 2]
                positive_distance = clearances[axis * 2 + 1]
                current_velocity = float(velocities[agent_id, axis])
                invariant_invalid = False
                for distance, toward_speed in (
                    (negative_distance, max(0.0, -current_velocity)),
                    (positive_distance, max(0.0, current_velocity)),
                ):
                    available = distance - self.surface_clearance_m - self.invariant_buffer_m
                    if available < self.stopping_distance(
                        toward_speed, dt=dt, max_acceleration=max_acceleration
                    ) - 1e-9:
                        invariant_invalid = True
                        break
                reason: str | None = None
                if safe_brake:
                    next_velocity = math.copysign(
                        max(0.0, abs(current_velocity) - max_acceleration * dt),
                        current_velocity,
                    ) if abs(current_velocity) > 1e-12 else 0.0
                    reason = "safe_brake"
                elif invariant_invalid:
                    next_velocity = current_velocity - math.copysign(
                        min(abs(current_velocity), max_acceleration * dt),
                        current_velocity,
                    ) if abs(current_velocity) > 1e-12 else 0.0
                    reason = "shield_invariant_violation"
                    violations += 1
                else:
                    next_velocity = float(projected[agent_id, axis])
                    if next_velocity < 0.0:
                        available = negative_distance - self.surface_clearance_m - self.invariant_buffer_m
                        limit = self._axis_limit(
                            max(0.0, available), dt=dt,
                            max_acceleration=max_acceleration, max_speed=max_speed,
                        )
                        if -next_velocity > limit + 1e-9:
                            next_velocity = -limit
                            reason = "boundary_action_projection_negative"
                    elif next_velocity > 0.0:
                        available = positive_distance - self.surface_clearance_m - self.invariant_buffer_m
                        limit = self._axis_limit(
                            max(0.0, available), dt=dt,
                            max_acceleration=max_acceleration, max_speed=max_speed,
                        )
                        if next_velocity > limit + 1e-9:
                            next_velocity = limit
                            reason = "boundary_action_projection_positive"
                action = (next_velocity - current_velocity) / (
                    max_acceleration * dt
                )
                executed[agent_id, axis] = float(np.clip(action, -1.0, 1.0))
                projected[agent_id, axis] = float(np.clip(
                    current_velocity
                    + executed[agent_id, axis] * max_acceleration * dt,
                    -max_speed,
                    max_speed,
                ))
                flags.append(reason is not None)
                reasons.append(reason)
            all_axes.append((flags[0], flags[1]))
            all_reasons.append((reasons[0], reasons[1]))
        return BoundaryGuardResult(
            requested_actions=requested.astype(np.float32),
            executed_actions=executed.astype(np.float32),
            projected_next_velocities=projected.astype(np.float32),
            surface_clearances=np.asarray(all_clearances, dtype=np.float32),
            shielded_axes=tuple(all_axes),
            reasons=tuple(all_reasons),
            invariant_violations=int(violations),
            safe_brake=bool(safe_brake),
        )


def _swept_pair_distance(
    left_start: np.ndarray,
    left_end: np.ndarray,
    right_start: np.ndarray,
    right_end: np.ndarray,
) -> float:
    """Minimum centre distance for two linear one-step trajectories."""
    relative_start = np.asarray(left_start, dtype=np.float64) - np.asarray(
        right_start, dtype=np.float64
    )
    relative_delta = (
        np.asarray(left_end, dtype=np.float64) - np.asarray(left_start, dtype=np.float64)
        - np.asarray(right_end, dtype=np.float64)
        + np.asarray(right_start, dtype=np.float64)
    )
    denominator = float(np.dot(relative_delta, relative_delta))
    fraction = 0.0 if denominator <= 1e-15 else float(
        np.clip(-np.dot(relative_start, relative_delta) / denominator, 0.0, 1.0)
    )
    return float(np.linalg.norm(relative_start + fraction * relative_delta))


@dataclass(frozen=True)
class JointGuardResult(BoundaryGuardResult):
    """BoundaryGuardResult plus execution-level pair-safety diagnostics."""

    pair_iterations: int
    requested_pair_min_distance: float
    executed_pair_min_distance: float
    minimum_pair_margin: float
    pair_invariant_violations: int
    pair_infeasible: bool
    recovery_required: bool
    offending_pairs: tuple[tuple[int, int], ...]
    affected_agent_mask: tuple[bool, ...]
    reference_safe_actions: np.ndarray
    reference_safe_action_available: bool
    reference_safe_action_norm: float


class JointSafetyGuardV2(BoundarySafetyGuardV1):
    """Joint boundary and inter-agent action projection for shared safety.

    Candidate-line geometry remains a planning filter.  This guard reasons over
    the actual velocity-first environment update and therefore owns the online
    execution-safety contract.
    """

    VERSION = "joint_safety_guard_v2"

    def __init__(self, config: Mapping[str, Any] | None = None):
        super().__init__(config)
        cfg = dict(config or {})
        self.pair_invariant_buffer_m = float(
            cfg.get("pair_invariant_buffer_m", 0.25)
        )
        self.pair_projection_iterations = int(
            cfg.get("pair_projection_iterations", 8)
        )
        if self.pair_invariant_buffer_m < 0.0:
            raise ValueError("pair_invariant_buffer_m must be nonnegative")
        if self.pair_projection_iterations <= 0:
            raise ValueError("pair_projection_iterations must be positive")

    @staticmethod
    def _pair_minimum(
        positions: np.ndarray,
        next_positions: np.ndarray,
    ) -> float:
        if len(positions) < 2:
            return float("inf")
        return min(
            _swept_pair_distance(
                positions[left], next_positions[left],
                positions[right], next_positions[right],
            )
            for left in range(len(positions))
            for right in range(left + 1, len(positions))
        )

    def project_actions(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        requested_actions: np.ndarray,
        *,
        map_size: np.ndarray,
        agent_radius: float,
        dt: float,
        max_acceleration: float,
        max_speed: float,
        safe_brake: bool = False,
        teammate_safety_distance: float = 6.0,
    ) -> JointGuardResult:
        positions = np.asarray(positions, dtype=np.float64)
        velocities = np.asarray(velocities, dtype=np.float64)
        requested = np.asarray(requested_actions, dtype=np.float64)
        if teammate_safety_distance < 2.0 * float(agent_radius):
            raise ValueError("teammate_safety_distance cannot be below contact distance")

        boundary = super().project_actions(
            positions,
            velocities,
            requested,
            map_size=map_size,
            agent_radius=agent_radius,
            dt=dt,
            max_acceleration=max_acceleration,
            max_speed=max_speed,
            safe_brake=safe_brake,
        )
        executed = np.asarray(boundary.executed_actions, dtype=np.float64).copy()
        required = float(teammate_safety_distance) + self.pair_invariant_buffer_m

        requested_velocities = np.clip(
            velocities + np.clip(requested, -1.0, 1.0) * max_acceleration * dt,
            -max_speed,
            max_speed,
        )
        requested_next = positions + requested_velocities * dt
        requested_minimum = self._pair_minimum(positions, requested_next)

        iterations = 0
        for iteration in range(self.pair_projection_iterations):
            changed = False
            projected_velocities = np.clip(
                velocities + executed * max_acceleration * dt,
                -max_speed,
                max_speed,
            )
            next_positions = positions + projected_velocities * dt
            for left in range(len(positions)):
                for right in range(left + 1, len(positions)):
                    relative = positions[left] - positions[right]
                    distance = float(np.linalg.norm(relative))
                    if distance <= 1e-9:
                        normal = np.asarray([1.0, 0.0], dtype=np.float64)
                    else:
                        normal = relative / distance
                    relative_velocity = projected_velocities[left] - projected_velocities[right]
                    closing_speed = max(0.0, -float(np.dot(relative_velocity, normal)))
                    braking = self.stopping_distance(
                        closing_speed,
                        dt=dt,
                        max_acceleration=2.0 * max_acceleration,
                    )
                    radial_next = float(
                        np.dot(next_positions[left] - next_positions[right], normal)
                    )
                    deficit = required + braking - radial_next
                    swept = _swept_pair_distance(
                        positions[left], next_positions[left],
                        positions[right], next_positions[right],
                    )
                    deficit = max(deficit, required - swept)
                    if deficit <= 1e-8:
                        continue
                    # Equal and opposite correction minimizes the joint action
                    # change for this linearized pair constraint.
                    action_delta = deficit / (
                        max(2.0 * max_acceleration * dt * dt, 1e-12)
                    )
                    executed[left] += action_delta * normal
                    executed[right] -= action_delta * normal
                    executed = np.clip(executed, -1.0, 1.0)
                    changed = True

            iterations = iteration + 1
            if changed:
                # Pair projection can push an axis toward a wall.  Reapply the
                # unchanged V1 boundary projection before checking all pairs.
                reprojection = super().project_actions(
                    positions,
                    velocities,
                    executed,
                    map_size=map_size,
                    agent_radius=agent_radius,
                    dt=dt,
                    max_acceleration=max_acceleration,
                    max_speed=max_speed,
                    safe_brake=safe_brake,
                )
                executed = np.asarray(
                    reprojection.executed_actions, dtype=np.float64
                )
            else:
                break

        projected = np.clip(
            velocities + executed * max_acceleration * dt,
            -max_speed,
            max_speed,
        )
        next_positions = positions + projected * dt
        executed_minimum = self._pair_minimum(positions, next_positions)
        offending_pairs = tuple(
            (left, right)
            for left in range(len(positions))
            for right in range(left + 1, len(positions))
            if _swept_pair_distance(
                positions[left], next_positions[left],
                positions[right], next_positions[right],
            ) < required - 1e-6
        )
        pair_infeasible = bool(offending_pairs)
        pair_violations = len(offending_pairs)
        affected = [False] * len(positions)
        for left, right in offending_pairs:
            affected[left] = True
            affected[right] = True

        if pair_infeasible:
            # Deterministic last-resort separation command.  It is still passed
            # through the boundary guard, so an impossible joint constraint is
            # reported rather than silently violating the wall invariant.
            emergency = executed.copy()
            for left in range(len(positions)):
                for right in range(left + 1, len(positions)):
                    relative = positions[left] - positions[right]
                    norm = float(np.linalg.norm(relative))
                    normal = (
                        np.asarray([1.0, 0.0], dtype=np.float64)
                        if norm <= 1e-9 else relative / norm
                    )
                    emergency[left] += normal
                    emergency[right] -= normal
            emergency = np.clip(emergency, -1.0, 1.0)
            reprojection = super().project_actions(
                positions,
                velocities,
                emergency,
                map_size=map_size,
                agent_radius=agent_radius,
                dt=dt,
                max_acceleration=max_acceleration,
                max_speed=max_speed,
                safe_brake=False,
            )
            executed = np.asarray(reprojection.executed_actions, dtype=np.float64)
            projected = np.clip(
                velocities + executed * max_acceleration * dt,
                -max_speed,
                max_speed,
            )
            next_positions = positions + projected * dt
            executed_minimum = self._pair_minimum(positions, next_positions)

        if safe_brake:
            reference = self.project_actions(
                positions,
                velocities,
                requested,
                map_size=map_size,
                agent_radius=agent_radius,
                dt=dt,
                max_acceleration=max_acceleration,
                max_speed=max_speed,
                safe_brake=False,
                teammate_safety_distance=teammate_safety_distance,
            )
            reference_actions = np.asarray(
                reference.executed_actions, dtype=np.float64
            )
            reference_available = bool(
                not reference.recovery_required
                and reference.invariant_violations == 0
                and np.linalg.norm(reference_actions) > 1e-6
            )
        else:
            reference_actions = executed.copy()
            reference_available = bool(
                not pair_infeasible
                and boundary.invariant_violations == 0
                and np.linalg.norm(reference_actions) > 1e-6
            )

        return JointGuardResult(
            requested_actions=requested.astype(np.float32),
            executed_actions=executed.astype(np.float32),
            projected_next_velocities=projected.astype(np.float32),
            surface_clearances=boundary.surface_clearances,
            shielded_axes=tuple(
                tuple(bool(abs(executed[row, axis] - requested[row, axis]) > 1e-7)
                      for axis in range(2))
                for row in range(len(positions))
            ),
            reasons=boundary.reasons,
            invariant_violations=int(boundary.invariant_violations),
            safe_brake=bool(safe_brake),
            pair_iterations=int(iterations),
            requested_pair_min_distance=float(requested_minimum),
            executed_pair_min_distance=float(executed_minimum),
            minimum_pair_margin=float(executed_minimum - required),
            pair_invariant_violations=pair_violations,
            pair_infeasible=pair_infeasible,
            recovery_required=pair_infeasible,
            offending_pairs=offending_pairs,
            affected_agent_mask=tuple(affected),
            reference_safe_actions=reference_actions.astype(np.float32),
            reference_safe_action_available=reference_available,
            reference_safe_action_norm=float(np.linalg.norm(reference_actions)),
        )
