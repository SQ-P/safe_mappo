"""Environment-aligned velocity-first dynamics for UCR-JSF-L."""

from __future__ import annotations

import numpy as np


def velocity_first_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    actions: np.ndarray,
    *,
    dt: float,
    max_acceleration: float,
    max_speed: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Match ``UAVState.apply_acceleration`` before boundary clipping."""

    p = np.asarray(positions, dtype=np.float64)
    v = np.asarray(velocities, dtype=np.float64)
    a = np.asarray(actions, dtype=np.float64)
    if p.shape != v.shape or p.shape != a.shape or p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("positions, velocities and actions must be aligned Nx2 matrices")
    if not np.isfinite((p, v, a)).all():
        raise ValueError("dynamics inputs must be finite")
    next_velocity = np.clip(
        v + np.clip(a, -1.0, 1.0) * float(max_acceleration) * float(dt),
        -float(max_speed), float(max_speed),
    )
    next_position = p + next_velocity * float(dt)
    return next_position, next_velocity


def rollout_constant_action(
    positions: np.ndarray,
    velocities: np.ndarray,
    actions: np.ndarray,
    *,
    horizon_steps: int,
    dt: float,
    max_acceleration: float,
    max_speed: float,
) -> tuple[np.ndarray, np.ndarray]:
    if int(horizon_steps) < 1:
        raise ValueError("horizon_steps must be positive")
    points = [np.asarray(positions, dtype=np.float64).copy()]
    speeds = [np.asarray(velocities, dtype=np.float64).copy()]
    current_p, current_v = points[0], speeds[0]
    for _ in range(int(horizon_steps)):
        current_p, current_v = velocity_first_step(
            current_p, current_v, actions, dt=dt,
            max_acceleration=max_acceleration, max_speed=max_speed,
        )
        points.append(current_p)
        speeds.append(current_v)
    return np.stack(points), np.stack(speeds)


def rollout_with_stop_hold(
    positions: np.ndarray,
    velocities: np.ndarray,
    actions: np.ndarray,
    *,
    braking_mask: np.ndarray,
    horizon_steps: int,
    dt: float,
    max_acceleration: float,
    max_speed: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out backup braking without reversing after the UAV has stopped."""

    mask = np.asarray(braking_mask, dtype=bool)
    base = np.asarray(actions, dtype=np.float64)
    if mask.shape != (base.shape[0],):
        raise ValueError("braking mask must align with candidate actions")
    points = [np.asarray(positions, dtype=np.float64).copy()]
    speeds = [np.asarray(velocities, dtype=np.float64).copy()]
    current_p, current_v = points[0], speeds[0]
    for _ in range(int(horizon_steps)):
        step_actions = base.copy()
        step_actions[mask] = np.clip(
            -current_v[mask] / (float(max_acceleration) * float(dt)), -1.0, 1.0
        )
        current_p, current_v = velocity_first_step(
            current_p, current_v, step_actions, dt=dt,
            max_acceleration=max_acceleration, max_speed=max_speed,
        )
        points.append(current_p)
        speeds.append(current_v)
    return np.stack(points), np.stack(speeds)
