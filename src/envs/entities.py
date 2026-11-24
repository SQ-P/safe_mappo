"""连续 UAV 与障碍物实体。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _vec2(value, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite two-dimensional vector")
    return result.copy()


@dataclass
class UAVState:
    position: np.ndarray
    velocity: np.ndarray
    radius: float = 4.0
    max_speed: float = 1.3

    def __post_init__(self) -> None:
        self.position = _vec2(self.position, "position")
        self.velocity = _vec2(self.velocity, "velocity")
        self.radius = float(self.radius)
        self.max_speed = float(self.max_speed)
        if self.radius <= 0.0 or self.max_speed <= 0.0:
            raise ValueError("radius and max_speed must be positive")

    def apply_action(self, action, bounds: tuple[float, float]) -> bool:
        """Apply a bounded velocity command and report a boundary contact.

        The returned flag is ``True`` when the *unclipped* next position would
        touch or cross the workspace boundary. The position is still clipped so
        that observations remain finite, while the environment can terminate the
        episode as a boundary-collision failure.
        """
        command = np.clip(_vec2(action, "action"), -1.0, 1.0)

        # Box(-1, 1) constrains each component separately. Normalize the command
        # again so that [1, 1] cannot exceed the configured physical max speed.
        command_norm = float(np.linalg.norm(command))
        if command_norm > 1.0:
            command = command / command_norm

        low = np.full(2, self.radius, dtype=np.float32)
        high = _vec2(bounds, "bounds") - self.radius
        if np.any(high <= low):
            raise ValueError("workspace bounds must exceed the UAV diameter")

        self.velocity = command * self.max_speed
        candidate = self.position + self.velocity

        # Touching the physical wall counts as a collision. Using the candidate
        # before clipping avoids silently turning an invalid action into a valid
        # boundary-hugging trajectory.
        boundary_hit = bool(np.any(candidate <= low) or np.any(candidate >= high))
        self.position = np.clip(candidate, low, high).astype(np.float32)
        return boundary_hit

    def apply_acceleration(
        self,
        acceleration,
        dt: float,
        bounds: tuple[float, float],
        max_acceleration: float = 1.0,
    ) -> bool:
        """按离散运动学更新速度与位置，并返回是否触碰地图边界。"""
        action = np.clip(_vec2(acceleration, "acceleration"), -1.0, 1.0)
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        max_acceleration = float(max_acceleration)
        if not np.isfinite(max_acceleration) or max_acceleration <= 0.0:
            raise ValueError("max_acceleration must be finite and positive")

        # 规格要求对两个速度分量分别裁剪，而不是按速度向量模长裁剪。
        self.velocity = np.clip(
            self.velocity + action * max_acceleration * dt,
            -self.max_speed,
            self.max_speed,
        ).astype(np.float32)
        candidate = self.position + self.velocity * dt
        low = np.full(2, self.radius, dtype=np.float32)
        high = _vec2(bounds, "bounds") - self.radius
        if np.any(high <= low):
            raise ValueError("workspace bounds must exceed the UAV diameter")
        boundary_hit = bool(np.any(candidate < low) or np.any(candidate > high))
        self.position = np.clip(candidate, low, high).astype(np.float32)
        return boundary_hit


@dataclass
class CircularObstacle:
    """圆形静态/动态障碍物；动态障碍物在地图边缘镜面反弹。"""

    position: np.ndarray
    radius: float
    velocity: np.ndarray
    dynamic: bool = False

    def __post_init__(self) -> None:
        self.position = _vec2(self.position, "position")
        self.velocity = _vec2(self.velocity, "velocity")
        self.radius = float(self.radius)
        self.dynamic = bool(self.dynamic)
        if self.radius <= 0.0:
            raise ValueError("obstacle radius must be positive")
        if not self.dynamic:
            self.velocity[:] = 0.0

    @property
    def size(self) -> np.ndarray:
        """兼容现有回放接口，返回直径组成的二维向量。"""
        return np.full(2, 2.0 * self.radius, dtype=np.float32)

    def step(self, dt: float, bounds: tuple[float, float]) -> None:
        if not self.dynamic:
            return
        dt = float(dt)
        workspace = _vec2(bounds, "bounds")
        low = np.full(2, self.radius, dtype=np.float32)
        high = workspace - self.radius
        if dt <= 0.0 or np.any(high <= low):
            raise ValueError("dt and obstacle workspace must be positive")

        candidate = self.position + self.velocity * dt
        for axis in range(2):
            if candidate[axis] < low[axis] or candidate[axis] > high[axis]:
                self.velocity[axis] *= -1.0
                candidate[axis] = float(np.clip(candidate[axis], low[axis], high[axis]))
        self.position = candidate.astype(np.float32)


@dataclass
class MovingRectangle:
    position: np.ndarray
    size: np.ndarray
    velocity: np.ndarray
    speed: float
    waypoint: np.ndarray

    def __post_init__(self) -> None:
        self.position = _vec2(self.position, "position")
        self.size = _vec2(self.size, "size")
        self.velocity = _vec2(self.velocity, "velocity")
        self.waypoint = _vec2(self.waypoint, "waypoint")
        self.speed = float(self.speed)
        if np.any(self.size <= 0.0) or self.speed < 0.0:
            raise ValueError("rectangle size must be positive and speed nonnegative")

    def step(self, dt: float, bounds: tuple[float, float], rng: np.random.Generator) -> None:
        dt = float(dt)
        workspace = _vec2(bounds, "bounds")
        high = workspace - self.size
        if dt <= 0.0 or np.any(high <= 0.0):
            raise ValueError("dt and available rectangle workspace must be positive")

        # A zero-speed rectangle is a static obstacle. Keeping the same entity
        # type avoids branching throughout observation and collision code.
        if self.speed == 0.0:
            self.velocity = np.zeros(2, dtype=np.float32)
            return

        delta = self.waypoint - self.position
        distance = float(np.linalg.norm(delta))
        if distance <= self.speed * dt:
            self.position = np.clip(self.waypoint, 0.0, high)
            self.waypoint = rng.uniform(np.zeros(2), high).astype(np.float32)
            delta = self.waypoint - self.position
            distance = float(np.linalg.norm(delta))

        direction = delta / max(distance, 1e-8)
        self.velocity = direction.astype(np.float32) * self.speed
        candidate = self.position + self.velocity * dt
        for axis in range(2):
            if candidate[axis] < 0.0 or candidate[axis] > high[axis]:
                candidate[axis] = float(np.clip(candidate[axis], 0.0, high[axis]))
                self.velocity[axis] *= -1.0
                remaining = abs(float(delta[axis]))
                self.waypoint[axis] = float(
                    np.clip(candidate[axis] + np.sign(self.velocity[axis]) * remaining, 0.0, high[axis])
                )
        self.position = candidate.astype(np.float32)


def point_to_rectangle_distance(
    point, rect_position, rect_size
) -> tuple[np.ndarray, float, np.ndarray]:
    """Return obstacle-to-point vector, distance, and closest rectangle point."""
    value = _vec2(point, "point")
    lower = _vec2(rect_position, "rect_position")
    size = _vec2(rect_size, "rect_size")
    if np.any(size <= 0.0):
        raise ValueError("rect_size must be positive")
    closest = np.clip(value, lower, lower + size)
    vector = value - closest
    return vector.astype(np.float32), float(np.linalg.norm(vector)), closest.astype(np.float32)


def segment_intersects_expanded_rectangle(
    start, end, rect_position, rect_size, expansion: float
) -> bool:
    """Return whether a segment intersects an axis-aligned expanded rectangle."""
    origin = _vec2(start, "start")
    target = _vec2(end, "end")
    lower = _vec2(rect_position, "rect_position") - float(expansion)
    upper = _vec2(rect_position, "rect_position") + _vec2(rect_size, "rect_size") + float(expansion)
    if expansion < 0.0 or np.any(upper <= lower):
        raise ValueError("expansion must be nonnegative and rectangle size positive")

    direction = target - origin
    t_min, t_max = 0.0, 1.0
    for axis in range(2):
        if abs(float(direction[axis])) < 1e-12:
            if origin[axis] < lower[axis] or origin[axis] > upper[axis]:
                return False
            continue
        inverse = 1.0 / float(direction[axis])
        t1 = float(lower[axis] - origin[axis]) * inverse
        t2 = float(upper[axis] - origin[axis]) * inverse
        t_min = max(t_min, min(t1, t2))
        t_max = min(t_max, max(t1, t2))
        if t_min > t_max:
            return False
    return True
