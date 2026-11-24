import numpy as np

from src.envs.entities import (
    MovingRectangle,
    segment_intersects_expanded_rectangle,
)


def test_rectangle_reaches_waypoint_and_resamples():
    rng = np.random.default_rng(7)
    obstacle = MovingRectangle(
        position=np.array([10.0, 10.0], dtype=np.float32),
        size=np.array([20.0, 30.0], dtype=np.float32),
        velocity=np.zeros(2, dtype=np.float32),
        speed=2.0,
        waypoint=np.array([11.0, 10.0], dtype=np.float32),
    )

    obstacle.step(1.0, (100.0, 80.0), rng)

    assert not np.allclose(obstacle.waypoint, [11.0, 10.0])


def test_rectangle_reflects_at_boundary():
    rng = np.random.default_rng(1)
    obstacle = MovingRectangle(
        position=np.array([79.5, 20.0], dtype=np.float32),
        size=np.array([20.0, 10.0], dtype=np.float32),
        velocity=np.array([2.0, 0.0], dtype=np.float32),
        speed=2.0,
        waypoint=np.array([100.0, 20.0], dtype=np.float32),
    )

    obstacle.step(1.0, (100.0, 80.0), rng)

    assert obstacle.position[0] <= 80.0
    assert obstacle.velocity[0] <= 0.0


def test_swept_segment_detects_expanded_rectangle():
    assert segment_intersects_expanded_rectangle(
        start=np.array([0.0, 15.0], dtype=np.float32),
        end=np.array([30.0, 15.0], dtype=np.float32),
        rect_position=np.array([12.0, 10.0], dtype=np.float32),
        rect_size=np.array([5.0, 10.0], dtype=np.float32),
        expansion=4.0,
    )


def test_segment_outside_rectangle_is_clear():
    assert not segment_intersects_expanded_rectangle(
        start=np.array([0.0, 40.0], dtype=np.float32),
        end=np.array([30.0, 40.0], dtype=np.float32),
        rect_position=np.array([12.0, 10.0], dtype=np.float32),
        rect_size=np.array([5.0, 10.0], dtype=np.float32),
        expansion=4.0,
    )
