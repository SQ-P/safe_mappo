"""Independent 8-UAV cross-navigation scenario; existing task code is untouched."""
from collections import deque
from copy import deepcopy

import numpy as np

from src.envs.cooperative_uav_env import CooperativeUAVEnv
from src.envs.entities import UAVState

SCENARIO = "cross8u_160_4s6d_v1"
STARTS = np.array([[12, 30], [12, 60], [12, 100], [12, 130],
                   [30, 12], [60, 12], [100, 12], [130, 12]], dtype=np.float32)
GOALS = np.array([[148, 130], [148, 100], [148, 60], [148, 30],
                  [130, 148], [100, 148], [60, 148], [30, 148]], dtype=np.float32)


def initial_geometric_routes(obstacles, *, spacing=2., clearance=2.):
    """Conservative 4-connected free-cell paths, not joint/dynamic feasibility.

    Every free square cell is obstacle-free after UAV-radius/clearance inflation.
    Four-neighbour cell paths cannot cut diagonally through obstacle corners.
    """
    margin = 1.5 + clearance
    axis = np.arange(margin, 160.-margin+1e-6, spacing)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    free = np.ones(xx.shape, dtype=bool)
    for obstacle in obstacles:
        inflated = obstacle.radius + 1.5 + clearance + spacing/np.sqrt(2.)
        free &= (xx-obstacle.position[0])**2 + (yy-obstacle.position[1])**2 > inflated**2
    labels = np.full(free.shape, -1, dtype=np.int32)
    def cell(point):
        return tuple(np.argmin(abs(axis-v)).item() for v in point)
    component = 0
    lengths = []
    for start, goal in zip(STARTS, GOALS):
        source, target = cell(start), cell(goal)
        if not free[source] or not free[target]:
            return {"all_routes_exist": False, "scope": "initial inflated-obstacle geometry"}
        if labels[source] < 0:
            labels[source] = component
            queue = deque([source])
            while queue:
                x, y = queue.popleft()
                for nx, ny in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                    if 0 <= nx < len(axis) and 0 <= ny < len(axis) and free[nx,ny] and labels[nx,ny] < 0:
                        labels[nx,ny] = component; queue.append((nx,ny))
            component += 1
        lengths.append(bool(labels[source] == labels[target]))
    return {"all_routes_exist": all(lengths), "per_agent_route_exists": lengths,
        "grid_spacing_m": spacing, "surface_clearance_m": clearance,
        "scope": "initial inflated-obstacle geometry; not moving-obstacle or joint-policy success"}


class Cross8U160Env(CooperativeUAVEnv):
    """Override scenario initialization and add direction-neutral telemetry only."""

    def __init__(self, config):
        cfg = deepcopy(dict(config))
        expected = {"num_agents": 8, "map_size": [160.,160.], "static_obstacle_num": 4,
            "dynamic_obstacle_num": 6, "agent_radius": 1.5, "agent_max_speed": 2., "dt": .5,
            "goal_radius": 3., "max_episode_steps": 500, "obstacle_slots": 10,
            "detection_radius": 30., "obstacle_radius_range": [2.,5.],
            "obstacle_spawn_x_range": [45.,115.], "obstacle_spawn_y_range": [45.,115.],
            "start_y_jitter": 0., "observation_schema_version": "tracked_obstacle_v2",
            "verification_scenario": SCENARIO}
        for key, value in expected.items():
            if cfg.get(key) != value:
                raise ValueError("Independent cross8u scenario contract mismatch: "+key)
        if not np.array_equal(cfg.get("start_positions"), STARTS) or not np.array_equal(cfg.get("goal_positions"), GOALS):
            raise ValueError("Explicit crossing endpoints differ from fixed scenario")
        super().__init__(cfg)
        self.layout_generation = {}

    def _sample_agents(self):
        return [UAVState(p.copy(), np.zeros(2,dtype=np.float32), self.agent_radius, self.agent_max_speed) for p in STARTS]

    def _build_goals(self):
        return GOALS.copy()

    def _sample_obstacles(self):
        for attempt in range(1, 101):
            obstacles = super()._sample_obstacles()
            routes = initial_geometric_routes(obstacles)
            if routes["all_routes_exist"]:
                self.layout_generation = {"attempts": attempt, "routes": routes,
                    "spawn_region": [[45.,115.],[45.,115.]], "dynamic_motion": "inherited full-map reflection"}
                return obstacles
        raise RuntimeError("No initial geometrically passable cross8u layout within fixed budget")

    def directional_diagnostics(self):
        direction = GOALS-STARTS
        progress = np.sum((self.positions-STARTS)*direction, axis=1)/np.sum(direction*direction, axis=1)
        remaining = np.linalg.norm(self.positions-GOALS, axis=1)
        return {"goal_axis_fraction": progress.tolist(), "remaining_goal_distance_m": remaining.tolist(),
            "horizontal_group_mean_fraction": float(progress[:4].mean()),
            "vertical_group_mean_fraction": float(progress[4:].mean()),
            "halfway_along_own_goal_axis": (progress >= .5).tolist(),
            "legacy_x_only_telemetry_applies_to_vertical_group": False}

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        info["verification_scenario"] = SCENARIO
        info["cross8u_layout"] = deepcopy(self.layout_generation)
        info["cross8u_directional"] = self.directional_diagnostics()
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        info["verification_scenario"] = SCENARIO
        info["cross8u_directional"] = self.directional_diagnostics()
        return observation, reward, terminated, truncated, info
