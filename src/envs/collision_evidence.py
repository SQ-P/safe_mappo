"""Continuous contact times and bounded, offline pre-collision evidence."""

from __future__ import annotations

import numpy as np


def sphere_contact_fraction(start, end, radius: float) -> float | None:
    """First contact on a relative linear sweep, including tangency, in [0, 1]."""
    start = np.asarray(start, dtype=np.float64)
    delta = np.asarray(end, dtype=np.float64) - start
    c = float(start @ start - radius * radius)
    if c <= 0:
        return 0.0
    a = float(delta @ delta)
    b = float(2 * (start @ delta))
    disc = b * b - 4 * a * c
    if a <= 1e-15 or disc < 0:
        return None
    fraction = (-b - np.sqrt(max(0.0, disc))) / (2 * a)
    return float(fraction) if 0 <= fraction <= 1 else None


def transition_evidence(env, actions) -> dict:
    """Pre-action world snapshot; privileged fields never enter the actor input.

    TTC assumes constant relative velocity and is evidence, not a causal label.
    Visibility is recorded so offline diagnosis can detect sensing limitations.
    """
    obstacles = []
    visible_ids = env._visible_obstacle_ids()
    for oid, obstacle in enumerate(env.obstacles):
        relative = obstacle.position - env.positions
        velocity = obstacle.velocity - env.velocities
        radius = env.agent_radius + obstacle.radius
        distances = np.linalg.norm(relative, axis=1)
        ttc = [sphere_contact_fraction(p, p + v * 60.0, radius)
               for p, v in zip(relative, velocity)]
        obstacles.append({
            "id": oid, "dynamic": obstacle.dynamic,
            "position": obstacle.position.tolist(), "radius": obstacle.radius,
            "velocity": obstacle.velocity.tolist(),
            "clearance_by_agent": (distances - radius).tolist(),
            "in_detection_radius_by_agent": (distances <= env.detection_radius).tolist(),
            "visible_by_agent": [oid in ids for ids in visible_ids],
            "constant_velocity_ttc_seconds": [None if t is None else t * 60 for t in ttc],
        })
    return {
        "step": env.step_count + 1, "dt": env.dt,
        "positions": env.positions.tolist(), "velocities": env.velocities.tolist(),
        "actions": np.asarray(actions).tolist(), "subgoals": env.subgoals.tolist(),
        "goals": env.goals.tolist(), "obstacles": obstacles,
        "evidence_scope": "offline_simulator_truth_with_visibility",
    }


def compact_failure_examples(records, limit=3, steps=6):
    """Bound prompt size; full JSONL remains the audit source, not this sample."""
    from copy import deepcopy
    examples = []
    for row in records:
        if row["safe_success"]:
            continue
        window = row["first_failure_window"] or row["terminal_window"]
        excerpt = deepcopy(window[-steps:])
        for frame in excerpt:
            contacts = {event.get("obstacle_id") for event in frame["collision_events"]}
            nearest = sorted(frame["obstacles"], key=lambda obj: min(obj["clearance_by_agent"]))[:2]
            keep = contacts | {obj["id"] for obj in nearest}
            frame["obstacles"] = [obj for obj in frame["obstacles"] if obj["id"] in keep]
        examples.append({"seed": row["seed"], "outcome": row["outcome"],
                         "full_window_steps": len(window), "window": excerpt,
                         "sampling": "first_failures_not_representative; last_steps; nearest_and_contact_obstacles"})
        if len(examples) >= limit:
            break
    return examples
