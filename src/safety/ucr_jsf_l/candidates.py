"""Deterministic bounded joint-candidate generation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from .contracts import JointSafetyInputV1, SafetyMode
from .liveness import LivenessMemoryV1


@dataclass(frozen=True)
class CandidateConfigV1:
    max_local_candidates_per_agent: int = 8
    max_joint_candidates: int = 64
    tangent_strength: float = 0.8
    emergency_candidate_reserve: int = 4

    @classmethod
    def from_mapping(cls, payload: dict) -> "CandidateConfigV1":
        return cls(
            max_local_candidates_per_agent=int(
                payload.get("max_local_candidates_per_agent", 8)
            ),
            max_joint_candidates=int(payload.get("max_joint_candidates", 64)),
            tangent_strength=float(payload.get("tangent_strength", 0.8)),
            emergency_candidate_reserve=int(
                payload.get("emergency_candidate_reserve", 4)
            ),
        )


@dataclass(frozen=True)
class JointCandidateV1:
    candidate_id: str
    actions: np.ndarray
    source: str
    braking_mask: tuple[bool, ...] = ()


def braking_action(velocity: np.ndarray, state: JointSafetyInputV1) -> np.ndarray:
    return np.clip(
        -np.asarray(velocity, dtype=float) / (state.max_acceleration * state.dt),
        -1.0, 1.0,
    )


class CandidateGeneratorV1:
    def __init__(self, config: CandidateConfigV1) -> None:
        self.config = config
        if not (1 <= config.max_local_candidates_per_agent <= 8):
            raise ValueError("local candidate cap must be in [1, 8]")
        if not (1 <= config.max_joint_candidates <= 64):
            raise ValueError("joint candidate cap must be in [1, 64]")
        if not (1 <= config.emergency_candidate_reserve < config.max_joint_candidates):
            raise ValueError("emergency reserve must fit inside the joint cap")

    @staticmethod
    def _deduplicate(rows: list[tuple[str, np.ndarray]]) -> list[tuple[str, np.ndarray]]:
        result: list[tuple[str, np.ndarray]] = []
        seen: set[tuple[float, float]] = set()
        for label, row in rows:
            value = np.clip(np.asarray(row, dtype=float), -1.0, 1.0)
            key = tuple(float(x) for x in np.round(value, 8))
            if key not in seen:
                seen.add(key)
                result.append((label, value))
        return result

    def _local(
        self,
        state: JointSafetyInputV1,
        requested: np.ndarray,
        agent_id: int,
        memory: LivenessMemoryV1,
        mode: SafetyMode,
        arrived: np.ndarray,
    ) -> list[tuple[str, np.ndarray]]:
        velocity = np.asarray(state.velocities[agent_id], dtype=float)
        brake = braking_action(velocity, state)
        goal_delta = np.asarray(state.final_goals[agent_id]) - np.asarray(state.positions[agent_id])
        goal_norm = float(np.linalg.norm(goal_delta))
        desired_velocity = (
            np.zeros(2) if goal_norm <= 1e-12
            else goal_delta / goal_norm * state.max_speed
        )
        goal_action = np.clip(
            (desired_velocity - velocity) / (state.max_acceleration * state.dt),
            -1.0, 1.0,
        )
        rows: list[tuple[str, np.ndarray]] = [
            ("actor", requested[agent_id]),
            ("brake", brake),
            ("half_actor", 0.5 * requested[agent_id]),
            ("goal", goal_action),
        ]
        blocker = memory.blocking_track(state, agent_id)
        if blocker is not None:
            radial = np.asarray(blocker.relative_position, dtype=float)
            radial_norm = float(np.linalg.norm(radial))
            if radial_norm > 1e-12:
                side = memory.preferred_passing_side(
                    state, agent_id, blocker.track_id, radial, commit=False
                )
                tangent = side * np.asarray([-radial[1], radial[0]]) / radial_norm
                tangent_action = np.clip(
                    tangent * self.config.tangent_strength
                    - velocity / max(state.max_speed, 1e-6) * 0.25,
                    -1.0, 1.0,
                )
                rows.extend([
                    ("committed_tangent", tangent_action),
                    ("tangent_goal_blend", 0.65 * tangent_action + 0.35 * goal_action),
                    ("opposite_tangent_probe", -tangent_action),
                ])
        if bool(arrived[agent_id]):
            rows.insert(0, ("arrival_hold", brake))
        if mode is SafetyMode.STALL_ESCAPE:
            committed = next(
                (row for label, row in rows if label == "committed_tangent"),
                goal_action,
            )
            rows.insert(0, ("stall_escape", committed))
        elif mode is SafetyMode.SAFETY_RECOVERY:
            rows.insert(0, ("recovery_brake", brake))
        return self._deduplicate(rows)[: self.config.max_local_candidates_per_agent]

    def generate(
        self,
        state: JointSafetyInputV1,
        requested_actions: np.ndarray,
        memory: LivenessMemoryV1,
        mode: SafetyMode,
        arrived: np.ndarray,
    ) -> tuple[list[JointCandidateV1], bool]:
        requested = np.clip(np.asarray(requested_actions, dtype=float), -1.0, 1.0)
        local = [
            self._local(state, requested, agent_id, memory, mode, arrived)
            for agent_id in range(state.num_agents)
        ]
        candidates: list[JointCandidateV1] = []
        seen: set[tuple[float, ...]] = set()

        def add(combination: tuple[tuple[str, np.ndarray], ...]) -> None:
            normal_budget = (
                self.config.max_joint_candidates
                - self.config.emergency_candidate_reserve
            )
            if len(candidates) >= normal_budget:
                return
            labels, rows = zip(*combination)
            actions = np.asarray(rows, dtype=np.float64)
            key = tuple(float(value) for value in np.round(actions.reshape(-1), 8))
            if key in seen:
                return
            seen.add(key)
            candidates.append(JointCandidateV1(
                candidate_id=f"joint:{len(candidates):03d}",
                actions=actions,
                source="+".join(labels),
                braking_mask=tuple(
                    label in {"brake", "arrival_hold", "recovery_brake"}
                    for label in labels
                ),
            ))

        # Register coordinated actor/brake/goal/tangent choices before the
        # bounded product so full-team fallbacks cannot be starved by order.
        for label in (
            "actor", "brake", "arrival_hold", "goal",
            "committed_tangent", "tangent_goal_blend", "stall_escape",
            "recovery_brake",
        ):
            aligned = []
            for rows in local:
                match = next((item for item in rows if item[0] == label), None)
                if match is None:
                    aligned = []
                    break
                aligned.append(match)
            if aligned:
                add(tuple(aligned))
        total_product = int(np.prod([len(row) for row in local], dtype=np.int64))
        for combination in product(*local):
            if len(candidates) >= (
                self.config.max_joint_candidates
                - self.config.emergency_candidate_reserve
            ):
                break
            add(tuple(combination))
        return candidates, total_product > len(candidates)

    def emergency_candidates(
        self,
        state: JointSafetyInputV1,
        requested_actions: np.ndarray,
        memory: LivenessMemoryV1,
    ) -> list[JointCandidateV1]:
        """Bounded backup candidates used only after normal rejection."""

        requested = np.clip(np.asarray(requested_actions, dtype=float), -1.0, 1.0)
        velocities = np.asarray(state.velocities, dtype=float)
        brake = np.asarray([
            braking_action(row, state) for row in velocities
        ], dtype=np.float64)
        rows = [JointCandidateV1(
            candidate_id="emergency:all_stop_hold",
            actions=brake,
            source="+".join("emergency_brake" for _ in range(state.num_agents)),
            braking_mask=tuple(True for _ in range(state.num_agents)),
        )]
        for agent_id, tracks in enumerate(state.visible_obstacles):
            if not tracks or len(rows) >= self.config.emergency_candidate_reserve:
                continue
            nearest = min(
                tracks,
                key=lambda item: (np.linalg.norm(item.relative_position), item.track_id),
            )
            radial = np.asarray(nearest.relative_position, dtype=float)
            norm = float(np.linalg.norm(radial))
            if norm <= 1e-12:
                continue
            away = -radial / norm
            side = memory.preferred_passing_side(
                state, agent_id, nearest.track_id, radial, commit=False
            )
            tangent = side * np.asarray([-radial[1], radial[0]]) / norm
            for label, action in (
                ("emergency_away", away),
                ("emergency_tangent", tangent),
            ):
                if len(rows) >= self.config.emergency_candidate_reserve:
                    break
                joint = brake.copy()
                joint[agent_id] = np.clip(action, -1.0, 1.0)
                labels = ["emergency_brake"] * state.num_agents
                labels[agent_id] = label
                mask = [True] * state.num_agents
                mask[agent_id] = False
                rows.append(JointCandidateV1(
                    candidate_id=f"emergency:{agent_id}:{label}",
                    actions=joint,
                    source="+".join(labels),
                    braking_mask=tuple(mask),
                ))
        return rows[: self.config.emergency_candidate_reserve]
