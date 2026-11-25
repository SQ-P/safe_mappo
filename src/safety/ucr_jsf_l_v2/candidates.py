"""Balanced bounded joint-candidate search for UCR-JSF-L V2."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

import numpy as np

from .contracts import JointCandidateV2, JointSafetyInputV2, SafetyMode
from .liveness import LivenessMemoryV2
from .ordering import CanonicalOrderingV2, canonical_action_signature


@dataclass(frozen=True)
class CandidateConfigV2:
    max_local_candidates_per_agent: int = 8
    max_joint_candidates: int = 64
    tangent_strength: float = 0.8
    emergency_candidate_reserve: int = 4
    team_anchor_limit: int = 8
    pair_candidate_quota: int = 15
    triple_candidate_quota: int = 16

    @classmethod
    def from_mapping(cls, payload: dict) -> "CandidateConfigV2":
        return cls(
            max_local_candidates_per_agent=int(
                payload.get("max_local_candidates_per_agent", 8)
            ),
            max_joint_candidates=int(payload.get("max_joint_candidates", 64)),
            tangent_strength=float(payload.get("tangent_strength", 0.8)),
            emergency_candidate_reserve=int(
                payload.get("emergency_candidate_reserve", 4)
            ),
            team_anchor_limit=int(payload.get("team_anchor_limit", 8)),
            pair_candidate_quota=int(payload.get("pair_candidate_quota", 15)),
            triple_candidate_quota=int(payload.get("triple_candidate_quota", 16)),
        )

    @property
    def normal_budget(self) -> int:
        return self.max_joint_candidates - self.emergency_candidate_reserve


def braking_action(velocity: np.ndarray, state: JointSafetyInputV2) -> np.ndarray:
    return np.clip(
        -np.asarray(velocity, dtype=np.float64)
        / (state.max_acceleration * state.dt),
        -1.0,
        1.0,
    )


class CandidateGeneratorV2:
    ANCHOR_LABELS = (
        "actor",
        "brake",
        "goal",
        "committed_tangent",
        "tangent_goal_blend",
        "arrival_hold",
        "stall_escape",
        "recovery_brake",
    )

    def __init__(self, config: CandidateConfigV2) -> None:
        self.config = config
        if not 1 <= config.max_local_candidates_per_agent <= 8:
            raise ValueError("local candidate cap must be in [1, 8]")
        if config.max_joint_candidates != 64:
            raise ValueError("V2 requires a 64-candidate total budget")
        if config.emergency_candidate_reserve != 4:
            raise ValueError("V2 requires four emergency candidates")
        if config.normal_budget != 60:
            raise ValueError("V2 requires a 60-candidate normal budget")
        if config.team_anchor_limit != 8:
            raise ValueError("V2 requires at most eight team anchors")
        if config.pair_candidate_quota != 15 or config.triple_candidate_quota != 16:
            raise ValueError("V2 pair/triple quotas must be 15/16")
        self.last_local_candidates: tuple[tuple[tuple[str, np.ndarray], ...], ...] = ()

    @staticmethod
    def _deduplicate(
        rows: list[tuple[str, np.ndarray]],
    ) -> list[tuple[str, np.ndarray]]:
        result: list[tuple[str, np.ndarray]] = []
        seen: set[tuple[float, float]] = set()
        for label, row in rows:
            value = np.clip(np.asarray(row, dtype=np.float64), -1.0, 1.0)
            key = tuple(float(item) for item in np.round(value, 12))
            if key not in seen:
                seen.add(key)
                result.append((label, value))
        return result

    def local_candidates(
        self,
        state: JointSafetyInputV2,
        requested: np.ndarray,
        agent_id: int,
        memory: LivenessMemoryV2,
        mode: SafetyMode,
        arrived: np.ndarray,
    ) -> list[tuple[str, np.ndarray]]:
        velocity = np.asarray(state.velocities[agent_id], dtype=np.float64)
        brake = braking_action(velocity, state)
        goal_delta = (
            np.asarray(state.final_goals[agent_id], dtype=np.float64)
            - np.asarray(state.positions[agent_id], dtype=np.float64)
        )
        goal_norm = float(np.linalg.norm(goal_delta))
        desired_velocity = (
            np.zeros(2, dtype=np.float64)
            if goal_norm <= 1e-12
            else goal_delta / goal_norm * state.max_speed
        )
        goal_action = np.clip(
            (desired_velocity - velocity) / (state.max_acceleration * state.dt),
            -1.0,
            1.0,
        )
        rows: list[tuple[str, np.ndarray]] = [
            ("actor", requested[agent_id]),
            ("brake", brake),
            ("half_actor", 0.5 * requested[agent_id]),
            ("goal", goal_action),
        ]
        blocker = memory.blocking_track(state, agent_id)
        if blocker is not None:
            radial = np.asarray(blocker.relative_position, dtype=np.float64)
            radial_norm = float(np.linalg.norm(radial))
            if radial_norm > 1e-12:
                side = memory.preferred_passing_side(
                    state, agent_id, blocker.track_id, radial, commit=False
                )
                tangent = side * np.asarray([-radial[1], radial[0]]) / radial_norm
                tangent_action = np.clip(
                    tangent * self.config.tangent_strength
                    - velocity / max(state.max_speed, 1e-6) * 0.25,
                    -1.0,
                    1.0,
                )
                rows.extend((
                    ("committed_tangent", tangent_action),
                    ("tangent_goal_blend", 0.65 * tangent_action + 0.35 * goal_action),
                    ("opposite_tangent_probe", -tangent_action),
                ))
        rows = self._deduplicate(rows)
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
        return self._deduplicate(rows)[:self.config.max_local_candidates_per_agent]

    @staticmethod
    def _rank_key(
        ranks: tuple[int, ...], ordering: CanonicalOrderingV2
    ) -> tuple:
        canonical = tuple(ranks[agent_id] for agent_id in ordering.flat)
        nonzero = tuple(value for value in canonical if value != 0)
        return (
            sum(nonzero),
            len(nonzero),
            max(nonzero, default=0),
            canonical,
        )

    def generate(
        self,
        state: JointSafetyInputV2,
        requested_actions: np.ndarray,
        memory: LivenessMemoryV2,
        mode: SafetyMode,
        arrived: np.ndarray,
        ordering: CanonicalOrderingV2,
    ) -> tuple[list[JointCandidateV2], bool]:
        requested = np.clip(
            np.asarray(requested_actions, dtype=np.float64), -1.0, 1.0
        )
        local = tuple(tuple(self.local_candidates(
            state, requested, agent_id, memory, mode, arrived
        )) for agent_id in range(state.num_agents))
        self.last_local_candidates = local
        candidates: list[JointCandidateV2] = []
        seen: set[tuple[tuple[float, float], ...]] = set()

        def add(ranks: tuple[int, ...], layer: str) -> bool:
            if len(candidates) >= self.config.normal_budget:
                return False
            if not ordering.ranks_are_symmetric(ranks):
                return False
            rows = [local[agent_id][ranks[agent_id]] for agent_id in range(state.num_agents)]
            labels = tuple(item[0] for item in rows)
            actions = np.asarray([item[1] for item in rows], dtype=np.float64)
            signature = canonical_action_signature(actions, ordering)
            if signature in seen:
                return False
            seen.add(signature)
            canonical_ranks = tuple(ranks[agent_id] for agent_id in ordering.flat)
            candidates.append(JointCandidateV2(
                candidate_id=f"normal:{len(candidates):03d}",
                actions=actions,
                source="+".join(labels),
                braking_mask=tuple(
                    label in {"brake", "arrival_hold", "recovery_brake"}
                    for label in labels
                ),
                layer=layer,
                canonical_ranks=canonical_ranks,
            ))
            return True

        baseline = tuple(0 for _ in range(state.num_agents))
        for label in self.ANCHOR_LABELS[:self.config.team_anchor_limit]:
            ranks = []
            for agent_id, rows in enumerate(local):
                ranks.append(next(
                    (rank for rank, item in enumerate(rows) if item[0] == label),
                    0,
                ))
            add(tuple(ranks), "anchor")

        for agent_id in ordering.flat:
            for rank in range(1, len(local[agent_id])):
                ranks = list(baseline)
                ranks[agent_id] = rank
                add(tuple(ranks), "single")

        canonical_agents = ordering.flat
        pair_rows: dict[tuple[int, int], list[tuple[int, ...]]] = {}
        for left, right in combinations(canonical_agents, 2):
            rows = []
            for left_rank in range(1, len(local[left])):
                for right_rank in range(1, len(local[right])):
                    ranks = list(baseline)
                    ranks[left] = left_rank
                    ranks[right] = right_rank
                    rank_tuple = tuple(ranks)
                    if ordering.ranks_are_symmetric(rank_tuple):
                        rows.append(rank_tuple)
            pair_rows[(left, right)] = sorted(
                rows, key=lambda item: self._rank_key(item, ordering)
            )

        pair_selected: dict[tuple[int, int], set[tuple[int, ...]]] = {
            pair: set() for pair in pair_rows
        }
        pair_count_per_pair = self.config.pair_candidate_quota // max(len(pair_rows), 1)
        for pair in sorted(
            pair_rows,
            key=lambda item: tuple(ordering.rank_by_agent[index] for index in item),
        ):
            for ranks in pair_rows[pair]:
                if len(pair_selected[pair]) >= pair_count_per_pair:
                    break
                if add(ranks, "pair"):
                    pair_selected[pair].add(ranks)

        triple_rows: list[tuple[int, ...]] = []
        if state.num_agents == 3:
            for rank_values in product(*(range(1, len(rows)) for rows in local)):
                ranks = tuple(int(value) for value in rank_values)
                if ordering.ranks_are_symmetric(ranks):
                    triple_rows.append(ranks)
            triple_rows.sort(key=lambda item: self._rank_key(item, ordering))
        triple_selected: set[tuple[int, ...]] = set()
        for ranks in triple_rows:
            if len(triple_selected) >= self.config.triple_candidate_quota:
                break
            if add(ranks, "triple"):
                triple_selected.add(ranks)

        # Reallocate anchor/dedup vacancies without reducing the guaranteed
        # per-pair or triple quotas established above.
        pair_leftovers = sorted(
            (
                ranks for pair, rows in pair_rows.items() for ranks in rows
                if ranks not in pair_selected[pair]
            ),
            key=lambda item: self._rank_key(item, ordering),
        )
        triple_leftovers = [
            ranks for ranks in triple_rows if ranks not in triple_selected
        ]
        for ranks in (*pair_leftovers, *triple_leftovers):
            if len(candidates) >= self.config.normal_budget:
                break
            add(ranks, "pair_fill" if sum(value != 0 for value in ranks) == 2 else "triple_fill")

        valid_product = 1
        for group in ordering.groups:
            if len(group) == 1:
                valid_product *= len(local[group[0]])
            else:
                valid_product *= min(len(local[agent_id]) for agent_id in group)
        return candidates, valid_product > len(candidates)

    def emergency_candidates(
        self,
        state: JointSafetyInputV2,
        requested_actions: np.ndarray,
        memory: LivenessMemoryV2,
        ordering: CanonicalOrderingV2,
    ) -> list[JointCandidateV2]:
        del requested_actions
        velocities = np.asarray(state.velocities, dtype=np.float64)
        brake = np.asarray(
            [braking_action(row, state) for row in velocities], dtype=np.float64
        )
        rows = [JointCandidateV2(
            candidate_id="emergency:all_stop_hold",
            actions=brake,
            source="+".join("emergency_brake" for _ in range(state.num_agents)),
            braking_mask=tuple(True for _ in range(state.num_agents)),
            layer="emergency_stop",
        )]

        valid: list[tuple[int, object, np.ndarray, float]] = []
        for canonical_rank, agent_id in enumerate(ordering.flat):
            tracks = state.visible_obstacles[agent_id]
            choices = sorted(
                (
                    float(np.linalg.norm(track.relative_position)),
                    tuple(float(value) for value in track.relative_position),
                    int(track.track_id),
                    track,
                )
                for track in tracks
                if float(np.linalg.norm(track.relative_position)) > 1e-12
            )
            if choices:
                track = choices[0][3]
                radial = np.asarray(track.relative_position, dtype=np.float64)
                valid.append((canonical_rank, track, radial, float(np.linalg.norm(radial))))

        def append_agent_action(
            canonical_rank: int,
            agent_id: int,
            label: str,
            action: np.ndarray,
        ) -> None:
            if len(rows) >= self.config.emergency_candidate_reserve:
                return
            joint = brake.copy()
            joint[agent_id] = np.clip(action, -1.0, 1.0)
            labels = ["emergency_brake"] * state.num_agents
            labels[agent_id] = label
            mask = [True] * state.num_agents
            mask[agent_id] = False
            rows.append(JointCandidateV2(
                candidate_id=f"emergency:c{canonical_rank:03d}:{label.removeprefix('emergency_')}",
                actions=joint,
                source="+".join(labels),
                braking_mask=tuple(mask),
                layer=label,
            ))

        # Breadth first: one valid Away action per physical canonical UAV.
        for canonical_rank, _track, radial, norm in valid:
            agent_id = ordering.flat[canonical_rank]
            append_agent_action(
                canonical_rank, agent_id, "emergency_away", -radial / norm
            )

        # Only remaining reserve is used for tangents.
        for canonical_rank, track, radial, norm in valid:
            if len(rows) >= self.config.emergency_candidate_reserve:
                break
            agent_id = ordering.flat[canonical_rank]
            side = memory.preferred_passing_side(
                state, agent_id, track.track_id, radial, commit=False
            )
            tangent = side * np.asarray([-radial[1], radial[0]]) / norm
            append_agent_action(
                canonical_rank, agent_id, "emergency_tangent", tangent
            )
        return rows[:self.config.emergency_candidate_reserve]
