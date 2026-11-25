from __future__ import annotations

from itertools import combinations, permutations

import numpy as np

from src.safety.ucr_jsf_l_v2.candidates import CandidateConfigV2, CandidateGeneratorV2
from src.safety.ucr_jsf_l_v2.contracts import SafetyMode
from src.safety.ucr_jsf_l_v2.liveness import LivenessConfigV2, LivenessMemoryV2
from src.safety.ucr_jsf_l_v2.ordering import build_canonical_ordering

from .conftest import blocked_state, make_state, map_actions_to_original, permute_state


def _generate(state, requested, mode=SafetyMode.NORMAL):
    memory = LivenessMemoryV2(3, LivenessConfigV2())
    arrived = memory.observe(state)[1]
    ordering = build_canonical_ordering(state, requested, liveness_memory=memory)
    generator = CandidateGeneratorV2(CandidateConfigV2())
    candidates, exhausted = generator.generate(
        state, requested, memory, mode, arrived, ordering
    )
    return memory, ordering, generator, candidates, exhausted


def test_complete_single_coverage_balanced_joint_quotas_and_budget():
    state = blocked_state()
    requested = np.asarray([[0.8, 0.2], [0.6, -0.3], [0.9, 0.1]])
    _memory, ordering, generator, candidates, exhausted = _generate(state, requested)
    assert len(candidates) == 60
    assert exhausted
    assert sum(item.layer == "anchor" for item in candidates) <= 8
    assert sum(item.layer == "single" for item in candidates) == sum(
        len(rows) - 1 for rows in generator.last_local_candidates
    )
    for canonical_rank, agent_id in enumerate(ordering.flat):
        covered = {
            item.canonical_ranks[canonical_rank]
            for item in candidates if item.layer == "single"
            and item.canonical_ranks[canonical_rank] != 0
            and sum(rank != 0 for rank in item.canonical_ranks) == 1
        }
        assert covered == set(range(1, len(generator.last_local_candidates[agent_id])))
    for left, right in combinations(range(3), 2):
        count = sum(
            item.layer == "pair"
            and tuple(index for index, rank in enumerate(item.canonical_ranks) if rank)
            == (left, right)
            for item in candidates
        )
        assert count == 5
    assert sum(item.layer == "triple" for item in candidates) == 16


def test_emergency_breadth_first_covers_each_physical_uav_and_total_budget():
    state = blocked_state()
    requested = np.asarray([[0.8, 0.2], [0.6, -0.3], [0.9, 0.1]])
    memory, ordering, generator, candidates, _ = _generate(state, requested)
    emergency = generator.emergency_candidates(state, requested, memory, ordering)
    assert [item.candidate_id for item in emergency] == [
        "emergency:all_stop_hold",
        "emergency:c000:away",
        "emergency:c001:away",
        "emergency:c002:away",
    ]
    assert len(candidates) + len(emergency) == 64

    baseline = [map_actions_to_original(row.actions, (0, 1, 2)) for row in emergency]
    for permutation in permutations(range(3)):
        pstate = permute_state(state, permutation)
        prequested = requested[list(permutation)]
        pmemory, porder, pgenerator, _rows, _ = _generate(pstate, prequested)
        pemergency = pgenerator.emergency_candidates(
            pstate, prequested, pmemory, porder
        )
        assert [item.candidate_id for item in pemergency] == [
            "emergency:all_stop_hold",
            "emergency:c000:away",
            "emergency:c001:away",
            "emergency:c002:away",
        ]
        for left, right in zip(baseline, pemergency):
            np.testing.assert_allclose(
                left, map_actions_to_original(right.actions, permutation)
            )


def test_heterogeneous_anchor_uses_per_agent_fallback_instead_of_disappearing():
    state = make_state(
        goals=[[20, 20], [90, 50], [90, 80]],
        tracks=((), (), ()),
    )
    requested = np.asarray([[0.8, 0.2], [0.6, -0.3], [0.9, 0.1]])
    _memory, _ordering, _generator, candidates, _ = _generate(state, requested)
    anchors = [item for item in candidates if item.layer == "anchor"]
    assert anchors
    assert any(len(set(item.source.split("+"))) > 1 for item in anchors)
