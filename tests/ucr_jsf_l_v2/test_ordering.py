from __future__ import annotations

from itertools import permutations
from pathlib import Path

import numpy as np

from src.safety.ucr_jsf_l_v2.liveness import LivenessConfigV2, LivenessMemoryV2
from src.safety.ucr_jsf_l_v2.ordering import (
    build_canonical_ordering,
    canonical_action_signature,
)
from src.safety.ucr_jsf_l_v2.reachability import ReachabilityConfigV2, ReachabilityMemoryV2

from .conftest import blocked_state, make_state, map_actions_to_original, permute_state


def test_physical_order_invariant_across_all_six_permutations_with_memory():
    state = blocked_state(step=4)
    requested = np.asarray([[0.8, 0.2], [0.6, -0.3], [0.9, 0.1]])
    liveness = LivenessMemoryV2(3, LivenessConfigV2())
    liveness.previous_distances = np.asarray([70.0, 71.0, 72.0])
    liveness.stall_counts = np.asarray([2, 4, 1])
    liveness.passing_sides = {(0, 101): (1, 3), (1, 102): (-1, 3)}
    reachability = ReachabilityMemoryV2(ReachabilityConfigV2(2.0, 2))
    reachability.update(state)
    baseline = build_canonical_ordering(
        state, requested, liveness_memory=liveness, reachability_memory=reachability
    )
    baseline_physical = baseline.flat

    for permutation in permutations(range(3)):
        permuted = permute_state(state, permutation)
        order = build_canonical_ordering(
            permuted,
            requested[list(permutation)],
            liveness_memory=liveness.permuted_copy(permutation),
            reachability_memory=reachability.permuted_copy(permutation),
        )
        physical = tuple(permutation[new_id] for new_id in order.flat)
        assert physical == baseline_physical


def test_degenerate_group_is_explicit_and_requires_symmetric_ranks():
    state = make_state(
        positions=[[20, 20], [20, 20], [20, 20]],
        velocities=np.zeros((3, 2)),
        goals=[[90, 20], [90, 20], [90, 20]],
    )
    requested = np.zeros((3, 2))
    order = build_canonical_ordering(state, requested)
    assert order.groups == ((0, 1, 2),)
    assert order.ranks_are_symmetric((2, 2, 2))
    assert not order.ranks_are_symmetric((1, 2, 1))
    actions = np.asarray([[0.2, 0.1], [-0.2, 0.1], [0.0, -0.3]])
    expected = canonical_action_signature(actions, order)
    for permutation in permutations(range(3)):
        assert canonical_action_signature(actions[list(permutation)], order) == expected


def test_v2_liveness_contains_no_agent_parity_fallback():
    source = Path("src/safety/ucr_jsf_l_v2/liveness.py").read_text(encoding="utf-8")
    assert "agent_id %" not in source
    assert "% 2" not in source
