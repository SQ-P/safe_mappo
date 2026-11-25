from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from src.safety.ucr_jsf_l_v2 import SafetyMode, UCRJSFLExecutorV2

from .conftest import (
    blocked_state,
    executor_config,
    make_state,
    make_track,
    map_actions_to_original,
    permute_state,
)


@pytest.mark.parametrize("mode", tuple(SafetyMode))
def test_full_pipeline_is_equivariant_for_all_permutations_and_history(mode):
    state = blocked_state(step=5)
    requested = np.asarray([[0.8, 0.2], [0.6, -0.3], [0.9, 0.1]])
    seed_executor = UCRJSFLExecutorV2(executor_config(), num_agents=3)
    seed_executor.liveness.previous_distances = np.asarray([70.2, 70.1, 70.3])
    seed_executor.liveness.stall_counts = np.asarray([3, 2, 1])
    seed_executor.liveness.passing_sides = {(0, 101): (1, 4), (1, 102): (-1, 4)}
    seed_executor.reachability.update(make_state(
        step=1,
        tracks=(
            (
                make_track(0, 101),
                make_track(0, 999, relative=(15.0, 2.0)),
            ),
            (make_track(1, 102),),
            (make_track(2, 103),),
        ),
    ))
    seed_executor.recovery.mode = mode

    expected = None
    for permutation in permutations(range(3)):
        executor = UCRJSFLExecutorV2(executor_config(), num_agents=3)
        executor.liveness = seed_executor.liveness.permuted_copy(permutation)
        executor.reachability = seed_executor.reachability.permuted_copy(permutation)
        executor.recovery.mode = mode
        output = executor.execute(
            permute_state(state, permutation), requested[list(permutation)]
        )
        actual = (
            map_actions_to_original(output.actions, permutation),
            output.trace.selected_candidate_id,
            output.trace.selected_canonical_action_signature,
            output.trace.selected_progress_score,
            output.trace.selected_actor_deviation,
            output.trace.minimum_boundary_margin_m,
            output.trace.minimum_pair_margin_m,
            output.trace.minimum_obstacle_margin_m,
        )
        assert any("999:lost_beyond_ttl" in row for row in output.trace.unknown_evidence)
        if expected is None:
            expected = actual
            continue
        np.testing.assert_allclose(actual[0], expected[0], rtol=0.0, atol=1e-7)
        assert actual[1:3] == expected[1:3]
        for observed, reference in zip(actual[3:], expected[3:]):
            if reference is None:
                assert observed is None
            else:
                assert observed == pytest.approx(reference, abs=1e-12)
