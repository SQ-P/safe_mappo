from __future__ import annotations

import numpy as np

from src.algorithms.intervention_advantage_certification import _best_reference


def test_best_reference_is_stage_specific_and_requires_valid_improvement():
    arrays = {
        "raw_stage_costs": np.asarray([[1.5, 1.2, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
        "reference_stage_costs": np.asarray([
            [[1.0, 1.1, 0.0], [1.4, 0.2, 0.0], [0.1, 0.1, 0.0],
             [2.0, 2.0, 0.0], [2.0, 2.0, 0.0], [2.0, 2.0, 0.0]],
            [[0.0, 0.0, 0.0]] * 6,
        ], dtype=np.float32),
        "reference_valid": np.asarray([
            [True, True, False, False, False, False],
            [True, True, True, True, True, True],
        ], dtype=bool),
    }
    index, mask = _best_reference(arrays)
    assert index[0].tolist() == [0, 1]
    assert mask[0].tolist() == [True, True]
    assert mask[1].tolist() == [False, False]


def test_equal_cost_reference_is_not_a_causal_better_pair():
    arrays = {
        "raw_stage_costs": np.ones((1, 3), dtype=np.float32),
        "reference_stage_costs": np.ones((1, 6, 3), dtype=np.float32),
        "reference_valid": np.ones((1, 6), dtype=bool),
    }
    _, mask = _best_reference(arrays)
    assert not mask.any()
