import numpy as np

from src.algorithms.ia_mappo_gates import evaluate_formal_r5_gate


def test_formal_r5_gate_uses_two_level_paired_bootstrap():
    train_seeds = np.repeat(np.arange(5), 50)
    layouts = np.tile(np.arange(50), 5)
    result = evaluate_formal_r5_gate(
        deltas={
            "material_modification": np.full(250, -0.06),
            "cfms": np.zeros(250),
            "progress": np.zeros(250),
            "collision": np.zeros(250),
            "timeout": np.zeros(250),
        },
        train_seeds=train_seeds,
        layout_ids=layouts,
        pair_collision_count=0,
        boundary_collision_count=0,
        replicates=100,
        seed=23,
    )
    assert result.passed
    assert result.metrics["material_modification"].upper_95 < 0.0


def test_formal_r5_gate_rejects_task_regression():
    train_seeds = np.repeat(np.arange(5), 20)
    layouts = np.tile(np.arange(20), 5)
    result = evaluate_formal_r5_gate(
        deltas={
            "material_modification": np.full(100, -0.06),
            "cfms": np.full(100, -0.03),
            "progress": np.zeros(100),
            "collision": np.zeros(100),
            "timeout": np.zeros(100),
        },
        train_seeds=train_seeds,
        layout_ids=layouts,
        pair_collision_count=0,
        boundary_collision_count=0,
        replicates=50,
        seed=41,
    )
    assert not result.passed
    assert "cfms_noninferiority_failed" in result.failures
