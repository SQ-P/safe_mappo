import json
from hashlib import sha256
from pathlib import Path

import numpy as np

from src.algorithms.intervention_r3 import (
    _critic_features,
    _severity_baseline_diagnostic,
    _split_layouts,
    R3_DATASET_SCHEMA,
    R3_FEATURE_CONTRACT,
    evaluate_r3_shadow_step,
    select_validation_operating_threshold,
    train_and_certify_r3_twin_critic,
)
from src.algorithms.intervention_team_data import ACTION_TTC_VERSION
from src.envs.cooperative_uav_env import CooperativeUAVEnv
from src.utils.io_utils import load_yaml


def test_r3_200_layout_panel_is_unique_and_disjoint_from_r2():
    r3 = json.loads(Path(
        "configs/evaluation/ia_mappo_v1_r3_200_layouts.json"
    ).read_text(encoding="utf-8"))["layout_seeds"]
    r2 = json.loads(Path(
        "configs/evaluation/ia_mappo_v1_r1_40_layouts.json"
    ).read_text(encoding="utf-8"))["layout_seeds"]
    assert len(r3) == 200
    assert len(set(r3)) == 200
    assert not set(r3) & set(r2)


def test_r3_1_panel_is_fresh_and_disjoint_from_opened_panels():
    r3_1 = json.loads(Path(
        "configs/evaluation/ia_mappo_v1_r3_1_200_layouts.json"
    ).read_text(encoding="utf-8"))
    r3 = json.loads(Path(
        "configs/evaluation/ia_mappo_v1_r3_200_layouts.json"
    ).read_text(encoding="utf-8"))["layout_seeds"]
    r2 = json.loads(Path(
        "configs/evaluation/ia_mappo_v1_r1_40_layouts.json"
    ).read_text(encoding="utf-8"))["layout_seeds"]
    assert r3_1["schema_version"] == "IAMAPPOR3LayoutPanelV2"
    layouts = r3_1["layout_seeds"]
    assert len(layouts) == len(set(layouts)) == 200
    assert not set(layouts) & set(r3)
    assert not set(layouts) & set(r2)


def test_r3_2_panel_is_fresh_and_disjoint_from_all_opened_panels():
    r3_2 = json.loads(Path(
        "configs/evaluation/ia_mappo_v1_r3_2_200_layouts.json"
    ).read_text(encoding="utf-8"))
    prior = []
    for path in (
        "configs/evaluation/ia_mappo_v1_r1_40_layouts.json",
        "configs/evaluation/ia_mappo_v1_r3_200_layouts.json",
        "configs/evaluation/ia_mappo_v1_r3_1_200_layouts.json",
    ):
        prior.extend(json.loads(Path(path).read_text(encoding="utf-8"))["layout_seeds"])
    assert r3_2["schema_version"] == "IAMAPPOR3LayoutPanelV3"
    layouts = r3_2["layout_seeds"]
    assert len(layouts) == len(set(layouts)) == 200
    assert not set(layouts) & set(prior)


def test_r3_layout_split_is_deterministic_isolated_60_10_10_20():
    layouts = list(range(51000, 51200))
    first = _split_layouts(layouts)
    second = _split_layouts(layouts)
    assert first == second
    assert {name: len(values) for name, values in first.items()} == {
        "train": 120, "calibration": 20,
        "threshold_selection": 20, "test": 40,
    }
    groups = [set(values) for values in first.values()]
    assert all(
        not left & right
        for index, left in enumerate(groups)
        for right in groups[index + 1:]
    )
    assert set().union(*groups) == set(layouts)


def test_r3_real_31_candidate_panel_is_state_immutable():
    config = load_yaml("configs/active/ia_mappo_v1_3u_6s1d_standard.yaml")
    env = CooperativeUAVEnv({
        **config["env"], "reward": config["reward"],
        "oracle_subgoal": config["oracle_subgoal"],
        "subgoal_selector": config["subgoal_selector"],
        "safety_execution": config["safety_execution"],
    })
    env.reset(seed=51000)
    raw = np.zeros((env.num_agents, 2), dtype=np.float32)
    record = evaluate_r3_shadow_step(
        env,
        raw_joint_action=raw,
        global_state_normalized=np.asarray(
            env.get_global_state(), dtype=np.float32
        ),
        actor_hidden=np.zeros((1, env.num_agents, 8), dtype=np.float32),
    )
    assert int(record["shadow_candidate_count"]) == 31
    assert record["snapshot_digest_before"] == record["snapshot_digest_after"]
    assert record["environment_fingerprint_before"] == (
        record["environment_fingerprint_after"]
    )
    assert record["actor_hidden_digest_before"] == (
        record["actor_hidden_digest_after"]
    )
    assert np.isclose(
        np.mean(record["stage_avoidable_costs"]),
        record["avoidable_total_cost"],
    )
    assert record["per_agent_margins"].shape == (env.num_agents, 3)
    assert record["action_conditioned_ttc_s"].shape == (env.num_agents,)
    assert record["ttc_warning"].shape == (env.num_agents,)
    assert record["ttc_emergency"].shape == (env.num_agents,)
    assert record["minimum_predicted_clearance_m"].shape == (env.num_agents,)
    assert np.isfinite(record["action_conditioned_ttc_s"]).all()
    assert np.isfinite(record["minimum_predicted_clearance_m"]).all()


def test_r3_shadow_panel_does_not_change_real_transition():
    config = load_yaml("configs/active/ia_mappo_v1_3u_6s1d_standard.yaml")
    env_config = {
        **config["env"], "reward": config["reward"],
        "oracle_subgoal": config["oracle_subgoal"],
        "subgoal_selector": config["subgoal_selector"],
        "safety_execution": config["safety_execution"],
    }
    shadowed = CooperativeUAVEnv(env_config)
    control = CooperativeUAVEnv(env_config)
    shadowed.reset(seed=51001)
    control.reset(seed=51001)
    raw = np.asarray(
        [[0.2, -0.1], [0.0, 0.15], [-0.2, 0.1]], dtype=np.float32
    )
    evaluate_r3_shadow_step(
        shadowed,
        raw_joint_action=raw,
        global_state_normalized=np.asarray(
            shadowed.get_global_state(), dtype=np.float32
        ),
        actor_hidden=np.zeros((1, shadowed.num_agents, 8), dtype=np.float32),
    )
    left = shadowed.step(raw)
    right = control.step(raw)
    np.testing.assert_allclose(left[0], right[0], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(left[1], right[1], atol=0.0, rtol=0.0)
    assert left[2:4] == right[2:4]
    assert left[4]["reason"] == right[4]["reason"]
    np.testing.assert_allclose(
        left[4]["executed_actions"], right[4]["executed_actions"],
        atol=0.0, rtol=0.0,
    )


def test_r3_1_critic_features_include_action_conditioned_ttc_contract():
    rows, agents = 2, 3
    data = {
        "layout_id": np.arange(rows),
        "global_state_normalized": np.zeros((rows, 5), np.float32),
        "raw_joint_mean_action": np.zeros((rows, agents, 2), np.float32),
        "causal_safety_context": np.zeros((rows, 4), np.float32),
        "current_safety_margin": np.zeros(rows, np.float32),
        "per_agent_margins": np.zeros((rows, agents, 3), np.float32),
        "action_conditioned_ttc_s": np.full((rows, agents), 7.0, np.float32),
        "ttc_warning": np.zeros((rows, agents), bool),
        "ttc_emergency": np.zeros((rows, agents), bool),
        "minimum_predicted_clearance_m": np.full(
            (rows, agents), 2.0, np.float32
        ),
    }
    features = _critic_features(data)
    expected = 5 + agents * 2 + 4 + 1 + agents * 3 + agents * 4
    assert features.shape == (rows, expected)
    assert np.all(features[:, -4 * agents:-3 * agents] == 7.0)


def test_r3_2_threshold_uses_layout_bootstrap_and_recall_margin():
    labels = np.asarray([1, 0] * 10)
    probabilities = np.asarray([0.9, 0.1] * 10)
    layouts = np.repeat(np.arange(10), 2)
    selection = select_validation_operating_threshold(
        labels, probabilities, layouts,
        bootstrap_replicates=100,
    )
    assert selection["selection_partition"] == "threshold_selection_layouts_only"
    assert selection["selection_passed"] is True
    assert np.isclose(selection["selected"]["threshold"], 0.9)
    assert np.isclose(selection["selected"]["recall"], 1.0)
    assert selection["selected"]["recall_lower_95_layout_bootstrap"] >= 0.8
    assert selection["selected"]["precision_lower_95_layout_bootstrap"] >= 0.5


def test_r3_2_severity_must_beat_train_median_baseline_by_layout():
    result = _severity_baseline_diagnostic(
        train_targets=np.asarray([0.5, 1.0, 1.5]),
        test_targets=np.asarray([0.5, 1.0, 1.5] * 10),
        test_predictions=np.asarray([0.55, 1.05, 1.45] * 10),
        test_layout_ids=np.repeat(np.arange(10), 3),
        bootstrap_replicates=100,
        seed=7,
    )
    assert result["passed"] is True
    assert result["model_mae"] < result["baseline_mae"]
    assert result["mae_improvement_lower_95_layout_bootstrap"] > 0.0


def test_r3_2_training_smoke_writes_prediction_audit(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    layouts = np.repeat(np.arange(40), 20)
    rows = len(layouts)
    binary = np.tile(np.asarray([0.0, 1.0], np.float32), rows // 2)
    severity = np.stack((binary, 0.5 * binary, np.zeros(rows)), axis=1)
    digest = np.asarray(["0" * 64] * rows)
    arrays = {
        "layout_id": layouts,
        "global_state_normalized": binary[:, None],
        "raw_joint_mean_action": np.zeros((rows, 3, 2), np.float32),
        "causal_safety_context": np.zeros((rows, 8), np.float32),
        "current_safety_margin": np.zeros(rows, np.float32),
        "per_agent_margins": np.zeros((rows, 3, 3), np.float32),
        "action_conditioned_ttc_s": np.ones((rows, 3), np.float32),
        "ttc_warning": np.zeros((rows, 3), bool),
        "ttc_emergency": np.zeros((rows, 3), bool),
        "minimum_predicted_clearance_m": np.ones((rows, 3), np.float32),
        "stage_avoidable_costs": severity,
        "avoidable_total_cost": severity.mean(axis=1),
        "stage_eligible": np.ones((rows, 3), bool),
        "snapshot_digest_before": digest,
        "snapshot_digest_after": digest,
        "environment_fingerprint_before": digest,
        "environment_fingerprint_after": digest,
        "actor_hidden_digest_before": digest,
        "actor_hidden_digest_after": digest,
    }
    events = dataset / "shadow_events.npz"
    np.savez_compressed(events, **arrays)
    split = {
        "train": list(range(0, 10)),
        "calibration": list(range(10, 20)),
        "threshold_selection": list(range(20, 30)),
        "test": list(range(30, 40)),
    }
    split_path = dataset / "layout_splits.json"
    split_path.write_text(json.dumps({"partitions": split}), encoding="utf-8")
    support_path = dataset / "support.json"
    support_path.write_text(json.dumps({"heads": {
        "initial_joint_guard": {"supported": True, "reason": None},
        "obstacle_ttc": {"supported": True, "reason": None},
        "final_joint_guard": {
            "supported": False,
            "reason": "r2r_final_head_frozen_unsupported",
        },
    }}), encoding="utf-8")
    episodes_path = dataset / "episodes.json"
    episodes_path.write_text(json.dumps({"episodes": []}), encoding="utf-8")
    artifacts = {}
    for path in (events, split_path, support_path, episodes_path):
        artifacts[path.name] = {
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
    manifest = {
        "schema_version": R3_DATASET_SCHEMA,
        "feature_contract": R3_FEATURE_CONTRACT,
        "action_conditioned_ttc_version": ACTION_TTC_VERSION,
        "invariance": {
            "snapshot_digest_equal": True,
            "environment_fingerprint_equal": True,
            "actor_hidden_digest_equal": True,
        },
        "artifacts": artifacts,
    }
    (dataset / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    output = tmp_path / "critic"
    report = train_and_certify_r3_twin_critic(
        dataset_dir=dataset,
        output_dir=output,
        device="cpu",
        epochs=1,
        batch_size=128,
        bootstrap_replicates=20,
        seed=7,
    )
    assert report["schema_version"] == "IAMAPPOR3TwinCriticCertificationV3"
    assert (output / "prediction_audit.npz").is_file()
    assert report["layout_partitions"] == {
        "train": 10, "calibration": 10,
        "threshold_selection": 10, "test": 10,
    }
