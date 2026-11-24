from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from src.algorithms.intervention_critic import (
    TwinInterventionCriticV1,
    TwinScalarInterventionCriticV1,
    fit_stage_temperatures,
    ia_actor_risk_penalty,
    intervention_critic_loss,
    save_twin_critic_artifact,
    save_twin_scalar_critic_artifact,
)
from src.algorithms.intervention_r3 import (
    R3_FEATURE_CONTRACT,
    R4_1_ACTOR_GRADIENT_CONTRACT,
    R4_2_ACTOR_GRADIENT_CONTRACT,
    R4_2_1_ACTOR_GRADIENT_CONTRACT,
    R4_3_1_ACTOR_GRADIENT_CONTRACT,
    R4_ACTOR_GRADIENT_CONTRACT,
)
from src.algorithms.factory import create_trainer
from src.safety.contracts import CausalSafetySnapshotV1, InterventionLevel
from src.safety.counterfactual import (
    ShadowSafetyResultV2,
    ShadowSafetyResultV1,
    evaluate_avoidable_intervention,
    evaluate_avoidable_intervention_v2,
    next_goal_progress_velocity,
    registered_shadow_actions,
)
from src.utils.io_utils import load_yaml


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _snapshot() -> CausalSafetySnapshotV1:
    return CausalSafetySnapshotV1(
        step=0, team_observation=((0.0, 0.0), (1.0, 1.0)),
        requested_actions=((1.0, 0.0), (1.0, 0.0)), safety_state=(1.0,),
        obstacle_tracks=((), ()), rng_state_digest=_digest("rng"),
        recovery_state_digest=_digest("recovery"),
    )


def test_registered_shadow_panel_and_immutable_evaluation():
    raw = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    goal_actions = np.asarray([[0.9, 0.0], [0.9, 0.0]], dtype=np.float32)
    shadows = registered_shadow_actions(raw, goal_actions, sobol_seed=7)
    assert len(shadows) == 31  # raw + HOLD + brake + four goal scales + 24 Sobol

    def stack(snapshot, action):
        modified = bool(np.allclose(action, raw))
        level = InterventionLevel.MATERIAL if modified else InterventionLevel.NONE
        delta = 0.10 if modified else 0.0
        return ShadowSafetyResultV1(
            requested_actions=tuple(map(tuple, action)),
            stage_delta_v_mps=(delta, 0.0, 0.0),
            stage_levels=(level, InterventionLevel.NONE, InterventionLevel.NONE),
            executed_actions=tuple(map(tuple, action)),
        )

    snapshot = _snapshot()
    before = snapshot.digest
    label = evaluate_avoidable_intervention(
        snapshot, shadows, stack, goal_directions=raw,
        reached_mask=np.asarray([False, False]),
        acceleration_scale=1.0, dt=0.5,
    )
    assert snapshot.digest == before
    assert label.avoidable_cost > 0.0
    assert label.valid_shadow_count > 0


def test_v2_shadow_cost_uses_team_mean_and_conserves_stage_attribution():
    raw = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    safe = np.asarray([[0.8, 0.0], [0.8, 0.0]], dtype=np.float32)

    def stack(snapshot, action):
        del snapshot
        modified = bool(np.allclose(action, raw))
        return ShadowSafetyResultV2(
            requested_actions=tuple(map(tuple, action)),
            per_agent_stage_delta_v_mps=(
                (0.10 if modified else 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            ),
            per_agent_stage_levels=(
                (
                    InterventionLevel.MATERIAL if modified
                    else InterventionLevel.NONE,
                    InterventionLevel.NONE,
                    InterventionLevel.NONE,
                ),
                (
                    InterventionLevel.NONE,
                    InterventionLevel.NONE,
                    InterventionLevel.NONE,
                ),
            ),
            executed_actions=tuple(map(tuple, action)),
        )

    raw_result = stack(_snapshot(), raw)
    assert np.isclose(raw_result.stage_costs[0], 0.7)
    label = evaluate_avoidable_intervention_v2(
        _snapshot(), (raw, safe), stack,
        goal_directions=raw,
        current_velocities=np.zeros_like(raw),
        reached_mask=np.asarray([False, False]),
        acceleration_scale=1.0,
        dt=0.5,
        maximum_speed=2.0,
        maximum_progress_loss_mps=0.10,
    )
    assert label.best_shadow_index == 1
    assert label.avoidable_cost > 0.0
    assert np.isclose(np.mean(label.stage_avoidable_costs), label.avoidable_cost)


def test_v2_progress_validity_uses_clipped_next_velocity_not_action_delta():
    progress = next_goal_progress_velocity(
        actions=np.asarray([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32),
        current_velocities=np.asarray(
            [[1.8, 0.0], [0.4, 0.0]], dtype=np.float32
        ),
        goal_directions=np.asarray(
            [[1.0, 0.0], [1.0, 0.0]], dtype=np.float32
        ),
        acceleration_scale=1.0,
        dt=0.5,
        maximum_speed=2.0,
    )
    np.testing.assert_allclose(progress, [2.0, -0.1], atol=1e-7)


def test_critic_loss_and_frozen_actor_gradient_path():
    critic = TwinInterventionCriticV1(input_dim=8)
    features = torch.randn(5, 8)
    logits, severity = critic.critics[0](features)
    targets = torch.tensor([[0, 1, 0]] * 5, dtype=torch.float32)
    delta_v = torch.tensor([[0.0, 0.1, 0.0]] * 5)
    loss = intervention_critic_loss(
        logits, severity, targets, delta_v, positive_weights=(1.0, 2.0, 1.0)
    )
    assert torch.isfinite(loss.total)

    critic.freeze_for_actor()
    action_features = torch.randn(5, 8, requires_grad=True)
    penalty = ia_actor_risk_penalty(
        critic, action_features, intervention_lambda=0.01
    )
    penalty.backward()
    assert action_features.grad is not None
    assert all(parameter.grad is None for parameter in critic.parameters())


def test_twin_risk_applies_weighted_bce_prior_offsets_and_head_mask():
    critic = TwinInterventionCriticV1(input_dim=4)
    critic.head_enabled.copy_(torch.tensor([True, True, False]))
    critic.prior_logit_offsets.copy_(torch.tensor([
        [1.0, -1.0, 0.0], [1.0, -1.0, 0.0]
    ]))
    features = torch.zeros(2, 4)
    risk_with_offsets = critic.conservative_risk(features)
    stage_risk = critic.conservative_stage_risk(features)
    critic.prior_logit_offsets.zero_()
    risk_without_offsets = critic.conservative_risk(features)
    assert risk_with_offsets.shape == (2,)
    assert stage_risk.shape == (2, 3)
    torch.testing.assert_close(stage_risk[:, 2], torch.zeros(2))
    assert not torch.allclose(risk_with_offsets, risk_without_offsets)


def test_stage_audit_api_does_not_change_registered_twin_total_risk():
    critic = TwinInterventionCriticV1(input_dim=4)
    critic.head_enabled.copy_(torch.tensor([True, True, False]))
    features = torch.randn(7, 4)
    normalized = critic.normalize_features(features)
    original = torch.stack([
        module.risk(
            normalized,
            critic.temperatures[index],
            critic.prior_logit_offsets[index],
            critic.head_enabled,
        )
        for index, module in enumerate(critic.critics)
    ]).max(dim=0).values
    torch.testing.assert_close(critic.conservative_risk(features), original)


def test_twin_actor_query_uses_artifact_feature_normalization():
    critic = TwinInterventionCriticV1(input_dim=4)
    critic.feature_mean.copy_(torch.tensor([10.0, 20.0, 30.0, 40.0]))
    critic.feature_std.copy_(torch.tensor([2.0, 4.0, 5.0, 10.0]))
    normalized = critic.normalize_features(torch.tensor([
        [12.0, 24.0, 25.0, 50.0]
    ]))
    torch.testing.assert_close(
        normalized, torch.tensor([[1.0, 1.0, -1.0, 1.0]])
    )


def test_temperature_calibration_returns_three_positive_values():
    logits = torch.tensor([
        [-2.0, 1.0, 0.2], [2.0, -1.0, -0.2],
        [-1.0, 2.0, 0.5], [1.0, -2.0, -0.5],
    ])
    targets = torch.tensor([
        [0.0, 1.0, 1.0], [1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0], [1.0, 0.0, 0.0],
    ])
    temperatures = fit_stage_temperatures(logits, targets, maximum_steps=10)
    assert temperatures.shape == (3,)
    assert torch.all(temperatures > 0.0)


def test_certified_critic_artifact_loads_frozen_for_ia_actor(tmp_path):
    config = load_yaml("configs/experiments/ia_mappo_v1_smoke_cpu.yaml")
    config["output"] = {
        "output_dir": str(tmp_path / "standard"),
        "checkpoint_dir": str(tmp_path / "standard" / "checkpoints"),
        "reject_nonempty": True,
    }
    standard = create_trainer(config, project_root=Path.cwd())
    input_dim = standard.global_state_dim + 2 * standard.num_agents + 8 + 1 + 7 * standard.num_agents
    artifact = tmp_path / "certified_critic.pt"
    critic = TwinInterventionCriticV1(input_dim)
    save_twin_critic_artifact(
        artifact, critic, input_dim=input_dim, intervention_lambda=0.01,
        risk_to_ppo_gradient_ratio=0.04, dataset_manifest_sha256="0" * 64,
        lambda_calibration_sha256="2" * 64,
        source_certification_sha256="3" * 64,
        feature_contract=R3_FEATURE_CONTRACT,
        actor_gradient_contract=R4_ACTOR_GRADIENT_CONTRACT,
    )

    config["train"]["intervention_aware"] = {
        "mode": "stage_counterfactual_v1",
        "intervention_lambda": 0.01,
        "critic": {"artifact": str(artifact)},
    }
    config["output"] = {
        "output_dir": str(tmp_path / "ia"),
        "checkpoint_dir": str(tmp_path / "ia" / "checkpoints"),
        "reject_nonempty": True,
    }
    ia_trainer = create_trainer(config, project_root=Path.cwd())
    assert ia_trainer.intervention_critic is not None
    assert all(
        not parameter.requires_grad
        for parameter in ia_trainer.intervention_critic.parameters()
    )


def test_scalar_material_baseline_loads_its_own_artifact_schema(tmp_path):
    config = load_yaml("configs/experiments/ia_mappo_v1_smoke_cpu.yaml")
    config["output"] = {
        "output_dir": str(tmp_path / "probe"),
        "checkpoint_dir": str(tmp_path / "probe" / "checkpoints"),
        "reject_nonempty": True,
    }
    probe = create_trainer(config, project_root=Path.cwd())
    input_dim = probe.global_state_dim + 2 * probe.num_agents + 8 + 1 + 7 * probe.num_agents
    artifact = tmp_path / "scalar.pt"
    critic = TwinScalarInterventionCriticV1(input_dim)
    save_twin_scalar_critic_artifact(
        artifact, critic, input_dim=input_dim, intervention_lambda=0.003,
        risk_to_ppo_gradient_ratio=0.05, dataset_manifest_sha256="1" * 64,
        lambda_calibration_sha256="2" * 64,
        source_certification_sha256="3" * 64,
        feature_contract=R3_FEATURE_CONTRACT,
        actor_gradient_contract=R4_ACTOR_GRADIENT_CONTRACT,
    )
    config["train"]["intervention_aware"] = {
        "mode": "scalar_material_v1",
        "intervention_lambda": 0.003,
        "critic": {"artifact": str(artifact)},
    }
    config["output"] = {
        "output_dir": str(tmp_path / "scalar"),
        "checkpoint_dir": str(tmp_path / "scalar" / "checkpoints"),
        "reject_nonempty": True,
    }
    trainer = create_trainer(config, project_root=Path.cwd())
    assert isinstance(trainer.intervention_critic, TwinScalarInterventionCriticV1)


def test_r4_1_progress_preserving_actor_path_smoke(tmp_path):
    config = load_yaml("configs/experiments/ia_mappo_v1_smoke_cpu.yaml")
    config["output"] = {
        "output_dir": str(tmp_path / "probe"),
        "checkpoint_dir": str(tmp_path / "probe" / "checkpoints"),
        "reject_nonempty": True,
    }
    probe = create_trainer(config, project_root=Path.cwd())
    input_dim = (
        probe.global_state_dim + 2 * probe.num_agents + 8 + 1
        + 7 * probe.num_agents
    )
    artifact = tmp_path / "stage_v3.pt"
    critic = TwinInterventionCriticV1(input_dim)
    save_twin_critic_artifact(
        artifact, critic, input_dim=input_dim, intervention_lambda=0.003,
        risk_to_ppo_gradient_ratio=0.04,
        dataset_manifest_sha256="1" * 64,
        lambda_calibration_sha256="2" * 64,
        source_certification_sha256="3" * 64,
        feature_contract=R3_FEATURE_CONTRACT,
        actor_gradient_contract=R4_1_ACTOR_GRADIENT_CONTRACT,
        schema_version="TwinInterventionCriticArtifactV3",
    )
    config["train"]["intervention_aware"] = {
        "mode": "stage_counterfactual_v2",
        "intervention_lambda": 0.003,
        "action_query": "rollout_straight_through",
        "support_z_limit": 6.0,
        "support_penalty_scale": 0.05,
        "critic": {"artifact": str(artifact)},
    }
    config["output"] = {
        "output_dir": str(tmp_path / "ia_v2"),
        "checkpoint_dir": str(tmp_path / "ia_v2" / "checkpoints"),
        "reject_nonempty": True,
    }
    trainer = create_trainer(config, project_root=Path.cwd())
    buffer, _ = trainer.collect_rollout()
    metrics = trainer.update(buffer)
    assert metrics["intervention_aware_mode"] == "stage_counterfactual_v2"
    assert np.isfinite(metrics["intervention_risk_penalty"])
    assert np.isfinite(metrics["intervention_risk_to_ppo_gradient_ratio_p95"])
    assert np.isfinite(metrics["intervention_risk_ppo_gradient_cosine_p05"])
    assert np.isfinite(metrics["intervention_action_ood_rate_max"])


def test_r4_2_warmup_and_parameter_constrained_actor_path_smoke(tmp_path):
    config = load_yaml("configs/experiments/ia_mappo_v1_smoke_cpu.yaml")
    config["output"] = {
        "output_dir": str(tmp_path / "probe_v4"),
        "checkpoint_dir": str(tmp_path / "probe_v4" / "checkpoints"),
        "reject_nonempty": True,
    }
    probe = create_trainer(config, project_root=Path.cwd())
    input_dim = (
        probe.global_state_dim + 2 * probe.num_agents + 8 + 1
        + 7 * probe.num_agents
    )
    artifact = tmp_path / "stage_v4.pt"
    critic = TwinInterventionCriticV1(input_dim)
    save_twin_critic_artifact(
        artifact, critic, input_dim=input_dim, intervention_lambda=0.003,
        risk_to_ppo_gradient_ratio=0.04,
        dataset_manifest_sha256="1" * 64,
        lambda_calibration_sha256="2" * 64,
        source_certification_sha256="3" * 64,
        feature_contract=R3_FEATURE_CONTRACT,
        actor_gradient_contract=R4_2_ACTOR_GRADIENT_CONTRACT,
        schema_version="TwinInterventionCriticArtifactV4",
    )
    config["train"]["intervention_aware"] = {
        "mode": "stage_counterfactual_v3",
        "intervention_lambda": 0.003,
        "action_query": "rollout_straight_through",
        "activation_update": 1,
        "support_z_limit": 6.0,
        "minimum_support_fraction": 0.90,
        "support_penalty_scale": 0.0,
        "critic": {"artifact": str(artifact)},
    }
    config["output"] = {
        "output_dir": str(tmp_path / "ia_v3"),
        "checkpoint_dir": str(tmp_path / "ia_v3" / "checkpoints"),
        "reject_nonempty": True,
    }
    trainer = create_trainer(config, project_root=Path.cwd())
    trainer.current_update = 2
    buffer, _ = trainer.collect_rollout()
    metrics = trainer.update(buffer)
    assert metrics["intervention_aware_mode"] == "stage_counterfactual_v3"
    assert metrics["intervention_risk_active"] == 1.0
    assert metrics["intervention_progress_constraint_violations"] == 0.0
    assert metrics["intervention_risk_ppo_gradient_cosine_p05"] >= -1e-6
    assert np.isfinite(metrics["intervention_support_fraction_min"])


def test_r4_2_risk_off_warmup_matches_standard_exactly(tmp_path):
    base = load_yaml("configs/experiments/ia_mappo_v1_smoke_cpu.yaml")
    base["output"] = {
        "output_dir": str(tmp_path / "shape_probe"),
        "checkpoint_dir": str(tmp_path / "shape_probe" / "checkpoints"),
        "reject_nonempty": True,
    }
    shape_probe = create_trainer(base, project_root=Path.cwd())
    input_dim = (
        shape_probe.global_state_dim + 2 * shape_probe.num_agents + 8 + 1
        + 7 * shape_probe.num_agents
    )
    artifact = tmp_path / "warmup_stage_v4.pt"
    critic = TwinInterventionCriticV1(input_dim)
    save_twin_critic_artifact(
        artifact, critic, input_dim=input_dim, intervention_lambda=0.003,
        risk_to_ppo_gradient_ratio=0.04,
        dataset_manifest_sha256="1" * 64,
        lambda_calibration_sha256="2" * 64,
        source_certification_sha256="3" * 64,
        feature_contract=R3_FEATURE_CONTRACT,
        actor_gradient_contract=R4_2_ACTOR_GRADIENT_CONTRACT,
        schema_version="TwinInterventionCriticArtifactV4",
    )

    standard_config = load_yaml("configs/experiments/ia_mappo_v1_smoke_cpu.yaml")
    standard_config["output"] = {
        "output_dir": str(tmp_path / "standard_warmup"),
        "checkpoint_dir": str(tmp_path / "standard_warmup" / "checkpoints"),
        "reject_nonempty": True,
    }
    ia_config = load_yaml("configs/experiments/ia_mappo_v1_smoke_cpu.yaml")
    ia_config["train"]["intervention_aware"] = {
        "mode": "stage_counterfactual_v3",
        "intervention_lambda": 0.003,
        "action_query": "rollout_straight_through",
        "activation_update": 1,
        "support_z_limit": 6.0,
        "minimum_support_fraction": 0.90,
        "critic": {"artifact": str(artifact)},
    }
    ia_config["output"] = {
        "output_dir": str(tmp_path / "ia_warmup"),
        "checkpoint_dir": str(tmp_path / "ia_warmup" / "checkpoints"),
        "reject_nonempty": True,
    }
    standard = create_trainer(standard_config, project_root=Path.cwd())
    standard.current_update = 1
    standard_buffer, _ = standard.collect_rollout()
    standard_metrics = standard.update(standard_buffer)

    ia = create_trainer(ia_config, project_root=Path.cwd())
    ia.current_update = 1
    ia_buffer, _ = ia.collect_rollout()
    ia_metrics = ia.update(ia_buffer)
    np.testing.assert_array_equal(standard_buffer.raw_actions, ia_buffer.raw_actions)
    np.testing.assert_array_equal(standard_buffer.rewards, ia_buffer.rewards)
    for name, value in standard.actor.state_dict().items():
        torch.testing.assert_close(value, ia.actor.state_dict()[name], rtol=0, atol=0)
    for name, value in standard.critic.state_dict().items():
        torch.testing.assert_close(value, ia.critic.state_dict()[name], rtol=0, atol=0)
    assert ia_metrics["intervention_risk_active"] == 0.0
    assert ia_metrics["intervention_risk_penalty"] == 0.0
    assert standard_metrics["policy_loss"] == ia_metrics["policy_loss"]


def test_r4_2_1_v5_artifact_freezes_u100_activation(tmp_path):
    config = load_yaml("configs/experiments/ia_mappo_v1_smoke_cpu.yaml")
    config["output"] = {
        "output_dir": str(tmp_path / "probe_v5"),
        "checkpoint_dir": str(tmp_path / "probe_v5" / "checkpoints"),
        "reject_nonempty": True,
    }
    probe = create_trainer(config, project_root=Path.cwd())
    input_dim = (
        probe.global_state_dim + 2 * probe.num_agents + 8 + 1
        + 7 * probe.num_agents
    )
    artifact = tmp_path / "stage_v5.pt"
    critic = TwinInterventionCriticV1(input_dim)
    save_twin_critic_artifact(
        artifact, critic, input_dim=input_dim, intervention_lambda=0.01,
        risk_to_ppo_gradient_ratio=0.02,
        dataset_manifest_sha256="1" * 64,
        lambda_calibration_sha256="2" * 64,
        source_certification_sha256="3" * 64,
        feature_contract=R3_FEATURE_CONTRACT,
        actor_gradient_contract=R4_2_1_ACTOR_GRADIENT_CONTRACT,
        schema_version="TwinInterventionCriticArtifactV5",
    )
    config["train"]["intervention_aware"] = {
        "mode": "stage_counterfactual_v4",
        "intervention_lambda": 0.01,
        "action_query": "rollout_straight_through",
        "activation_update": 100,
        "support_z_limit": 6.0,
        "minimum_support_fraction": 0.90,
        "critic": {"artifact": str(artifact)},
    }
    config["output"] = {
        "output_dir": str(tmp_path / "ia_v4"),
        "checkpoint_dir": str(tmp_path / "ia_v4" / "checkpoints"),
        "reject_nonempty": True,
    }
    trainer = create_trainer(config, project_root=Path.cwd())
    assert trainer.intervention_aware_mode == "stage_counterfactual_v4"
    assert trainer.intervention_activation_update == 100
    trainer.current_update = 100
    buffer, _ = trainer.collect_rollout()
    metrics = trainer.update(buffer)
    assert metrics["intervention_risk_active"] == 0.0
    assert metrics["intervention_risk_penalty"] == 0.0


def test_r4_3_1_v6_masks_ineligible_team_steps_without_using_labels(tmp_path):
    config = load_yaml("configs/experiments/ia_mappo_v1_smoke_cpu.yaml")
    config["output"] = {
        "output_dir": str(tmp_path / "probe_v6"),
        "checkpoint_dir": str(tmp_path / "probe_v6" / "checkpoints"),
        "reject_nonempty": True,
    }
    probe = create_trainer(config, project_root=Path.cwd())
    input_dim = (
        probe.global_state_dim + 2 * probe.num_agents + 8 + 1
        + 7 * probe.num_agents
    )
    artifact = tmp_path / "stage_v6.pt"
    critic = TwinInterventionCriticV1(input_dim)
    save_twin_critic_artifact(
        artifact, critic, input_dim=input_dim, intervention_lambda=0.01,
        risk_to_ppo_gradient_ratio=0.02,
        dataset_manifest_sha256="1" * 64,
        lambda_calibration_sha256="2" * 64,
        source_certification_sha256="3" * 64,
        feature_contract=R3_FEATURE_CONTRACT,
        actor_gradient_contract=R4_3_1_ACTOR_GRADIENT_CONTRACT,
        schema_version="TwinInterventionCriticArtifactV6",
    )
    config["train"]["intervention_aware"] = {
        "mode": "stage_counterfactual_v5",
        "intervention_lambda": 0.01,
        "action_query": "rollout_straight_through",
        "activation_update": 100,
        "support_z_limit": 100.0,
        "minimum_support_fraction": 0.90,
        "critic": {"artifact": str(artifact)},
    }
    config["output"] = {
        "output_dir": str(tmp_path / "ia_v5"),
        "checkpoint_dir": str(tmp_path / "ia_v5" / "checkpoints"),
        "reject_nonempty": True,
    }
    trainer = create_trainer(config, project_root=Path.cwd())
    trainer.current_update = 101
    buffer, _ = trainer.collect_rollout()
    assert buffer.actor_risk_eligible.shape == (len(buffer.rewards),)
    assert buffer.actor_risk_eligible.dtype == np.bool_
    indices = np.arange(len(buffer.rewards), dtype=np.int64)
    actor = trainer.actor.actor_modules()[0]
    buffer.actor_risk_eligible[:] = True
    eligible_risk = trainer._intervention_raw_risk(buffer, indices, actor)
    buffer.actor_risk_eligible[:] = False
    excluded_risk = trainer._intervention_raw_risk(buffer, indices, actor)
    assert float(eligible_risk.detach()) > 0.0
    assert float(excluded_risk.detach()) == 0.0
    assert trainer._latest_intervention_eligibility_fraction == 0.0
