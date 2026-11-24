from copy import deepcopy

import numpy as np
import pytest
import torch

from src.algorithms.intervention_advantage import (
    StructuredFeatureSpecV1,
    TwinInterventionAdvantageCriticV1,
)
from src.algorithms.ppo.networks import RecurrentActor
from src.algorithms.ppo.rollout_buffer import RecurrentRolloutBuffer
from src.algorithms.safety_alignment import (
    counterfactual_alignment_objective,
    select_alignment_lambda,
)
from src.diagnostics.csa_paper_report import compare_csa_formal


def test_zero_initialized_adapter_preserves_action_and_log_probability():
    torch.manual_seed(7)
    base = RecurrentActor(12, 128, initial_action_log_std=-1.2)
    adapted = RecurrentActor(
        12, 128, initial_action_log_std=-1.2,
        safety_adapter_rank=16, safety_adapter_scale=0.15,
    )
    loaded = adapted.load_state_dict(base.state_dict(), strict=False)
    assert loaded.unexpected_keys == []
    assert loaded.missing_keys and all(
        "safety_adapter." in key for key in loaded.missing_keys
    )
    obs = torch.randn(3, 4, 12)
    hidden = torch.randn(1, 4, 128)
    raw = torch.randn(3, 4, 2)
    left = base.evaluate_sequence(obs, hidden, raw)
    right = adapted.evaluate_sequence(obs, hidden, raw)
    torch.testing.assert_close(left.log_prob, right.log_prob, rtol=0, atol=0)
    torch.testing.assert_close(
        base.mean_action_sequence(obs, hidden),
        adapted.mean_action_sequence(obs, hidden), rtol=0, atol=0,
    )


def test_adapter_freeze_budget_and_gradient_contract():
    torch.manual_seed(11)
    actor = RecurrentActor(12, 128, safety_adapter_rank=16)
    actor.freeze_backbone_for_adapter()
    ratio = actor.adapter_parameter_count() / (
        sum(parameter.numel() for parameter in actor.parameters())
        - actor.adapter_parameter_count()
    )
    assert ratio < 0.03
    assert all(
        parameter.requires_grad == name.startswith("safety_adapter.")
        for name, parameter in actor.named_parameters()
    )
    loss = actor.mean_action_sequence(
        torch.randn(2, 3, 12), torch.zeros(1, 3, 128)
    ).square().mean()
    loss.backward()
    adapter_gradients = [
        parameter.grad for name, parameter in actor.named_parameters()
        if name.startswith("safety_adapter.")
    ]
    assert any(
        gradient is not None and torch.isfinite(gradient).all()
        and bool(torch.count_nonzero(gradient))
        for gradient in adapter_gradients
    )
    assert all(
        parameter.grad is None for name, parameter in actor.named_parameters()
        if not name.startswith("safety_adapter.")
    )


def _features(action):
    batch, agents = action.shape[:2]
    return (
        torch.randn(batch, agents * 2), action,
        torch.randn(batch, 8), torch.randn(batch),
        torch.randn(batch, agents, 3), torch.ones(batch, agents),
        torch.zeros(batch, agents), torch.zeros(batch, agents),
        torch.ones(batch, agents),
    )


def test_reference_branch_is_detached_and_ineligible_rows_have_zero_gradient():
    torch.manual_seed(5)
    critic = TwinInterventionAdvantageCriticV1(
        StructuredFeatureSpecV1(agent_count=2, global_state_dim=4, safety_context_dim=8)
    )
    critic.freeze_for_actor()
    raw_action = torch.randn(3, 2, 2, requires_grad=True)
    reference_actions = [
        torch.randn(3, 2, 2, requires_grad=True) for _ in range(6)
    ]
    objective, _ = counterfactual_alignment_objective(
        critic,
        raw_features=_features(raw_action),
        reference_features=[_features(action) for action in reference_actions],
        reference_valid=torch.ones(3, 6, dtype=torch.bool),
        actor_eligible=torch.ones(3, dtype=torch.bool),
        deadzone=torch.full((2,), -1.0),
        support_z_limit=100.0,
    )
    objective.backward()
    assert raw_action.grad is not None and torch.isfinite(raw_action.grad).all()
    assert all(action.grad is None for action in reference_actions)
    assert all(parameter.grad is None for parameter in critic.parameters())

    blocked = torch.randn(2, 2, 2, requires_grad=True)
    zero, diagnostics = counterfactual_alignment_objective(
        critic,
        raw_features=_features(blocked),
        reference_features=[_features(torch.zeros_like(blocked)) for _ in range(6)],
        reference_valid=torch.ones(2, 6, dtype=torch.bool),
        actor_eligible=torch.zeros(2, dtype=torch.bool),
        deadzone=torch.zeros(2), support_z_limit=100.0,
    )
    zero.backward()
    assert float(zero) == 0.0
    assert diagnostics["alignment_valid_fraction"] == 0.0
    assert blocked.grad is not None and bool(torch.all(blocked.grad == 0))


def test_rollout_buffer_rejects_invalid_reference_bank_shape():
    buffer = RecurrentRolloutBuffer(3, team_gae=True)
    common = dict(
        local_obs=np.zeros((3, 4)), global_state=np.zeros(6),
        hidden=np.zeros((1, 3, 8)), actions=np.zeros((3, 2)),
        raw_actions=np.zeros((3, 2)), log_probs=np.zeros(3),
        rewards=np.zeros(3), dones=np.zeros(3, dtype=bool), values=np.zeros(3),
    )
    with pytest.raises(ValueError, match="reference_actions"):
        buffer.add(**common, reference_actions=np.zeros((5, 3, 2)))


def _formal_rows(method, avoidable):
    return [
        {
            "train_seed": train_seed, "seed": layout,
            "task_config_fingerprint": "clean-8s2d", "method": method,
            "avoidable_intervention_rate": avoidable,
            "goal_progress_ratio": 0.8, "safe_success": True,
            "collision_count": 0, "truncated": False,
            "inter_agent_collision_count": 0, "boundary_collision_count": 0,
        }
        for train_seed in (23, 41, 67) for layout in range(120000, 120004)
    ]


def test_formal_report_uses_hierarchical_pairs_and_applies_all_gates():
    report = compare_csa_formal(
        _formal_rows("standard_mappo", 0.4),
        _formal_rows("csa_mappo", 0.2),
        expected_layout_count=4, bootstrap_samples=200,
    )
    assert report["all_gates_pass"]
    assert report["avoidable_relative_reduction"] == pytest.approx(0.5)
    assert report["estimates"]["avoidable_intervention_rate"][
        "per_seed_differences"
    ] == pytest.approx([-0.2, -0.2, -0.2])


@pytest.mark.parametrize("problem", ["duplicate", "missing", "fingerprint", "layout"])
def test_formal_report_fails_closed_on_pairing_or_provenance_error(problem):
    baseline = _formal_rows("standard_mappo", 0.4)
    candidate = _formal_rows("csa_mappo", 0.2)
    if problem == "duplicate":
        candidate.append(deepcopy(candidate[0]))
    elif problem == "missing":
        candidate.pop()
    elif problem == "fingerprint":
        candidate[0]["task_config_fingerprint"] = "wrong"
    else:
        candidate[0]["seed"] = 999
    with pytest.raises(ValueError):
        compare_csa_formal(
            baseline, candidate, expected_layout_count=4,
            bootstrap_samples=100,
        )


def test_lambda_selection_uses_smallest_candidate_in_registered_band():
    assert select_alignment_lambda(2.0) == 0.03
    with pytest.raises(ValueError):
        select_alignment_lambda(0.01)
