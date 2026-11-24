from __future__ import annotations

import numpy as np
import torch

from src.algorithms.intervention_advantage import (
    StructuredFeatureSpecV1,
    TwinInterventionAdvantageCriticV1,
    advantage_risk,
    weighted_probability_pair_loss,
)


def _features(batch: int = 8):
    return (
        torch.randn(batch, 198),
        torch.randn(batch, 3, 2).clamp(-1, 1),
        torch.randn(batch, 8),
        torch.rand(batch),
        torch.rand(batch, 3, 3),
        torch.rand(batch, 3),
        torch.zeros(batch, 3),
        torch.zeros(batch, 3),
        torch.rand(batch, 3),
    )


def test_twin_advantage_critic_shapes_and_independent_initialization():
    spec = StructuredFeatureSpecV1(3, 198, 8)
    model = TwinInterventionAdvantageCriticV1(spec)
    members = model.member_probabilities(*_features())
    assert members.shape == (2, 8, 2)
    assert model.conservative_stage_probability(*_features()).shape == (8, 2)
    left = next(model.critics[0].parameters()).detach()
    right = next(model.critics[1].parameters()).detach()
    assert not torch.equal(left, right)


def test_advantage_reference_branch_is_detached():
    raw = torch.tensor([[0.8, 0.6]], requires_grad=True)
    references = torch.tensor([[[0.2, 0.3], [0.4, 0.1]]], requires_grad=True)
    valid = torch.tensor([[True, True]])
    risk = advantage_risk(raw, references, valid, deadzone=torch.tensor([0.05, 0.05]))
    risk.sum().backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    assert references.grad is None


def test_pair_loss_prefers_raw_risk_above_better_reference():
    labels = torch.ones(4, 2)
    eligible = torch.ones(4, 2)
    mask = torch.ones(4, 2)
    good_raw = torch.full((4, 2), 2.0, requires_grad=True)
    bad_raw = torch.full((4, 2), -2.0, requires_grad=True)
    reference = torch.zeros(4, 2)
    good, _ = weighted_probability_pair_loss(
        good_raw, labels, eligible, reference_logits=reference,
        pair_mask=mask, alpha_pair=1.0,
    )
    bad, _ = weighted_probability_pair_loss(
        bad_raw, labels, eligible, reference_logits=reference,
        pair_mask=mask, alpha_pair=1.0,
    )
    assert float(good.detach()) < float(bad.detach())


def test_structured_spec_rejects_unpartitionable_global_state():
    arrays = {
        "raw_joint_mean_action": np.zeros((2, 3, 2), dtype=np.float32),
        "global_state_normalized": np.zeros((2, 197), dtype=np.float32),
        "causal_safety_context": np.zeros((2, 8), dtype=np.float32),
        "per_agent_margins": np.zeros((2, 3, 3), dtype=np.float32),
    }
    try:
        StructuredFeatureSpecV1.from_arrays(arrays)
    except ValueError as exc:
        assert "partitioned" in str(exc)
    else:
        raise AssertionError("invalid global-state partition was accepted")
