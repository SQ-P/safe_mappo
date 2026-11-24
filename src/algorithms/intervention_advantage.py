"""Probability-only intervention-advantage critic used by R4.4.

The module is intentionally independent from the historical severity critics.
Only Initial Guard and TTC are supported; Final Guard remains unsupported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


SUPPORTED_STAGES = ("initial_joint_guard", "obstacle_ttc")
REFERENCE_NAMES = ("hold", "brake", "goal_025", "goal_050", "goal_075", "goal_100")
REFERENCE_BANK_CONTRACT = "hold_brake_goal_scales_025_050_075_100_v1"
FEATURE_CONTRACT = "r4_4_structured_agent_mean_max_action_ttc_v1"


@dataclass(frozen=True)
class StructuredFeatureSpecV1:
    agent_count: int
    global_state_dim: int
    safety_context_dim: int
    per_agent_margin_dim: int = 3

    @property
    def local_state_dim(self) -> int:
        if self.global_state_dim % self.agent_count:
            raise ValueError("global state cannot be evenly partitioned by UAV")
        return self.global_state_dim // self.agent_count

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, np.ndarray]) -> "StructuredFeatureSpecV1":
        action = np.asarray(arrays["raw_joint_mean_action"])
        state = np.asarray(arrays["global_state_normalized"])
        context = np.asarray(arrays["causal_safety_context"])
        margins = np.asarray(arrays["per_agent_margins"])
        if action.ndim != 3 or action.shape[-1] != 2:
            raise ValueError("R4.4 joint action must have shape [row,UAV,2]")
        if state.ndim != 2 or context.ndim != 2 or margins.ndim != 3:
            raise ValueError("R4.4 structured feature ranks are invalid")
        spec = cls(action.shape[1], state.shape[1], context.shape[1], margins.shape[2])
        _ = spec.local_state_dim
        return spec


class InterventionAdvantageCriticV1(nn.Module):
    """Shared UAV encoder followed by mean/max team pooling and two heads."""

    def __init__(self, spec: StructuredFeatureSpecV1) -> None:
        super().__init__()
        self.spec = spec
        local_input = spec.local_state_dim + 2 + spec.per_agent_margin_dim + 4
        self.agent_encoder = nn.Sequential(
            nn.Linear(local_input, 256), nn.SiLU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.SiLU(), nn.LayerNorm(256),
        )
        team_input = 512 + spec.safety_context_dim + 1
        self.team_encoder = nn.Sequential(
            nn.Linear(team_input, 256), nn.SiLU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.SiLU(), nn.LayerNorm(256),
        )
        self.heads = nn.ModuleList(nn.Linear(256, 1) for _ in SUPPORTED_STAGES)

    def forward(
        self,
        global_state: torch.Tensor,
        action: torch.Tensor,
        safety_context: torch.Tensor,
        current_margin: torch.Tensor,
        per_agent_margins: torch.Tensor,
        action_ttc: torch.Tensor,
        ttc_warning: torch.Tensor,
        ttc_emergency: torch.Tensor,
        predicted_clearance: torch.Tensor,
    ) -> torch.Tensor:
        batch = global_state.shape[0]
        local_state = global_state.reshape(
            batch, self.spec.agent_count, self.spec.local_state_dim
        )
        agent_input = torch.cat((
            local_state, action, per_agent_margins,
            action_ttc.unsqueeze(-1), ttc_warning.unsqueeze(-1),
            ttc_emergency.unsqueeze(-1), predicted_clearance.unsqueeze(-1),
        ), dim=-1)
        encoded = self.agent_encoder(agent_input)
        pooled = torch.cat((encoded.mean(dim=1), encoded.max(dim=1).values), dim=-1)
        team = self.team_encoder(torch.cat((
            pooled, safety_context, current_margin.reshape(batch, 1),
        ), dim=-1))
        return torch.cat([head(team) for head in self.heads], dim=-1)


class TwinInterventionAdvantageCriticV1(nn.Module):
    def __init__(self, spec: StructuredFeatureSpecV1) -> None:
        super().__init__()
        self.spec = spec
        self.critics = nn.ModuleList([
            InterventionAdvantageCriticV1(spec),
            InterventionAdvantageCriticV1(spec),
        ])
        self.register_buffer("temperatures", torch.ones(2, len(SUPPORTED_STAGES)))
        self.register_buffer("action_mean", torch.zeros(spec.agent_count, 2))
        self.register_buffer("action_std", torch.ones(spec.agent_count, 2))

    def action_in_support(self, action: torch.Tensor, z_limit: float = 6.0) -> torch.Tensor:
        normalized = (action - self.action_mean) / self.action_std.clamp_min(1e-6)
        return normalized.abs().amax(dim=(-2, -1)) <= float(z_limit)

    def logits(self, *features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return tuple(critic(*features) for critic in self.critics)  # type: ignore[return-value]

    def member_probabilities(self, *features: torch.Tensor) -> torch.Tensor:
        logits = self.logits(*features)
        return torch.stack([
            torch.sigmoid(value / self.temperatures[index].clamp_min(0.05))
            for index, value in enumerate(logits)
        ], dim=0)

    def conservative_stage_probability(self, *features: torch.Tensor) -> torch.Tensor:
        return self.member_probabilities(*features).max(dim=0).values

    def conservative_total_probability(self, *features: torch.Tensor) -> torch.Tensor:
        return self.conservative_stage_probability(*features).mean(dim=-1)

    def freeze_for_actor(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)


def weighted_probability_pair_loss(
    raw_logits: torch.Tensor,
    labels: torch.Tensor,
    eligible: torch.Tensor,
    *,
    reference_logits: torch.Tensor | None = None,
    pair_mask: torch.Tensor | None = None,
    positive_weights: torch.Tensor | None = None,
    alpha_pair: float = 0.0,
    pair_temperature: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if raw_logits.shape != labels.shape or eligible.shape != labels.shape:
        raise ValueError("R4.4 raw logits, labels and eligibility must align")
    weights = torch.ones(raw_logits.shape[-1], device=raw_logits.device)
    if positive_weights is not None:
        weights = positive_weights.to(raw_logits.device)
    element = F.binary_cross_entropy_with_logits(
        raw_logits, labels, reduction="none", pos_weight=weights
    )
    classification = (element * eligible).sum() / eligible.sum().clamp_min(1.0)
    pairwise = raw_logits.new_zeros(())
    if alpha_pair > 0.0:
        if reference_logits is None or pair_mask is None:
            raise ValueError("pairwise loss requires reference logits and mask")
        if reference_logits.shape != raw_logits.shape or pair_mask.shape != raw_logits.shape:
            raise ValueError("R4.4 pair tensors must align with raw logits")
        difference = (raw_logits - reference_logits) / float(pair_temperature)
        pair_element = F.softplus(-difference)
        pairwise = (pair_element * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)
    total = classification + float(alpha_pair) * pairwise
    return total, {"classification": classification.detach(), "pairwise": pairwise.detach()}


def advantage_risk(
    raw_stage_probability: torch.Tensor,
    reference_stage_probability: torch.Tensor,
    reference_valid: torch.Tensor,
    *,
    deadzone: torch.Tensor,
) -> torch.Tensor:
    """Actor-facing relative risk; reference selection is fully detached."""

    if reference_stage_probability.ndim != 3:
        raise ValueError("reference probabilities must be [batch,reference,stage]")
    if reference_valid.shape != reference_stage_probability.shape[:2]:
        raise ValueError("reference validity mask has the wrong shape")
    masked = reference_stage_probability.detach().masked_fill(
        ~reference_valid.unsqueeze(-1), float("inf")
    )
    baseline = masked.min(dim=1).values
    fallback = raw_stage_probability.detach()
    baseline = torch.where(torch.isfinite(baseline), baseline, fallback)
    return F.relu(raw_stage_probability - baseline - deadzone).mean(dim=-1)


def pairwise_accuracy(
    raw: np.ndarray, reference: np.ndarray, mask: np.ndarray
) -> float:
    raw = np.asarray(raw, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if raw.shape != reference.shape or raw.shape != mask.shape or not mask.any():
        return 0.0
    return float(np.mean(raw[mask] > reference[mask]))
