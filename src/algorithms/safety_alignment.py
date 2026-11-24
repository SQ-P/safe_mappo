"""Actor-facing objectives for CSA-MAPPO and its registered ablations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from src.algorithms.intervention_advantage import (
    TwinInterventionAdvantageCriticV1,
    advantage_risk,
)
from src.algorithms.intervention_team_data import (
    torch_action_conditioned_obstacle_features,
    torch_smooth_minimum_obstacle_clearance,
)


@dataclass(frozen=True)
class AlignmentDynamics:
    agent_radius: float
    obstacle_safety_margin: float
    invariant_buffer: float
    max_acceleration: float
    max_speed: float
    dt: float
    horizon_seconds: float
    warning_ttc_seconds: float
    emergency_ttc_seconds: float


def select_alignment_lambda(
    unscaled_risk_to_ppo_ratio: float,
    *,
    candidates: tuple[float, ...] = (0.01, 0.03, 0.10, 0.30),
    target: tuple[float, float] = (0.05, 0.10),
) -> float:
    """Select the smallest preregistered coefficient entering the ratio band."""

    ratio = float(unscaled_risk_to_ppo_ratio)
    if not torch.isfinite(torch.tensor(ratio)) or ratio <= 0.0:
        raise ValueError("unscaled gradient ratio must be finite and positive")
    lower, upper = map(float, target)
    if not 0.0 < lower <= upper:
        raise ValueError("gradient-ratio target is invalid")
    for candidate in candidates:
        scaled = float(candidate) * ratio
        if lower <= scaled <= upper:
            return float(candidate)
    raise ValueError("no preregistered lambda enters the gradient-ratio band")


def structured_critic_features(
    *,
    global_state: torch.Tensor,
    action: torch.Tensor,
    safety_context: torch.Tensor,
    current_margin: torch.Tensor,
    per_agent_margins: torch.Tensor,
    current_velocity: torch.Tensor,
    obstacle_tracks: torch.Tensor,
    dynamics: AlignmentDynamics,
) -> tuple[torch.Tensor, ...]:
    """Build the differentiable R4.4 structured feature tuple."""

    ttc, warning, emergency, clearance = torch_action_conditioned_obstacle_features(
        current_velocity=current_velocity,
        raw_action=action,
        obstacle_tracks=obstacle_tracks,
        agent_radius=dynamics.agent_radius,
        obstacle_safety_margin=dynamics.obstacle_safety_margin,
        invariant_buffer=dynamics.invariant_buffer,
        max_acceleration=dynamics.max_acceleration,
        max_speed=dynamics.max_speed,
        dt=dynamics.dt,
        horizon_seconds=dynamics.horizon_seconds,
        warning_ttc_seconds=dynamics.warning_ttc_seconds,
        emergency_ttc_seconds=dynamics.emergency_ttc_seconds,
    )
    smooth_clearance = torch_smooth_minimum_obstacle_clearance(
        current_velocity=current_velocity,
        raw_action=action,
        obstacle_tracks=obstacle_tracks,
        agent_radius=dynamics.agent_radius,
        obstacle_safety_margin=dynamics.obstacle_safety_margin,
        invariant_buffer=dynamics.invariant_buffer,
        max_acceleration=dynamics.max_acceleration,
        max_speed=dynamics.max_speed,
        dt=dynamics.dt,
        horizon_seconds=dynamics.horizon_seconds,
        time_samples=33,
        softmin_temperature_m=0.25,
    )
    clearance_feature = clearance.detach() + smooth_clearance - smooth_clearance.detach()
    return (
        global_state,
        action,
        safety_context,
        current_margin,
        per_agent_margins,
        ttc.detach(),
        warning.detach(),
        emergency.detach(),
        clearance_feature,
    )


def counterfactual_alignment_objective(
    critic: TwinInterventionAdvantageCriticV1,
    *,
    raw_features: tuple[torch.Tensor, ...],
    reference_features: list[tuple[torch.Tensor, ...]],
    reference_valid: torch.Tensor,
    actor_eligible: torch.Tensor,
    deadzone: torch.Tensor,
    support_z_limit: float,
    use_counterfactual: bool = True,
    use_confidence_gate: bool = True,
) -> tuple[torch.Tensor, Mapping[str, float]]:
    """Return a masked relative-risk mean and inspectable diagnostics."""

    action = raw_features[1]
    raw_stage = critic.conservative_stage_probability(*raw_features)
    reference_stage = torch.stack(
        [critic.conservative_stage_probability(*features) for features in reference_features],
        dim=1,
    )
    valid = reference_valid.bool()
    if use_confidence_gate:
        raw_supported = critic.action_in_support(action, z_limit=support_z_limit)
        reference_supported = torch.stack(
            [critic.action_in_support(features[1], z_limit=support_z_limit) for features in reference_features],
            dim=1,
        )
        valid = valid & reference_supported
        row_valid = actor_eligible.bool() & raw_supported
        if use_counterfactual:
            row_valid = row_valid & valid.any(dim=1)
    else:
        row_valid = torch.ones_like(actor_eligible, dtype=torch.bool)

    if use_counterfactual:
        per_row = advantage_risk(
            raw_stage,
            reference_stage,
            valid,
            deadzone=deadzone,
        )
    else:
        per_row = torch.relu(raw_stage - deadzone).mean(dim=-1)
    denominator = row_valid.float().sum().clamp_min(1.0)
    objective = (per_row * row_valid.to(per_row.dtype)).sum() / denominator
    diagnostics = {
        "alignment_valid_fraction": float(row_valid.float().mean().detach().cpu()),
        "alignment_reference_valid_fraction": float(valid.float().mean().detach().cpu()),
        "alignment_raw_risk_mean": float(raw_stage.mean().detach().cpu()),
        "alignment_advantage_mean": float(per_row.mean().detach().cpu()),
    }
    return objective, diagnostics
