"""Twin stage-wise intervention critic for IA-MAPPO V1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


STAGE_COUNT = 3


class InterventionCriticV1(nn.Module):
    """Shared encoder with three classification and severity heads."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.input_dim = int(input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
        )
        self.classification_heads = nn.ModuleList(
            nn.Linear(256, 1) for _ in range(STAGE_COUNT)
        )
        self.severity_heads = nn.ModuleList(
            nn.Linear(256, 1) for _ in range(STAGE_COUNT)
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.shape[-1] != self.input_dim:
            raise ValueError("intervention critic feature dimension mismatch")
        encoded = self.encoder(features)
        logits = torch.cat([head(encoded) for head in self.classification_heads], dim=-1)
        severity = F.softplus(torch.cat([head(encoded) for head in self.severity_heads], dim=-1))
        return logits, severity

    def risk(
        self,
        features: torch.Tensor,
        temperatures: torch.Tensor | None = None,
        logit_offsets: torch.Tensor | None = None,
        head_enabled: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits, severity = self(features)
        if logit_offsets is not None:
            if logit_offsets.shape != (STAGE_COUNT,):
                raise ValueError("logit_offsets must contain one value per stage")
            logits = logits - logit_offsets
        if temperatures is not None:
            if temperatures.shape != (STAGE_COUNT,):
                raise ValueError("temperatures must contain one value per stage")
            logits = logits / temperatures.clamp_min(1e-3)
        probability = torch.sigmoid(logits)
        expected_stage_cost = probability * torch.clamp(severity, 0.0, 2.0)
        if head_enabled is None:
            enabled = torch.ones(
                STAGE_COUNT, dtype=torch.bool, device=features.device
            )
        else:
            enabled = torch.as_tensor(
                head_enabled, dtype=torch.bool, device=features.device
            )
        if enabled.shape != (STAGE_COUNT,) or not bool(enabled.any()):
            raise ValueError("head_enabled must enable at least one of three stages")
        return expected_stage_cost[..., enabled].mean(dim=-1)


class TwinInterventionCriticV1(nn.Module):
    """Two independently initialized critics; actor uses conservative max risk."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.critics = nn.ModuleList(
            [InterventionCriticV1(input_dim), InterventionCriticV1(input_dim)]
        )
        self.register_buffer("temperatures", torch.ones(2, STAGE_COUNT))
        self.register_buffer("prior_logit_offsets", torch.zeros(2, STAGE_COUNT))
        self.register_buffer("head_enabled", torch.ones(STAGE_COUNT, dtype=torch.bool))
        self.register_buffer("feature_mean", torch.zeros(self.input_dim))
        self.register_buffer("feature_std", torch.ones(self.input_dim))

    def normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.input_dim:
            raise ValueError("twin intervention critic feature dimension mismatch")
        return (features - self.feature_mean) / self.feature_std.clamp_min(1e-6)

    def forward(self, features: torch.Tensor):
        normalized = self.normalize_features(features)
        return tuple(critic(normalized) for critic in self.critics)

    def _twin_stage_risks(self, features: torch.Tensor) -> torch.Tensor:
        normalized = self.normalize_features(features)
        twin_stage_risks = []
        for index, critic in enumerate(self.critics):
            logits, severity = critic(normalized)
            logits = (
                logits - self.prior_logit_offsets[index]
            ) / self.temperatures[index].clamp_min(1e-3)
            expected = torch.sigmoid(logits) * torch.clamp(
                severity, 0.0, 2.0
            )
            expected = expected * self.head_enabled.to(expected.dtype)
            twin_stage_risks.append(expected)
        return torch.stack(twin_stage_risks, dim=0)

    def conservative_stage_risk(self, features: torch.Tensor) -> torch.Tensor:
        """Return the conservative expected cost for each registered stage.

        This is the audit-facing form of the frozen critic.  Keeping the
        stage axis is important when checking whether an online policy has
        drifted differently for Initial Guard and TTC.  Disabled heads are
        returned as zero and therefore cannot silently enter the actor loss.
        """

        return self._twin_stage_risks(features).max(dim=0).values

    def conservative_risk(self, features: torch.Tensor) -> torch.Tensor:
        twin_stage_risk = self._twin_stage_risks(features)
        enabled = self.head_enabled.to(twin_stage_risk.dtype)
        twin_total = (
            twin_stage_risk.sum(dim=-1) / enabled.sum().clamp_min(1.0)
        )
        return twin_total.max(dim=0).values

    def freeze_for_actor(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)


def rebase_feature_normalizer_preserving_function(
    critic: TwinInterventionCriticV1,
    new_mean: torch.Tensor,
    new_std: torch.Tensor,
) -> None:
    """Replace feature normalization without changing the represented function.

    The first encoder layer is analytically reparameterized so that every raw
    feature vector produces the same hidden pre-activation before and after the
    normalizer change.  This lets R4.3.2 learn support statistics from sampled
    actions without silently changing the already-certified TTC head.
    """

    mean = torch.as_tensor(
        new_mean, dtype=critic.feature_mean.dtype,
        device=critic.feature_mean.device,
    )
    std = torch.as_tensor(
        new_std, dtype=critic.feature_std.dtype,
        device=critic.feature_std.device,
    )
    if mean.shape != critic.feature_mean.shape or std.shape != critic.feature_std.shape:
        raise ValueError("replacement feature normalization has the wrong shape")
    if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
        raise ValueError("replacement feature normalization must be finite")
    if bool((std <= 0.0).any()):
        raise ValueError("replacement feature std must be positive")

    old_mean = critic.feature_mean.detach().clone()
    old_std = critic.feature_std.detach().clone().clamp_min(1e-6)
    scale = std / old_std
    shift = (mean - old_mean) / old_std
    with torch.no_grad():
        for member in critic.critics:
            first = member.encoder[0]
            if not isinstance(first, nn.Linear):
                raise TypeError("intervention critic encoder must start with Linear")
            old_weight = first.weight.detach().clone()
            first.weight.copy_(old_weight * scale.unsqueeze(0))
            first.bias.add_(old_weight @ shift)
        critic.feature_mean.copy_(mean)
        critic.feature_std.copy_(std)


def initial_head_repair_parameters(
    critic: TwinInterventionCriticV1,
) -> list[nn.Parameter]:
    """Freeze the shared/TTC network and expose only both Initial heads."""

    for parameter in critic.parameters():
        parameter.requires_grad_(False)
    parameters: list[nn.Parameter] = []
    for member in critic.critics:
        for head in (member.classification_heads[0], member.severity_heads[0]):
            for parameter in head.parameters():
                parameter.requires_grad_(True)
                parameters.append(parameter)
    return parameters


def joint_stage_repair_parameter_groups(
    critic: TwinInterventionCriticV1,
) -> list[dict[str, list[nn.Parameter]]]:
    """Expose the last encoder block and Initial/TTC heads for each twin."""

    for parameter in critic.parameters():
        parameter.requires_grad_(False)
    groups = []
    for member in critic.critics:
        encoder_parameters: list[nn.Parameter] = []
        for layer_index in (3, 5):
            for parameter in member.encoder[layer_index].parameters():
                parameter.requires_grad_(True)
                encoder_parameters.append(parameter)
        head_parameters: list[nn.Parameter] = []
        for stage_index in (0, 1):
            for head in (
                member.classification_heads[stage_index],
                member.severity_heads[stage_index],
            ):
                for parameter in head.parameters():
                    parameter.requires_grad_(True)
                    head_parameters.append(parameter)
        groups.append({"encoder": encoder_parameters, "heads": head_parameters})
    return groups


def full_encoder_stage_repair_parameter_groups(
    critic: TwinInterventionCriticV1,
) -> list[dict[str, list[nn.Parameter]]]:
    """Expose the full encoder and Initial/TTC heads for robust repair.

    R4.3.4 only exposed the second encoder block.  That is sufficient for
    probability repair, but it leaves the first projection fixed to the old
    representation.  R4.3.5 uses a lower encoder learning rate and exposes
    both Linear/LayerNorm blocks so action-conditioned TTC severity can alter
    the complete representation.  Final Guard remains frozen.
    """

    for parameter in critic.parameters():
        parameter.requires_grad_(False)
    groups = []
    for member in critic.critics:
        encoder_parameters: list[nn.Parameter] = []
        for layer_index in (0, 2, 3, 5):
            for parameter in member.encoder[layer_index].parameters():
                parameter.requires_grad_(True)
                encoder_parameters.append(parameter)
        head_parameters: list[nn.Parameter] = []
        for stage_index in (0, 1):
            for head in (
                member.classification_heads[stage_index],
                member.severity_heads[stage_index],
            ):
                for parameter in head.parameters():
                    parameter.requires_grad_(True)
                    head_parameters.append(parameter)
        groups.append({"encoder": encoder_parameters, "heads": head_parameters})
    return groups


class ScalarInterventionCriticV1(nn.Module):
    """EGPO-style scalar Material-intervention baseline critic."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 256), nn.SiLU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.SiLU(), nn.LayerNorm(256),
        )
        self.classification_head = nn.Linear(256, 1)
        self.severity_head = nn.Linear(256, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.shape[-1] != self.input_dim:
            raise ValueError("scalar intervention critic feature dimension mismatch")
        encoded = self.encoder(features)
        return self.classification_head(encoded), F.softplus(
            self.severity_head(encoded)
        )

    def risk(self, features: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        logits, severity = self(features)
        probability = torch.sigmoid(logits / temperature.clamp_min(1e-3))
        return (probability * torch.clamp(severity, 0.0, 2.0)).squeeze(-1)

    def risk_with_offset(
        self, features: torch.Tensor, temperature: torch.Tensor,
        logit_offset: torch.Tensor,
    ) -> torch.Tensor:
        logits, severity = self(features)
        probability = torch.sigmoid(
            (logits - logit_offset) / temperature.clamp_min(1e-3)
        )
        return (probability * torch.clamp(severity, 0.0, 2.0)).squeeze(-1)


class TwinScalarInterventionCriticV1(nn.Module):
    """Two independently initialized scalar critics for the direct baseline."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.critics = nn.ModuleList(
            [ScalarInterventionCriticV1(input_dim), ScalarInterventionCriticV1(input_dim)]
        )
        self.register_buffer("temperatures", torch.ones(2, 1))
        self.register_buffer("prior_logit_offsets", torch.zeros(2, 1))
        self.register_buffer("feature_mean", torch.zeros(self.input_dim))
        self.register_buffer("feature_std", torch.ones(self.input_dim))

    def normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.input_dim:
            raise ValueError("twin scalar critic feature dimension mismatch")
        return (features - self.feature_mean) / self.feature_std.clamp_min(1e-6)

    def conservative_risk(self, features: torch.Tensor) -> torch.Tensor:
        normalized = self.normalize_features(features)
        risks = torch.stack([
            critic.risk_with_offset(
                normalized, self.temperatures[index],
                self.prior_logit_offsets[index],
            )
            for index, critic in enumerate(self.critics)
        ])
        return risks.max(dim=0).values

    def freeze_for_actor(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)


@dataclass(frozen=True)
class InterventionCriticLoss:
    total: torch.Tensor
    classification: torch.Tensor
    severity: torch.Tensor


def intervention_critic_loss(
    logits: torch.Tensor,
    severity: torch.Tensor,
    material_targets: torch.Tensor,
    delta_v_targets: torch.Tensor,
    *,
    positive_weights: Sequence[float],
    head_enabled: Sequence[bool] = (True, True, True),
) -> InterventionCriticLoss:
    """Frequency-weighted BCE plus Material-only Huber severity loss."""

    expected = logits.shape
    if expected[-1] != STAGE_COUNT or severity.shape != expected:
        raise ValueError("critic outputs must have three equal-shaped heads")
    if material_targets.shape != expected or delta_v_targets.shape != expected:
        raise ValueError("critic targets must match output shape")
    weights = torch.as_tensor(positive_weights, dtype=logits.dtype, device=logits.device)
    enabled = torch.as_tensor(head_enabled, dtype=torch.bool, device=logits.device)
    if weights.shape != (STAGE_COUNT,) or enabled.shape != (STAGE_COUNT,):
        raise ValueError("weights and head_enabled must contain three values")
    if not bool(enabled.any()):
        raise ValueError("at least one stage head must be enabled")
    bce = F.binary_cross_entropy_with_logits(
        logits[..., enabled], material_targets[..., enabled],
        pos_weight=weights[enabled], reduction="mean",
    )
    material_mask = material_targets[..., enabled] > 0.5
    if bool(material_mask.any()):
        huber = F.huber_loss(
            severity[..., enabled][material_mask],
            delta_v_targets[..., enabled][material_mask],
            delta=0.05,
            reduction="mean",
        )
    else:
        huber = logits.new_zeros(())
    return InterventionCriticLoss(total=bce + huber, classification=bce, severity=huber)


def ia_actor_risk_penalty(
    twin_critic: TwinInterventionCriticV1 | TwinScalarInterventionCriticV1,
    critic_features: torch.Tensor,
    *,
    intervention_lambda: float,
) -> torch.Tensor:
    """Return λ·max(Q1,Q2), preserving gradients only to input features."""

    if intervention_lambda not in {0.003, 0.01, 0.03}:
        raise ValueError("intervention_lambda must be one of {0.003, 0.01, 0.03}")
    if any(parameter.requires_grad for parameter in twin_critic.parameters()):
        raise RuntimeError("critic must be frozen during actor optimization")
    return float(intervention_lambda) * twin_critic.conservative_risk(critic_features).mean()


def fit_stage_temperatures(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    head_enabled: Sequence[bool] = (True, True, True),
    maximum_steps: int = 100,
) -> torch.Tensor:
    """Fit one positive temperature per stage on validation layouts only."""

    if logits.shape != targets.shape or logits.shape[-1] != STAGE_COUNT:
        raise ValueError("temperature calibration requires matching three-head tensors")
    enabled = torch.as_tensor(head_enabled, dtype=torch.bool, device=logits.device)
    log_temperature = torch.zeros(
        STAGE_COUNT, dtype=logits.dtype, device=logits.device,
        requires_grad=True,
    )
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.1, max_iter=int(maximum_steps),
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.binary_cross_entropy_with_logits(
            logits[..., enabled] / temperature[enabled],
            targets[..., enabled],
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return log_temperature.detach().exp().clamp(0.05, 20.0)


def save_twin_critic_artifact(
    path: str | Path,
    critic: TwinInterventionCriticV1,
    *,
    input_dim: int,
    intervention_lambda: float,
    risk_to_ppo_gradient_ratio: float,
    dataset_manifest_sha256: str,
    lambda_calibration_sha256: str | None = None,
    source_certification_sha256: str | None = None,
    feature_contract: str | None = None,
    actor_gradient_contract: str | None = None,
    schema_version: str = "TwinInterventionCriticArtifactV2",
) -> None:
    """Write the frozen R3 artifact required by active IA actor training."""

    if intervention_lambda not in {0.003, 0.01, 0.03}:
        raise ValueError("unregistered intervention lambda")
    if not 0.0 <= risk_to_ppo_gradient_ratio <= 0.05:
        raise ValueError("risk gradient ratio exceeds the preregistered 5% limit")
    if len(dataset_manifest_sha256) != 64:
        raise ValueError("dataset manifest SHA256 is required")
    if not lambda_calibration_sha256 or len(lambda_calibration_sha256) != 64:
        raise ValueError("lambda calibration SHA256 is required")
    if not source_certification_sha256 or len(source_certification_sha256) != 64:
        raise ValueError("source certification SHA256 is required")
    if not feature_contract:
        raise ValueError("causal feature contract is required")
    if not actor_gradient_contract:
        raise ValueError("actor gradient contract is required")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": str(schema_version),
        "input_dim": int(input_dim),
        "state_dict": critic.state_dict(),
        "temperatures": critic.temperatures.detach().cpu(),
        "intervention_lambda": float(intervention_lambda),
        "risk_to_ppo_gradient_ratio": float(risk_to_ppo_gradient_ratio),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "lambda_calibration_sha256": lambda_calibration_sha256,
        "source_certification_sha256": source_certification_sha256,
        "feature_contract": feature_contract,
        "actor_gradient_contract": actor_gradient_contract,
        "certified_for_actor_training": True,
    }, destination)


def save_twin_scalar_critic_artifact(
    path: str | Path,
    critic: TwinScalarInterventionCriticV1,
    *,
    input_dim: int,
    intervention_lambda: float,
    risk_to_ppo_gradient_ratio: float,
    dataset_manifest_sha256: str,
    lambda_calibration_sha256: str | None = None,
    source_certification_sha256: str | None = None,
    feature_contract: str | None = None,
    actor_gradient_contract: str | None = None,
) -> None:
    """Write a calibrated scalar-baseline artifact for R4."""

    if intervention_lambda not in {0.003, 0.01, 0.03}:
        raise ValueError("unregistered intervention lambda")
    if not 0.0 <= risk_to_ppo_gradient_ratio <= 0.05:
        raise ValueError("risk gradient ratio exceeds the preregistered 5% limit")
    if len(dataset_manifest_sha256) != 64:
        raise ValueError("dataset manifest SHA256 is required")
    if not lambda_calibration_sha256 or len(lambda_calibration_sha256) != 64:
        raise ValueError("lambda calibration SHA256 is required")
    if not source_certification_sha256 or len(source_certification_sha256) != 64:
        raise ValueError("source certification SHA256 is required")
    if not feature_contract:
        raise ValueError("causal feature contract is required")
    if not actor_gradient_contract:
        raise ValueError("actor gradient contract is required")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": "TwinScalarInterventionCriticArtifactV2",
        "input_dim": int(input_dim),
        "state_dict": critic.state_dict(),
        "temperatures": critic.temperatures.detach().cpu(),
        "intervention_lambda": float(intervention_lambda),
        "risk_to_ppo_gradient_ratio": float(risk_to_ppo_gradient_ratio),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "lambda_calibration_sha256": lambda_calibration_sha256,
        "source_certification_sha256": source_certification_sha256,
        "feature_contract": feature_contract,
        "actor_gradient_contract": actor_gradient_contract,
        "certified_for_actor_training": True,
    }, destination)
