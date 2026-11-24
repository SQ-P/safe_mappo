"""Recurrent MAPPO/HAPPO trainers for the two-UAV team-reward task."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import optim

from src.algorithms.multi_agent.policy import (
    MultiAgentPolicy,
    OptimizerBundle,
    TeamCentralizedCritic,
)
from src.algorithms.intervention_critic import (
    TwinInterventionCriticV1,
    TwinScalarInterventionCriticV1,
    ia_actor_risk_penalty,
)
from src.algorithms.intervention_r3 import (
    R3_FEATURE_CONTRACT,
    R4_1_ACTOR_GRADIENT_CONTRACT,
    R4_2_1_ACTOR_GRADIENT_CONTRACT,
    R4_3_1_ACTOR_GRADIENT_CONTRACT,
    R4_2_ACTOR_GRADIENT_CONTRACT,
    R4_ACTOR_GRADIENT_CONTRACT,
)
from src.algorithms.intervention_team_data import (
    torch_action_conditioned_obstacle_features,
    torch_smooth_minimum_obstacle_clearance,
)
from src.algorithms.intervention_advantage_certification import (
    load_advantage_artifact,
)
from src.algorithms.safety_alignment import (
    AlignmentDynamics,
    counterfactual_alignment_objective,
    structured_critic_features,
)
from src.algorithms.ppo.rollout_buffer import RecurrentRolloutBuffer
from src.algorithms.ppo.trainer import PPOTrainer
from src.utils.io_utils import save_json


def straight_through_rollout_action(
    raw_mean: torch.Tensor,
    behavior_raw: torch.Tensor,
) -> torch.Tensor:
    """Use behavior actions in forward while differentiating only the mean."""

    if raw_mean.shape != behavior_raw.shape:
        raise ValueError("raw mean and behavior action shapes must match")
    raw_query = raw_mean + (behavior_raw - raw_mean).detach()
    return torch.tanh(raw_query)


def project_risk_gradient_preserving_progress(
    gradient: torch.Tensor,
    progress_gradient: torch.Tensor,
) -> torch.Tensor:
    """Remove risk-descent components that would reduce per-UAV progress."""

    if gradient.shape != progress_gradient.shape or gradient.shape[-1] != 2:
        raise ValueError("risk and progress gradients must share [..., UAV,2] shape")
    norm = torch.linalg.vector_norm(
        progress_gradient, dim=-1, keepdim=True
    )
    unit = torch.where(
        norm > 1e-8,
        progress_gradient / norm.clamp_min(1e-8),
        torch.zeros_like(progress_gradient),
    )
    harmful = (gradient * unit).sum(dim=-1, keepdim=True).clamp_min(0.0)
    return gradient - harmful * unit


def project_parameter_risk_gradient(
    risk_gradient: torch.Tensor,
    task_gradient: torch.Tensor,
    progress_gradients: list[torch.Tensor],
    *,
    passes: int = 12,
) -> torch.Tensor:
    """Project risk gradient into PPO-compatible per-agent progress halfspaces.

    The optimizer applies ``-gradient``.  Therefore the projected risk
    gradient must have a non-negative dot product with the task gradient and
    a non-positive dot product with every progress-score gradient.
    """

    if risk_gradient.ndim != 1 or task_gradient.shape != risk_gradient.shape:
        raise ValueError("task and risk gradients must be equal flat vectors")
    if any(item.shape != risk_gradient.shape for item in progress_gradients):
        raise ValueError("all progress gradients must match the flat risk vector")
    projected = risk_gradient.clone()
    constraints: list[tuple[torch.Tensor, float]] = [
        (task_gradient, 1.0),
        *((item, -1.0) for item in progress_gradients),
    ]
    for _ in range(max(int(passes), 1)):
        changed = False
        for normal, required_sign in constraints:
            norm_squared = normal.square().sum()
            if float(norm_squared.detach().cpu()) <= 1e-20:
                continue
            signed_dot = required_sign * torch.dot(projected, normal)
            if float(signed_dot.detach().cpu()) < -1e-10:
                projected = projected - (
                    torch.dot(projected, normal) / norm_squared
                ) * normal
                changed = True
        if not changed:
            break
    task_ok = float(torch.dot(projected, task_gradient).detach().cpu()) >= -1e-8
    progress_ok = all(
        float(torch.dot(projected, item).detach().cpu()) <= 1e-8
        for item in progress_gradients
    )
    if not task_ok or not progress_ok:
        # The origin belongs to every registered halfspace.  Falling back to
        # zero risk is preferable to silently applying a conflicting update.
        projected = torch.zeros_like(projected)
    if not bool(torch.isfinite(projected).all()):
        raise FloatingPointError("non-finite parameter-space risk projection")
    return projected


class MultiAgentPPOTrainer(PPOTrainer):
    """Common CTDE implementation for MAPPO and HAPPO.

    The environment still exposes the legacy broadcast reward array, while
    this trainer verifies that it is an exact team reward and computes one
    team GAE before broadcasting the resulting advantage to the actors.
    """

    POLICY_SCHEMA_VERSION = 10
    JOINT_POLICY = False
    TEAM_GAE = True

    def __init__(self, config, project_root=None):
        super().__init__(config, project_root=project_root)
        if self.safety_alignment_mode != "disabled":
            self._load_alignment_parent_checkpoint()

    def _initialize_networks(
        self,
        model_cfg: Mapping[str, Any],
        train_cfg: Mapping[str, Any],
    ) -> None:
        if self.config.get("env", {}).get("reward_mode") != "team_v4_subgoal":
            raise ValueError(
                "MAPPO/HAPPO comparison requires reward_mode=team_v4_subgoal"
            )
        if bool(dict(train_cfg.get("capture_sampler", {})).get("enabled", False)):
            raise ValueError(
                "capture-bank starts are not part of the MAPPO/HAPPO comparison contract"
            )
        if "projection_aware_execution" in train_cfg:
            raise ValueError(
                "Projection-aware V1/V2 losses are archived and cannot initialize active training"
            )

        multi_cfg = dict(self.config.get("multi_agent", {}))
        default_sharing = self.algorithm == "mappo"
        self.actor_parameter_sharing = bool(
            multi_cfg.get("actor_parameter_sharing", default_sharing)
        )
        self.agent_update_order = str(
            multi_cfg.get("agent_update_order", "random")
        ).lower()
        if self.algorithm == "happo":
            if self.actor_parameter_sharing:
                raise ValueError("HAPPO requires independent actors")
            if self.agent_update_order != "random":
                raise ValueError("HAPPO requires agent_update_order=random")
        elif self.algorithm == "mappo":
            if self.agent_update_order not in {"random", "simultaneous"}:
                raise ValueError(
                    "MAPPO agent_update_order must be simultaneous or random"
                )
            self.agent_update_order = "simultaneous"
        else:
            raise ValueError("multi-agent trainer supports MAPPO or HAPPO")

        self.config["multi_agent"] = {
            "actor_parameter_sharing": self.actor_parameter_sharing,
            "agent_update_order": self.agent_update_order,
        }
        alignment_cfg = dict(train_cfg.get("safety_alignment", {}))
        self.safety_alignment_mode = str(
            alignment_cfg.get("mode", "disabled")
        ).lower()
        if self.safety_alignment_mode not in {
            "disabled", "counterfactual_adapter", "lagrangian"
        }:
            raise ValueError("unsupported train.safety_alignment.mode")
        if self.safety_alignment_mode != "disabled" and (
            self.algorithm != "mappo" or not self.actor_parameter_sharing
        ):
            raise ValueError("safety alignment requires shared-actor MAPPO")
        if self.safety_alignment_mode != "disabled" and "intervention_aware" in train_cfg:
            ia_mode = str(dict(train_cfg.get("intervention_aware", {})).get("mode", "disabled"))
            if ia_mode != "disabled":
                raise ValueError("safety_alignment and archived intervention_aware modes are mutually exclusive")
        self.safety_alignment_config = alignment_cfg
        self.safety_alignment_parent_sha256: str | None = None
        self.safety_alignment_parent_update: int | None = None
        self.safety_alignment_critic_sha256: str | None = None
        self.safety_alignment_lambda = float(alignment_cfg.get("lambda_a", 0.0))
        self.safety_alignment_calibration: dict[str, Any] | None = None
        calibration_value = alignment_cfg.get("lambda_calibration")
        if calibration_value:
            calibration_path = self._resolve(calibration_value).resolve()
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            if (
                calibration.get("schema_version") != "CSALambdaCalibrationV1"
                or int(calibration.get("train_seed", -1)) != 23
                or not bool(calibration.get("fixed_rollout", False))
                or not bool(calibration.get("frozen_for_all_training_seeds", False))
            ):
                raise ValueError("CSA lambda calibration contract is invalid")
            selected_lambda = float(calibration["selected_lambda_a"])
            if self.safety_alignment_lambda not in {0.0, selected_lambda}:
                raise ValueError("configured lambda_a differs from frozen calibration")
            self.safety_alignment_lambda = selected_lambda
            self.safety_alignment_config["lambda_a"] = selected_lambda
            self.safety_alignment_calibration = calibration
        self.safety_alignment_deadzone = torch.as_tensor(
            alignment_cfg.get("deadzone", [0.05, 0.05]),
            dtype=torch.float32, device=self.device,
        )
        if self.safety_alignment_deadzone.shape != (2,):
            raise ValueError("safety_alignment.deadzone must contain Initial/TTC values")
        self.safety_alignment_support_z_limit = float(
            alignment_cfg.get("support_z_limit", 6.0)
        )
        self.safety_alignment_use_counterfactual = bool(
            alignment_cfg.get("use_counterfactual", True)
        )
        self.safety_alignment_use_confidence_gate = bool(
            alignment_cfg.get("use_confidence_gate", True)
        )
        self.safety_alignment_train_backbone = bool(
            alignment_cfg.get("train_backbone", False)
        )
        adapter_rank = int(alignment_cfg.get("adapter_rank", 16))
        if self.safety_alignment_mode != "counterfactual_adapter":
            adapter_rank = 0
        if self.safety_alignment_train_backbone:
            adapter_rank = int(alignment_cfg.get("adapter_rank", 0))
        self.actor = MultiAgentPolicy(
            num_agents=self.num_agents,
            obs_dim=self.obs_dim,
            hidden_dim=self.hidden_dim,
            action_dim=2,
            parameter_sharing=self.actor_parameter_sharing,
            initial_action_log_std=float(
                model_cfg.get("initial_action_log_std", -1.2)
            ),
            log_std_min=float(train_cfg.get("log_std_min", -4.0)),
            log_std_max=float(train_cfg.get("log_std_max", -0.3)),
            safety_adapter_rank=adapter_rank,
            safety_adapter_scale=float(alignment_cfg.get("adapter_scale", 0.15)),
        ).to(self.device)
        self.critic = TeamCentralizedCritic(
            self.global_state_dim, self.hidden_dim, self.num_agents
        ).to(self.device)

        actor_lr = float(train_cfg.get("actor_learning_rate", 3e-4))
        if self.safety_alignment_mode == "counterfactual_adapter" and not self.safety_alignment_train_backbone:
            self.actor.freeze_backbone_for_adapter()
            adapter_parameters = self.actor.adapter_parameter_count()
            if adapter_parameters <= 0:
                raise ValueError("CSA-MAPPO requires a non-empty adapter")
            backbone_parameters = self.actor.parameter_count() - adapter_parameters
            if adapter_parameters / max(backbone_parameters, 1) >= 0.03:
                raise ValueError("safety adapter exceeds the registered 3% parameter budget")
        actor_optimizers = []
        for actor in self.actor.actor_modules():
            trainable = [parameter for parameter in actor.parameters() if parameter.requires_grad]
            if not trainable:
                raise ValueError("actor has no trainable parameters")
            actor_optimizers.append(optim.Adam(trainable, lr=actor_lr))
        self.actor_optimizer = OptimizerBundle(actor_optimizers)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=float(train_cfg.get("critic_learning_rate", 3e-4)),
        )
        self.last_agent_update_order: list[int] = []
        self.capture_safety_alignment_references = (
            self.safety_alignment_mode == "counterfactual_adapter"
        )
        self.capture_safety_alignment_costs = (
            self.safety_alignment_mode in {"counterfactual_adapter", "lagrangian"}
        )
        self.safety_alignment_critic = None
        if self.safety_alignment_mode == "counterfactual_adapter":
            if self.safety_alignment_lambda <= 0.0:
                raise ValueError("counterfactual_adapter requires lambda_a > 0")
            critic_value = alignment_cfg.get("critic_artifact")
            certification_value = alignment_cfg.get("critic_certification")
            if not critic_value or not certification_value:
                raise ValueError("CSA-MAPPO requires critic_artifact and critic_certification")
            artifact_path = self._resolve(critic_value).resolve()
            certification_path = self._resolve(certification_value).resolve()
            certification = json.loads(certification_path.read_text(encoding="utf-8"))
            artifact_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            if not bool(certification.get("certified", False)):
                raise ValueError("CSA critic has not passed fresh certification")
            if certification.get("artifact_sha256") != artifact_sha:
                raise ValueError("CSA critic certification/artifact hash mismatch")
            self.safety_alignment_critic, critic_payload = load_advantage_artifact(
                artifact_path, device=str(self.device)
            )
            spec = self.safety_alignment_critic.spec
            if (
                spec.agent_count != self.num_agents
                or spec.global_state_dim != self.global_state_dim
                or spec.safety_context_dim != 8
            ):
                raise ValueError("CSA critic structured feature contract mismatch")
            if float(critic_payload.get("alpha_pair", -1.0)) != 0.25:
                raise ValueError("CSA main method requires alpha_pair=0.25")
            self.safety_alignment_critic.freeze_for_actor()
            self.safety_alignment_critic_sha256 = artifact_sha
            if (
                self.safety_alignment_calibration is not None
                and self.safety_alignment_calibration.get("critic_artifact_sha256")
                != artifact_sha
            ):
                raise ValueError("lambda calibration used a different critic artifact")
        self.lagrangian_multiplier = float(alignment_cfg.get("lambda_init", 0.0))
        self.lagrangian_learning_rate = float(alignment_cfg.get("lambda_learning_rate", 0.01))
        self.lagrangian_max = float(alignment_cfg.get("lambda_max", 10.0))
        self.lagrangian_cost_limit = float(alignment_cfg.get("cost_limit", 0.0))
        self.lagrangian_cost_budget: dict[str, Any] | None = None
        budget_value = alignment_cfg.get("cost_budget_manifest")
        if budget_value:
            budget_path = self._resolve(budget_value).resolve()
            budget = json.loads(budget_path.read_text(encoding="utf-8"))
            if (
                budget.get("schema_version") != "MAPPOLagrangianCostBudgetV1"
                or float(budget.get("budget_fraction", -1.0)) != 0.90
                or not bool(budget.get("frozen_for_all_training_seeds", False))
            ):
                raise ValueError("Lagrangian cost-budget contract is invalid")
            manifest_limit = float(budget["cost_limit"])
            if self.lagrangian_cost_limit not in {0.0, manifest_limit}:
                raise ValueError("configured cost_limit differs from frozen manifest")
            self.lagrangian_cost_limit = manifest_limit
            self.safety_alignment_config["cost_limit"] = manifest_limit
            self.lagrangian_cost_budget = budget
        if self.safety_alignment_mode == "lagrangian" and not 0.0 <= self.lagrangian_cost_limit <= 1.0:
            raise ValueError("lagrangian cost_limit must be in [0,1]")
        ia_cfg = dict(train_cfg.get("intervention_aware", {}))
        self.intervention_aware_mode = str(ia_cfg.get("mode", "disabled"))
        self.capture_intervention_features = bool(
            self.capture_safety_alignment_costs
            or
            ia_cfg.get("capture_preaction_features", False)
            or self.intervention_aware_mode != "disabled"
        )
        self.intervention_lambda = float(ia_cfg.get("intervention_lambda", 0.0))
        self.intervention_action_query = str(
            ia_cfg.get("action_query", "bounded_mean")
        )
        self.intervention_support_z_limit = float(
            ia_cfg.get("support_z_limit", 6.0)
        )
        self.intervention_support_penalty_scale = float(
            ia_cfg.get("support_penalty_scale", 0.05)
        )
        self.intervention_activation_update = int(
            ia_cfg.get("activation_update", 0)
        )
        self.intervention_minimum_support_fraction = float(
            ia_cfg.get("minimum_support_fraction", 0.90)
        )
        self.intervention_smooth_clearance_samples = int(
            ia_cfg.get("smooth_clearance_samples", 33)
        )
        self.intervention_smooth_clearance_temperature = float(
            ia_cfg.get("smooth_clearance_temperature_m", 0.25)
        )
        self._latest_intervention_ood_rate = 0.0
        self._latest_intervention_support_fraction = 1.0
        self._latest_intervention_eligibility_fraction = 1.0
        self.intervention_critic: (
            TwinInterventionCriticV1 | TwinScalarInterventionCriticV1 | None
        ) = None
        if self.intervention_aware_mode != "disabled":
            if self.intervention_aware_mode not in {
                "stage_counterfactual_v1", "stage_counterfactual_v2",
                "stage_counterfactual_v3",
                "stage_counterfactual_v4",
                "stage_counterfactual_v5",
                "scalar_material_v1",
            }:
                raise ValueError("unsupported train.intervention_aware.mode")
            if self.algorithm != "mappo" or not self.actor_parameter_sharing:
                raise ValueError("IA-MAPPO V1 requires shared-actor MAPPO")
            critic_cfg = dict(ia_cfg.get("critic", {}))
            artifact_value = critic_cfg.get("artifact")
            if not artifact_value:
                raise ValueError(
                    f"{self.intervention_aware_mode} requires a frozen certified critic artifact"
                )
            artifact_path = self._resolve(artifact_value)
            payload = torch.load(
                artifact_path, map_location=self.device, weights_only=False
            )
            if self.intervention_aware_mode == "stage_counterfactual_v2":
                expected_schema = "TwinInterventionCriticArtifactV3"
                expected_gradient_contract = R4_1_ACTOR_GRADIENT_CONTRACT
                if self.intervention_action_query != "rollout_straight_through":
                    raise ValueError(
                        "stage_counterfactual_v2 requires action_query=rollout_straight_through"
                    )
            elif self.intervention_aware_mode == "stage_counterfactual_v3":
                expected_schema = "TwinInterventionCriticArtifactV4"
                expected_gradient_contract = R4_2_ACTOR_GRADIENT_CONTRACT
                if self.intervention_action_query != "rollout_straight_through":
                    raise ValueError(
                        "stage_counterfactual_v3 requires action_query=rollout_straight_through"
                    )
                if self.intervention_activation_update < 1:
                    raise ValueError(
                        "stage_counterfactual_v3 requires a positive activation_update"
                    )
                if not 0.0 < self.intervention_minimum_support_fraction <= 1.0:
                    raise ValueError("minimum_support_fraction must be in (0,1]")
            elif self.intervention_aware_mode == "stage_counterfactual_v4":
                expected_schema = "TwinInterventionCriticArtifactV5"
                expected_gradient_contract = R4_2_1_ACTOR_GRADIENT_CONTRACT
                if self.intervention_action_query != "rollout_straight_through":
                    raise ValueError(
                        "stage_counterfactual_v4 requires action_query=rollout_straight_through"
                    )
                if self.intervention_activation_update != 100:
                    raise ValueError(
                        "stage_counterfactual_v4 freezes activation_update=100"
                    )
                if not 0.0 < self.intervention_minimum_support_fraction <= 1.0:
                    raise ValueError("minimum_support_fraction must be in (0,1]")
            elif self.intervention_aware_mode == "stage_counterfactual_v5":
                expected_schema = "TwinInterventionCriticArtifactV6"
                expected_gradient_contract = R4_3_1_ACTOR_GRADIENT_CONTRACT
                if self.intervention_action_query != "rollout_straight_through":
                    raise ValueError(
                        "stage_counterfactual_v5 requires action_query=rollout_straight_through"
                    )
                if self.intervention_activation_update != 100:
                    raise ValueError(
                        "stage_counterfactual_v5 freezes activation_update=100"
                    )
                if not 0.0 < self.intervention_minimum_support_fraction <= 1.0:
                    raise ValueError("minimum_support_fraction must be in (0,1]")
            elif self.intervention_aware_mode == "stage_counterfactual_v1":
                expected_schema = "TwinInterventionCriticArtifactV2"
                expected_gradient_contract = R4_ACTOR_GRADIENT_CONTRACT
            else:
                expected_schema = "TwinScalarInterventionCriticArtifactV2"
                expected_gradient_contract = R4_ACTOR_GRADIENT_CONTRACT
            if payload.get("schema_version") != expected_schema:
                raise ValueError("unsupported intervention critic artifact schema")
            expected_input = (
                self.global_state_dim + 2 * self.num_agents + 8 + 1
                + 3 * self.num_agents + 4 * self.num_agents
            )
            if int(payload.get("input_dim", -1)) != expected_input:
                raise ValueError("intervention critic input contract mismatch")
            if payload.get("feature_contract") != R3_FEATURE_CONTRACT:
                raise ValueError("intervention critic causal feature contract mismatch")
            if payload.get("actor_gradient_contract") != expected_gradient_contract:
                raise ValueError("intervention critic actor-gradient contract mismatch")
            if not bool(payload.get("certified_for_actor_training", False)):
                raise ValueError("intervention critic is not certified for actor training")
            for evidence_key in (
                "dataset_manifest_sha256", "lambda_calibration_sha256",
                "source_certification_sha256",
            ):
                evidence = str(payload.get(evidence_key, ""))
                if len(evidence) != 64:
                    raise ValueError(
                        f"intervention critic is missing {evidence_key} evidence"
                    )
            certified_lambda = float(payload.get("intervention_lambda", -1.0))
            gradient_ratio = float(payload.get("risk_to_ppo_gradient_ratio", 1.0))
            if certified_lambda != self.intervention_lambda or gradient_ratio > 0.05:
                raise ValueError("critic artifact lacks the preregistered lambda calibration")
            critic_type = (
                TwinInterventionCriticV1
                if self.intervention_aware_mode in {
                    "stage_counterfactual_v1", "stage_counterfactual_v2",
                    "stage_counterfactual_v3",
                    "stage_counterfactual_v4",
                    "stage_counterfactual_v5",
                }
                else TwinScalarInterventionCriticV1
            )
            # A frozen auxiliary model must not perturb the policy sampling
            # RNG.  This is essential for the R4.2 risk-off warm-up parity
            # contract against fresh-init Standard MAPPO.
            fork_devices = []
            if self.device.type == "cuda":
                fork_devices = [
                    self.device.index
                    if self.device.index is not None
                    else torch.cuda.current_device()
                ]
            with torch.random.fork_rng(devices=fork_devices):
                self.intervention_critic = critic_type(expected_input).to(
                    self.device
                )
            self.intervention_critic.load_state_dict(payload["state_dict"])
            temperatures = torch.as_tensor(
                payload.get("temperatures"), dtype=torch.float32,
                device=self.device,
            )
            expected_temperature_shape = (
                (2, 3)
                if self.intervention_aware_mode in {
                    "stage_counterfactual_v1", "stage_counterfactual_v2",
                    "stage_counterfactual_v3",
                    "stage_counterfactual_v4",
                    "stage_counterfactual_v5",
                }
                else (2, 1)
            )
            if temperatures.shape != expected_temperature_shape:
                raise ValueError(
                    "critic artifact temperatures do not match its registered mode"
                )
            self.intervention_critic.temperatures.copy_(temperatures)
            self.intervention_critic.freeze_for_actor()

    @property
    def algorithm_variant(self) -> str:
        sharing = "shared" if self.actor_parameter_sharing else "independent"
        return f"{self.algorithm}_{sharing}"

    def actor_parameter_count(self) -> int:
        return self.actor.parameter_count()

    def critic_parameter_count(self) -> int:
        return self.critic.parameter_count()

    def policy_rewards(self, rewards) -> np.ndarray:
        values = np.asarray(rewards, dtype=np.float32).reshape(-1)
        if values.shape != (self.num_agents,):
            raise ValueError("team reward must contain one broadcast row per agent")
        if not np.isfinite(values).all():
            raise ValueError("team reward must be finite")
        if not np.all(values == values[0]):
            raise ValueError(
                "MAPPO/HAPPO require an exactly broadcast team reward"
            )
        return values.copy()

    def _task_config_payload(self) -> dict[str, Any]:
        payload = super()._task_config_payload()
        payload["task_contract_version"] = 5
        payload["algorithm"] = self.algorithm
        payload["multi_agent"] = deepcopy(
            self.config.get("multi_agent", {})
        )
        return payload

    def _learner_contract_payload(self) -> dict[str, Any]:
        payload = super()._learner_contract_payload()
        payload["multi_agent"] = deepcopy(
            self.config.get("multi_agent", {})
        )
        payload["team_value_head_count"] = 1
        payload["team_gae"] = True
        payload["safety_alignment"] = {
            "mode": self.safety_alignment_mode,
            "adapter_rank": int(self.safety_alignment_config.get("adapter_rank", 0)),
            "adapter_scale": float(self.safety_alignment_config.get("adapter_scale", 0.15)),
            "deadzone": [float(item) for item in self.safety_alignment_deadzone.detach().cpu()],
            "support_z_limit": self.safety_alignment_support_z_limit,
            "use_counterfactual": self.safety_alignment_use_counterfactual,
            "use_confidence_gate": self.safety_alignment_use_confidence_gate,
            "train_backbone": self.safety_alignment_train_backbone,
            "lambda_a": self.safety_alignment_lambda,
            "lagrangian_cost_limit": self.lagrangian_cost_limit,
            "lagrangian_learning_rate": self.lagrangian_learning_rate,
            "lagrangian_max": self.lagrangian_max,
        }
        return payload

    def _load_alignment_parent_checkpoint(self) -> None:
        value = self.safety_alignment_config.get("parent_checkpoint")
        if not value:
            raise ValueError("safety_alignment requires parent_checkpoint")
        path = self._resolve(value).resolve()
        payload = torch.load(path, map_location=self.device, weights_only=False)
        parent_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = {
            "algorithm": "mappo",
            "num_agents": self.num_agents,
            "obs_dim": self.obs_dim,
            "global_state_dim": self.global_state_dim,
            "action_dim": self.policy_action_dim,
            "hidden_dim": self.hidden_dim,
            "task_config_fingerprint": self.task_config_fingerprint(),
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"alignment parent checkpoint {key} is incompatible")
        result = self.actor.load_state_dict(payload["actor"], strict=False)
        missing = set(result.missing_keys)
        unexpected = set(result.unexpected_keys)
        if unexpected or any("safety_adapter." not in key for key in missing):
            raise ValueError(
                "alignment parent actor differs beyond the registered adapter"
            )
        if self.safety_alignment_mode == "lagrangian" and missing:
            raise ValueError("Lagrangian parent actor must load strictly")
        self.critic.load_state_dict(payload["critic"], strict=True)
        self.safety_alignment_parent_sha256 = parent_sha
        self.safety_alignment_parent_update = int(payload["update"])
        if self.safety_alignment_parent_update != int(
            self.safety_alignment_config.get("parent_update", 400)
        ):
            raise ValueError("alignment parent checkpoint is not the registered u400 parent")
        expected_sha = self.safety_alignment_config.get("parent_checkpoint_sha256")
        if expected_sha is not None and str(expected_sha) != parent_sha:
            raise ValueError("alignment parent checkpoint SHA256 mismatch")
        if self.safety_alignment_calibration is not None:
            if self.safety_alignment_calibration.get("parent_checkpoint_sha256") != parent_sha:
                raise ValueError("lambda calibration used a different parent checkpoint")
            if self.safety_alignment_calibration.get("task_config_fingerprint") != self.task_config_fingerprint():
                raise ValueError("lambda calibration used a different task contract")
        if (
            self.lagrangian_cost_budget is not None
            and self.lagrangian_cost_budget.get("task_config_fingerprint")
            != self.task_config_fingerprint()
        ):
            raise ValueError("Lagrangian budget used a different task contract")
        if self.safety_alignment_mode == "counterfactual_adapter":
            adapter = self.actor.actor_modules()[0].safety_adapter
            if adapter is not None:
                if any(bool(torch.count_nonzero(parameter).item()) for name, parameter in adapter.named_parameters() if name.startswith("up.")):
                    raise ValueError("CSA adapter must be zero initialized")

    def _checkpoint_extension_payload(self) -> dict[str, Any]:
        if self.safety_alignment_mode == "disabled":
            return {}
        return {
            "safety_alignment_checkpoint": {
                "schema_version": "CSAMAPPOCheckpointV1",
                "contract": self._learner_contract_payload()["safety_alignment"],
                "parent_checkpoint_sha256": self.safety_alignment_parent_sha256,
                "parent_checkpoint_update": self.safety_alignment_parent_update,
                "critic_artifact_sha256": self.safety_alignment_critic_sha256,
                "adapter_parameter_count": self.actor.adapter_parameter_count(),
                "frozen_actor_parameters": sorted(
                    name for name, parameter in self.actor.named_parameters()
                    if not parameter.requires_grad
                ),
                "lagrangian_multiplier": self.lagrangian_multiplier,
            }
        }

    def _load_checkpoint_extension(self, payload: Mapping[str, Any]) -> None:
        if self.safety_alignment_mode == "disabled":
            return
        saved = payload.get("safety_alignment_checkpoint")
        if not isinstance(saved, Mapping) or saved.get("schema_version") != "CSAMAPPOCheckpointV1":
            raise ValueError("alignment checkpoint extension is missing or incompatible")
        if saved.get("contract") != self._learner_contract_payload()["safety_alignment"]:
            raise ValueError("alignment checkpoint contract is incompatible")
        if saved.get("parent_checkpoint_sha256") != self.safety_alignment_parent_sha256:
            raise ValueError("alignment parent provenance changed during resume")
        if saved.get("critic_artifact_sha256") != self.safety_alignment_critic_sha256:
            raise ValueError("alignment critic provenance changed during resume")
        self.lagrangian_multiplier = float(
            saved.get("lagrangian_multiplier", self.lagrangian_multiplier)
        )

    def _alignment_dynamics(self) -> AlignmentDynamics:
        return AlignmentDynamics(
            agent_radius=float(self.env.agent_radius),
            obstacle_safety_margin=float(self.env.oracle_config["obstacle_safety_margin"]),
            invariant_buffer=float(self.env.boundary_guard.invariant_buffer_m),
            max_acceleration=float(self.env.agent_max_acceleration),
            max_speed=float(self.env.agent_max_speed),
            dt=float(self.env.dt),
            horizon_seconds=float(self.env.obstacle_guard_horizon_seconds),
            warning_ttc_seconds=float(self.env.obstacle_guard_warning_ttc_seconds),
            emergency_ttc_seconds=float(self.env.obstacle_guard_emergency_ttc_seconds),
        )

    def _safety_alignment_actor_penalty(
        self, buffer: RecurrentRolloutBuffer, batch, actor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if self.safety_alignment_critic is None:
            raise RuntimeError("CSA objective requires a frozen advantage critic")
        indices = np.unique(batch.time_indices[batch.mask])
        index = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        local_obs = torch.as_tensor(buffer.local_obs[indices], device=self.device)
        hidden = torch.as_tensor(buffer.hidden[indices], device=self.device)
        flat_obs = local_obs.reshape(-1, self.obs_dim)
        flat_hidden = hidden.permute(1, 0, 2, 3).reshape(1, -1, self.hidden_dim)
        distribution, _ = actor._distribution_and_state(flat_obs, flat_hidden)
        action = torch.tanh(distribution.base.mean).reshape(-1, self.num_agents, 2)

        common = {
            "global_state": torch.as_tensor(buffer.global_states[indices], device=self.device),
            "safety_context": torch.as_tensor(buffer.safety_context[indices], device=self.device),
            "current_margin": torch.as_tensor(buffer.safety_margin[indices], device=self.device),
            "per_agent_margins": torch.as_tensor(buffer.preaction_per_agent_margins[indices], device=self.device),
            "current_velocity": torch.as_tensor(buffer.preaction_team_velocity[indices], device=self.device),
            "obstacle_tracks": torch.as_tensor(buffer.preaction_obstacle_tracks[indices], device=self.device),
            "dynamics": self._alignment_dynamics(),
        }
        raw_features = structured_critic_features(action=action, **common)
        references = torch.as_tensor(buffer.reference_actions[indices], device=self.device)
        reference_features = [
            structured_critic_features(action=references[:, ref].detach(), **common)
            for ref in range(references.shape[1])
        ]
        objective, diagnostics = counterfactual_alignment_objective(
            self.safety_alignment_critic,
            raw_features=raw_features,
            reference_features=reference_features,
            reference_valid=torch.as_tensor(buffer.reference_valid[indices], device=self.device),
            actor_eligible=torch.as_tensor(buffer.actor_risk_eligible[indices], device=self.device),
            deadzone=self.safety_alignment_deadzone,
            support_z_limit=self.safety_alignment_support_z_limit,
            use_counterfactual=self.safety_alignment_use_counterfactual,
            use_confidence_gate=self.safety_alignment_use_confidence_gate,
        )
        return self.safety_alignment_lambda * objective, dict(diagnostics)

    @staticmethod
    def _lagrangian_advantage_batch(buffer, batch) -> np.ndarray:
        values = np.zeros(batch.mask.shape, dtype=np.float32)
        for column, agent_id in enumerate(batch.agent_ids):
            valid = batch.mask[:, column]
            values[valid, column] = buffer.safety_cost_advantages[
                batch.time_indices[valid, column], int(agent_id)
            ]
        return values

    @staticmethod
    def _masked_factor(
        factors: np.ndarray,
        time_indices: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        result = np.ones(mask.shape, dtype=np.float32)
        result[mask] = factors[time_indices[mask]].astype(np.float32)
        return result

    def _update_actor(
        self,
        buffer: RecurrentRolloutBuffer,
        *,
        actor_index: int,
        agent_id: int | None,
        factors: np.ndarray,
        advantage_mean: float,
        advantage_std: float,
        entropy_coef: float,
    ) -> dict[str, list[float]]:
        clip_coef = float(self.train_config.get("clip_coef", 0.2))
        max_grad_norm = float(self.train_config.get("max_grad_norm", 0.5))
        sequence_length = int(self.train_config.get("sequence_length", 16))
        minibatch_size = int(
            self.train_config.get("sequence_minibatch_size", 32)
        )
        target_kl = float(self.train_config.get("target_kl", 0.03))
        ppo_epochs = int(self.train_config.get("ppo_epochs", 4))
        actor = self.actor.actor_modules()[actor_index]
        optimizer = self.actor_optimizer.optimizers[actor_index]
        result: dict[str, list[float]] = {
            "policy_loss": [],
            "action_entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
            "actor_grad_norm": [],
            "intervention_risk_penalty": [],
            "intervention_ppo_gradient_norm": [],
            "intervention_risk_gradient_norm": [],
            "intervention_risk_to_ppo_gradient_ratio": [],
            "intervention_risk_ppo_gradient_cosine": [],
            "intervention_raw_risk_ppo_gradient_cosine": [],
            "intervention_progress_constraint_violations": [],
            "intervention_action_ood_rate": [],
            "intervention_support_fraction": [],
            "intervention_eligibility_fraction": [],
            "intervention_risk_active": [],
            "alignment_penalty": [],
            "alignment_valid_fraction": [],
            "alignment_reference_valid_fraction": [],
            "alignment_raw_risk_mean": [],
            "alignment_advantage_mean": [],
            "alignment_ppo_gradient_norm": [],
            "alignment_risk_gradient_norm": [],
            "alignment_risk_to_ppo_gradient_ratio": [],
            "lagrangian_penalty": [],
        }

        for _ in range(ppo_epochs):
            epoch_kls: list[float] = []
            for batch in buffer.sequence_minibatches(
                sequence_length,
                minibatch_size,
                self.training_rng,
                agent_id=agent_id,
            ):
                obs = torch.as_tensor(batch.local_obs, device=self.device)
                hidden = torch.as_tensor(
                    batch.initial_hidden, device=self.device
                )
                raw_actions = torch.as_tensor(
                    batch.raw_actions, device=self.device
                )
                old_log_probs = torch.as_tensor(
                    batch.old_log_probs, device=self.device
                )
                advantages = torch.as_tensor(
                    (batch.advantages - advantage_mean)
                    / (advantage_std + 1e-8),
                    device=self.device,
                )
                mask = torch.as_tensor(
                    batch.mask, dtype=torch.bool, device=self.device
                )
                factor_batch = torch.as_tensor(
                    self._masked_factor(
                        factors, batch.time_indices, batch.mask
                    ),
                    device=self.device,
                )

                evaluated = actor.evaluate_sequence(
                    obs, hidden, raw_actions
                )
                log_ratio = evaluated.log_prob - old_log_probs
                ratio = log_ratio.exp()
                unclipped = ratio * advantages
                clipped = torch.clamp(
                    ratio, 1.0 - clip_coef, 1.0 + clip_coef
                ) * advantages
                policy_loss = -(
                    factor_batch * torch.minimum(unclipped, clipped)
                )[mask].mean()
                action_entropy = evaluated.action_entropy[mask].mean()
                approximate_kl = (
                    (ratio - 1.0) - log_ratio
                )[mask].mean()
                clip_fraction = (
                    (ratio - 1.0).abs() > clip_coef
                )[mask].float().mean()

                intervention_penalty = policy_loss.new_zeros(())
                alignment_penalty = policy_loss.new_zeros(())
                lagrangian_penalty = policy_loss.new_zeros(())
                alignment_diagnostics = {
                    "alignment_valid_fraction": 0.0,
                    "alignment_reference_valid_fraction": 0.0,
                    "alignment_raw_risk_mean": 0.0,
                    "alignment_advantage_mean": 0.0,
                    "alignment_ppo_gradient_norm": 0.0,
                    "alignment_risk_gradient_norm": 0.0,
                    "alignment_risk_to_ppo_gradient_ratio": 0.0,
                }
                gradient_diagnostics = {
                    "intervention_ppo_gradient_norm": 0.0,
                    "intervention_risk_gradient_norm": 0.0,
                    "intervention_risk_to_ppo_gradient_ratio": 0.0,
                    "intervention_risk_ppo_gradient_cosine": 0.0,
                    "intervention_raw_risk_ppo_gradient_cosine": 0.0,
                    "intervention_progress_constraint_violations": 0.0,
                    "intervention_action_ood_rate": 0.0,
                    "intervention_support_fraction": 1.0,
                    "intervention_eligibility_fraction": 1.0,
                    "intervention_risk_active": 0.0,
                }
                risk_active = bool(
                    self.intervention_critic is not None
                    and (
                        self.intervention_aware_mode not in {
                            "stage_counterfactual_v3",
                            "stage_counterfactual_v4",
                            "stage_counterfactual_v5",
                        }
                        or self.current_update > self.intervention_activation_update
                    )
                )
                constrained_update = False
                if risk_active:
                    intervention_penalty = self._intervention_actor_penalty(
                        buffer, batch, actor
                    )
                    if self.intervention_aware_mode == "stage_counterfactual_v2":
                        gradient_diagnostics.update(
                            self._intervention_gradient_diagnostics(
                                policy_loss, intervention_penalty, actor
                            )
                        )
                        gradient_diagnostics["intervention_action_ood_rate"] = (
                            float(self._latest_intervention_ood_rate)
                        )
                    elif self.intervention_aware_mode in {
                        "stage_counterfactual_v3",
                        "stage_counterfactual_v4",
                        "stage_counterfactual_v5",
                    }:
                        gradient_diagnostics["intervention_action_ood_rate"] = (
                            float(self._latest_intervention_ood_rate)
                        )
                        gradient_diagnostics["intervention_support_fraction"] = (
                            float(self._latest_intervention_support_fraction)
                        )
                        gradient_diagnostics["intervention_eligibility_fraction"] = (
                            float(self._latest_intervention_eligibility_fraction)
                        )
                        gradient_diagnostics["intervention_risk_active"] = 1.0
                        constrained_update = True
                if self.safety_alignment_mode == "counterfactual_adapter":
                    alignment_penalty, alignment_diagnostics = (
                        self._safety_alignment_actor_penalty(buffer, batch, actor)
                    )
                    measured = self._intervention_gradient_diagnostics(
                        policy_loss, alignment_penalty, actor
                    )
                    alignment_diagnostics.update({
                        "alignment_ppo_gradient_norm": measured["intervention_ppo_gradient_norm"],
                        "alignment_risk_gradient_norm": measured["intervention_risk_gradient_norm"],
                        "alignment_risk_to_ppo_gradient_ratio": measured["intervention_risk_to_ppo_gradient_ratio"],
                    })
                if self.safety_alignment_mode == "lagrangian":
                    cost_advantage = torch.as_tensor(
                        self._lagrangian_advantage_batch(buffer, batch),
                        device=self.device,
                    )
                    lagrangian_penalty = self.lagrangian_multiplier * (
                        ratio * cost_advantage
                    )[mask].mean()
                actor_loss = (
                    policy_loss - entropy_coef * action_entropy
                    + intervention_penalty + alignment_penalty
                    + lagrangian_penalty
                )

                optimizer.zero_grad()
                if not bool(torch.isfinite(actor_loss)):
                    raise FloatingPointError(
                        "non-finite IA actor loss detected before backward"
                    )
                if constrained_update:
                    # The constrained helper computes gradients before this
                    # zero_grad call, so recompute them after clearing any
                    # stale optimizer state.
                    gradient_diagnostics.update(
                        self._apply_r4_2_constrained_actor_gradients(
                            task_loss=policy_loss - entropy_coef * action_entropy,
                            intervention_penalty=intervention_penalty,
                            buffer=buffer,
                            batch=batch,
                            actor=actor,
                            assign=True,
                        )
                    )
                else:
                    actor_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    actor.parameters(), max_grad_norm,
                    error_if_nonfinite=True,
                )
                optimizer.step()

                values = {
                    "policy_loss": policy_loss,
                    "action_entropy": action_entropy,
                    "approx_kl": approximate_kl,
                    "clip_fraction": clip_fraction,
                    "actor_grad_norm": grad_norm,
                    "intervention_risk_penalty": intervention_penalty,
                    "alignment_penalty": alignment_penalty,
                    "lagrangian_penalty": lagrangian_penalty,
                    **{
                        key: policy_loss.new_tensor(value)
                        for key, value in gradient_diagnostics.items()
                    },
                    **{
                        key: policy_loss.new_tensor(value)
                        for key, value in alignment_diagnostics.items()
                    },
                }
                for key, value in values.items():
                    result[key].append(float(value.detach().cpu()))
                epoch_kls.append(float(approximate_kl.detach().cpu()))
            if epoch_kls and float(np.mean(epoch_kls)) > target_kl:
                break
        return result

    @staticmethod
    def _intervention_gradient_diagnostics(
        policy_loss: torch.Tensor,
        intervention_penalty: torch.Tensor,
        actor,
    ) -> dict[str, float]:
        """Measure the actual minibatch gradient relationship before clipping."""

        parameters = [
            parameter for parameter in actor.parameters()
            if parameter.requires_grad
        ]
        policy_gradients = torch.autograd.grad(
            policy_loss, parameters, retain_graph=True, allow_unused=True
        )
        risk_gradients = torch.autograd.grad(
            intervention_penalty, parameters,
            retain_graph=True, allow_unused=True,
        )
        policy_squared = policy_loss.new_zeros(())
        risk_squared = policy_loss.new_zeros(())
        dot = policy_loss.new_zeros(())
        for parameter, policy_gradient, risk_gradient in zip(
            parameters, policy_gradients, risk_gradients
        ):
            left = (
                torch.zeros_like(parameter)
                if policy_gradient is None else policy_gradient
            )
            right = (
                torch.zeros_like(parameter)
                if risk_gradient is None else risk_gradient
            )
            policy_squared = policy_squared + left.square().sum()
            risk_squared = risk_squared + right.square().sum()
            dot = dot + (left * right).sum()
        policy_norm = policy_squared.sqrt()
        risk_norm = risk_squared.sqrt()
        denominator = (policy_norm * risk_norm).clamp_min(1e-12)
        return {
            "intervention_ppo_gradient_norm": float(policy_norm.detach().cpu()),
            "intervention_risk_gradient_norm": float(risk_norm.detach().cpu()),
            "intervention_risk_to_ppo_gradient_ratio": float(
                (risk_norm / policy_norm.clamp_min(1e-12)).detach().cpu()
            ),
            "intervention_risk_ppo_gradient_cosine": float(
                (dot / denominator).detach().cpu()
            ),
        }

    @staticmethod
    def _flat_gradients(
        parameters: list[torch.nn.Parameter],
        gradients: tuple[torch.Tensor | None, ...],
    ) -> torch.Tensor:
        return torch.cat([
            (
                torch.zeros_like(parameter).reshape(-1)
                if gradient is None else gradient.reshape(-1)
            )
            for parameter, gradient in zip(parameters, gradients)
        ])

    @staticmethod
    def _assign_flat_gradients(
        parameters: list[torch.nn.Parameter], gradient: torch.Tensor,
    ) -> None:
        offset = 0
        for parameter in parameters:
            count = parameter.numel()
            parameter.grad = gradient[offset:offset + count].reshape_as(
                parameter
            ).clone()
            offset += count
        if offset != gradient.numel():
            raise RuntimeError("flat actor gradient length mismatch")

    def _per_agent_subgoal_progress_scores(
        self,
        buffer: RecurrentRolloutBuffer,
        indices: np.ndarray,
        actor,
    ) -> torch.Tensor:
        """Differentiable pre-safety velocity toward each Oracle subgoal."""

        local_obs = torch.as_tensor(
            buffer.local_obs[indices], dtype=torch.float32, device=self.device
        )
        hidden = torch.as_tensor(
            buffer.hidden[indices], dtype=torch.float32, device=self.device
        )
        step_count = local_obs.shape[0]
        distribution, _ = actor._distribution_and_state(
            local_obs.reshape(step_count * self.num_agents, self.obs_dim),
            hidden.reshape(1, step_count * self.num_agents, self.hidden_dim),
        )
        bounded_mean = torch.tanh(distribution.base.mean).reshape(
            step_count, self.num_agents, 2
        )
        base_env = self._base_env()
        velocity = torch.as_tensor(
            buffer.preaction_team_velocity[indices],
            dtype=torch.float32, device=self.device,
        )
        direction = torch.as_tensor(
            buffer.preaction_subgoal_directions[indices],
            dtype=torch.float32, device=self.device,
        )
        candidate = torch.clamp(
            velocity
            + bounded_mean
            * float(base_env.agent_max_acceleration)
            * float(base_env.dt),
            min=-float(base_env.agent_max_speed),
            max=float(base_env.agent_max_speed),
        )
        return (candidate * direction).sum(dim=-1).mean(dim=0)

    def _apply_r4_2_constrained_actor_gradients(
        self,
        *,
        task_loss: torch.Tensor,
        intervention_penalty: torch.Tensor,
        buffer: RecurrentRolloutBuffer,
        batch,
        actor,
        assign: bool = True,
    ) -> dict[str, float]:
        """Apply parameter-space PCGrad and per-UAV subgoal constraints."""

        parameters = [
            parameter for parameter in actor.parameters()
            if parameter.requires_grad
        ]
        task_parts = torch.autograd.grad(
            task_loss, parameters, retain_graph=True, allow_unused=True
        )
        risk_parts = torch.autograd.grad(
            intervention_penalty, parameters,
            retain_graph=True, allow_unused=True,
        )
        task_gradient = self._flat_gradients(parameters, task_parts)
        raw_risk_gradient = self._flat_gradients(parameters, risk_parts)
        indices = np.unique(batch.time_indices[batch.mask])
        indices = indices[indices >= 0]
        progress_scores = self._per_agent_subgoal_progress_scores(
            buffer, indices, actor
        )
        progress_gradients = []
        for agent_id in range(self.num_agents):
            parts = torch.autograd.grad(
                progress_scores[agent_id], parameters,
                retain_graph=agent_id + 1 < self.num_agents,
                allow_unused=True,
            )
            progress_gradients.append(
                self._flat_gradients(parameters, parts)
            )
        projected_risk = project_parameter_risk_gradient(
            raw_risk_gradient, task_gradient, progress_gradients
        )
        if assign:
            self._assign_flat_gradients(
                parameters, task_gradient + projected_risk
            )

        task_norm = torch.linalg.vector_norm(task_gradient)
        risk_norm = torch.linalg.vector_norm(projected_risk)
        raw_norm = torch.linalg.vector_norm(raw_risk_gradient)
        projected_denominator = (task_norm * risk_norm).clamp_min(1e-12)
        raw_denominator = (task_norm * raw_norm).clamp_min(1e-12)
        progress_violations = sum(
            int(float(torch.dot(projected_risk, gradient).detach().cpu()) > 1e-7)
            for gradient in progress_gradients
            if float(gradient.square().sum().detach().cpu()) > 1e-20
        )
        return {
            "intervention_ppo_gradient_norm": float(task_norm.detach().cpu()),
            "intervention_risk_gradient_norm": float(risk_norm.detach().cpu()),
            "intervention_risk_to_ppo_gradient_ratio": float(
                (risk_norm / task_norm.clamp_min(1e-12)).detach().cpu()
            ),
            "intervention_risk_ppo_gradient_cosine": float(
                (torch.dot(task_gradient, projected_risk)
                 / projected_denominator).detach().cpu()
            ),
            "intervention_raw_risk_ppo_gradient_cosine": float(
                (torch.dot(task_gradient, raw_risk_gradient)
                 / raw_denominator).detach().cpu()
            ),
            "intervention_progress_constraint_violations": float(
                progress_violations
            ),
        }

    def _intervention_actor_penalty(
        self, buffer: RecurrentRolloutBuffer, batch, actor,
    ) -> torch.Tensor:
        """Query the frozen critic with the registered R4/R4.1 action path."""

        if self.intervention_critic is None:
            raise RuntimeError("intervention critic is not initialized")
        indices = np.unique(batch.time_indices[batch.mask])
        indices = indices[indices >= 0]
        if not len(indices):
            raise ValueError("risk query minibatch contains no valid rollout steps")
        return float(self.intervention_lambda) * self._intervention_raw_risk(
            buffer, indices, actor
        )

    def _intervention_raw_risk(
        self,
        buffer: RecurrentRolloutBuffer,
        indices: np.ndarray,
        actor,
    ) -> torch.Tensor:
        """Return unscaled frozen-Critic risk for unique rollout steps."""

        if self.intervention_critic is None:
            raise RuntimeError("intervention critic is not initialized")
        local_obs = torch.as_tensor(
            buffer.local_obs[indices], dtype=torch.float32, device=self.device
        )
        hidden = torch.as_tensor(
            buffer.hidden[indices], dtype=torch.float32, device=self.device
        )
        step_count = local_obs.shape[0]
        distribution, _ = actor._distribution_and_state(
            local_obs.reshape(step_count * self.num_agents, self.obs_dim),
            hidden.reshape(1, step_count * self.num_agents, self.hidden_dim),
        )
        raw_actor_mean = distribution.base.mean.reshape(
            step_count, self.num_agents, 2
        )
        if self.intervention_aware_mode in {
            "stage_counterfactual_v2", "stage_counterfactual_v3",
            "stage_counterfactual_v4",
            "stage_counterfactual_v5",
        }:
            behavior_raw = torch.as_tensor(
                buffer.raw_actions[indices], dtype=torch.float32,
                device=self.device,
            )
            raw_mean = raw_actor_mean
            # Forward equals the sampled rollout action, while the straight-
            # through derivative is restricted to the current Actor mean.
            joint_action = straight_through_rollout_action(
                raw_mean, behavior_raw
            )
            if self.intervention_aware_mode == "stage_counterfactual_v2":
                joint_action.register_hook(
                    self._goal_progress_preserving_hook(
                        buffer, indices, joint_action
                    )
                )
        else:
            joint_action = torch.tanh(raw_actor_mean)
        features = self._intervention_critic_features(
            buffer, indices, joint_action
        )
        risk = self.intervention_critic.conservative_risk(features)
        self._latest_intervention_ood_rate = 0.0
        self._latest_intervention_support_fraction = 1.0
        if self.intervention_aware_mode == "stage_counterfactual_v2":
            start = int(self.global_state_dim)
            end = start + 2 * int(self.num_agents)
            mean = self.intervention_critic.feature_mean[start:end]
            std = self.intervention_critic.feature_std[start:end].clamp_min(1e-6)
            normalized_action = (
                features[:, start:end] - mean
            ) / std
            excess = (
                normalized_action.abs() - self.intervention_support_z_limit
            ).clamp_min(0.0)
            support_penalty = excess.square().mean(dim=-1)
            risk = risk + (
                self.intervention_support_penalty_scale * support_penalty
            )
            self._latest_intervention_ood_rate = float(
                (normalized_action.abs().amax(dim=-1)
                 > self.intervention_support_z_limit)
                .float().mean().detach().cpu()
            )
        elif self.intervention_aware_mode in {
            "stage_counterfactual_v3", "stage_counterfactual_v4",
            "stage_counterfactual_v5",
        }:
            start = int(self.global_state_dim)
            end = start + 2 * int(self.num_agents)
            mean = self.intervention_critic.feature_mean[start:end]
            std = self.intervention_critic.feature_std[start:end].clamp_min(1e-6)
            normalized_action = (features[:, start:end] - mean) / std
            in_support = (
                normalized_action.abs().amax(dim=-1)
                <= self.intervention_support_z_limit
            )
            actor_eligible = torch.ones_like(in_support)
            if self.intervention_aware_mode == "stage_counterfactual_v5":
                actor_eligible = torch.as_tensor(
                    buffer.actor_risk_eligible[indices], dtype=torch.bool,
                    device=self.device,
                )
            valid = in_support & actor_eligible
            support_count = valid.float().sum()
            self._latest_intervention_support_fraction = float(
                in_support.float().mean().detach().cpu()
            )
            self._latest_intervention_eligibility_fraction = float(
                actor_eligible.float().mean().detach().cpu()
            )
            self._latest_intervention_ood_rate = (
                1.0 - self._latest_intervention_support_fraction
            )
            # Never extrapolate Critic risk through an unsupported action.
            # OOD rows are excluded rather than pulled toward the late-policy
            # action mean by the R4.1 support penalty.
            risk = (risk * valid.to(risk.dtype)).sum() / support_count.clamp_min(1.0)
            return risk
        return risk.mean()

    def _goal_progress_preserving_hook(
        self,
        buffer: RecurrentRolloutBuffer,
        indices: np.ndarray,
        bounded_action: torch.Tensor,
    ):
        """Project risk gradients so descent cannot reduce any UAV's progress."""

        base_env = self._base_env()
        direction = torch.as_tensor(
            buffer.preaction_goal_directions[indices],
            dtype=torch.float32, device=self.device,
        )
        velocity = torch.as_tensor(
            buffer.preaction_team_velocity[indices],
            dtype=torch.float32, device=self.device,
        )
        candidate = (
            velocity + bounded_action.detach()
            * float(base_env.agent_max_acceleration) * float(base_env.dt)
        )
        unsaturated = (
            candidate.abs() < float(base_env.agent_max_speed) - 1e-6
        ).to(direction.dtype)
        progress_gradient = direction * unsaturated
        def project(gradient: torch.Tensor) -> torch.Tensor:
            # Gradient descent uses -gradient.  A positive gradient component
            # along goal progress would therefore reduce progress; remove it.
            projected = project_risk_gradient_preserving_progress(
                gradient, progress_gradient
            )
            if not bool(torch.isfinite(projected).all()):
                raise FloatingPointError(
                    "non-finite progress-projected intervention gradient"
                )
            return projected

        return project

    def _intervention_critic_features(
        self, buffer: RecurrentRolloutBuffer, indices: np.ndarray,
        joint_mean: torch.Tensor,
    ) -> torch.Tensor:
        """Build the exact R3.2 pre-action 234-D contract for current means."""

        if not self.capture_intervention_features:
            raise RuntimeError("pre-action intervention features were not captured")
        base_env = self._base_env()
        ttc, warning, emergency, clearance = (
            torch_action_conditioned_obstacle_features(
                current_velocity=torch.as_tensor(
                    buffer.preaction_team_velocity[indices],
                    dtype=torch.float32, device=self.device,
                ),
                raw_action=joint_mean,
                obstacle_tracks=torch.as_tensor(
                    buffer.preaction_obstacle_tracks[indices],
                    dtype=torch.float32, device=self.device,
                ),
                agent_radius=float(base_env.agent_radius),
                obstacle_safety_margin=float(
                    base_env.oracle_config["obstacle_safety_margin"]
                ),
                invariant_buffer=float(base_env.boundary_guard.invariant_buffer_m),
                max_acceleration=float(base_env.agent_max_acceleration),
                max_speed=float(base_env.agent_max_speed),
                dt=float(base_env.dt),
                horizon_seconds=float(base_env.obstacle_guard_horizon_seconds),
                warning_ttc_seconds=float(
                    base_env.obstacle_guard_warning_ttc_seconds
                ),
                emergency_ttc_seconds=float(
                    base_env.obstacle_guard_emergency_ttc_seconds
                ),
            )
        )
        if self.intervention_aware_mode in {
            "stage_counterfactual_v2", "stage_counterfactual_v3",
            "stage_counterfactual_v4",
            "stage_counterfactual_v5",
        }:
            smooth_clearance = torch_smooth_minimum_obstacle_clearance(
                current_velocity=torch.as_tensor(
                    buffer.preaction_team_velocity[indices],
                    dtype=torch.float32, device=self.device,
                ),
                raw_action=joint_mean,
                obstacle_tracks=torch.as_tensor(
                    buffer.preaction_obstacle_tracks[indices],
                    dtype=torch.float32, device=self.device,
                ),
                agent_radius=float(base_env.agent_radius),
                obstacle_safety_margin=float(
                    base_env.oracle_config["obstacle_safety_margin"]
                ),
                invariant_buffer=float(base_env.boundary_guard.invariant_buffer_m),
                max_acceleration=float(base_env.agent_max_acceleration),
                max_speed=float(base_env.agent_max_speed),
                dt=float(base_env.dt),
                horizon_seconds=float(base_env.obstacle_guard_horizon_seconds),
                time_samples=self.intervention_smooth_clearance_samples,
                softmin_temperature_m=(
                    self.intervention_smooth_clearance_temperature
                ),
            )
            # Exact R3 feature forward value; smooth bounded backward path.
            clearance_feature = (
                clearance.detach()
                + smooth_clearance - smooth_clearance.detach()
            )
        else:
            clearance_feature = clearance.detach()
        features = torch.cat((
            torch.as_tensor(
                buffer.global_states[indices], dtype=torch.float32,
                device=self.device,
            ),
            joint_mean.reshape(len(indices), -1),
            torch.as_tensor(
                buffer.safety_context[indices], dtype=torch.float32,
                device=self.device,
            ),
            torch.as_tensor(
                buffer.safety_margin[indices, None], dtype=torch.float32,
                device=self.device,
            ),
            torch.as_tensor(
                buffer.preaction_per_agent_margins[indices],
                dtype=torch.float32, device=self.device,
            ).reshape(len(indices), -1),
            # TTC and clearance are recomputed from the current bounded mean,
            # but treated as local critic context.  Their quadratic-root
            # derivative is singular at tangential contact; the registered
            # direct raw-action feature remains the Actor gradient path.
            ttc.detach(), warning.detach(), emergency.detach(),
            clearance_feature,
        ), dim=-1)
        if features.shape[-1] != int(self.intervention_critic.input_dim):
            raise RuntimeError("online intervention feature contract dimension mismatch")
        return features

    def _updated_agent_log_probs(
        self,
        buffer: RecurrentRolloutBuffer,
        agent_id: int,
    ) -> np.ndarray:
        sequence_length = int(self.train_config.get("sequence_length", 16))
        actor = self.actor.actor_for(agent_id)
        result = np.full(len(buffer.rewards), np.nan, dtype=np.float64)
        with torch.no_grad():
            for sequence in buffer.sequence_batches(sequence_length):
                if sequence.agent_id != int(agent_id):
                    continue
                evaluated = actor.evaluate_sequence(
                    torch.as_tensor(
                        sequence.local_obs[:, None, :],
                        device=self.device,
                    ),
                    torch.as_tensor(
                        sequence.initial_hidden, device=self.device
                    ),
                    torch.as_tensor(
                        sequence.raw_actions[:, None, :],
                        device=self.device,
                    ),
                )
                result[sequence.time_indices] = (
                    evaluated.log_prob[:, 0].cpu().numpy()
                )
        if not np.isfinite(result).all():
            raise RuntimeError("updated HAPPO log probabilities are not finite")
        return result

    def _write_happo_factor_failure(
        self,
        *,
        order: list[int],
        agent_id: int,
        old_factors: np.ndarray,
        log_ratio: np.ndarray,
    ) -> None:
        path = (
            Path(self.output_dir)
            / f"happo_factor_failure_update_{self.current_update:06d}.json"
        )
        if path.exists():
            return
        save_json(
            {
                "schema_version": "happo_factor_failure_v1",
                "update": int(self.current_update),
                "agent_update_order": order,
                "failed_agent_id": int(agent_id),
                "old_factor_min": float(np.nanmin(old_factors)),
                "old_factor_max": float(np.nanmax(old_factors)),
                "log_ratio_min": float(np.nanmin(log_ratio)),
                "log_ratio_max": float(np.nanmax(log_ratio)),
            },
            path,
        )

    def _update_critic(
        self, buffer: RecurrentRolloutBuffer
    ) -> tuple[list[float], list[float]]:
        value_coef = float(self.train_config.get("value_coef", 0.5))
        max_grad_norm = float(self.train_config.get("max_grad_norm", 0.5))
        minibatch_size = int(
            self.train_config.get("critic_minibatch_size", 512)
        )
        ppo_epochs = int(self.train_config.get("ppo_epochs", 4))
        value_loss_type = str(
            self.train_config.get("value_loss_type", "huber")
        ).lower()
        huber_delta = float(self.train_config.get("huber_delta", 10.0))
        if value_loss_type not in {"huber", "mse"}:
            raise ValueError("train.value_loss_type must be 'huber' or 'mse'")
        if huber_delta <= 0.0:
            raise ValueError("train.huber_delta must be positive")

        states = torch.as_tensor(
            buffer.global_states, dtype=torch.float32, device=self.device
        )
        returns = torch.as_tensor(
            buffer.returns[:, 0], dtype=torch.float32, device=self.device
        )
        losses: list[float] = []
        grad_norms: list[float] = []
        for _ in range(ppo_epochs):
            chunks = max(1, int(np.ceil(len(states) / minibatch_size)))
            for indices in np.array_split(
                self.training_rng.permutation(len(states)), chunks
            ):
                index = torch.as_tensor(
                    indices, dtype=torch.long, device=self.device
                )
                predicted = self.critic.team_value(states[index])
                if value_loss_type == "huber":
                    value_loss = torch.nn.functional.huber_loss(
                        predicted, returns[index], delta=huber_delta
                    )
                else:
                    value_loss = 0.5 * (
                        predicted - returns[index]
                    ).square().mean()
                self.critic_optimizer.zero_grad()
                (value_coef * value_loss).backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), max_grad_norm
                )
                self.critic_optimizer.step()
                losses.append(float(value_loss.detach().cpu()))
                grad_norms.append(float(grad_norm.detach().cpu()))
        return losses, grad_norms

    @staticmethod
    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    def update(self, buffer: RecurrentRolloutBuffer) -> dict[str, Any]:
        if not buffer.team_gae:
            raise ValueError("multi-agent algorithms require team GAE")
        mean_lagrangian_cost = 0.0
        if self.safety_alignment_mode == "lagrangian":
            team_cost = np.any(
                np.asarray(buffer.raw_stage_costs[:, :2]) > 1e-8,
                axis=1,
            ).astype(np.float32)
            mean_lagrangian_cost = float(team_cost.mean())
            discounted = np.zeros_like(team_cost)
            running = 0.0
            gamma = float(self.train_config.get("gamma", 0.99))
            team_done = np.asarray(buffer.dones[:, 0], dtype=bool)
            for time in range(len(team_cost) - 1, -1, -1):
                running = float(team_cost[time]) + gamma * running * (1.0 - float(team_done[time]))
                discounted[time] = running
            normalized = (discounted - discounted.mean()) / (discounted.std() + 1e-8)
            buffer.safety_cost_advantages = np.repeat(
                normalized[:, None], self.num_agents, axis=1
            ).astype(np.float32)
        team_advantages = np.asarray(buffer.advantages[:, 0], dtype=np.float32)
        advantage_mean = float(team_advantages.mean())
        advantage_std = float(team_advantages.std())
        entropy_coef = self._scheduled_coefficient(
            "action_entropy_coef",
            "action_entropy_coef_start",
            "action_entropy_coef_end",
            "action_entropy_anneal_updates",
            1e-4,
        )
        factors = np.ones(len(team_advantages), dtype=np.float64)
        if self.algorithm == "happo":
            order = [
                int(item)
                for item in self.training_rng.permutation(self.num_agents)
            ]
        else:
            order = list(range(self.num_agents))
        self.last_agent_update_order = order.copy()

        all_actor_stats: list[dict[str, list[float]]] = []
        per_agent_stats: dict[str, float] = {}
        if self.actor_parameter_sharing:
            shared = self._update_actor(
                buffer,
                actor_index=0,
                agent_id=None,
                factors=factors,
                advantage_mean=advantage_mean,
                advantage_std=advantage_std,
                entropy_coef=entropy_coef,
            )
            all_actor_stats.append(shared)
        else:
            for agent_id in order:
                actor_stats = self._update_actor(
                    buffer,
                    actor_index=agent_id,
                    agent_id=agent_id,
                    factors=factors,
                    advantage_mean=advantage_mean,
                    advantage_std=advantage_std,
                    entropy_coef=entropy_coef,
                )
                all_actor_stats.append(actor_stats)
                for key, values in actor_stats.items():
                    per_agent_stats[f"agent_{agent_id}_{key}"] = self._mean(
                        values
                    )
                if self.algorithm == "happo":
                    new_log_probs = self._updated_agent_log_probs(
                        buffer, agent_id
                    )
                    old_log_probs = np.asarray(
                        buffer.log_probs[:, agent_id], dtype=np.float64
                    )
                    log_ratio = new_log_probs - old_log_probs
                    candidate = factors * np.exp(log_ratio)
                    if not np.isfinite(candidate).all():
                        self._write_happo_factor_failure(
                            order=order,
                            agent_id=agent_id,
                            old_factors=factors,
                            log_ratio=log_ratio,
                        )
                        raise RuntimeError(
                            "non-finite HAPPO cumulative factor"
                        )
                    factors = candidate

        value_losses, critic_grad_norms = self._update_critic(buffer)
        combined = {
            key: [value for stats in all_actor_stats for value in stats[key]]
            for key in (
                "policy_loss",
                "action_entropy",
                "approx_kl",
                "clip_fraction",
                "actor_grad_norm",
                "intervention_risk_penalty",
                "intervention_ppo_gradient_norm",
                "intervention_risk_gradient_norm",
                "intervention_risk_to_ppo_gradient_ratio",
                "intervention_risk_ppo_gradient_cosine",
                "intervention_raw_risk_ppo_gradient_cosine",
                "intervention_progress_constraint_violations",
                "intervention_action_ood_rate",
                "intervention_support_fraction",
                "intervention_eligibility_fraction",
                "intervention_risk_active",
                "alignment_penalty",
                "alignment_valid_fraction",
                "alignment_reference_valid_fraction",
                "alignment_raw_risk_mean",
                "alignment_advantage_mean",
                "alignment_ppo_gradient_norm",
                "alignment_risk_gradient_norm",
                "alignment_risk_to_ppo_gradient_ratio",
                "lagrangian_penalty",
            )
        }
        if self.safety_alignment_mode == "lagrangian":
            self.lagrangian_multiplier = float(np.clip(
                self.lagrangian_multiplier
                + self.lagrangian_learning_rate
                * (mean_lagrangian_cost - self.lagrangian_cost_limit),
                0.0,
                self.lagrangian_max,
            ))
        result: dict[str, Any] = {
            "policy_loss": self._mean(combined["policy_loss"]),
            "value_loss": self._mean(value_losses),
            "action_entropy": self._mean(combined["action_entropy"]),
            "entropy": self._mean(combined["action_entropy"]),
            "approx_kl": self._mean(combined["approx_kl"]),
            "clip_fraction": self._mean(combined["clip_fraction"]),
            "actor_grad_norm": self._mean(combined["actor_grad_norm"]),
            "critic_grad_norm": self._mean(critic_grad_norms),
            "intervention_risk_penalty": self._mean(
                combined["intervention_risk_penalty"]
            ),
            "intervention_ppo_gradient_norm": self._mean(
                combined["intervention_ppo_gradient_norm"]
            ),
            "intervention_risk_gradient_norm": self._mean(
                combined["intervention_risk_gradient_norm"]
            ),
            "intervention_risk_to_ppo_gradient_ratio": self._mean(
                combined["intervention_risk_to_ppo_gradient_ratio"]
            ),
            "intervention_risk_to_ppo_gradient_ratio_p95": float(
                np.quantile(
                    combined["intervention_risk_to_ppo_gradient_ratio"], 0.95
                )
            ),
            "intervention_risk_ppo_gradient_cosine": self._mean(
                combined["intervention_risk_ppo_gradient_cosine"]
            ),
            "intervention_risk_ppo_gradient_cosine_p05": float(
                np.quantile(
                    combined["intervention_risk_ppo_gradient_cosine"], 0.05
                )
            ),
            "intervention_raw_risk_ppo_gradient_cosine": self._mean(
                combined["intervention_raw_risk_ppo_gradient_cosine"]
            ),
            "intervention_raw_risk_ppo_gradient_cosine_p05": float(
                np.quantile(
                    combined["intervention_raw_risk_ppo_gradient_cosine"],
                    0.05,
                )
            ),
            "intervention_progress_constraint_violations": float(
                np.sum(
                    combined["intervention_progress_constraint_violations"]
                )
            ),
            "intervention_action_ood_rate": self._mean(
                combined["intervention_action_ood_rate"]
            ),
            "intervention_action_ood_rate_max": float(
                np.max(combined["intervention_action_ood_rate"])
            ),
            "intervention_support_fraction": self._mean(
                combined["intervention_support_fraction"]
            ),
            "intervention_support_fraction_min": float(
                np.min(combined["intervention_support_fraction"])
            ),
            "intervention_eligibility_fraction": self._mean(
                combined["intervention_eligibility_fraction"]
            ),
            "intervention_eligibility_fraction_min": float(
                np.min(combined["intervention_eligibility_fraction"])
            ),
            "intervention_risk_active": float(
                np.max(combined["intervention_risk_active"])
            ),
            "alignment_penalty": self._mean(combined["alignment_penalty"]),
            "alignment_valid_fraction": self._mean(combined["alignment_valid_fraction"]),
            "alignment_reference_valid_fraction": self._mean(combined["alignment_reference_valid_fraction"]),
            "alignment_raw_risk_mean": self._mean(combined["alignment_raw_risk_mean"]),
            "alignment_advantage_mean": self._mean(combined["alignment_advantage_mean"]),
            "alignment_ppo_gradient_norm": self._mean(combined["alignment_ppo_gradient_norm"]),
            "alignment_risk_gradient_norm": self._mean(combined["alignment_risk_gradient_norm"]),
            "alignment_risk_to_ppo_gradient_ratio": self._mean(combined["alignment_risk_to_ppo_gradient_ratio"]),
            "lagrangian_penalty": self._mean(combined["lagrangian_penalty"]),
            "lagrangian_cost": mean_lagrangian_cost,
            "lagrangian_cost_limit": self.lagrangian_cost_limit,
            "lagrangian_multiplier": self.lagrangian_multiplier,
            "safety_alignment_mode": self.safety_alignment_mode,
            "intervention_aware_mode": self.intervention_aware_mode,
            "intervention_lambda": self.intervention_lambda,
            "action_entropy_coef": float(entropy_coef),
            "agent_update_order": order,
            "happo_factor_min": float(np.min(factors)),
            "happo_factor_max": float(np.max(factors)),
            "happo_factor_mean": float(np.mean(factors)),
            **per_agent_stats,
            **self.actor.distribution_stats(),
        }
        if not all(
            np.isfinite(value)
            for value in result.values()
            if isinstance(value, (float, int))
        ):
            raise RuntimeError("non-finite MAPPO/HAPPO update metric")
        return result

