"""双无人机中央联合循环 PPO 训练与评估循环。"""

from __future__ import annotations

import json
import hashlib
from copy import deepcopy
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import optim

from src.algorithms.ppo.networks import CentralizedCritic, RecurrentActor
from src.algorithms.ppo.capture_sampler import (
    file_sha256, load_anchor, load_bank_manifest, warm_capture_hidden,
)
from src.algorithms.ppo.capture_schedule import CaptureSchedule
from src.algorithms.ppo.observation_normalizer import UAVObservationNormalizer
from src.algorithms.ppo.rollout_buffer import RecurrentRolloutBuffer
from src.algorithms.ppo.training_logger import (
    configure_training_logger,
    format_duration,
)
from src.algorithms.intervention_data import (
    InterventionDatasetAccumulatorV1,
    TRAINABLE_STAGES,
    _causal_context_from_snapshot,
)
from src.algorithms.intervention_team_data import (
    padded_snapshot_obstacle_tracks,
    per_agent_snapshot_margins,
)
from src.algorithms.intervention_r3 import (
    R3ShadowDatasetAccumulatorV1,
    evaluate_r3_shadow_step,
)
from src.envs.cooperative_uav_env import CooperativeUAVEnv
from src.safety.staged_execution.fixed_point_attribution import (
    ACTION_STAGE_ORDER,
    classify_fixed_point_attribution,
)
from src.safety.counterfactual import (
    evaluate_avoidable_intervention_v2,
    registered_reference_actions,
)
from src.utils.io_utils import (
    append_jsonl,
    ensure_dir,
    read_jsonl,
    save_json,
    save_jsonl,
)
from src.utils.seed_utils import set_global_seeds


GUIDANCE_METRIC_KEYS = (
    "subgoal_reach_rate",
    "subgoal_progress_ratio",
    "mean_subgoal_distance",
    "mean_steps_per_subgoal",
    "subgoal_timeout_rate",
    "invalid_subgoal_rate",
    "subgoal_replan_count",
    "early_replan_count",
    "subgoal_collision_rate",
    "max_goal_progress_ratio",
    "halfway_reach_rate",
    "near_goal_reach_rate",
    "mean_max_x",
    "half_map_entry_rate",
    "oracle_recovery_attempt_count",
    "oracle_recovery_candidate_count",
    "oracle_recovery_success_count",
    "oracle_recovery_success_rate",
    "previous_target_hold_count",
    "previous_target_hold_rate",
    "blocked_final_goal_fallback_count",
    "blocked_primary_candidate_rate",
    "recovery_activation_rate",
    "stagnation_trigger_rate",
    "repeated_direction_rate",
    "shield_intervention_count",
    "joint_guard_intervention_count",
    "final_joint_guard_residual_count",
    "obstacle_ttc_projection_count",
    "obstacle_ttc_fallback_count",
    "obstacle_ttc_post_action_goal_velocity_sum",
    "obstacle_ttc_post_action_goal_velocity_count",
    "total_action_modification_count",
    "goal_corridor_projection_count",
    "goal_corridor_agent_projection_count",
    "goal_corridor_tangential_progress_m",
    "goal_corridor_nonpositive_progress_count",
    "shield_invariant_violation_count",
    "safe_brake_step_count",
    "safe_brake_entry_count",
    "safe_brake_exit_count",
    "max_safe_brake_streak",
    "pair_infeasible_count",
    "pair_infeasible_entry_count",
    "safe_recover_step_count",
    "safe_recover_entry_count",
    "safe_recover_exit_count",
    "safe_recovery_failure_count",
    "safe_recovery_unavailable_count",
    "safe_recovery_retry_count",
    "safe_recovery_abort_count",
    "safe_recovery_hard_abort_count",
    "safe_recovery_refresh_count",
    "safe_recovery_reentry_count",
    "safe_recovery_abort_then_success_count",
    "safe_recovery_affected_agent_count",
    "recovery_incident_count",
    "recovery_incident_merge_count",
    "recovery_incident_reentry_reset_count",
    "recovery_duplicate_hard_abort_count",
    "max_recovery_incident_steps",
    "terminal_recovery_unresolved",
    "recovery_watchdog_step_count",
    "max_recovery_direction_unavailable_streak",
    "recovery_incident_close_count",
    "recovery_incident_reopen_count",
    "hazard_retrigger_count",
    "same_hazard_reentry_count",
    "terminal_incident_reconciled_count",
    "clearance_pending_steps",
    "pending_clearance_reappearance_count",
    "post_certificate_retrigger_count",
    "close_without_certificate_count",
    "stale_closure_bypass_count",
    "release_certificate_count",
    "escape_steps",
    "consolidation_steps",
    "release_shadow_check_count",
    "release_shadow_check_failure_count",
    "bounded_recovery_exhaustion_count",
    "bounded_exhaustion_then_success_count",
    "rapid_same_hazard_retrigger_count",
    "recovery_lineage_failure_count",
    "recovery_lineage_budget_reset_count",
    "recovery_duplicate_failure_record_count",
    "max_recovery_lineage_steps",
    "max_rapid_retriggers_per_lineage",
    "recovery_candidate_selection_count",
    "recovery_candidate_rejection_count",
    "max_safe_recover_streak",
    "boundary_recovery_trigger_count",
)


_SAFETY_ABLATION_SWITCHES = (
    ("recovery_enabled",),
    ("initial_joint_guard", "enabled"),
    ("initial_joint_guard", "action_shield_enabled"),
    ("obstacle_ttc", "enabled"),
    ("final_joint_guard", "enabled"),
)


def _registered_safety_ablation_changes(
    checkpoint_contract: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
) -> dict[str, dict[str, bool]] | None:
    """Accept only preregistered Boolean safety-module switches."""

    parent = deepcopy(dict(checkpoint_contract))
    child = deepcopy(dict(runtime_contract))
    parent_safety = parent.setdefault("safety_execution", {})
    child_safety = child.setdefault("safety_execution", {})
    changes: dict[str, dict[str, bool]] = {}
    for path in _SAFETY_ABLATION_SWITCHES:
        parent_cursor = parent_safety
        child_cursor = child_safety
        for key in path[:-1]:
            parent_cursor = parent_cursor.setdefault(key, {})
            child_cursor = child_cursor.setdefault(key, {})
        leaf = path[-1]
        parent_value = bool(parent_cursor.pop(leaf, False))
        child_value = bool(child_cursor.pop(leaf, False))
        if parent_value != child_value:
            changes["safety_execution." + ".".join(path)] = {
                "checkpoint": parent_value,
                "runtime": child_value,
            }
    if parent != child or not changes:
        return None
    return changes


TRAIN_METRICS_SCHEMA_VERSION = "train_metrics_v2_compact"

# These aliases are retained inside the environment/trainer for historical
# diagnostics, but storing both copies in every training row makes long runs
# unnecessarily noisy.  The scale expresses alias == scale * canonical.
TRAIN_METRIC_ALIASES = (
    ("success_rate", "reach_success_rate", 1.0),
    ("entropy", "action_entropy", 1.0),
)
TRAIN_REWARD_COMPONENT_ALIASES = (
    ("individual_navigation_reward", "progress_reward", 1.0),
    ("navigation_reward", "progress_reward", 1.0),
    ("cooperation_bonus", "cooperation_reward", 1.0),
    ("obstacle_collision_penalty", "obstacle_reward", -1.0),
    ("inter_agent_collision_penalty", "inter_agent_reward", -1.0),
    ("boundary_collision_penalty", "boundary_reward", -1.0),
    ("timeout_penalty", "timeout_reward", -1.0),
    ("proximity_risk_penalty", "proximity_risk_reward", -1.0),
    ("control_effort_penalty", "control_effort_reward", -1.0),
    ("terminal_bonus", "goal_bonus", 1.0),
    ("terminal_reward", "goal_bonus", 1.0),
)


OOD_EVALUATION_ENV_KEYS = frozenset({"static_obstacle_num"})
BOUNDARY_SIDES = ("left", "right", "top", "bottom")


class PPOTrainer:
    """中央联合循环 PPO 训练器。

    重要重构：
    * ``num_agents`` 全部从环境读取，不再硬编码4；
    * Actor只优化物理动作，不再存在通信消息分支；
    * checkpoint 会记录当前智能体数量和观测维度，改变这些配置后旧模型会被拒绝；
    * 日志保留奖励分量，方便后续分析奖励量级是否失衡。
    """

    POLICY_SCHEMA_VERSION = 9
    ALGORITHM = "ppo"
    JOINT_POLICY = True
    TEAM_GAE = False

    def __init__(
        self,
        config: Mapping[str, Any],
        project_root: str | Path | None = None,
    ):
        self.config = deepcopy(dict(config))
        self.algorithm = str(self.config.get("algorithm", "ppo")).lower()
        if self.algorithm != self.ALGORITHM:
            raise ValueError(f"algorithm must be {self.ALGORITHM}")
        self.joint_policy = bool(self.JOINT_POLICY)
        if self.config.get("env", {}).get("reward_mode") not in {
            "team_v1",
            "team_v2_risk",
            "team_v3_potential",
            "team_v4_subgoal",
        }:
            raise ValueError(
                "centralized PPO requires a team reward mode"
            )

        self.project_root = Path(project_root or Path.cwd()).resolve()
        self.seed = int(self.config.get("seed", 0))
        set_global_seeds(self.seed)

        device_name = str(self.config.get("device", "cpu"))
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA training was requested but no CUDA device is available"
            )
        self.device = torch.device(device_name)

        # ---------------------------- 环境 ----------------------------
        env_config = dict(self.config.get("env", {}))
        runtime_cfg = dict(self.config.get("runtime", {}))
        fastpath_cfg = dict(runtime_cfg.get("serial_fastpath", {}))
        self.serial_fastpath_enabled = bool(fastpath_cfg.get("enabled", False))
        self.training_telemetry_level = str(
            fastpath_cfg.get("telemetry_level", "full")
        )
        self.evaluation_telemetry_level = str(
            fastpath_cfg.get("evaluation_telemetry_level", "full")
        )
        valid_telemetry_levels = {"full", "train_minimal"}
        if self.training_telemetry_level not in valid_telemetry_levels:
            raise ValueError("invalid training telemetry level")
        if self.evaluation_telemetry_level != "full":
            raise ValueError("evaluation telemetry level must be full")
        self.consolidate_rollout_device_transfer = bool(
            fastpath_cfg.get("consolidate_rollout_device_transfer", False)
        )
        env_config["runtime"] = runtime_cfg
        env_config["reward"] = dict(
            self.config.get("reward", env_config.get("reward", {}))
        )
        # Oracle 配置只透传给环境，不进入 Actor/Critic 或 PPO 超参数。
        env_config["oracle_subgoal"] = dict(
            self.config.get(
                "oracle_subgoal", env_config.get("oracle_subgoal", {})
            )
        )
        env_config["subgoal_selector"] = dict(
            self.config.get(
                "subgoal_selector", env_config.get("subgoal_selector", {})
            )
        )
        # Shared safety is a top-level experimental condition, like the
        # frozen Oracle.  Explicitly transmit it into the environment contract
        # so the environment cannot silently fall back to a disabled stack.
        env_config["safety_execution"] = deepcopy(dict(
            self.config.get(
                "safety_execution", env_config.get("safety_execution", {})
            )
        ))
        if any(
            env_config["subgoal_selector"].get(key)
            for key in (
                "competence_model_artifact", "pairwise_progress_artifact",
            )
        ):
            raise ValueError("archived selector artifacts cannot be loaded")
        self.env = CooperativeUAVEnv(env_config)
        requested_safety = dict(env_config["safety_execution"])
        requested_safety_enabled = bool(requested_safety.get("enabled", False))
        actual_safety_contract = {
            "safety_execution_enabled": bool(
                self.env.safety_execution_enabled
            ),
            "initial_joint_guard_enabled": bool(
                self.env.initial_joint_guard_enabled
            ),
            "obstacle_ttc_enabled": bool(
                self.env.obstacle_execution_guard_enabled
            ),
            "final_joint_guard_enabled": bool(
                self.env.final_joint_guard_enabled
            ),
            "recovery_enabled": bool(self.env.safety_recovery_enabled),
        }
        requested_safety_contract = {
            "safety_execution_enabled": requested_safety_enabled,
            "initial_joint_guard_enabled": bool(
                requested_safety.get("initial_joint_guard", {}).get(
                    "enabled", False
                )
            ),
            "obstacle_ttc_enabled": bool(
                requested_safety.get("obstacle_ttc", {}).get(
                    "enabled", False
                )
            ),
            "final_joint_guard_enabled": bool(
                requested_safety.get("final_joint_guard", {}).get(
                    "enabled", False
                )
            ),
            "recovery_enabled": bool(
                requested_safety.get("recovery_enabled", False)
            ),
        }
        self.runtime_contract_assertion = {
            "schema_version": "shared_safety_runtime_contract_v1",
            "requested": requested_safety_contract,
            "actual": actual_safety_contract,
            "passed": True,
        }
        if requested_safety_enabled:
            mismatches = [
                key for key, expected in requested_safety_contract.items()
                if expected and not actual_safety_contract[key]
            ]
            if mismatches:
                self.runtime_contract_assertion["passed"] = False
                raise RuntimeError(
                    "shared safety runtime contract mismatch: "
                    + ", ".join(mismatches)
                )
        initial_obs, _ = self.env.reset(seed=self.seed)
        self.num_agents = self.env.num_agents
        self.policy_rows = 1 if self.joint_policy else self.num_agents
        self.obs_dim = int(initial_obs.shape[1])
        self.global_state_dim = int(self.env.global_state_dim)
        self.policy_obs_dim = self.global_state_dim if self.joint_policy else self.obs_dim
        self.policy_action_dim = 2 * self.num_agents if self.joint_policy else 2
        from src.llm_refinement.reward_wrapper import wrap_configured_reward
        self.env = wrap_configured_reward(self.env, self.config)
        default_normalization_version = (
            UAVObservationNormalizer.TRACKED_VERSION
            if self.env.observation_schema_version == "tracked_obstacle_v2"
            else UAVObservationNormalizer.VERSION
        )
        normalization_cfg = {
            "enabled": True,
            "version": default_normalization_version,
        } | dict(self.config.get("observation_normalization", {}))
        if normalization_cfg["version"] not in UAVObservationNormalizer.SUPPORTED_VERSIONS:
            raise ValueError("unsupported observation normalization version")
        self.config["observation_normalization"] = normalization_cfg
        self.observation_normalizer = UAVObservationNormalizer(
            env_config,
            enabled=bool(normalization_cfg["enabled"]),
            version=str(normalization_cfg["version"]),
        )

        # ---------------------------- 网络 ----------------------------
        model_cfg = dict(self.config.get("model", {}))
        train_cfg = dict(self.config.get("train", {}))
        self.train_config = train_cfg
        self.hidden_dim = int(model_cfg.get("hidden_dim", 128))

        self._initialize_networks(model_cfg, train_cfg)

        # ---------------------------- 输出 ----------------------------
        output_cfg = dict(self.config.get("output", {}))
        self.output_dir = ensure_dir(
            self._resolve(output_cfg.get("output_dir", "results/ppo"))
        )
        self.checkpoint_dir = ensure_dir(
            self._resolve(
                output_cfg.get(
                    "checkpoint_dir",
                    "results/ppo/checkpoints",
                )
            )
        )
        self.log_every = int(self.train_config.get("log_every", 1))
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")
        self.logger = configure_training_logger(self.output_dir, self.algorithm)

        self.current_update = 0
        self.training_rng = np.random.default_rng(self.seed + 2_027)
        self.best_model_key: tuple[float, ...] | None = None
        self.loaded_checkpoint_task_config_mismatch = False
        self.generic_task_config_override_used = False
        self.selector_intervention_used = False
        self.ood_evaluation_intervention_used = False
        self.ood_evaluation_changes: dict[str, dict[str, Any]] = {}
        self.safety_ablation_intervention_used = False
        self.safety_ablation_changes: dict[str, dict[str, bool]] = {}
        self.parent_task_config_fingerprint: str | None = None
        self.curriculum_history: list[dict[str, Any]] = []
        self._resumed_from_checkpoint = False
        self._manifest_written = False
        self._loaded_checkpoint_hash: str | None = None
        self.capture_bank_path: Path | None = None
        self.capture_bank: dict[str, Any] | None = None
        self.capture_schedule: CaptureSchedule | None = None
        self.capture_steps_cumulative = 0
        capture_cfg = dict(self.train_config.get("capture_sampler", {}))
        if bool(capture_cfg.get("enabled", False)):
            probability = float(capture_cfg.get("probability", 0.5))
            scheduled = bool(capture_cfg.get("schedule"))
            if not (0.0 < probability <= 1.0 if scheduled
                    else 0.0 < probability < 1.0):
                raise ValueError("train.capture_sampler.probability is outside valid range")
            max_step_fraction = float(capture_cfg.get("max_step_fraction", 0.2))
            if not 0.0 < max_step_fraction <= 0.5:
                raise ValueError("train.capture_sampler.max_step_fraction must be in (0, 0.5]")
            if scheduled:
                self.capture_schedule = CaptureSchedule.from_config(
                    dict(capture_cfg["schedule"]),
                    int(self.train_config.get("total_updates", 1)),
                    max_step_fraction,
                )
            self.capture_bank_path = self._resolve(capture_cfg["bank_manifest"]).resolve()
            capture_cfg["bank_manifest"] = str(self.capture_bank_path)
            capture_cfg["bank_manifest_sha256"] = file_sha256(self.capture_bank_path)
            capture_cfg["probability"] = probability
            capture_cfg["max_step_fraction"] = max_step_fraction
            self.train_config["capture_sampler"] = capture_cfg
            self.config["train"] = deepcopy(self.train_config)
            self.capture_bank = load_bank_manifest(
                self.capture_bank_path, self.base_task_config_fingerprint()
            )
            if self.env.env is not self._base_env():
                raise ValueError("A7 capture sampler requires one reward wrapper")
            if self.env.reward_fn is not None:
                raise ValueError("A7 capture bank does not version diagnostic reward state")

    def _initialize_networks(
        self,
        model_cfg: Mapping[str, Any],
        train_cfg: Mapping[str, Any],
    ) -> None:
        """Build the legacy joint policy without changing its checkpoint schema."""
        self.actor = RecurrentActor(
            self.policy_obs_dim,
            self.hidden_dim,
            action_dim=self.policy_action_dim,
            initial_action_log_std=float(
                model_cfg.get("initial_action_log_std", -1.2)
            ),
            log_std_min=float(train_cfg.get("log_std_min", -4.0)),
            log_std_max=float(train_cfg.get("log_std_max", -0.3)),
        ).to(self.device)
        self.critic = CentralizedCritic(
            self.global_state_dim,
            self.hidden_dim,
            self.policy_rows,
        ).to(self.device)
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=float(train_cfg.get("actor_learning_rate", 3e-4)),
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=float(train_cfg.get("critic_learning_rate", 1e-4)),
        )

    @property
    def algorithm_variant(self) -> str:
        return "joint_ppo"

    def actor_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.actor.parameters())

    def critic_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.critic.parameters())

    def _environment_contract_payload(self) -> dict[str, Any]:
        payload = self._task_config_payload()
        result: dict[str, Any] = {"environment_contract_version": 2}
        for key in (
            "env",
            "reward",
            "oracle_subgoal",
            "subgoal_selector",
            "safety_execution",
            "observation_normalization",
            "terminal_contact_semantics_version",
        ):
            if key in payload:
                result[key] = deepcopy(payload[key])
        return result

    def environment_contract_fingerprint(self) -> str:
        canonical = json.dumps(
            self._environment_contract_payload(),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _learner_contract_payload(self) -> dict[str, Any]:
        task = self._task_config_payload()
        return {
            "learner_contract_version": 1,
            "algorithm": self.algorithm,
            "algorithm_variant": self.algorithm_variant,
            "model": deepcopy(task.get("model", {})),
            "ppo_contract": deepcopy(task.get("ppo_contract", {})),
        }

    def learner_contract_fingerprint(self) -> str:
        canonical = json.dumps(
            self._learner_contract_payload(),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _write_run_manifest(self) -> None:
        """Persist the resolved training task instead of relying on YAML names."""
        save_json(
            {
                "schema_version": 1,
                "algorithm": self.algorithm,
                "algorithm_variant": self.algorithm_variant,
                "seed": self.seed,
                "device": str(self.device),
                "num_agents": self.num_agents,
                "obs_dim": self.obs_dim,
                "global_state_dim": self.global_state_dim,
                "action_dim": self.policy_action_dim,
                "task_config_fingerprint": self.task_config_fingerprint(),
                "environment_contract_fingerprint": (
                    self.environment_contract_fingerprint()
                ),
                "learner_contract_fingerprint": (
                    self.learner_contract_fingerprint()
                ),
                "actor_parameter_count": self.actor_parameter_count(),
                "critic_parameter_count": self.critic_parameter_count(),
                "parent_task_config_fingerprint": self.parent_task_config_fingerprint,
                "selector_intervention_used": self.selector_intervention_used,
                "runtime_contract_assertion": deepcopy(
                    getattr(self, "runtime_contract_assertion", None)
                ),
                "curriculum_history": self.curriculum_history,
                "task_config": self._task_config_payload(),
                "resolved_config": self.config,
            },
            self.output_dir / "run_manifest.json",
        )

    def _base_env(self) -> CooperativeUAVEnv:
        env = self.env
        while hasattr(env, "env"):
            env = env.env
        return env

    @staticmethod
    def _rollout_step_host_arrays(
        hidden_before: torch.Tensor,
        action: torch.Tensor,
        raw_action: torch.Tensor,
        log_prob: torch.Tensor,
        values: torch.Tensor,
    ) -> dict[str, np.ndarray]:
        """Copy all per-step policy tensors to host with one synchronization."""

        tensors = {
            "hidden": hidden_before,
            "action": action,
            "raw_action": raw_action,
            "log_prob": log_prob,
            "values": values,
        }
        shapes = {key: tuple(value.shape) for key, value in tensors.items()}
        sizes = {key: int(value.numel()) for key, value in tensors.items()}
        packed = torch.cat([
            value.detach().reshape(-1) for value in tensors.values()
        ]).cpu().numpy()
        result: dict[str, np.ndarray] = {}
        offset = 0
        for key in tensors:
            next_offset = offset + sizes[key]
            result[key] = packed[offset:next_offset].reshape(shapes[key])
            offset = next_offset
        return result

    def policy_fingerprint(self) -> str:
        """Hash the exact actor weights used to collect an attempt."""
        digest = hashlib.sha256()
        for name, tensor in sorted(self.actor.state_dict().items()):
            values = tensor.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(values.shape)).encode("ascii"))
            digest.update(values.numpy().tobytes())
        return digest.hexdigest()

    @staticmethod
    def _directory_has_material_content(path: Path) -> bool:
        for child in path.iterdir():
            # Logger construction necessarily creates this file before train().
            if child.name == "train.log" and child.is_file():
                continue
            if child.name == "checkpoints" and child.is_dir() and not any(child.iterdir()):
                continue
            return True
        return False

    def _assert_output_contract(self) -> None:
        """Reject accidental evidence mixing before any training write."""
        if not bool(self.config.get("output", {}).get("reject_nonempty", True)):
            return
        manifest_path = self.output_dir / "run_manifest.json"
        if self._resumed_from_checkpoint:
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest_fingerprint = manifest.get("task_config_fingerprint")
                allowed = {self.task_config_fingerprint()}
                if self.selector_intervention_used and self.parent_task_config_fingerprint:
                    allowed.add(self.parent_task_config_fingerprint)
                if manifest_fingerprint not in allowed:
                    raise ValueError(
                        "resume output manifest does not match checkpoint/runtime task fingerprint"
                    )
                if (
                    self.selector_intervention_used
                    and manifest_fingerprint == self.parent_task_config_fingerprint
                    and manifest_fingerprint != self.task_config_fingerprint()
                ):
                    raise ValueError(
                        "selector intervention requires a new child output directory"
                    )
            elif self._directory_has_material_content(self.output_dir):
                raise ValueError(
                    "resume into a non-empty output directory requires run_manifest.json"
                )
            return
        if self._manifest_written:
            return
        if self._directory_has_material_content(self.output_dir):
            raise FileExistsError(
                f"refusing to train into non-empty output directory: {self.output_dir}"
            )

    def _set_target_obstacles(self) -> None:
        """Apply the fixed target obstacle counts for training and evaluation."""
        counts = dict(self.config.get("env", {}))
        env = self.env
        while hasattr(env, "env"):
            env = env.env
        env.static_obstacle_num = int(counts.get("static_obstacle_num", counts.get("obstacle_num", env.static_obstacle_num)))
        env.dynamic_obstacle_num = int(counts.get("dynamic_obstacle_num", 0 if "obstacle_num" in counts else env.dynamic_obstacle_num))
        env.obstacle_num = env.static_obstacle_num + env.dynamic_obstacle_num
        env.static_obstacles = env.dynamic_obstacle_num == 0

    def policy_observations(self, observations) -> np.ndarray:
        normalized = self.normalize_local_observations(observations)
        return normalized.reshape(1, -1) if self.joint_policy else normalized

    def policy_rewards(self, rewards) -> np.ndarray:
        values = np.asarray(rewards, dtype=np.float32)
        # team_v1 broadcasts the full team collision penalty before any shaping.
        return np.asarray([values.mean()], dtype=np.float32) if self.joint_policy else values

    def _resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.project_root / path

    def normalize_local_observations(self, observations) -> np.ndarray:
        """训练、评估和回放共用同一无状态缩放。"""
        return self.observation_normalizer.transform(observations)

    def normalize_global_state(self, state) -> np.ndarray:
        values = np.asarray(state, dtype=np.float32).reshape(
            self.num_agents, self.obs_dim
        )
        return self.normalize_local_observations(values).reshape(-1)

    def _task_config_payload(self) -> dict[str, Any]:
        """只包含会改变任务语义或策略输入的配置。"""
        env_config = dict(self.config.get("env", {}))
        # 渲染分辨率不改变 MDP，避免仅修改视频尺寸就错误拒绝 checkpoint。
        for key in ("render_width", "render_height", "render_mode"):
            env_config.pop(key, None)
        ppo_contract_keys = (
            "rollout_steps", "ppo_epochs", "sequence_length",
            "sequence_minibatch_size", "critic_minibatch_size", "target_kl",
            "gamma", "gae_lambda", "clip_coef", "value_coef",
            "action_entropy_coef", "action_entropy_coef_start",
            "action_entropy_coef_end", "action_entropy_anneal_updates",
            "value_loss_type", "huber_delta", "log_std_min", "log_std_max",
            "actor_learning_rate", "critic_learning_rate", "max_grad_norm",
            "capture_sampler",
        )
        train_config = dict(self.config.get("train", {}))
        payload = {
            "task_contract_version": 4,
            "env": env_config,
            "reward": dict(self.config.get("reward", {})),
            "oracle_subgoal": {
                key: value
                for key, value in dict(self.config.get("oracle_subgoal", {})).items()
                if key != "generation_enabled" or value is False
            },
            "subgoal_selector": dict(self.config.get("subgoal_selector", {})),
            "safety_execution": deepcopy(
                self.config.get("safety_execution", {})
            ),
            "observation_normalization": dict(
                self.config.get("observation_normalization", {})
            ),
            "model": dict(self.config.get("model", {})),
            "ppo_contract": {
                key: train_config[key]
                for key in ppo_contract_keys
                if key in train_config
            },
        }
        if self.env.collision_mode == "terminal":
            # Version exact endpoint wall contact and first-side-only telemetry.
            # Intermediate terminal checkpoints from before this correction are
            # not interchangeable even if their YAML files look identical.
            payload["terminal_contact_semantics_version"] = 2
        return payload

    def _checkpoint_extension_payload(self) -> dict[str, Any]:
        """Algorithm-specific checkpoint fields.

        The base PPO schema remains unchanged when subclasses do not opt in.
        """

        return {}

    def _load_checkpoint_extension(self, payload: Mapping[str, Any]) -> None:
        """Validate and restore subclass state after the common payload."""

        del payload

    def task_config_fingerprint(self) -> str:
        canonical = json.dumps(
            self._task_config_payload(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def base_task_config_fingerprint(self) -> str:
        """Compare a capture bank to the original PPO/MDP contract."""
        payload = self._task_config_payload()
        payload["ppo_contract"].pop("capture_sampler", None)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _start_training_episode(self, rollout_seed: int, full_reset_count: int,
                                capture_steps: int, capture_step_budget: int,
                                *, force_full: bool = False):
        """Return a full reset or a legal terminal reset with warmed Actor GRU."""
        if self.capture_schedule is None:
            # Preserve the A7 random draw sequence when no A8 schedule is used.
            use_anchor = self.capture_bank is not None and not force_full and (
                self.training_rng.random()
                < float(self.train_config["capture_sampler"]["probability"])
            )
        else:
            probability = float(self.train_config["capture_sampler"]["probability"])
            use_anchor = (self.capture_bank is not None and not force_full
                          and capture_steps < capture_step_budget
                          and (probability == 1.0 or self.training_rng.random() < probability))
        if use_anchor:
            assert self.capture_bank_path is not None
            entries = [item for item in self.capture_bank["anchors"]
                       if int(item["remaining_steps"]) <= capture_step_budget - capture_steps]
            if not entries:
                use_anchor = False
        if use_anchor:
            entry = entries[int(self.training_rng.integers(len(entries)))]
            env, obs, history = load_anchor(self.capture_bank_path, entry)
            self.env.env = env
            self.env._reward_state = {}
            hidden = warm_capture_hidden(self, history)
            return obs, hidden, "capture_bank", entry
        obs, _ = self.env.reset(seed=rollout_seed + full_reset_count)
        hidden = self.actor.initial_hidden(self.policy_rows, self.device)
        return obs, hidden, "full_reset", None

    def _scheduled_coefficient(
        self,
        legacy_name: str,
        start_name: str,
        end_name: str,
        anneal_name: str,
        default: float,
    ) -> float:
        """线性退火某个训练系数，同时兼容旧的单值配置。"""
        start = float(
            self.train_config.get(
                start_name,
                self.train_config.get(legacy_name, default),
            )
        )
        end = float(self.train_config.get(end_name, start))
        anneal_updates = max(
            1, int(self.train_config.get(anneal_name, 1))
        )
        fraction = min(
            max(self.current_update, 0) / anneal_updates,
            1.0,
        )
        return start + fraction * (end - start)

    # ==================================================================
    # Rollout收集
    # ==================================================================
    def collect_rollout(
        self,
    ) -> tuple[RecurrentRolloutBuffer, dict[str, Any]]:
        """用当前策略与环境交互并构造一次 on-policy rollout。"""
        self._base_env().set_runtime_telemetry_level(
            self.training_telemetry_level
        )
        rollout_steps = int(self.train_config.get("rollout_steps", 1024))
        if rollout_steps <= 0:
            raise ValueError("rollout_steps must be positive")

        self._set_target_obstacles()
        buffer = RecurrentRolloutBuffer(
            self.policy_rows, team_gae=bool(self.TEAM_GAE)
        )
        rollout_seed = self.seed + self.current_update * 100_003
        max_capture_fraction = float(
            self.train_config.get("capture_sampler", {}).get("max_step_fraction", 0.0)
        )
        capture_step_budget = (
            self.capture_schedule.budget(
                self.current_update, rollout_steps, self.capture_steps_cumulative,
                max_capture_fraction,
            ) if self.capture_schedule is not None else
            int(rollout_steps * max_capture_fraction)
        )
        obs, hidden, episode_source, anchor_entry = self._start_training_episode(
            rollout_seed, 0, 0, capture_step_budget, force_full=True
        )
        full_reset_count = int(episode_source == "full_reset")
        capture_steps = 0
        capture_completed = 0
        full_completed = 0
        capture_safe_successes = 0
        full_safe_successes = 0
        sampler_rows: list[dict[str, Any]] = []
        if self.capture_bank is not None:
            sampler_rows.append({
                "update": self.current_update, "start_rollout_step": 1,
                "source": episode_source,
                "anchor_id": None if anchor_entry is None else anchor_entry["anchor_id"],
                "layout_seed": None if anchor_entry is None else anchor_entry["layout_seed"],
                "physical_start_step": int(self._base_env().step_count),
            })

        episode_returns: list[float] = []
        current_return = 0.0
        successes = 0
        safe_successes = 0
        episode_reach_successes: list[bool] = []
        completed_episodes = 0
        reason_counts: Counter[str] = Counter()
        episode_lengths: list[int] = []
        final_max_distances: list[float] = []
        final_required_distances: list[float] = []
        goal_progress_ratios: list[float] = []
        team_arrival_flags: list[float] = []
        agent_arrival_counts = np.zeros(self.num_agents, dtype=np.float64)
        min_obstacle_clearances: list[float] = []
        min_inter_agent_distances: list[float] = []
        episode_collision_counts: list[int] = []
        episode_obstacle_collision_counts: list[int] = []
        episode_static_obstacle_collision_counts: list[int] = []
        episode_dynamic_obstacle_collision_counts: list[int] = []
        episode_inter_agent_collision_counts: list[int] = []
        episode_boundary_collision_counts: list[int] = []
        episode_boundary_side_counts: defaultdict[str, list[int]] = defaultdict(list)
        episode_boundary_contact_steps: list[int] = []
        episode_max_boundary_contact_streaks: list[int] = []
        episode_has_collisions: list[bool] = []
        first_collision_steps: list[int | None] = []
        first_collision_types: list[str | None] = []
        first_collision_goal_progress_ratios: list[float | None] = []
        guidance_metrics: defaultdict[str, list[float]] = defaultdict(list)
        reward_component_sums: defaultdict[str, float] = defaultdict(float)
        reward_component_abs_sums: defaultdict[str, float] = defaultdict(float)
        reward_component_steps = 0
        current_length = 0
        last_done = False
        action_sum = np.zeros((self.num_agents, 2), dtype=np.float64)
        action_positive_count = np.zeros((self.num_agents, 2), dtype=np.float64)
        action_sample_count = 0
        velocity_saturation_count = np.zeros(self.num_agents, dtype=np.float64)
        velocity_sample_count = 0

        gamma = float(self.train_config.get("gamma", 0.995))

        # Optional, read-only evidence from the *actual sampled* PPO rollout.
        # This is deliberately outside the task fingerprint and changes no
        # reward, action, optimizer input, or checkpoint state.
        credit_cfg = dict(self.train_config.get("credit_audit", {}))
        credit_updates = [int(value) for value in credit_cfg.get("updates", [])]
        credit_max_steps = int(credit_cfg.get("max_steps", 0))
        if credit_max_steps < 0:
            raise ValueError("train.credit_audit.max_steps must be nonnegative")
        credit_enabled = bool(credit_cfg.get("enabled", False)) and (
            not credit_updates or self.current_update in credit_updates
        )
        credit_path = self.output_dir / f"credit_audit_update_{self.current_update:06d}.jsonl"
        if credit_enabled and credit_path.exists():
            raise FileExistsError(f"credit audit output exists: {credit_path}")
        credit_rows: list[dict[str, Any]] = []
        credit_policy_hash = self.policy_fingerprint() if credit_enabled else None
        credit_task_hash = self.task_config_fingerprint() if credit_enabled else None

        for rollout_index in range(rollout_steps):
            policy_obs = self.policy_observations(obs)
            state = self.normalize_global_state(self.env.get_global_state())
            obs_tensor = torch.as_tensor(
                policy_obs, dtype=torch.float32, device=self.device
            )
            state_tensor = torch.as_tensor(
                state[None, :], dtype=torch.float32, device=self.device
            )
            hidden_before = hidden.detach().clone()
            record_credit = credit_enabled and (
                credit_max_steps == 0 or len(credit_rows) < credit_max_steps
            )
            if record_credit:
                base_before = self._base_env()
                credit_positions_before = base_before.positions.copy()
                credit_velocities_before = base_before.velocities.copy()
                credit_subgoals_before = base_before.subgoals.copy()
                credit_goals = base_before.goals.copy()
                credit_true_obstacles = [
                    {"position": item.position.tolist(), "radius": float(item.radius)}
                    for item in base_before.obstacles
                ]

            with torch.no_grad():
                actor_step = self.actor.forward_step(
                    obs_tensor, hidden_before, deterministic=False
                )
                values = self.critic(state_tensor).squeeze(0)

            host_step = None
            if self.consolidate_rollout_device_transfer:
                host_step = self._rollout_step_host_arrays(
                    hidden_before,
                    actor_step.action,
                    actor_step.raw_action,
                    actor_step.log_prob,
                    values,
                )
                actions = host_step["action"].reshape(self.num_agents, 2)
            else:
                actions = actor_step.action.cpu().numpy().reshape(
                    self.num_agents, 2
                )
            action_sum += actions
            action_positive_count += actions > 0.0
            action_sample_count += 1
            preaction_context = None
            preaction_margin = None
            preaction_velocity = None
            preaction_goal_directions = None
            preaction_subgoal_directions = None
            preaction_agent_margins = None
            preaction_tracks = None
            reference_actions = None
            reference_valid = None
            reference_stage_costs = None
            raw_stage_costs = None
            if bool(getattr(self, "capture_intervention_features", False)):
                base_before = self._base_env()
                causal_snapshot = base_before._causal_safety_snapshot(actions)
                snapshot_payload = causal_snapshot.canonical_payload()
                preaction_context, preaction_margin = _causal_context_from_snapshot(
                    causal_snapshot,
                    agent_radius=float(base_before.agent_radius),
                    map_size=tuple(float(value) for value in base_before.map_size),
                )
                team_observation = np.asarray(
                    causal_snapshot.team_observation, dtype=np.float32
                )
                preaction_velocity = team_observation[:, 2:4]
                goal_delta = np.asarray(
                    base_before.goals - base_before.positions,
                    dtype=np.float32,
                )
                goal_norm = np.linalg.norm(
                    goal_delta, axis=1, keepdims=True
                )
                preaction_goal_directions = np.divide(
                    goal_delta,
                    goal_norm,
                    out=np.zeros_like(goal_delta),
                    where=goal_norm > 1e-8,
                )
                reached = np.asarray(
                    base_before._goal_distances() < base_before.goal_radius,
                    dtype=bool,
                )
                preaction_goal_directions[reached] = 0.0
                subgoal_delta = np.asarray(
                    base_before.subgoals - base_before.positions,
                    dtype=np.float32,
                )
                subgoal_norm = np.linalg.norm(
                    subgoal_delta, axis=1, keepdims=True
                )
                preaction_subgoal_directions = np.divide(
                    subgoal_delta,
                    subgoal_norm,
                    out=np.zeros_like(subgoal_delta),
                    where=subgoal_norm > 1e-8,
                )
                subgoal_reached = np.asarray(
                    subgoal_norm[:, 0] < base_before.subgoal_reach_radius,
                    dtype=bool,
                )
                preaction_subgoal_directions[subgoal_reached] = 0.0
                preaction_agent_margins = per_agent_snapshot_margins(
                    snapshot_payload,
                    agent_radius=float(base_before.agent_radius),
                    map_size=tuple(float(value) for value in base_before.map_size),
                )
                preaction_tracks = padded_snapshot_obstacle_tracks(
                    snapshot_payload, obstacle_slots=int(base_before.obstacle_slots)
                )
                if bool(getattr(self, "capture_safety_alignment_references", False)):
                    references = registered_reference_actions(
                        actions, preaction_goal_directions
                    )
                    reference_panel = evaluate_avoidable_intervention_v2(
                        causal_snapshot,
                        (np.asarray(actions, dtype=np.float32), *references),
                        base_before.evaluate_shadow_action_v2,
                        goal_directions=goal_delta,
                        current_velocities=preaction_velocity,
                        reached_mask=reached,
                        acceleration_scale=float(base_before.agent_max_acceleration),
                        dt=float(base_before.dt),
                        maximum_speed=float(base_before.agent_max_speed),
                        maximum_progress_loss_mps=0.10,
                    )
                    reference_actions = np.asarray(references, dtype=np.float32)
                    reference_valid = np.asarray(
                        reference_panel.valid_shadow_mask[1:7], dtype=bool
                    )
                    reference_stage_costs = np.asarray(
                        reference_panel.shadow_stage_costs[1:7], dtype=np.float32
                    )
                    raw_stage_costs = np.asarray(
                        reference_panel.raw_stage_costs, dtype=np.float32
                    )
                elif bool(getattr(self, "capture_safety_alignment_costs", False)):
                    raw_stage_costs = np.asarray(
                        base_before.evaluate_shadow_action_v2(
                            causal_snapshot,
                            np.asarray(actions, dtype=np.float32),
                        ).stage_costs,
                        dtype=np.float32,
                    )
            next_obs, rewards, terminated, truncated, info = self.env.step(
                actions
            )
            executed_actions = np.asarray(
                info.get("executed_actions", actions), dtype=np.float32
            )
            projection_delta_norm = np.asarray(
                info.get(
                    "projection_delta_norm",
                    np.linalg.norm(executed_actions - actions, axis=1),
                ),
                dtype=np.float32,
            )
            intervention_mask = np.asarray(
                info.get("intervention_mask", projection_delta_norm > 1e-7),
                dtype=bool,
            )
            execution_target_eligible = np.asarray(
                info.get(
                    "execution_target_eligible",
                    np.zeros(self.policy_rows, dtype=bool),
                ),
                dtype=bool,
            )
            stage_trace = info.get("stage_intervention_trace")
            if stage_trace is not None:
                risk_eligibility = [
                    bool(record.eligible_for_actor_risk)
                    for record in stage_trace.records
                    if record.stage in TRAINABLE_STAGES
                ]
                expected_risk_records = int(self.num_agents) * len(
                    TRAINABLE_STAGES
                )
                if len(risk_eligibility) != expected_risk_records:
                    raise RuntimeError(
                        "stage trace is incomplete for actor-risk eligibility"
                    )
                actor_risk_eligible = bool(
                    all(risk_eligibility)
                    and not stage_trace.recovery_active
                    and not stage_trace.safe_brake_active
                    and not stage_trace.terminal_transition
                    and not terminated
                    and not truncated
                )
            else:
                actor_risk_eligible = bool(
                    not terminated
                    and not truncated
                    and not info.get("recovery_control_applied", False)
                    and not info.get("safe_brake_active", False)
                )
            execution_stage_codes = {
                "actor": 0,
                "joint_guard": 1,
                "ttc_projection": 2,
                "obstacle_fallback": 3,
                "safe_brake": 4,
                "recovery": 5,
            }
            execution_stage = np.full(
                self.policy_rows,
                execution_stage_codes.get(str(info.get("execution_stage", "actor")), 6),
                dtype=np.int16,
            )
            if episode_source == "capture_bank":
                capture_steps += 1
            velocity_saturation_count += np.any(
                np.isclose(
                    np.abs(self.env.velocities),
                    self.env.agent_max_speed,
                    atol=1e-5,
                ),
                axis=1,
            )
            velocity_sample_count += 1
            last_done = bool(terminated or truncated)

            # 时间上限 truncation 不是MDP真实终止。
            # 将下一状态价值并入训练奖励，避免critic把超时状态错误视为价值0。
            training_rewards = self.policy_rewards(rewards)
            timeout_value = None
            if truncated and not terminated:
                with torch.no_grad():
                    next_state = torch.as_tensor(
                        self.normalize_global_state(
                            self.env.get_global_state()
                        )[None, :],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    timeout_value = (
                        self.critic(next_state)
                        .squeeze(0)
                        .cpu()
                        .numpy()
                    )
                training_rewards = training_rewards + gamma * timeout_value

            hidden_host = (
                host_step["hidden"]
                if host_step is not None else hidden_before.cpu().numpy()
            )
            action_host = (
                host_step["action"]
                if host_step is not None else actor_step.action.cpu().numpy()
            )
            raw_action_host = (
                host_step["raw_action"]
                if host_step is not None else actor_step.raw_action.cpu().numpy()
            )
            log_prob_host = (
                host_step["log_prob"]
                if host_step is not None else actor_step.log_prob.cpu().numpy()
            )
            value_host = (
                host_step["values"]
                if host_step is not None else values.cpu().numpy()
            )
            if self.joint_policy:
                buffer_executed_actions = executed_actions.reshape(1, -1)
                buffer_projection_delta_norm = np.asarray([
                    float(np.linalg.norm(projection_delta_norm))
                ], dtype=np.float32)
                buffer_intervention_mask = np.asarray([
                    bool(np.any(intervention_mask))
                ])
                buffer_execution_target_eligible = np.asarray([
                    bool(np.all(execution_target_eligible))
                ])
                buffer_execution_stage = np.asarray([
                    int(np.max(execution_stage))
                ], dtype=np.int16)
            else:
                buffer_executed_actions = executed_actions
                buffer_projection_delta_norm = projection_delta_norm
                buffer_intervention_mask = intervention_mask
                buffer_execution_target_eligible = execution_target_eligible
                buffer_execution_stage = execution_stage
            buffer.add(
                local_obs=policy_obs,
                global_state=state,
                hidden=hidden_host,
                actions=action_host,
                raw_actions=raw_action_host,
                log_probs=log_prob_host,
                rewards=training_rewards,
                dones=np.full(
                    self.policy_rows, last_done, dtype=bool
                ),
                values=value_host,
                executed_actions=buffer_executed_actions,
                projection_delta_norm=buffer_projection_delta_norm,
                intervention_mask=buffer_intervention_mask,
                execution_target_eligible=buffer_execution_target_eligible,
                actor_risk_eligible=actor_risk_eligible,
                execution_stage=buffer_execution_stage,
                safety_context=(
                    preaction_context if preaction_context is not None
                    else info.get("causal_safety_context")
                ),
                safety_margin=(
                    preaction_margin if preaction_margin is not None
                    else info.get("current_safety_margin")
                ),
                preaction_team_velocity=preaction_velocity,
                preaction_goal_directions=preaction_goal_directions,
                preaction_subgoal_directions=preaction_subgoal_directions,
                preaction_per_agent_margins=preaction_agent_margins,
                preaction_obstacle_tracks=preaction_tracks,
                reference_actions=reference_actions,
                reference_valid=reference_valid,
                reference_stage_costs=reference_stage_costs,
                raw_stage_costs=raw_stage_costs,
            )
            if record_credit:
                base_after = self._base_env()
                components = info.get("reward_components", {})
                component_names = (
                    "subgoal_progress_reward", "historical_final_progress_reward",
                    "proximity_risk_reward", "time_reward", "goal_bonus",
                    "obstacle_collision_penalty", "inter_agent_collision_penalty",
                    "boundary_collision_penalty", "timeout_reward", "total_reward",
                )
                credit_rows.append({
                    "schema_version": "ppo_onpolicy_credit_audit_v1",
                    "update": int(self.current_update),
                    "step_in_rollout": len(credit_rows) + 1,
                    "rollout_seed": int(rollout_seed),
                    "training_episode_source": episode_source,
                    "capture_anchor_id": None if anchor_entry is None else anchor_entry["anchor_id"],
                    "policy_fingerprint": credit_policy_hash,
                    "task_config_fingerprint": credit_task_hash,
                    "raw_observation": np.asarray(obs, dtype=np.float32).tolist(),
                    "policy_observation": np.asarray(policy_obs, dtype=np.float32).tolist(),
                    "hidden_before_sha256": hashlib.sha256(
                        hidden_before.cpu().contiguous().numpy().tobytes()
                    ).hexdigest(),
                    "positions_before": credit_positions_before.tolist(),
                    "velocities_before": credit_velocities_before.tolist(),
                    "subgoals_before": credit_subgoals_before.tolist(),
                    "goals": credit_goals.tolist(),
                    "true_obstacles_for_analysis_only": credit_true_obstacles,
                    "positions_after": base_after.positions.tolist(),
                    "subgoals_after": base_after.subgoals.tolist(),
                    "episode_step_after": int(base_after.step_count),
                    "sampled_action": actions.tolist(),
                    "raw_action": raw_action_host.tolist(),
                    "old_log_prob": log_prob_host.tolist(),
                    "critic_value": value_host.tolist(),
                    "environment_reward": float(np.mean(rewards)),
                    "training_reward": training_rewards.tolist(),
                    "timeout_bootstrap_value": None if timeout_value is None
                        else timeout_value.tolist(),
                    "reward_components": {
                        name: float(np.asarray(components[name], dtype=np.float64).mean())
                        for name in component_names if name in components
                    },
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "reason": str(info.get("reason", "ongoing")),
                    "collision_events": info.get("collision_events", []),
                })

            # 日志统计仍使用环境原始奖励，避免timeout bootstrap污染回报曲线。
            current_return += float(np.mean(rewards))
            current_length += 1
            self._accumulate_reward_components(
                info.get("reward_components", {}),
                reward_component_sums,
                reward_component_abs_sums,
            )
            reward_component_steps += 1

            if last_done:
                if episode_source == "capture_bank":
                    capture_completed += 1
                    capture_safe_successes += int(bool(info.get("safe_success", False)))
                else:
                    full_completed += 1
                    full_safe_successes += int(bool(info.get("safe_success", False)))
                if self.capture_bank is not None:
                    sampler_rows[-1].update({
                        "end_rollout_step": rollout_index + 1,
                        "segment_steps": current_length,
                        "completed": True,
                        "reason": str(info.get("reason", "unknown")),
                        "safe_success": bool(info.get("safe_success", False)),
                    })
                episode_returns.append(current_return)
                current_return = 0.0
                successes += int(bool(info.get("reach_success", info.get("success", False))))
                safe_successes += int(bool(info.get("safe_success", False)))
                episode_reach_successes.append(
                    bool(info.get("reach_success", info.get("success", False)))
                )
                completed_episodes += 1
                reason_counts[str(info.get("reason", "unknown"))] += 1
                episode_lengths.append(current_length)
                final_max_distances.append(
                    float(np.max(info["goal_distances"]))
                )
                final_required_distances.append(
                    float(info["required_goal_distance"])
                )
                goal_progress_ratios.append(
                    float(info.get("goal_progress_ratio", 0.0))
                )
                team_arrival_flags.append(
                    float(info.get("team_arrival_observed", False))
                )
                ever_in_goal = np.asarray(
                    info.get("agents_ever_in_goal", np.zeros(self.num_agents)),
                    dtype=np.float32,
                )
                if ever_in_goal.shape == (self.num_agents,):
                    agent_arrival_counts += ever_in_goal
                min_obstacle_clearances.append(
                    float(info.get("min_obstacle_clearance", np.inf))
                )
                min_inter_agent_distances.append(
                    float(info.get("min_inter_agent_distance", np.inf))
                )
                episode_collision_counts.append(int(info.get("collision_count", 0)))
                episode_obstacle_collision_counts.append(
                    int(info.get("obstacle_collision_count", 0))
                )
                episode_static_obstacle_collision_counts.append(
                    int(info.get("static_obstacle_collision_count", 0))
                )
                episode_dynamic_obstacle_collision_counts.append(
                    int(info.get("dynamic_obstacle_collision_count", 0))
                )
                episode_inter_agent_collision_counts.append(
                    int(info.get("inter_agent_collision_count", 0))
                )
                episode_boundary_collision_counts.append(
                    int(info.get("boundary_collision_count", 0))
                )
                for side in BOUNDARY_SIDES:
                    episode_boundary_side_counts[side].append(
                        int(info.get(f"{side}_boundary_collision_count", 0))
                    )
                episode_boundary_contact_steps.append(
                    int(info.get("boundary_contact_step_count", 0))
                )
                episode_max_boundary_contact_streaks.append(
                    int(info.get("max_boundary_contact_streak", 0))
                )
                episode_has_collisions.append(bool(info.get("episode_has_collision", False)))
                first_collision_steps.append(info.get("first_collision_step"))
                first_collision_types.append(info.get("first_collision_type"))
                first_collision_goal_progress_ratios.append(
                    info.get("first_collision_goal_progress_ratio")
                )
                for key in GUIDANCE_METRIC_KEYS:
                    guidance_metrics[key].append(float(info.get(key, 0.0)))
                current_length = 0

                if rollout_index + 1 < rollout_steps:
                    obs, hidden, episode_source, anchor_entry = self._start_training_episode(
                        rollout_seed, full_reset_count, capture_steps, capture_step_budget
                    )
                    full_reset_count += int(episode_source == "full_reset")
                    if self.capture_bank is not None:
                        sampler_rows.append({
                            "update": self.current_update,
                            "start_rollout_step": rollout_index + 2,
                            "source": episode_source,
                            "anchor_id": None if anchor_entry is None else anchor_entry["anchor_id"],
                            "layout_seed": None if anchor_entry is None else anchor_entry["layout_seed"],
                            "physical_start_step": int(self._base_env().step_count),
                        })
            else:
                obs = next_obs
                hidden = actor_step.hidden.detach()

        if last_done:
            last_value = np.zeros(self.policy_rows, dtype=np.float32)
        else:
            with torch.no_grad():
                state = torch.as_tensor(
                    self.normalize_global_state(
                        self.env.get_global_state()
                    )[None, :],
                    dtype=torch.float32,
                    device=self.device,
                )
                last_value = (
                    self.critic(state).squeeze(0).cpu().numpy()
                )

        buffer.compute_returns_and_advantages(
            last_value,
            gamma=gamma,
            gae_lambda=float(
                self.train_config.get("gae_lambda", 0.95)
            ),
        )
        if credit_enabled:
            # Mirror RecurrentRolloutBuffer's exact TD convention, including
            # timeout bootstrap already folded into training_reward.
            for index, row in enumerate(credit_rows):
                next_value = (buffer.values[index + 1]
                              if index + 1 < len(buffer.values) else last_value)
                continuation = 1.0 - buffer.dones[index].astype(np.float32)
                delta = (buffer.rewards[index] + gamma * next_value
                         * continuation - buffer.values[index])
                row["td_residual"] = delta.tolist()
                row["gae_advantage"] = buffer.advantages[index].tolist()
                row["value_target"] = buffer.returns[index].tolist()
            with credit_path.open("x", encoding="utf-8") as stream:
                for row in credit_rows:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

        if self.capture_bank is not None:
            if sampler_rows and "completed" not in sampler_rows[-1]:
                sampler_rows[-1].update({
                    "end_rollout_step": rollout_steps,
                    "segment_steps": current_length,
                    "completed": False,
                    "reason": "rollout_boundary",
                    "safe_success": False,
                })
            sampler_path = self.output_dir / f"capture_sampler_update_{self.current_update:06d}.jsonl"
            if sampler_path.exists():
                raise FileExistsError(f"A7 sampler output exists: {sampler_path}")
            save_jsonl(sampler_rows, sampler_path)

        denominator = max(completed_episodes, 1)
        collision_count = int(sum(episode_collision_counts))
        obstacle_collision_count = int(sum(episode_obstacle_collision_counts))
        static_obstacle_collision_count = int(sum(episode_static_obstacle_collision_counts))
        dynamic_obstacle_collision_count = int(sum(episode_dynamic_obstacle_collision_counts))
        inter_agent_collision_count = int(sum(episode_inter_agent_collision_counts))
        boundary_collision_count = int(sum(episode_boundary_collision_counts))

        rollout_metrics = {
            "mean_episode_return": (
                float(np.mean(episode_returns))
                if episode_returns
                else current_return
            ),
            "success_rate": successes / denominator,
            "reach_success_rate": successes / denominator,
            "safe_success_rate": safe_successes / denominator,
            "collision_rate": float(np.mean(episode_has_collisions)) if episode_has_collisions else 0.0,
            "collision_count": collision_count,
            "obstacle_collision_count": obstacle_collision_count,
            "static_obstacle_collision_count": static_obstacle_collision_count,
            "dynamic_obstacle_collision_count": dynamic_obstacle_collision_count,
            "inter_agent_collision_count": inter_agent_collision_count,
            "boundary_collision_count": boundary_collision_count,
            "mean_collision_count": collision_count / denominator,
            "first_collision_step": first_collision_steps,
            "first_collision_type_counts": dict(
                Counter(item for item in first_collision_types if item is not None)
            ),
            "mean_first_collision_step": (
                float(np.mean([item for item in first_collision_steps if item is not None]))
                if any(item is not None for item in first_collision_steps)
                else None
            ),
            "timeout_rate": reason_counts["timeout"] / denominator,
            "completed_episodes": float(completed_episodes),
            "mean_episode_length": (
                float(np.mean(episode_lengths))
                if episode_lengths
                else float(current_length)
            ),
            "mean_final_max_goal_distance": (
                float(np.mean(final_max_distances))
                if final_max_distances
                else float(np.max(self.env._goal_distances()))
            ),
            "mean_final_required_goal_distance": (
                float(np.mean(final_required_distances))
                if final_required_distances
                else float(np.max(self.env._goal_distances()))
            ),
            "mean_goal_progress_ratio": (
                float(np.mean(goal_progress_ratios))
                if goal_progress_ratios else 0.0
            ),
            "team_arrival_rate": (
                float(np.mean(team_arrival_flags)) if team_arrival_flags else 0.0
            ),
            "per_agent_arrival_rates": (
                agent_arrival_counts / denominator
            ).astype(float).tolist(),
            "mean_agent_arrival_rate": float(
                np.mean(agent_arrival_counts / denominator)
            ),
            "obstacle_collision_rate": float(np.mean(np.asarray(episode_obstacle_collision_counts) > 0)) if episode_obstacle_collision_counts else 0.0,
            "static_obstacle_collision_rate": float(np.mean(np.asarray(episode_static_obstacle_collision_counts) > 0)) if episode_static_obstacle_collision_counts else 0.0,
            "dynamic_obstacle_collision_rate": float(np.mean(np.asarray(episode_dynamic_obstacle_collision_counts) > 0)) if episode_dynamic_obstacle_collision_counts else 0.0,
            "agent_collision_rate": float(np.mean(np.asarray(episode_inter_agent_collision_counts) > 0)) if episode_inter_agent_collision_counts else 0.0,
            "boundary_collision_rate": float(np.mean(np.asarray(episode_boundary_collision_counts) > 0)) if episode_boundary_collision_counts else 0.0,
            "mean_min_obstacle_clearance": self._finite_mean(
                min_obstacle_clearances
            ),
            "mean_min_inter_agent_distance": self._finite_mean(
                min_inter_agent_distances
            ),
            "reason_counts": dict(reason_counts),
            "reward_components": {
                key: value / max(reward_component_steps, 1)
                for key, value in reward_component_sums.items()
            },
            "abs_reward_components": self._module_abs_reward_components(
                reward_component_abs_sums, reward_component_steps
            ),
            "reward_component_ratios": self._module_reward_ratios(
                reward_component_abs_sums, reward_component_steps
            ),
        }
        rollout_metrics.update(
            self._outcome_safety_metrics(
                episode_reach_successes,
                episode_obstacle_collision_counts,
                episode_inter_agent_collision_counts,
                episode_boundary_collision_counts,
                first_collision_steps,
                first_collision_goal_progress_ratios,
                episode_static_obstacle_collision_counts,
                episode_dynamic_obstacle_collision_counts,
            )
        )
        rollout_metrics.update(
            {
                key: float(np.mean(guidance_metrics[key]))
                if guidance_metrics[key]
                else 0.0
                for key in GUIDANCE_METRIC_KEYS
            }
        )
        rollout_metrics.update(
            {
                f"{side}_boundary_collision_count": int(
                    sum(episode_boundary_side_counts[side])
                )
                for side in BOUNDARY_SIDES
            }
        )
        rollout_metrics.update(
            {
                f"{side}_boundary_collision_rate": float(
                    np.mean(np.asarray(episode_boundary_side_counts[side]) > 0)
                )
                if episode_boundary_side_counts[side]
                else 0.0
                for side in BOUNDARY_SIDES
            }
        )
        rollout_metrics.update(
            {
                "mean_boundary_contact_steps": float(
                    np.mean(episode_boundary_contact_steps)
                )
                if episode_boundary_contact_steps
                else 0.0,
                "mean_max_boundary_contact_streak": float(
                    np.mean(episode_max_boundary_contact_streaks)
                )
                if episode_max_boundary_contact_streaks
                else 0.0,
                "per_agent_action_mean": (
                    action_sum / max(action_sample_count, 1)
                ).tolist(),
                "per_agent_action_positive_fraction": (
                    action_positive_count / max(action_sample_count, 1)
                ).tolist(),
                "per_agent_velocity_saturation_rate": (
                    velocity_saturation_count / max(velocity_sample_count, 1)
                ).tolist(),
                "mean_velocity_saturation_rate": float(
                    np.mean(
                        velocity_saturation_count
                        / max(velocity_sample_count, 1)
                    )
                ),
            }
        )
        if self.capture_bank is not None:
            rollout_metrics.update({
                "capture_sampler_enabled": True,
                "capture_bank_manifest_sha256": self.train_config["capture_sampler"]["bank_manifest_sha256"],
                "capture_anchor_steps": capture_steps,
                "capture_anchor_step_fraction": capture_steps / rollout_steps,
                "capture_anchor_episodes_started": sum(
                    row["source"] == "capture_bank" for row in sampler_rows
                ),
                "capture_anchor_episodes_completed": capture_completed,
                "capture_anchor_safe_successes": capture_safe_successes,
                "full_reset_episodes_completed": full_completed,
                "full_reset_safe_successes": full_safe_successes,
                "training_episode_metrics_use_mixed_reset_distribution": True,
            })
            if self.capture_schedule is not None:
                self.capture_steps_cumulative += capture_steps
                rollout_metrics.update({
                    "capture_schedule_active": self.capture_schedule.active(self.current_update),
                    "capture_schedule_rollout_budget": capture_step_budget,
                    "capture_schedule_planned_cumulative_steps":
                        self.capture_schedule.planned_cumulative_steps(
                            self.current_update, rollout_steps),
                    "capture_schedule_actual_cumulative_steps": self.capture_steps_cumulative,
                })
        return buffer, rollout_metrics

    @staticmethod
    def _finite_mean(values: list[float]) -> float:
        """忽略 inf/NaN 后计算均值；没有有限值时返回 inf。"""
        array = np.asarray(values, dtype=np.float64)
        finite = array[np.isfinite(array)]
        return float(np.mean(finite)) if finite.size else float("inf")

    @staticmethod
    def _outcome_safety_metrics(
        reach_successes,
        obstacle_counts,
        inter_agent_counts,
        boundary_counts,
        first_collision_steps,
        first_collision_goal_progress_ratios,
        static_obstacle_counts=None,
        dynamic_obstacle_counts=None,
    ) -> dict[str, float | int | None]:
        """把 Reach×Collision 分解为互斥结果桶和条件安全指标。"""
        reach = np.asarray(reach_successes, dtype=bool)
        obstacle = np.asarray(obstacle_counts, dtype=np.int64)
        inter_agent = np.asarray(inter_agent_counts, dtype=np.int64)
        boundary = np.asarray(boundary_counts, dtype=np.int64)
        static_obstacle = (
            np.asarray(static_obstacle_counts, dtype=np.int64)
            if static_obstacle_counts is not None
            else None
        )
        dynamic_obstacle = (
            np.asarray(dynamic_obstacle_counts, dtype=np.int64)
            if dynamic_obstacle_counts is not None
            else None
        )
        if not (
            len(reach)
            == len(obstacle)
            == len(inter_agent)
            == len(boundary)
        ):
            raise ValueError("episode outcome metric lengths must match")
        if reach.size == 0:
            return {
                "safe_success_count": 0,
                "unsafe_success_count": 0,
                "collision_failure_count": 0,
                "clean_failure_count": 0,
                "unsafe_success_rate": 0.0,
                "collision_failure_rate": 0.0,
                "clean_failure_rate": 0.0,
                "safe_conversion_rate": 0.0,
                "collision_given_success": 0.0,
                "collision_given_failure": 0.0,
            }

        total_counts = obstacle + inter_agent + boundary
        collision = total_counts > 0
        failure = ~reach
        safe_success = reach & ~collision
        unsafe_success = reach & collision
        collision_failure = failure & collision
        clean_failure = failure & ~collision

        def conditional_rate(events: np.ndarray, mask: np.ndarray) -> float:
            denominator = int(np.count_nonzero(mask))
            return (
                float(np.count_nonzero(events & mask) / denominator)
                if denominator
                else 0.0
            )

        def conditional_mean(values: np.ndarray, mask: np.ndarray) -> float:
            return float(np.mean(values[mask])) if np.any(mask) else 0.0

        def optional_conditional_mean(values, mask: np.ndarray) -> float | None:
            selected = [
                float(value)
                for value, keep in zip(values, mask)
                if keep and value is not None
            ]
            return float(np.mean(selected)) if selected else None

        result: dict[str, float | int | None] = {
            "safe_success_count": int(np.count_nonzero(safe_success)),
            "unsafe_success_count": int(np.count_nonzero(unsafe_success)),
            "collision_failure_count": int(np.count_nonzero(collision_failure)),
            "clean_failure_count": int(np.count_nonzero(clean_failure)),
            "unsafe_success_rate": float(np.mean(unsafe_success)),
            "collision_failure_rate": float(np.mean(collision_failure)),
            "clean_failure_rate": float(np.mean(clean_failure)),
            "safe_conversion_rate": (
                float(np.count_nonzero(safe_success) / np.count_nonzero(reach))
                if np.any(reach)
                else 0.0
            ),
            "collision_given_success": conditional_rate(collision, reach),
            "collision_given_failure": conditional_rate(collision, failure),
            "mean_collision_count_given_success": conditional_mean(
                total_counts, reach
            ),
            "mean_collision_count_given_failure": conditional_mean(
                total_counts, failure
            ),
            "mean_first_collision_step_given_success": optional_conditional_mean(
                first_collision_steps, reach & collision
            ),
            "mean_first_collision_step_given_failure": optional_conditional_mean(
                first_collision_steps, failure & collision
            ),
            "mean_first_collision_goal_progress_ratio_given_success": optional_conditional_mean(
                first_collision_goal_progress_ratios, reach & collision
            ),
            "mean_first_collision_goal_progress_ratio_given_failure": optional_conditional_mean(
                first_collision_goal_progress_ratios, failure & collision
            ),
        }
        for prefix, counts in (
            ("obstacle", obstacle),
            ("inter_agent", inter_agent),
            ("boundary", boundary),
        ):
            events = counts > 0
            result[f"{prefix}_collision_given_success"] = conditional_rate(
                events, reach
            )
            result[f"{prefix}_collision_given_failure"] = conditional_rate(
                events, failure
            )
            result[f"mean_{prefix}_collision_count_given_success"] = conditional_mean(
                counts, reach
            )
            result[f"mean_{prefix}_collision_count_given_failure"] = conditional_mean(
                counts, failure
            )

        # 动静态来源是 obstacle 指标的诊断细分；保持旧调用不传参数时完全兼容。
        for prefix, counts in (
            ("static_obstacle", static_obstacle),
            ("dynamic_obstacle", dynamic_obstacle),
        ):
            if counts is None:
                continue
            if len(counts) != len(reach):
                raise ValueError("episode obstacle source metric lengths must match")
            events = counts > 0
            result[f"{prefix}_collision_given_success"] = conditional_rate(events, reach)
            result[f"{prefix}_collision_given_failure"] = conditional_rate(events, failure)
            result[f"mean_{prefix}_collision_count_given_success"] = conditional_mean(counts, reach)
            result[f"mean_{prefix}_collision_count_given_failure"] = conditional_mean(counts, failure)

        for suffix in ("success", "failure"):
            key = f"mean_first_collision_goal_progress_ratio_given_{suffix}"
            progress = result[key]
            result[
                f"mean_remaining_goal_progress_at_first_collision_given_{suffix}"
            ] = None if progress is None else 1.0 - float(progress)
        return result

    @staticmethod
    def _accumulate_reward_components(
        components: Mapping[str, Any],
        sums: Mapping[str, float],
        abs_sums: Mapping[str, float],
    ) -> None:
        """Reward 遥测只接受有限数值，错误需在环境边界明确暴露。"""
        for key, value in components.items():
            if isinstance(value, Mapping):
                raise TypeError(f"reward component {key!r} must be numeric, got mapping")
            try:
                component_values = np.asarray(value, dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"reward component {key!r} must be numeric") from exc
            if not np.all(np.isfinite(component_values)):
                raise FloatingPointError(f"reward component {key!r} contains NaN/Inf")
            sums[key] += float(component_values.mean())
            abs_sums[key] += float(np.abs(component_values).mean())

    @staticmethod
    def _module_abs_reward_components(
        abs_sums: Mapping[str, float], steps: int
    ) -> dict[str, float]:
        """把逐项日志整理为诊断使用的模块级绝对奖励贡献。"""
        denom = max(int(steps), 1)
        mapping = {
            "navigation": "navigation_reward",
            "cooperation": "cooperation_reward",
            "obstacle": "obstacle_reward",
            "inter_agent": "inter_agent_reward",
            "boundary": "boundary_reward",
            "time": "time_reward",
            "timeout_failure": "timeout_reward",
            "risk": "proximity_risk_reward",
            "control": "control_effort_reward",
            "terminal": "terminal_reward",
        }
        return {
            module: float(abs_sums.get(key, 0.0)) / denom
            for module, key in mapping.items()
        }

    @classmethod
    def _module_reward_ratios(
        cls, abs_sums: Mapping[str, float], steps: int
    ) -> dict[str, float]:
        components = cls._module_abs_reward_components(abs_sums, steps)
        total = sum(components.values())
        if total <= 1e-12:
            return {key: 0.0 for key in components}
        return {key: value / total for key, value in components.items()}

    @staticmethod
    def _compact_train_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
        """Remove only aliases that are numerically identical in train JSONL.

        Environment ``info`` and in-memory metrics keep their historical keys
        for compatibility.  If an alias ever diverges from its canonical
        metric, both are retained so a future semantic change cannot silently
        discard evidence.
        """
        compact = deepcopy(dict(metrics))
        compact["metrics_schema_version"] = TRAIN_METRICS_SCHEMA_VERSION

        def remove_equal_aliases(
            values: dict[str, Any],
            aliases: tuple[tuple[str, str, float], ...],
        ) -> None:
            for alias, canonical, scale in aliases:
                if alias not in values or canonical not in values:
                    continue
                alias_value = np.asarray(values[alias], dtype=np.float64)
                canonical_value = np.asarray(
                    values[canonical], dtype=np.float64
                )
                if alias_value.shape != canonical_value.shape:
                    continue
                if np.allclose(
                    alias_value,
                    float(scale) * canonical_value,
                    rtol=1e-7,
                    atol=1e-12,
                    equal_nan=False,
                ):
                    values.pop(alias)

        remove_equal_aliases(compact, TRAIN_METRIC_ALIASES)
        reward_components = compact.get("reward_components")
        if isinstance(reward_components, Mapping):
            compact_components = dict(reward_components)
            remove_equal_aliases(
                compact_components,
                TRAIN_REWARD_COMPONENT_ALIASES,
            )
            compact["reward_components"] = compact_components
        return compact

    # ==================================================================
    # PPO更新
    # ==================================================================
    def update(self, buffer: RecurrentRolloutBuffer) -> dict[str, float]:
        """执行 recurrent PPO Actor/Critic 更新。"""
        clip_coef = float(self.train_config.get("clip_coef", 0.2))
        action_entropy_coef = self._scheduled_coefficient(
            "action_entropy_coef",
            "action_entropy_coef_start",
            "action_entropy_coef_end",
            "action_entropy_anneal_updates",
            1e-4,
        )
        value_coef = float(self.train_config.get("value_coef", 0.5))
        max_grad_norm = float(self.train_config.get("max_grad_norm", 0.5))
        sequence_length = int(self.train_config.get("sequence_length", 16))
        sequence_minibatch_size = int(
            self.train_config.get("sequence_minibatch_size", 32)
        )
        critic_minibatch_size = int(
            self.train_config.get("critic_minibatch_size", 512)
        )
        target_kl = float(self.train_config.get("target_kl", 0.03))
        ppo_epochs = int(self.train_config.get("ppo_epochs", 4))
        value_loss_type = str(
            self.train_config.get("value_loss_type", "huber")
        ).lower()
        huber_delta = float(self.train_config.get("huber_delta", 10.0))
        if value_loss_type not in {"huber", "mse"}:
            raise ValueError("train.value_loss_type must be 'huber' or 'mse'")
        if huber_delta <= 0.0:
            raise ValueError("train.huber_delta must be positive")
        if "projection_aware_execution" in self.train_config:
            raise ValueError("Projection-aware V1/V2 losses are archived")

        advantage_mean = float(buffer.advantages.mean())
        advantage_std = float(buffer.advantages.std())

        policy_losses: list[float] = []
        value_losses: list[float] = []
        action_entropies: list[float] = []
        approximate_kls: list[float] = []
        clip_fractions: list[float] = []
        actor_grad_norms: list[float] = []
        critic_grad_norms: list[float] = []

        for _ in range(ppo_epochs):
            epoch_kls: list[float] = []

            for batch in buffer.sequence_minibatches(
                sequence_length,
                sequence_minibatch_size,
                self.training_rng,
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

                evaluated = self.actor.evaluate_sequence(
                    obs, hidden, raw_actions
                )
                log_ratio = evaluated.log_prob - old_log_probs
                ratio = log_ratio.exp()

                unclipped = -advantages * ratio
                clipped = -advantages * torch.clamp(
                    ratio,
                    1.0 - clip_coef,
                    1.0 + clip_coef,
                )
                policy_loss = torch.maximum(
                    unclipped, clipped
                )[mask].mean()
                action_entropy = evaluated.action_entropy[mask].mean()
                approximate_kl = (
                    (ratio - 1.0) - log_ratio
                )[mask].mean()
                clip_fraction = (
                    (ratio - 1.0).abs() > clip_coef
                )[mask].float().mean()

                actor_loss = policy_loss - action_entropy_coef * action_entropy
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), max_grad_norm
                )
                self.actor_optimizer.step()

                policy_losses.append(float(policy_loss.detach().cpu()))
                action_entropies.append(
                    float(action_entropy.detach().cpu())
                )
                kl_value = float(approximate_kl.detach().cpu())
                approximate_kls.append(kl_value)
                epoch_kls.append(kl_value)
                clip_fractions.append(
                    float(clip_fraction.detach().cpu())
                )
                actor_grad_norms.append(
                    float(actor_grad_norm.detach().cpu())
                )

            # Critic使用所有时间步的全局状态。
            states = torch.as_tensor(
                buffer.global_states,
                dtype=torch.float32,
                device=self.device,
            )
            returns = torch.as_tensor(
                buffer.returns,
                dtype=torch.float32,
                device=self.device,
            )

            for indices in np.array_split(
                self.training_rng.permutation(len(states)),
                max(
                    1,
                    int(
                        np.ceil(
                            len(states) / critic_minibatch_size
                        )
                    ),
                ),
            ):
                index = torch.as_tensor(
                    indices, dtype=torch.long, device=self.device
                )
                predicted = self.critic(states[index])
                if value_loss_type == "huber":
                    # 碰撞奖励会带来较大的 TD 离群值；Huber 损失限制其梯度，
                    # 不改变奖励数值，也不改变 Critic 网络结构。
                    value_loss = torch.nn.functional.huber_loss(
                        predicted,
                        returns[index],
                        delta=huber_delta,
                    )
                else:
                    value_loss = 0.5 * (
                        predicted - returns[index]
                    ).square().mean()

                self.critic_optimizer.zero_grad()
                (value_coef * value_loss).backward()
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), max_grad_norm
                )
                self.critic_optimizer.step()

                value_losses.append(
                    float(value_loss.detach().cpu())
                )
                critic_grad_norms.append(
                    float(critic_grad_norm.detach().cpu())
                )

            if epoch_kls and float(np.mean(epoch_kls)) > target_kl:
                break

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "action_entropy": float(np.mean(action_entropies)),
            "entropy": float(np.mean(action_entropies)),
            "approx_kl": float(np.mean(approximate_kls)),
            "clip_fraction": float(np.mean(clip_fractions)),
            "actor_grad_norm": float(np.mean(actor_grad_norms)),
            "critic_grad_norm": float(np.mean(critic_grad_norms)),
            "action_saturation_fraction": float(
                np.mean(np.abs(buffer.actions) > 0.98)
            ),
            "action_entropy_coef": action_entropy_coef,
            **self.actor.distribution_stats(),
        }

    # ==================================================================
    # 训练/评估入口
    # ==================================================================
    def train(self) -> dict[str, Any]:
        if getattr(self, "recovery_study_evaluation_only", False):
            raise ValueError("recovery study checkpoint transition is evaluation-only")
        if str(self.config.get("subgoal_selector", {}).get("mode", "rule")) == "llm":
            raise ValueError("constrained LLM subgoal selection is evaluation-only")
        if self.ood_evaluation_intervention_used:
            raise ValueError("OOD environment intervention is evaluation-only")
        self._assert_output_contract()
        self._write_run_manifest()
        self._manifest_written = True
        total_updates = int(self.train_config.get("total_updates", 1))
        save_every = max(1, int(self.train_config.get("save_every", 1)))
        latest: dict[str, Any] = {}
        first_update = self.current_update + 1
        start_time = perf_counter()

        self.logger.info(
            "training start | device %s | agents %d | obstacles %d | "
            "update %d/%d | rollout_steps %d",
            self.device,
            self.env.num_agents,
            self.env.obstacle_num,
            self.current_update,
            total_updates,
            int(self.train_config.get("rollout_steps", 1024)),
        )

        if self.current_update >= total_updates:
            summary_path = self.output_dir / "train_summary.json"
            if summary_path.exists():
                return json.loads(summary_path.read_text(encoding="utf-8"))
            return {"updates_completed": self.current_update}

        for update in range(first_update, total_updates + 1):
            update_started = perf_counter()
            self.current_update = update
            rollout_started = perf_counter()
            buffer, rollout_stats = self.collect_rollout()
            rollout_seconds = perf_counter() - rollout_started
            learner_started = perf_counter()
            update_stats = self.update(buffer)
            learner_seconds = perf_counter() - learner_started
            latest = {
                **rollout_stats,
                **update_stats,
                "updates_completed": update,
                "environment_steps": update * int(self.train_config.get("rollout_steps", 1024)),
                "algorithm": self.algorithm,
                "algorithm_variant": self.algorithm_variant,
                "environment_contract_fingerprint": (
                    self.environment_contract_fingerprint()
                ),
                "learner_contract_fingerprint": (
                    self.learner_contract_fingerprint()
                ),
                "actor_parameter_count": self.actor_parameter_count(),
                "critic_parameter_count": self.critic_parameter_count(),
                "actor_learning_rate": float(self.actor_optimizer.param_groups[0]["lr"]),
                "critic_learning_rate": float(self.critic_optimizer.param_groups[0]["lr"]),
            }
            append_jsonl(
                self._compact_train_metrics(latest),
                self.output_dir / "train_metrics.jsonl",
            )

            completed_this_run = update - first_update + 1
            elapsed = perf_counter() - start_time
            eta = (
                elapsed
                / completed_this_run
                * max(total_updates - update, 0)
            )

            if update % self.log_every == 0 or update == total_updates:
                self.logger.info(
                    "update %d/%d | episodes %d | return %.3f | success %.3f | "
                    "policy %.4f | value %.4f | entropy %.4f | elapsed %s | eta %s",
                    update,
                    total_updates,
                    int(latest["completed_episodes"]),
                    latest["mean_episode_return"],
                    latest["success_rate"],
                    latest["policy_loss"],
                    latest["value_loss"],
                    latest["entropy"],
                    format_duration(elapsed),
                    format_duration(eta),
                )

            evaluation_cfg = dict(self.config.get("evaluation", {}))
            evaluation_updates = {
                int(value) for value in evaluation_cfg.get("updates", [])
            }
            if any(value <= 0 for value in evaluation_updates):
                raise ValueError("evaluation updates must be positive")
            evaluation_interval = max(
                1,
                int(
                    evaluation_cfg.get(
                        "interval_updates", save_every
                    )
                ),
            )
            should_evaluate = (
                update in evaluation_updates
                if evaluation_updates else update % evaluation_interval == 0
            ) or update == total_updates
            evaluation_seconds = 0.0
            if should_evaluate:
                evaluation_started = perf_counter()
                save_episode_records = bool(
                    evaluation_cfg.get("save_episode_records", False)
                )
                evaluation = self.evaluate(
                    episodes=int(evaluation_cfg.get("episodes", 10)),
                    seed_base=int(
                        evaluation_cfg.get("seed_base", 30_000)
                    ),
                    episode_records_path=(
                        self.output_dir / "evaluation_episodes.jsonl"
                        if save_episode_records
                        else None
                    ),
                    append_episode_records=save_episode_records,
                )
                append_jsonl(
                    {"update": update, **evaluation},
                    self.output_dir / "evaluation_metrics.jsonl",
                )
                self._save_best_checkpoints(evaluation)
                evaluation_seconds = perf_counter() - evaluation_started

            checkpoint_updates = {
                int(value)
                for value in self.train_config.get("checkpoint_updates", [])
            }
            if any(value <= 0 for value in checkpoint_updates):
                raise ValueError("checkpoint updates must be positive")
            should_checkpoint = (
                update in checkpoint_updates
                if checkpoint_updates else update % save_every == 0
            ) or update == total_updates
            checkpoint_seconds = 0.0
            if should_checkpoint:
                checkpoint_started = perf_counter()
                checkpoint_path = self.checkpoint_dir / "latest.pt"
                self.save_checkpoint(checkpoint_path)
                if bool(self.train_config.get("save_periodic_checkpoints", False)):
                    self.save_checkpoint(
                        self.checkpoint_dir / f"update_{update:04d}.pt"
                    )
                self.logger.info(
                    "checkpoint saved | %s", checkpoint_path
                )
                checkpoint_seconds = perf_counter() - checkpoint_started

            append_jsonl(
                {
                    "schema_version": "ppo_runtime_timing_v1",
                    "update": int(update),
                    "rollout_seconds": float(rollout_seconds),
                    "learner_seconds": float(learner_seconds),
                    "evaluation_seconds": float(evaluation_seconds),
                    "checkpoint_seconds": float(checkpoint_seconds),
                    "total_update_seconds": float(
                        perf_counter() - update_started
                    ),
                    "training_telemetry_level": (
                        self.training_telemetry_level
                    ),
                    "consolidated_rollout_device_transfer": bool(
                        self.consolidate_rollout_device_transfer
                    ),
                },
                self.output_dir / "runtime_metrics.jsonl",
            )

        save_json(latest, self.output_dir / "train_summary.json")
        self.logger.info(
            "training complete | updates %d/%d | elapsed %s",
            self.current_update,
            total_updates,
            format_duration(perf_counter() - start_time),
        )
        return latest

    def evaluate(
        self,
        episodes: int = 10,
        seed_base: int = 30_000,
        layout_seeds: Sequence[int] | None = None,
        episode_records_path: str | Path | None = None,
        append_episode_records: bool = False,
        summary_output_path: str | Path | None = None,
        selector_deterministic: bool = True,
        reward_audit_dir: str | Path | None = None,
        intervention_dataset_dir: str | Path | None = None,
        counterfactual_dataset_dir: str | Path | None = None,
        r2r_gate_report_path: str | Path | None = None,
        counterfactual_expected_layout_count: int = 200,
        policy_deterministic: bool = True,
    ) -> dict[str, Any]:
        """确定性策略评估。

        ``episode_records_path`` 仅增加逐回合审计产物，不改变策略执行和聚合指标。
        """
        self._base_env().set_runtime_telemetry_level(
            self.evaluation_telemetry_level
        )
        if layout_seeds is not None:
            frozen_layout_seeds = tuple(int(value) for value in layout_seeds)
            if not frozen_layout_seeds or len(set(frozen_layout_seeds)) != len(
                frozen_layout_seeds
            ):
                raise ValueError("layout_seeds must be a non-empty unique sequence")
            episodes = len(frozen_layout_seeds)
        else:
            frozen_layout_seeds = tuple(
                int(seed_base) + episode for episode in range(int(episodes))
            )
        if int(episodes) <= 0:
            raise ValueError("evaluation episodes must be positive")
        audit_dir = Path(reward_audit_dir) if reward_audit_dir is not None else None
        if audit_dir is not None and audit_dir.exists():
            raise FileExistsError(f"reward audit output already exists: {audit_dir}")
        dataset_dir = (
            Path(intervention_dataset_dir)
            if intervention_dataset_dir is not None else None
        )
        if dataset_dir is not None and dataset_dir.exists():
            raise FileExistsError(
                f"intervention dataset output already exists: {dataset_dir}"
            )
        counterfactual_dir = (
            Path(counterfactual_dataset_dir)
            if counterfactual_dataset_dir is not None else None
        )
        if counterfactual_dir is not None and counterfactual_dir.exists():
            raise FileExistsError(
                f"R3 counterfactual dataset output already exists: {counterfactual_dir}"
            )
        gate_report = (
            Path(r2r_gate_report_path)
            if r2r_gate_report_path is not None else None
        )
        if counterfactual_dir is not None:
            if gate_report is None or not gate_report.is_file():
                raise FileNotFoundError(
                    "R3 collection requires the passed R2R GateFix report"
                )
            gate_payload = json.loads(gate_report.read_text(encoding="utf-8"))
            if gate_payload.get("schema_version") != "IAMAPPOR2RCombinedDecisionV2":
                raise ValueError("R3 requires the versioned R2R GateFix V2 report")
            if gate_payload.get("primary_decision") not in {
                "advance_r3_flat_and_agent", "advance_r3_agent_encoder",
            }:
                raise RuntimeError("R2R GateFix decision does not permit R3")
        self._set_target_obstacles()
        intervention_dataset = None
        counterfactual_dataset = None
        if dataset_dir is not None:
            base_env = self._base_env()
            telemetry = dict(
                self.config.get("safety_execution", {}).get("telemetry", {})
            )
            intervention_dataset = InterventionDatasetAccumulatorV1(
                policy_seed=int(self.config.get("seed", 0)),
                checkpoint_update=int(self.current_update),
                task_config_fingerprint=self.task_config_fingerprint(),
                checkpoint_sha256=self._loaded_checkpoint_hash,
                agent_radius=float(base_env.agent_radius),
                map_size=tuple(float(value) for value in base_env.map_size),
                obstacle_safety_margin=float(
                    base_env.oracle_config["obstacle_safety_margin"]
                ),
                any_action_l2=float(telemetry.get("any_action_l2", 1e-7)),
                material_delta_v_mps=float(
                    telemetry.get("material_delta_v_mps", 0.05)
                ),
                large_delta_v_mps=float(
                    telemetry.get("large_delta_v_mps", 0.25)
                ),
            )
        if counterfactual_dir is not None:
            expected_counterfactual_layouts = int(
                counterfactual_expected_layout_count
            )
            if expected_counterfactual_layouts <= 0:
                raise ValueError(
                    "counterfactual_expected_layout_count must be positive"
                )
            if (
                layout_seeds is None
                or len(frozen_layout_seeds) != expected_counterfactual_layouts
            ):
                raise ValueError(
                    "counterfactual collection layout count differs from "
                    "its registered diagnostic panel"
                )
            if not self._loaded_checkpoint_hash:
                raise RuntimeError("R3 collection requires a loaded checkpoint hash")
            counterfactual_dataset = R3ShadowDatasetAccumulatorV1(
                policy_seed=int(self.config.get("seed", 0)),
                checkpoint_update=int(self.current_update),
                checkpoint_sha256=str(self._loaded_checkpoint_hash),
                task_config_fingerprint=self.task_config_fingerprint(),
                r2r_gate_report_sha256=hashlib.sha256(
                    gate_report.read_bytes()
                ).hexdigest(),
                expected_layout_count=expected_counterfactual_layouts,
                action_query_kind=(
                    "bounded_actor_mean" if policy_deterministic
                    else "sampled_bounded_rollout_action"
                ),
            )
        returns: list[float] = []
        successes: list[float] = []
        safe_successes: list[float] = []
        lengths: list[int] = []
        final_max_distances: list[float] = []
        final_required_distances: list[float] = []
        goal_progress_ratios: list[float] = []
        team_arrival_flags: list[float] = []
        agent_arrival_counts = np.zeros(self.num_agents, dtype=np.float64)
        min_obstacle_clearances: list[float] = []
        min_inter_agent_distances: list[float] = []
        visible_obstacle_truncation_rates: list[float] = []
        max_visible_obstacles_pre_truncation: list[int] = []
        selected_velocity_valid_rates: list[float] = []
        dynamic_obstacle_slot_coverages: list[float] = []
        near_goal_stall_episode_flags: list[bool] = []
        action_fixed_point_episode_flags: list[bool] = []
        actor_raw_fixed_point_episode_flags: list[bool] = []
        safety_absorbed_fixed_point_episode_flags: list[bool] = []
        fixed_point_unattributed_agent_counts: list[int] = []
        fixed_point_classification_conservation_flags: list[bool] = []
        raw_action_norm_samples: list[float] = []
        executed_action_norm_samples: list[float] = []
        shield_delta_norm_samples: list[float] = []
        executed_raw_norm_ratio_samples: list[float] = []
        requested_executed_cosine_samples: list[float] = []
        episode_collision_counts: list[int] = []
        episode_obstacle_collision_counts: list[int] = []
        episode_static_obstacle_collision_counts: list[int] = []
        episode_dynamic_obstacle_collision_counts: list[int] = []
        episode_inter_agent_collision_counts: list[int] = []
        episode_boundary_collision_counts: list[int] = []
        episode_boundary_side_counts: defaultdict[str, list[int]] = defaultdict(list)
        episode_boundary_contact_steps: list[int] = []
        episode_max_boundary_contact_streaks: list[int] = []
        episode_has_collisions: list[bool] = []
        first_collision_steps: list[int | None] = []
        first_collision_types: list[str | None] = []
        first_collision_goal_progress_ratios: list[float | None] = []
        guidance_metrics: defaultdict[str, list[float]] = defaultdict(list)
        reward_component_sums: defaultdict[str, float] = defaultdict(float)
        reward_component_abs_sums: defaultdict[str, float] = defaultdict(float)
        reward_component_steps = 0
        reason_counts: Counter[str] = Counter()
        action_sum = np.zeros((self.num_agents, 2), dtype=np.float64)
        action_positive_count = np.zeros((self.num_agents, 2), dtype=np.float64)
        action_sample_count = 0
        velocity_saturation_count = np.zeros(self.num_agents, dtype=np.float64)
        velocity_sample_count = 0
        episode_records: list[dict[str, Any]] = []
        reward_audit_episodes: list[dict[str, Any]] = []
        reward_audit_steps: list[dict[str, Any]] = []
        audit_gamma = float(self.train_config.get("gamma", 0.99))
        collect_alignment_metrics = bool(
            self.config.get("evaluation", {}).get(
                "safety_alignment_metrics", False
            )
        )
        avoidable_team_steps_total = 0
        evaluated_team_steps_total = 0
        material_intervention_team_steps_total = 0

        for episode, episode_seed in enumerate(frozen_layout_seeds):
            obs, _ = self.env.reset(seed=episode_seed)
            base_env = self._base_env()
            initial_goal_distances = np.asarray(
                base_env._goal_distances(), dtype=np.float64)
            minimum_goal_distances = initial_goal_distances.copy()
            minimum_joint_goal_distance = float(np.max(initial_goal_distances))
            first_6m_entry_steps = np.where(initial_goal_distances < 6.0, 0, -1).astype(np.int32)
            first_joint_6m_step = 0 if np.all(initial_goal_distances < 6.0) else -1
            hidden = self.actor.initial_hidden(
                self.policy_rows, self.device
            )
            total = 0.0
            discounted_total = 0.0
            episode_component_sums: defaultdict[str, float] = defaultdict(float)
            discounted_component_sums: defaultdict[str, float] = defaultdict(float)
            episode_raw_action_norms: list[np.ndarray] = []
            episode_executed_action_norms: list[np.ndarray] = []
            episode_shield_delta_norms: list[np.ndarray] = []
            episode_executed_raw_norm_ratios: list[np.ndarray] = []
            episode_requested_executed_cosines: list[np.ndarray] = []
            episode_stage_actions: dict[str, list[np.ndarray]] = {
                stage: [] for stage in ACTION_STAGE_ORDER
            }
            episode_positions = [base_env.positions.copy()]
            previous_subgoal_ids = base_env.subgoal_ids.copy()
            subgoal_switch_count = 0
            episode_avoidable_team_steps = 0
            episode_evaluated_team_steps = 0
            episode_material_intervention_team_steps = 0

            for step in range(self.env.max_episode_steps):
                collection_local_raw = None
                collection_local_normalized = None
                collection_global_raw = None
                collection_global_normalized = None
                if (
                    intervention_dataset is not None
                    or counterfactual_dataset is not None
                ):
                    collection_local_raw = np.asarray(obs, dtype=np.float32).copy()
                    collection_local_normalized = self.normalize_local_observations(
                        collection_local_raw
                    )
                    collection_global_raw = np.asarray(
                        self.env.get_global_state(), dtype=np.float32
                    ).copy()
                    collection_global_normalized = self.normalize_global_state(
                        collection_global_raw
                    )
                policy_obs = self.policy_observations(obs)
                with torch.no_grad():
                    output = self.actor.forward_step(
                        torch.as_tensor(
                            policy_obs,
                            dtype=torch.float32,
                            device=self.device,
                        ),
                        hidden,
                        bool(policy_deterministic),
                    )

                actions = output.action.cpu().numpy().reshape(self.num_agents, 2)
                counterfactual_record = None
                if counterfactual_dataset is not None:
                    counterfactual_record = evaluate_r3_shadow_step(
                        base_env,
                        raw_joint_action=actions,
                        global_state_normalized=collection_global_normalized,
                        actor_hidden=output.hidden,
                    )
                if collect_alignment_metrics:
                    snapshot = base_env._causal_safety_snapshot(actions)
                    goal_delta = np.asarray(
                        base_env.goals - base_env.positions,
                        dtype=np.float32,
                    )
                    norms = np.linalg.norm(goal_delta, axis=1, keepdims=True)
                    goal_actions = np.divide(
                        goal_delta, norms, out=np.zeros_like(goal_delta),
                        where=norms > 1e-8,
                    )
                    references = registered_reference_actions(
                        actions, goal_actions
                    )
                    label = evaluate_avoidable_intervention_v2(
                        snapshot,
                        (actions, *references),
                        base_env.evaluate_shadow_action_v2,
                        goal_directions=goal_delta,
                        current_velocities=np.asarray(
                            [row[2:4] for row in snapshot.team_observation],
                            dtype=np.float32,
                        ),
                        reached_mask=np.asarray(
                            base_env._goal_distances() < base_env.goal_radius,
                            dtype=bool,
                        ),
                        acceleration_scale=float(base_env.agent_max_acceleration),
                        dt=float(base_env.dt),
                        maximum_speed=float(base_env.agent_max_speed),
                    )
                    reference_costs = np.asarray(
                        label.shadow_stage_costs[1:7], dtype=np.float64
                    )
                    reference_valid = np.asarray(
                        label.valid_shadow_mask[1:7], dtype=bool
                    )
                    raw_costs = np.asarray(
                        label.raw_stage_costs, dtype=np.float64
                    )
                    episode_material_intervention_team_steps += int(
                        np.any(raw_costs[:2] > 1e-8)
                    )
                    avoidable = False
                    if bool(reference_valid.any()):
                        best = reference_costs[reference_valid, :2].min(axis=0)
                        avoidable = bool(np.any(raw_costs[:2] - best > 1e-8))
                    episode_avoidable_team_steps += int(avoidable)
                    episode_evaluated_team_steps += 1
                if audit_dir is not None:
                    positions_before = base_env.positions.copy()
                    velocities_before = base_env.velocities.copy()
                    subgoals_before = base_env.subgoals.copy()
                    subgoal_ids_before = base_env.subgoal_ids.copy()
                action_sum += actions
                action_positive_count += actions > 0.0
                action_sample_count += 1
                obs, rewards, terminated, truncated, info = self.env.step(actions)
                if counterfactual_dataset is not None:
                    trace = info.get("stage_intervention_trace")
                    if counterfactual_record is None or trace is None:
                        raise RuntimeError(
                            "R3 collection requires a shadow record and stage trace"
                        )
                    counterfactual_dataset.append_step(
                        counterfactual_record,
                        layout_id=int(episode_seed),
                        episode_index=int(episode),
                        step=int(step + 1),
                        trace=trace,
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                    )
                if intervention_dataset is not None:
                    snapshot = base_env._current_causal_safety_snapshot
                    trace = info.get("stage_intervention_trace")
                    if snapshot is None or trace is None:
                        raise RuntimeError(
                            "R2 collection requires causal snapshot and stage trace"
                        )
                    intervention_dataset.append_step(
                        layout_id=int(episode_seed),
                        episode_index=int(episode),
                        step=int(step + 1),
                        local_observations_raw=collection_local_raw,
                        local_observations_normalized=collection_local_normalized,
                        global_state_raw=collection_global_raw,
                        global_state_normalized=collection_global_normalized,
                        bounded_joint_action=actions,
                        causal_safety_context=np.asarray(
                            info["causal_safety_context"], dtype=np.float32
                        ),
                        current_safety_margin=float(
                            info["current_safety_margin"]
                        ),
                        snapshot=snapshot,
                        trace=trace,
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                    )
                requested_actions = np.asarray(
                    info.get("requested_actions", actions), dtype=np.float64
                )
                executed_actions = np.asarray(
                    info.get("executed_actions", actions), dtype=np.float64
                )
                stage_payload = dict(info.get("action_stage_actions", {}))
                stage_values: dict[str, np.ndarray] = {
                    "actor_raw": requested_actions,
                    "recovery_overlay": np.asarray(
                        stage_payload.get("recovery_overlay", requested_actions),
                        dtype=np.float64,
                    ),
                    "initial_joint_guard": np.asarray(
                        stage_payload.get("initial_joint_guard", executed_actions),
                        dtype=np.float64,
                    ),
                    "obstacle_ttc": np.asarray(
                        stage_payload.get("obstacle_ttc", executed_actions),
                        dtype=np.float64,
                    ),
                    "final_joint_guard": np.asarray(
                        stage_payload.get("final_joint_guard", executed_actions),
                        dtype=np.float64,
                    ),
                    "executed": executed_actions,
                }
                for stage in ACTION_STAGE_ORDER:
                    if stage_values[stage].shape != (self.num_agents, 2):
                        raise ValueError(
                            f"evaluation action stage {stage} has the wrong shape"
                        )
                    episode_stage_actions[stage].append(
                        stage_values[stage].copy()
                    )
                raw_norms = np.linalg.norm(requested_actions, axis=1)
                executed_norms = np.linalg.norm(executed_actions, axis=1)
                shield_deltas = np.linalg.norm(
                    executed_actions - requested_actions, axis=1
                )
                valid_raw = raw_norms > 1e-6
                norm_ratios = np.divide(
                    executed_norms,
                    raw_norms,
                    out=np.zeros_like(executed_norms),
                    where=valid_raw,
                )
                valid_angle = valid_raw & (executed_norms > 1e-6)
                cosines = np.divide(
                    np.sum(requested_actions * executed_actions, axis=1),
                    raw_norms * executed_norms,
                    out=np.zeros_like(executed_norms),
                    where=valid_angle,
                )
                episode_raw_action_norms.append(raw_norms)
                episode_executed_action_norms.append(executed_norms)
                episode_shield_delta_norms.append(shield_deltas)
                episode_executed_raw_norm_ratios.append(norm_ratios[valid_raw])
                episode_requested_executed_cosines.append(cosines[valid_angle])
                raw_action_norm_samples.extend(raw_norms.astype(float).tolist())
                executed_action_norm_samples.extend(executed_norms.astype(float).tolist())
                shield_delta_norm_samples.extend(shield_deltas.astype(float).tolist())
                executed_raw_norm_ratio_samples.extend(
                    norm_ratios[valid_raw].astype(float).tolist()
                )
                requested_executed_cosine_samples.extend(
                    cosines[valid_angle].astype(float).tolist()
                )
                episode_positions.append(base_env.positions.copy())
                current_subgoal_ids = base_env.subgoal_ids.copy()
                subgoal_switch_count += int(
                    np.any(current_subgoal_ids != previous_subgoal_ids)
                )
                previous_subgoal_ids = current_subgoal_ids
                goal_distances_after = np.asarray(info["goal_distances"], dtype=np.float64)
                minimum_goal_distances = np.minimum(
                    minimum_goal_distances, goal_distances_after)
                minimum_joint_goal_distance = min(
                    minimum_joint_goal_distance, float(np.max(goal_distances_after)))
                entered_6m = (goal_distances_after < 6.0) & (first_6m_entry_steps < 0)
                first_6m_entry_steps[entered_6m] = step + 1
                if first_joint_6m_step < 0 and np.all(goal_distances_after < 6.0):
                    first_joint_6m_step = step + 1
                velocity_saturation_count += np.any(
                    np.isclose(
                        np.abs(self.env.velocities),
                        self.env.agent_max_speed,
                        atol=1e-5,
                    ),
                    axis=1,
                )
                velocity_sample_count += 1
                hidden = output.hidden
                total += float(np.mean(rewards))
                if audit_dir is not None:
                    team_reward = float(np.mean(rewards))
                    discount = audit_gamma ** step
                    discounted_total += discount * team_reward
                    components = {
                        str(key): float(np.asarray(value, dtype=np.float64).mean())
                        for key, value in info.get("reward_components", {}).items()
                    }
                    if not np.isclose(components.get("total_reward", float("nan")),
                                      team_reward, atol=1e-5):
                        raise RuntimeError("reward audit total disagrees with environment reward")
                    for key, value in components.items():
                        episode_component_sums[key] += value
                        discounted_component_sums[key] += discount * value
                    reward_audit_steps.append({
                        "seed": int(episode_seed), "episode_index": int(episode),
                        "step": int(step + 1), "checkpoint_update": int(self.current_update),
                        "task_config_fingerprint": self.task_config_fingerprint(),
                        "positions_before": positions_before.tolist(),
                        "positions_after": base_env.positions.tolist(),
                        "velocities_before": velocities_before.tolist(),
                        "velocities_after": base_env.velocities.tolist(),
                        "requested_action": np.asarray(
                            info.get("requested_actions", actions)
                        ).tolist(),
                        "executed_action": np.asarray(
                            info.get("executed_actions", actions)
                        ).tolist(),
                        "boundary_guard": info.get("boundary_guard"),
                        "subgoals_before": subgoals_before.tolist(),
                        "subgoals_after": base_env.subgoals.tolist(),
                        "subgoal_ids_before": subgoal_ids_before.tolist(),
                        "subgoal_ids_after": base_env.subgoal_ids.tolist(),
                        "subgoal_changed": bool(np.any(subgoal_ids_before != base_env.subgoal_ids)
                                                or not np.array_equal(subgoals_before, base_env.subgoals)),
                        "goal_distances_after": np.asarray(info["goal_distances"]).tolist(),
                        "team_reward": team_reward,
                        "discounted_team_reward": discount * team_reward,
                        "reward_components_emitted_mean": components,
                        "terminated": bool(terminated), "truncated": bool(truncated),
                        "reason": str(info.get("reason", "unknown")),
                    })
                self._accumulate_reward_components(
                    info.get("reward_components", {}),
                    reward_component_sums,
                    reward_component_abs_sums,
                )
                reward_component_steps += 1

                if terminated or truncated:
                    if intervention_dataset is not None:
                        intervention_dataset.record_episode(
                            layout_id=int(episode_seed),
                            episode_index=int(episode),
                            outcome={
                                "length": int(step + 1),
                                "return": float(total),
                                "reason": str(info.get("reason", "unknown")),
                                "reach_success": bool(
                                    info.get(
                                        "reach_success",
                                        info.get("success", False),
                                    )
                                ),
                                "safe_success": bool(
                                    info.get("safe_success", False)
                                ),
                                "collision_count": int(
                                    info.get("collision_count", 0)
                                ),
                                "goal_progress_ratio": float(
                                    info.get("goal_progress_ratio", 0.0)
                                ),
                            },
                        )
                    if counterfactual_dataset is not None:
                        counterfactual_dataset.record_episode(
                            layout_id=int(episode_seed),
                            episode_index=int(episode),
                            outcome={
                                "length": int(step + 1),
                                "return": float(total),
                                "reason": str(info.get("reason", "unknown")),
                                "reach_success": bool(info.get(
                                    "reach_success", info.get("success", False)
                                )),
                                "safe_success": bool(
                                    info.get("safe_success", False)
                                ),
                                "collision_count": int(
                                    info.get("collision_count", 0)
                                ),
                                "goal_progress_ratio": float(
                                    info.get("goal_progress_ratio", 0.0)
                                ),
                            },
                        )
                    reach_success = bool(
                        info.get("reach_success", info.get("success", False))
                    )
                    safe_success = bool(info.get("safe_success", False))
                    returns.append(total)
                    successes.append(float(reach_success))
                    safe_successes.append(float(safe_success))
                    lengths.append(step + 1)
                    final_max_distances.append(
                        float(np.max(info["goal_distances"]))
                    )
                    final_required_distances.append(
                        float(info["required_goal_distance"])
                    )
                    goal_progress_ratios.append(
                        float(info.get("goal_progress_ratio", 0.0))
                    )
                    team_arrival_flags.append(
                        float(info.get("team_arrival_observed", False))
                    )
                    ever_in_goal = np.asarray(
                        info.get(
                            "agents_ever_in_goal", np.zeros(self.num_agents)
                        ),
                        dtype=np.float32,
                    )
                    if ever_in_goal.shape == (self.num_agents,):
                        agent_arrival_counts += ever_in_goal
                    min_obstacle_clearances.append(
                        float(info.get("min_obstacle_clearance", np.inf))
                    )
                    min_inter_agent_distances.append(
                        float(info.get("min_inter_agent_distance", np.inf))
                    )
                    visible_obstacle_truncation_rates.append(
                        float(
                            info.get(
                                "visible_obstacle_truncation_agent_step_rate",
                                0.0,
                            )
                        )
                    )
                    max_visible_obstacles_pre_truncation.append(
                        int(
                            info.get(
                                "max_visible_obstacles_pre_truncation", 0
                            )
                        )
                    )
                    selected_velocity_valid_rates.append(
                        float(info.get("episode_selected_velocity_valid_rate", 0.0))
                    )
                    dynamic_coverage = info.get("episode_dynamic_obstacle_slot_coverage")
                    if dynamic_coverage is not None:
                        dynamic_obstacle_slot_coverages.append(float(dynamic_coverage))
                    stall_flags = np.asarray(
                        info.get("near_goal_stall_triggered", np.zeros(self.num_agents)),
                        dtype=bool,
                    )
                    near_goal_stall_episode_flags.append(bool(np.any(stall_flags)))
                    episode_collision_counts.append(int(info.get("collision_count", 0)))
                    episode_obstacle_collision_counts.append(
                        int(info.get("obstacle_collision_count", 0))
                    )
                    episode_static_obstacle_collision_counts.append(
                        int(info.get("static_obstacle_collision_count", 0))
                    )
                    episode_dynamic_obstacle_collision_counts.append(
                        int(info.get("dynamic_obstacle_collision_count", 0))
                    )
                    episode_inter_agent_collision_counts.append(
                        int(info.get("inter_agent_collision_count", 0))
                    )
                    episode_boundary_collision_counts.append(
                        int(info.get("boundary_collision_count", 0))
                    )
                    for side in BOUNDARY_SIDES:
                        episode_boundary_side_counts[side].append(
                            int(info.get(f"{side}_boundary_collision_count", 0))
                        )
                    episode_boundary_contact_steps.append(
                        int(info.get("boundary_contact_step_count", 0))
                    )
                    episode_max_boundary_contact_streaks.append(
                        int(info.get("max_boundary_contact_streak", 0))
                    )
                    episode_has_collisions.append(bool(info.get("episode_has_collision", False)))
                    first_collision_steps.append(info.get("first_collision_step"))
                    first_collision_types.append(info.get("first_collision_type"))
                    first_collision_goal_progress_ratios.append(
                        info.get("first_collision_goal_progress_ratio")
                    )
                    for key in GUIDANCE_METRIC_KEYS:
                        guidance_metrics[key].append(float(info.get(key, 0.0)))
                    reason_counts[
                        str(info.get("reason", "unknown"))
                    ] += 1
                    final_distances = np.asarray(
                        info.get("goal_distances", np.full(self.num_agents, np.nan)),
                        dtype=np.float32,
                    )
                    first_entries = np.asarray(
                        info.get("first_goal_entry_steps", np.full(self.num_agents, -1)),
                        dtype=np.int32,
                    )
                    last_entries = np.asarray(
                        info.get("last_goal_entry_steps", np.full(self.num_agents, -1)),
                        dtype=np.int32,
                    )
                    exits = np.asarray(
                        info.get("goal_exit_counts", np.zeros(self.num_agents)),
                        dtype=np.int32,
                    )
                    in_goal_steps = np.asarray(
                        info.get("in_goal_step_counts", np.zeros(self.num_agents)),
                        dtype=np.int32,
                    )
                    collision_types = {
                        "obstacle": int(info.get("obstacle_collision_count", 0)) > 0,
                        "inter_agent": int(info.get("inter_agent_collision_count", 0)) > 0,
                        "boundary": int(info.get("boundary_collision_count", 0)) > 0,
                    }
                    valid_first_entries = first_entries[first_entries >= 0]
                    raw_norm_matrix = np.stack(episode_raw_action_norms)
                    executed_norm_matrix = np.stack(episode_executed_action_norms)
                    shield_delta_matrix = np.stack(episode_shield_delta_norms)
                    attribution = classify_fixed_point_attribution(
                        reason=str(info.get("reason", "unknown")),
                        final_goal_distances=final_distances,
                        unfinished_distance_threshold=float(
                            base_env.goal_radius
                            + base_env.near_goal_stall_distance_margin
                        ),
                        positions=episode_positions,
                        action_stage_traces=episode_stage_actions,
                        action_threshold=float(
                            base_env.near_goal_stall_action_threshold
                        ),
                        displacement_threshold_m=0.25,
                        tail_steps=20,
                    )
                    fixed_point_agents = np.asarray(
                        attribution["execution_fixed_point_agents"], dtype=bool
                    )
                    action_fixed_point = bool(
                        attribution["execution_fixed_point"]
                    )
                    action_fixed_point_episode_flags.append(action_fixed_point)
                    actor_raw_fixed_point_episode_flags.append(bool(
                        attribution["actor_raw_fixed_point"]
                    ))
                    safety_absorbed_fixed_point_episode_flags.append(bool(
                        attribution["safety_absorbed_fixed_point"]
                    ))
                    fixed_point_unattributed_agent_counts.append(int(
                        attribution["fixed_point_unattributed_count"]
                    ))
                    fixed_point_classification_conservation_flags.append(bool(
                        attribution["classification_count_conserved"]
                    ))
                    stage_norms = dict(
                        attribution[
                            "last_20_stage_action_norm_mean_per_agent"
                        ]
                    )
                    tail_action_mean_per_agent = np.asarray(
                        stage_norms["executed"], dtype=np.float64
                    )
                    tail_displacement_per_agent = np.asarray(
                        attribution["last_20_displacement_m"],
                        dtype=np.float64,
                    )
                    progress_ratio = float(info.get("goal_progress_ratio", 0.0))
                    failure_phase = (
                        "success" if safe_success else
                        "early" if progress_ratio < 1.0 / 3.0 else
                        "middle" if progress_ratio < 0.8 else "near_goal"
                    )
                    episode_records.append(
                        {
                            "update": int(self.current_update),
                            "train_seed": self.seed,
                            "algorithm": self.algorithm,
                            "algorithm_variant": self.algorithm_variant,
                            "method": str(
                                self.config.get("evaluation", {}).get(
                                    "paper_method",
                                    {
                                        "disabled": "standard_mappo",
                                        "counterfactual_adapter": "csa_mappo",
                                        "lagrangian": "mappo_lagrangian",
                                    }.get(
                                        str(getattr(self, "safety_alignment_mode", "disabled")),
                                        "standard_mappo",
                                    ),
                                )
                            ),
                            "task_config_fingerprint": self.task_config_fingerprint(),
                            "environment_contract_fingerprint": (
                                self.environment_contract_fingerprint()
                            ),
                            "learner_contract_fingerprint": (
                                self.learner_contract_fingerprint()
                            ),
                            "checkpoint_sha256": self._loaded_checkpoint_hash,
                            "collision_mode": self.env.collision_mode,
                            "outcome": info.get("outcome"),
                            "terminated": bool(terminated), "truncated": bool(truncated),
                            "first_failure_window": info.get("first_failure_window", []),
                            "terminal_window": info.get("terminal_window", []),
                            "episode_index": int(episode),
                            "seed": int(episode_seed),
                            "reach_success": reach_success,
                            "safe_success": safe_success,
                            "reason": str(info.get("reason", "unknown")),
                            "episode_length": int(step + 1),
                            "episode_return": float(total),
                            "goal_progress_ratio": progress_ratio,
                            "avoidable_intervention_team_steps": int(
                                episode_avoidable_team_steps
                            ),
                            "alignment_evaluated_team_steps": int(
                                episode_evaluated_team_steps
                            ),
                            "avoidable_intervention_rate": float(
                                episode_avoidable_team_steps
                                / max(episode_evaluated_team_steps, 1)
                            ),
                            "material_intervention_rate": float(
                                episode_material_intervention_team_steps
                                / max(episode_evaluated_team_steps, 1)
                            ),
                            "max_goal_progress_ratio": float(info.get("max_goal_progress_ratio", 0.0)),
                            "initial_goal_distances": initial_goal_distances.tolist(),
                            "minimum_goal_distances": minimum_goal_distances.tolist(),
                            "minimum_joint_goal_distance": minimum_joint_goal_distance,
                            "first_6m_entry_steps": first_6m_entry_steps.tolist(),
                            "first_joint_6m_step": first_joint_6m_step,
                            "agents_ever_in_goal": ever_in_goal.astype(bool).tolist(),
                            "first_goal_entry_steps": first_entries.tolist(),
                            "last_goal_entry_steps": last_entries.tolist(),
                            "goal_exit_counts": exits.tolist(),
                            "in_goal_step_counts": in_goal_steps.tolist(),
                            "arrival_step_gap": (
                                int(np.max(valid_first_entries) - np.min(valid_first_entries))
                                if valid_first_entries.size == self.num_agents else None
                            ),
                            "final_goal_distances": final_distances.tolist(),
                            "collision_count": int(info.get("collision_count", 0)),
                            "obstacle_collision_count": int(info.get("obstacle_collision_count", 0)),
                            "static_obstacle_collision_count": int(info.get("static_obstacle_collision_count", 0)),
                            "dynamic_obstacle_collision_count": int(info.get("dynamic_obstacle_collision_count", 0)),
                            "inter_agent_collision_count": int(info.get("inter_agent_collision_count", 0)),
                            "boundary_collision_count": int(info.get("boundary_collision_count", 0)),
                            "collision_types": collision_types,
                            "first_collision_step": info.get("first_collision_step"),
                            "first_collision_type": info.get("first_collision_type"),
                            "min_obstacle_clearance": float(info.get("min_obstacle_clearance", np.inf)),
                            "max_visible_obstacles_pre_truncation": int(
                                info.get("max_visible_obstacles_pre_truncation", 0)
                            ),
                            "visible_obstacle_truncation_agent_step_rate": float(
                                info.get(
                                    "visible_obstacle_truncation_agent_step_rate",
                                    0.0,
                                )
                            ),
                            "observation_schema_version": info.get("observation_schema_version"),
                            "obstacle_selection_version": info.get("obstacle_selection_version"),
                            "selected_velocity_valid_rate": float(
                                info.get("episode_selected_velocity_valid_rate", 0.0)
                            ),
                            "dynamic_obstacle_slot_coverage": info.get(
                                "episode_dynamic_obstacle_slot_coverage"
                            ),
                            "near_goal_stall_triggered": stall_flags.tolist(),
                            "near_goal_stall_first_steps": np.asarray(
                                info.get("near_goal_stall_first_steps", [-1] * self.num_agents),
                                dtype=np.int32,
                            ).tolist(),
                            "near_goal_stall_max_streaks": np.asarray(
                                info.get("near_goal_stall_max_streaks", [0] * self.num_agents),
                                dtype=np.int32,
                            ).tolist(),
                            "raw_action_norm_mean": float(np.mean(raw_norm_matrix)),
                            "raw_action_norm_p05": float(np.quantile(raw_norm_matrix, 0.05)),
                            "raw_action_norm_p50": float(np.quantile(raw_norm_matrix, 0.50)),
                            "raw_action_norm_p95": float(np.quantile(raw_norm_matrix, 0.95)),
                            "executed_action_norm_mean": float(np.mean(executed_norm_matrix)),
                            "executed_action_norm_p05": float(np.quantile(executed_norm_matrix, 0.05)),
                            "executed_action_norm_p50": float(np.quantile(executed_norm_matrix, 0.50)),
                            "executed_action_norm_p95": float(np.quantile(executed_norm_matrix, 0.95)),
                            "shield_delta_norm_mean": float(np.mean(shield_delta_matrix)),
                            "executed_raw_action_norm_ratio_p50": float(np.median(
                                np.concatenate(episode_executed_raw_norm_ratios)
                            )) if any(row.size for row in episode_executed_raw_norm_ratios) else None,
                            "requested_executed_action_cosine_p50": float(np.median(
                                np.concatenate(episode_requested_executed_cosines)
                            )) if any(row.size for row in episode_requested_executed_cosines) else None,
                            "last_20_action_norm_mean": float(
                                np.mean(tail_action_mean_per_agent)
                            ),
                            "last_20_action_norm_mean_per_agent": (
                                tail_action_mean_per_agent.astype(float).tolist()
                            ),
                            "last_20_displacement_m": (
                                tail_displacement_per_agent.astype(float).tolist()
                            ),
                            "last_20_raw_action_norm_mean_per_agent": stage_norms[
                                "actor_raw"
                            ],
                            "last_20_recovery_action_norm_mean_per_agent": stage_norms[
                                "recovery_overlay"
                            ],
                            "last_20_initial_guard_action_norm_mean_per_agent": stage_norms[
                                "initial_joint_guard"
                            ],
                            "last_20_ttc_action_norm_mean_per_agent": stage_norms[
                                "obstacle_ttc"
                            ],
                            "last_20_final_guard_action_norm_mean_per_agent": stage_norms[
                                "final_joint_guard"
                            ],
                            "last_20_executed_action_norm_mean_per_agent": stage_norms[
                                "executed"
                            ],
                            "last_20_stage_delta_norm_mean_per_agent": attribution[
                                "last_20_stage_delta_norm_mean_per_agent"
                            ],
                            "last_20_total_action_delta_norm_mean_per_agent": attribution[
                                "last_20_total_action_delta_norm_mean_per_agent"
                            ],
                            "action_fixed_point_agents": fixed_point_agents.tolist(),
                            "action_fixed_point": action_fixed_point,
                            "actor_raw_fixed_point_agents": attribution[
                                "actor_raw_fixed_point_agents"
                            ],
                            "actor_raw_fixed_point": attribution[
                                "actor_raw_fixed_point"
                            ],
                            "execution_fixed_point_agents": attribution[
                                "execution_fixed_point_agents"
                            ],
                            "execution_fixed_point": attribution[
                                "execution_fixed_point"
                            ],
                            "safety_absorbed_fixed_point_agents": attribution[
                                "safety_absorbed_fixed_point_agents"
                            ],
                            "safety_absorbed_fixed_point": attribution[
                                "safety_absorbed_fixed_point"
                            ],
                            "fixed_point_unattributed_agents": attribution[
                                "fixed_point_unattributed_agents"
                            ],
                            "fixed_point_unattributed_count": attribution[
                                "fixed_point_unattributed_count"
                            ],
                            "fixed_point_primary_stage": attribution[
                                "fixed_point_primary_stage"
                            ],
                            "fixed_point_primary_stage_per_agent": attribution[
                                "fixed_point_primary_stage_per_agent"
                            ],
                            "fixed_point_stage_counts": attribution[
                                "fixed_point_stage_counts"
                            ],
                            "fixed_point_classification_count_conserved": attribution[
                                "classification_count_conserved"
                            ],
                            "subgoal_switch_count": int(subgoal_switch_count),
                            "subgoal_reach_count": int(base_env._subgoal_reach_count),
                            "stall_steps": int(np.max(np.asarray(
                                info.get("near_goal_stall_max_streaks", [0] * self.num_agents),
                                dtype=np.int32,
                            ))),
                            "failure_phase": failure_phase,
                            "oracle_recovery_attempt_count": int(info.get("oracle_recovery_attempt_count", 0)),
                            "oracle_recovery_success_count": int(info.get("oracle_recovery_success_count", 0)),
                            "oracle_recovery_success_rate": float(info.get("oracle_recovery_success_rate", 0.0)),
                            "previous_target_hold_count": int(info.get("previous_target_hold_count", 0)),
                            "blocked_final_goal_fallback_count": int(info.get("blocked_final_goal_fallback_count", 0)),
                            "stagnation_trigger_count": int(info.get("stagnation_trigger_count", 0)),
                            "repeated_direction_rejection_count": int(info.get("repeated_direction_rejection_count", 0)),
                            "safe_brake_step_count": int(info.get("safe_brake_step_count", 0)),
                            "safe_brake_entry_count": int(info.get("safe_brake_entry_count", 0)),
                            "safe_brake_exit_count": int(info.get("safe_brake_exit_count", 0)),
                            "max_safe_brake_streak": int(info.get("max_safe_brake_streak", 0)),
                            "pair_infeasible_count": int(info.get("pair_infeasible_count", 0)),
                            "pair_infeasible_entry_count": int(info.get("pair_infeasible_entry_count", 0)),
                            "safe_brake_entry_steps": list(info.get("safe_brake_entry_steps", [])),
                            "safe_brake_exit_steps": list(info.get("safe_brake_exit_steps", [])),
                            "safe_brake_durations": list(info.get("safe_brake_durations", [])),
                            "safe_brake_entry_reasons": list(info.get("safe_brake_entry_reasons", [])),
                            "persistent_safe_brake_timeout": bool(
                                str(info.get("reason", "unknown")) == "timeout"
                                and info.get("safe_brake_active", False)
                            ),
                            "safe_recover_step_count": int(info.get("safe_recover_step_count", 0)),
                            "safe_recover_entry_count": int(info.get("safe_recover_entry_count", 0)),
                            "safe_recover_exit_count": int(info.get("safe_recover_exit_count", 0)),
                            "safe_recovery_failure_count": int(info.get("safe_recovery_failure_count", 0)),
                            "safe_recovery_unavailable_count": int(info.get("safe_recovery_unavailable_count", 0)),
                            "safe_recovery_retry_count": int(info.get("safe_recovery_retry_count", 0)),
                            "safe_recovery_abort_count": int(info.get("safe_recovery_abort_count", 0)),
                            "safe_recovery_hard_abort_count": int(info.get("safe_recovery_hard_abort_count", 0)),
                            "safe_recovery_refresh_count": int(info.get("safe_recovery_refresh_count", 0)),
                            "safe_recovery_reentry_count": int(info.get("safe_recovery_reentry_count", 0)),
                            "safe_recovery_abort_then_success_count": int(info.get("safe_recovery_abort_then_success_count", 0)),
                            "safe_recovery_episode_failure": bool(info.get("safe_recovery_episode_failure", False)),
                            "safe_recovery_affected_agent_count": int(info.get("safe_recovery_affected_agent_count", 0)),
                            "joint_guard_intervention_count": int(info.get("joint_guard_intervention_count", 0)),
                            "initial_joint_guard_intervention_steps": list(info.get("initial_joint_guard_intervention_steps", [])),
                            "final_joint_guard_residual_count": int(info.get("final_joint_guard_residual_count", 0)),
                            "final_joint_guard_residual_steps": list(info.get("final_joint_guard_residual_steps", [])),
                            "obstacle_ttc_projection_count": int(info.get("obstacle_ttc_projection_count", 0)),
                            "obstacle_ttc_projection_steps": list(info.get("obstacle_ttc_projection_steps", [])),
                            "obstacle_ttc_fallback_count": int(info.get("obstacle_ttc_fallback_count", 0)),
                            "obstacle_ttc_post_action_goal_velocity_sum": float(info.get("obstacle_ttc_post_action_goal_velocity_sum", 0.0)),
                            "obstacle_ttc_post_action_goal_velocity_count": int(info.get("obstacle_ttc_post_action_goal_velocity_count", 0)),
                            "obstacle_ttc_post_action_min_clearance_m": info.get("obstacle_ttc_post_action_min_clearance_m"),
                            "ttc_headroom_audit_events": list(info.get("ttc_headroom_audit_events", [])),
                            "ttc_headroom_intervention_event_count": int(info.get("ttc_headroom_intervention_event_count", 0)),
                            "ttc_headroom_positive_event_count": int(info.get("ttc_headroom_positive_event_count", 0)),
                            "ttc_v2_high_progress_selection_count": int(info.get("ttc_v2_high_progress_selection_count", 0)),
                            "total_action_modification_count": int(info.get("total_action_modification_count", 0)),
                            "goal_corridor_projection_count": int(info.get("goal_corridor_projection_count", 0)),
                            "goal_corridor_agent_projection_count": int(info.get("goal_corridor_agent_projection_count", 0)),
                            "goal_corridor_projection_steps": list(info.get("goal_corridor_projection_steps", [])),
                            "goal_corridor_tangential_progress_m": float(info.get("goal_corridor_tangential_progress_m", 0.0)),
                            "goal_corridor_nonpositive_progress_count": int(info.get("goal_corridor_nonpositive_progress_count", 0)),
                            "recovery_incident_count": int(info.get("recovery_incident_count", 0)),
                            "recovery_incident_merge_count": int(info.get("recovery_incident_merge_count", 0)),
                            "recovery_incident_reentry_reset_count": int(info.get("recovery_incident_reentry_reset_count", 0)),
                            "recovery_duplicate_hard_abort_count": int(info.get("recovery_duplicate_hard_abort_count", 0)),
                            "max_recovery_incident_steps": int(info.get("max_recovery_incident_steps", 0)),
                            "terminal_recovery_unresolved": bool(info.get("terminal_recovery_unresolved", False)),
                            "recovery_watchdog_step_count": int(info.get("recovery_watchdog_step_count", 0)),
                            "max_recovery_direction_unavailable_streak": int(info.get("max_recovery_direction_unavailable_streak", 0)),
                            "recovery_incident_close_count": int(info.get("recovery_incident_close_count", 0)),
                            "recovery_incident_reopen_count": int(info.get("recovery_incident_reopen_count", 0)),
                            "hazard_retrigger_count": int(info.get("hazard_retrigger_count", 0)),
                            "same_hazard_reentry_count": int(info.get("same_hazard_reentry_count", 0)),
                            "hazard_resolution_steps": list(info.get("hazard_resolution_steps", [])),
                            "incident_close_reason_counts": dict(info.get("incident_close_reason_counts", {})),
                            "terminal_incident_reconciled_count": int(info.get("terminal_incident_reconciled_count", 0)),
                            "recovery_telemetry_schema_version": info.get("recovery_telemetry_schema_version"),
                            "closure_state": info.get("closure_state", "ACTIVE"),
                            "clearance_pending_steps": int(info.get("clearance_pending_steps", 0)),
                            "pending_clearance_reappearance_count": int(info.get("pending_clearance_reappearance_count", 0)),
                            "post_certificate_retrigger_count": int(info.get("post_certificate_retrigger_count", 0)),
                            "close_without_certificate_count": int(info.get("close_without_certificate_count", 0)),
                            "stale_closure_bypass_count": int(info.get("stale_closure_bypass_count", 0)),
                            "release_certificate_count": int(info.get("release_certificate_count", 0)),
                            "release_certificate_id": int(info.get("release_certificate_id", 0)),
                            "recovery_lineage_id": info.get("recovery_lineage_id"),
                            "recovery_phase": info.get("recovery_phase", "IDLE"),
                            "escape_steps": int(info.get("escape_steps", 0)),
                            "consolidation_steps": int(info.get("consolidation_steps", 0)),
                            "release_shadow_check_passed": bool(info.get("release_shadow_check_passed", False)),
                            "release_shadow_check_count": int(info.get("release_shadow_check_count", 0)),
                            "release_shadow_check_failure_count": int(info.get("release_shadow_check_failure_count", 0)),
                            "release_margin_m": info.get("release_margin_m"),
                            "release_margin_delta_m": info.get("release_margin_delta_m"),
                            "bounded_recovery_exhaustion_count": int(info.get("bounded_recovery_exhaustion_count", 0)),
                            "bounded_exhaustion_then_success_count": int(info.get("bounded_exhaustion_then_success_count", 0)),
                            "rapid_same_hazard_retrigger_count": int(info.get("rapid_same_hazard_retrigger_count", 0)),
                            "recovery_lineage_failure_count": int(info.get("recovery_lineage_failure_count", 0)),
                            "recovery_attributable_mission_failure": bool(info.get("recovery_attributable_mission_failure", False)),
                            "recovery_attributable_failure_reason": info.get("recovery_attributable_failure_reason"),
                            "max_recovery_lineage_steps": int(info.get("max_recovery_lineage_steps", 0)),
                            "recovery_lineage_budget_reset_count": int(info.get("recovery_lineage_budget_reset_count", 0)),
                            "recovery_duplicate_failure_record_count": int(info.get("recovery_duplicate_failure_record_count", 0)),
                            "max_rapid_retriggers_per_lineage": int(info.get("max_rapid_retriggers_per_lineage", 0)),
                            "recovery_lineage_failure_emitted": bool(info.get("recovery_lineage_failure_emitted", False)),
                            "recovery_lineage_terminal_state": info.get("recovery_lineage_terminal_state", "IDLE"),
                            "recovery_candidate_selection_count": int(info.get("recovery_candidate_selection_count", 0)),
                            "recovery_candidate_rejection_count": int(info.get("recovery_candidate_rejection_count", 0)),
                            "recovery_candidate_rejection_reason_counts": dict(info.get("recovery_candidate_rejection_reason_counts", {})),
                            "recovery_candidate_post_projection_norm": info.get("recovery_candidate_post_projection_norm"),
                            "recovery_candidate_predicted_margin_gain": info.get("recovery_candidate_predicted_margin_gain"),
                            "recovery_candidate_fixed_point_streak": int(info.get("recovery_candidate_fixed_point_streak", 0)),
                            "release_shadow_horizon_min_margin": info.get("release_shadow_horizon_min_margin"),
                            "release_shadow_horizon_min_ttc": info.get("release_shadow_horizon_min_ttc"),
                            "action_stage_attribution": dict(info.get("action_stage_attribution", {})),
                            "max_safe_recover_streak": int(info.get("max_safe_recover_streak", 0)),
                            "boundary_recovery_trigger_count": int(info.get("boundary_recovery_trigger_count", 0)),
                            "safe_recovery_durations": list(info.get("safe_recovery_durations", [])),
                            "safe_recovery_events": list(info.get("safe_recovery_events", [])),
                            "persistent_safe_recovery_timeout": bool(
                                str(info.get("reason", "unknown")) == "timeout"
                                and (
                                    info.get("safety_mode") == "SAFE_RECOVER"
                                    or (
                                        info.get("star_safety_state_machine_version")
                                        not in {
                                            "star_persistent_recovery_v5_3",
                                            "star_persistent_recovery_v5_4",
                                            "star_closure_consistency_audit_v1",
                                        }
                                        and int(info.get("safe_recovery_failure_count", 0)) > 0
                                    )
                                    or int(info.get("recovery_lineage_failure_count", 0)) > 0
                                    or bool(info.get("terminal_recovery_unresolved", False))
                                )
                            ),
                            "minimum_observed_obstacle_ttc": info.get(
                                "minimum_observed_obstacle_ttc"
                            ),
                            "obstacle_ttc_risk_step_count": int(
                                info.get("obstacle_ttc_risk_step_count", 0)
                            ),
                            "first_obstacle_visible_steps": list(
                                info.get("first_obstacle_visible_steps", [])
                            ),
                            "star_safety_state_machine_version": info.get(
                                "star_safety_state_machine_version"
                            ),
                            "star_candidate_feasibility_version": info.get(
                                "star_candidate_feasibility_version"
                            ),
                        }
                    )
                    avoidable_team_steps_total += episode_avoidable_team_steps
                    evaluated_team_steps_total += episode_evaluated_team_steps
                    material_intervention_team_steps_total += (
                        episode_material_intervention_team_steps
                    )
                    if audit_dir is not None:
                        if not np.isclose(total, episode_component_sums["total_reward"], atol=1e-4):
                            raise RuntimeError("episode reward audit does not sum to episode return")
                        reward_audit_episodes.append({
                            "seed": int(episode_seed), "episode_index": int(episode),
                            "checkpoint_update": int(self.current_update),
                            "train_seed": int(self.seed),
                            "task_config_fingerprint": self.task_config_fingerprint(),
                            "outcome": info.get("outcome"),
                            "episode_length": int(step + 1),
                            "episode_return": total,
                            "discounted_episode_return": discounted_total,
                            "discount_gamma": audit_gamma,
                            "goal_progress_ratio": float(info.get("goal_progress_ratio", 0.0)),
                            "final_goal_distances": final_distances.tolist(),
                            "reward_components_emitted_mean_sum": dict(episode_component_sums),
                            "discounted_reward_components_emitted_mean_sum": dict(discounted_component_sums),
                            "warning": "emitted components include aliases and broadcast penalties; do not sum across keys",
                        })
                    break

        denominator = max(int(episodes), 1)
        finite_first_steps = [float(item) for item in first_collision_steps if item is not None]

        # Compact evaluation contract: one task score, distinct terminal causes,
        # one progress signal and three guidance-health signals. Detailed traces
        # are opt-in through episode_records_path instead of duplicated here.
        result = {
            "algorithm": self.algorithm,
            "algorithm_variant": self.algorithm_variant,
            "collision_mode": self.env.collision_mode,
            "episodes": int(episodes),
            "task_config_fingerprint": self.task_config_fingerprint(),
            "avoidable_intervention_rate": float(
                avoidable_team_steps_total
                / max(evaluated_team_steps_total, 1)
            ),
            "alignment_evaluated_team_steps": int(
                evaluated_team_steps_total
            ),
            "material_intervention_rate": float(
                material_intervention_team_steps_total
                / max(evaluated_team_steps_total, 1)
            ),
            "environment_contract_fingerprint": (
                self.environment_contract_fingerprint()
            ),
            "learner_contract_fingerprint": (
                self.learner_contract_fingerprint()
            ),
            "checkpoint_sha256": self._loaded_checkpoint_hash,
            "actor_parameter_count": self.actor_parameter_count(),
            "critic_parameter_count": self.critic_parameter_count(),
            **self.actor.distribution_stats(),
            "parent_task_config_fingerprint": self.parent_task_config_fingerprint,
            "curriculum_history": self.curriculum_history,
            "selector_intervention_used": bool(self.selector_intervention_used),
            "runtime_contract_assertion": deepcopy(
                getattr(self, "runtime_contract_assertion", None)
            ),
            "ood_evaluation_intervention_used": bool(
                self.ood_evaluation_intervention_used
            ),
            "ood_evaluation_changes": deepcopy(self.ood_evaluation_changes),
            "safety_ablation_intervention_used": bool(
                self.safety_ablation_intervention_used
            ),
            "safety_ablation_changes": deepcopy(
                self.safety_ablation_changes
            ),
            "task_config_override_used": bool(
                self.generic_task_config_override_used
            ),
            "cfms": float(np.mean(safe_successes)),
            "mean_episode_return": float(np.mean(returns)),
            "mean_episode_length": float(np.mean(lengths)),
            "mean_goal_progress_ratio": float(np.mean(goal_progress_ratios)),
            "team_arrival_rate": float(np.mean(team_arrival_flags)),
            "per_agent_arrival_rates": (
                agent_arrival_counts / denominator
            ).astype(float).tolist(),
            "mean_agent_arrival_rate": float(
                np.mean(agent_arrival_counts / denominator)
            ),
            "collision_rate": float(np.mean(episode_has_collisions)),
            "obstacle_collision_rate": float(np.mean(np.asarray(episode_obstacle_collision_counts) > 0)),
            "agent_collision_rate": float(np.mean(np.asarray(episode_inter_agent_collision_counts) > 0)),
            "boundary_collision_rate": float(np.mean(np.asarray(episode_boundary_collision_counts) > 0)),
            "mean_first_collision_step": (
                float(np.mean(finite_first_steps)) if finite_first_steps else None
            ),
            "timeout_rate": reason_counts["timeout"] / denominator,
            "mean_min_obstacle_clearance": self._finite_mean(
                min_obstacle_clearances
            ),
            "mean_min_inter_agent_distance": self._finite_mean(
                min_inter_agent_distances
            ),
            "mean_visible_obstacle_truncation_agent_step_rate": float(
                np.mean(visible_obstacle_truncation_rates)
            ),
            "episode_observation_capacity_exceeded_rate": float(
                np.mean(
                    np.asarray(max_visible_obstacles_pre_truncation)
                    > int(self.env.obstacle_slots)
                )
            ),
            "max_visible_obstacles_pre_truncation": int(
                max(max_visible_obstacles_pre_truncation, default=0)
            ),
            "mean_selected_velocity_valid_rate": float(
                np.mean(selected_velocity_valid_rates)
            ),
            "mean_dynamic_obstacle_slot_coverage": (
                float(np.mean(dynamic_obstacle_slot_coverages))
                if dynamic_obstacle_slot_coverages else None
            ),
            "near_goal_stall_episode_rate": float(
                np.mean(near_goal_stall_episode_flags)
            ),
            "action_fixed_point": bool(any(action_fixed_point_episode_flags)),
            "action_fixed_point_episode_rate": float(
                np.mean(action_fixed_point_episode_flags)
            ),
            "actor_raw_fixed_point": bool(
                any(actor_raw_fixed_point_episode_flags)
            ),
            "actor_raw_fixed_point_episode_rate": float(
                np.mean(actor_raw_fixed_point_episode_flags)
            ),
            "execution_fixed_point": bool(
                any(action_fixed_point_episode_flags)
            ),
            "execution_fixed_point_episode_rate": float(
                np.mean(action_fixed_point_episode_flags)
            ),
            "safety_absorbed_fixed_point": bool(
                any(safety_absorbed_fixed_point_episode_flags)
            ),
            "safety_absorbed_fixed_point_episode_rate": float(
                np.mean(safety_absorbed_fixed_point_episode_flags)
            ),
            "fixed_point_unattributed_count": int(
                np.sum(fixed_point_unattributed_agent_counts)
            ),
            "fixed_point_classification_count_conserved": bool(
                all(fixed_point_classification_conservation_flags)
            ),
            "raw_action_norm_mean": float(np.mean(raw_action_norm_samples)),
            "raw_action_norm_p05": float(np.quantile(raw_action_norm_samples, 0.05)),
            "raw_action_norm_p50": float(np.quantile(raw_action_norm_samples, 0.50)),
            "raw_action_norm_p95": float(np.quantile(raw_action_norm_samples, 0.95)),
            "executed_action_norm_mean": float(
                np.mean(executed_action_norm_samples)
            ),
            "executed_action_norm_p05": float(
                np.quantile(executed_action_norm_samples, 0.05)
            ),
            "executed_action_norm_p50": float(
                np.quantile(executed_action_norm_samples, 0.50)
            ),
            "executed_action_norm_p95": float(
                np.quantile(executed_action_norm_samples, 0.95)
            ),
            "shield_delta_norm_mean": float(np.mean(shield_delta_norm_samples)),
            "executed_raw_action_norm_ratio_p50": float(np.median(
                executed_raw_norm_ratio_samples
            )) if executed_raw_norm_ratio_samples else None,
            "requested_executed_action_cosine_p50": float(np.median(
                requested_executed_cosine_samples
            )) if requested_executed_cosine_samples else None,
            "subgoal_reach_rate": float(np.mean(guidance_metrics["subgoal_reach_rate"])),
            "subgoal_timeout_rate": float(np.mean(guidance_metrics["subgoal_timeout_rate"])),
            "invalid_subgoal_rate": float(np.mean(guidance_metrics["invalid_subgoal_rate"])),
            "blocked_primary_candidate_rate": float(
                np.mean(guidance_metrics["blocked_primary_candidate_rate"])
            ),
            "recovery_activation_rate": float(
                np.mean(guidance_metrics["recovery_activation_rate"])
            ),
            "stagnation_trigger_rate": float(
                np.mean(guidance_metrics["stagnation_trigger_rate"])
            ),
            "repeated_direction_rate": float(
                np.mean(guidance_metrics["repeated_direction_rate"])
            ),
            "shield_intervention_rate": float(
                np.sum(guidance_metrics["shield_intervention_count"])
                / max(np.sum(lengths), 1)
            ),
            "joint_guard_intervention_rate": float(
                np.sum(guidance_metrics["joint_guard_intervention_count"])
                / max(np.sum(lengths), 1)
            ),
            "initial_joint_guard_intervention_rate": float(
                np.sum(guidance_metrics["joint_guard_intervention_count"])
                / max(np.sum(lengths), 1)
            ),
            "final_joint_guard_residual_rate": float(
                np.sum(guidance_metrics["final_joint_guard_residual_count"])
                / max(np.sum(lengths), 1)
            ),
            "obstacle_ttc_projection_rate": float(
                np.sum(guidance_metrics["obstacle_ttc_projection_count"])
                / max(np.sum(lengths), 1)
            ),
            "obstacle_ttc_fallback_rate": float(
                np.sum(guidance_metrics["obstacle_ttc_fallback_count"])
                / max(np.sum(lengths), 1)
            ),
            "obstacle_ttc_post_action_goal_velocity_mean": float(
                np.sum(guidance_metrics[
                    "obstacle_ttc_post_action_goal_velocity_sum"
                ])
                / max(np.sum(guidance_metrics[
                    "obstacle_ttc_post_action_goal_velocity_count"
                ]), 1)
            ),
            "obstacle_ttc_post_action_min_clearance_m": min(
                (
                    float(row["obstacle_ttc_post_action_min_clearance_m"])
                    for row in episode_records
                    if row.get("obstacle_ttc_post_action_min_clearance_m")
                    is not None
                ),
                default=None,
            ),
            "total_action_modification_rate": float(
                np.sum(guidance_metrics["total_action_modification_count"])
                / max(np.sum(lengths), 1)
            ),
            "goal_corridor_projection_rate": float(
                np.sum(guidance_metrics["goal_corridor_projection_count"])
                / max(np.sum(lengths), 1)
            ),
            "goal_corridor_agent_projection_rate": float(
                np.sum(guidance_metrics["goal_corridor_agent_projection_count"])
                / max(np.sum(lengths) * self.num_agents, 1)
            ),
            "goal_corridor_tangential_progress_m": float(
                np.sum(guidance_metrics["goal_corridor_tangential_progress_m"])
            ),
            "goal_corridor_nonpositive_progress_count": int(
                np.sum(guidance_metrics["goal_corridor_nonpositive_progress_count"])
            ),
            "shield_invariant_violation_rate": float(
                np.sum(guidance_metrics["shield_invariant_violation_count"])
                / max(np.sum(lengths), 1)
            ),
            "safe_brake_rate": float(
                np.sum(guidance_metrics["safe_brake_step_count"])
                / max(np.sum(lengths), 1)
            ),
            "safe_brake_entry_count": int(
                np.sum(guidance_metrics["safe_brake_entry_count"])
            ),
            "safe_brake_exit_count": int(
                np.sum(guidance_metrics["safe_brake_exit_count"])
            ),
            "max_safe_brake_streak": int(
                np.max(guidance_metrics["max_safe_brake_streak"])
            ),
            "pair_infeasible_rate": float(
                np.sum(guidance_metrics["pair_infeasible_count"])
                / max(np.sum(lengths), 1)
            ),
            "pair_infeasible_entry_count": int(
                np.sum(guidance_metrics["pair_infeasible_entry_count"])
            ),
            "persistent_safe_brake_timeout_rate": float(np.mean([
                bool(row.get("persistent_safe_brake_timeout", False))
                for row in episode_records
            ])),
            "safe_recover_rate": float(
                np.sum(guidance_metrics["safe_recover_step_count"])
                / max(np.sum(lengths), 1)
            ),
            "safe_recover_entry_count": int(
                np.sum(guidance_metrics["safe_recover_entry_count"])
            ),
            "safe_recover_exit_count": int(
                np.sum(guidance_metrics["safe_recover_exit_count"])
            ),
            "safe_recovery_failure_rate": float(np.mean([
                int(row.get("safe_recovery_failure_count", 0)) > 0
                for row in episode_records
            ])),
            "safe_recovery_retry_count": int(
                np.sum(guidance_metrics["safe_recovery_retry_count"])
            ),
            "safe_recovery_abort_rate": float(np.mean([
                int(row.get("safe_recovery_abort_count", 0)) > 0
                for row in episode_records
            ])),
            "safe_recovery_hard_abort_rate": float(np.mean([
                int(row.get("safe_recovery_hard_abort_count", 0)) > 0
                for row in episode_records
            ])),
            "safe_recovery_episode_failure_rate": float(np.mean([
                bool(row.get("safe_recovery_episode_failure", False))
                for row in episode_records
            ])),
            "safe_recovery_refresh_count": int(
                np.sum(guidance_metrics["safe_recovery_refresh_count"])
            ),
            "safe_recovery_reentry_count": int(
                np.sum(guidance_metrics["safe_recovery_reentry_count"])
            ),
            "safe_recovery_abort_then_success_count": int(
                np.sum(guidance_metrics["safe_recovery_abort_then_success_count"])
            ),
            "recovery_incident_count": int(
                np.sum(guidance_metrics["recovery_incident_count"])
            ),
            "recovery_incident_merge_count": int(
                np.sum(guidance_metrics["recovery_incident_merge_count"])
            ),
            "recovery_incident_reentry_reset_count": int(
                np.sum(guidance_metrics["recovery_incident_reentry_reset_count"])
            ),
            "recovery_duplicate_hard_abort_count": int(
                np.sum(guidance_metrics["recovery_duplicate_hard_abort_count"])
            ),
            "max_recovery_incident_steps": int(
                np.max(guidance_metrics["max_recovery_incident_steps"])
            ),
            "terminal_recovery_unresolved_rate": float(np.mean([
                bool(row.get("terminal_recovery_unresolved", False))
                for row in episode_records
            ])),
            "recovery_watchdog_step_count": int(
                np.sum(guidance_metrics["recovery_watchdog_step_count"])
            ),
            "max_recovery_direction_unavailable_streak": int(
                np.max(guidance_metrics[
                    "max_recovery_direction_unavailable_streak"
                ])
            ),
            "recovery_incident_close_count": int(
                np.sum(guidance_metrics["recovery_incident_close_count"])
            ),
            "recovery_incident_reopen_count": int(
                np.sum(guidance_metrics["recovery_incident_reopen_count"])
            ),
            "hazard_retrigger_count": int(
                np.sum(guidance_metrics["hazard_retrigger_count"])
            ),
            "same_hazard_reentry_count": int(
                np.sum(guidance_metrics["same_hazard_reentry_count"])
            ),
            "terminal_incident_reconciled_count": int(
                np.sum(guidance_metrics["terminal_incident_reconciled_count"])
            ),
            "bounded_recovery_exhaustion_count": int(
                np.sum(guidance_metrics["bounded_recovery_exhaustion_count"])
            ),
            "bounded_recovery_exhaustion_episode_rate": float(np.mean([
                int(row.get("bounded_recovery_exhaustion_count", 0)) > 0
                for row in episode_records
            ])),
            "bounded_exhaustion_then_success_count": int(
                np.sum(guidance_metrics["bounded_exhaustion_then_success_count"])
            ),
            "rapid_same_hazard_retrigger_count": int(
                np.sum(guidance_metrics["rapid_same_hazard_retrigger_count"])
            ),
            "rapid_same_hazard_retrigger_episode_rate": float(np.mean([
                int(row.get("rapid_same_hazard_retrigger_count", 0)) > 0
                for row in episode_records
            ])),
            "rapid_retrigger_per_closed_incident_rate": float(
                np.sum(guidance_metrics["rapid_same_hazard_retrigger_count"])
                / max(np.sum(guidance_metrics["recovery_incident_close_count"]), 1)
            ),
            "recovery_lineage_failure_count": int(
                np.sum(guidance_metrics["recovery_lineage_failure_count"])
            ),
            "recovery_lineage_failure_rate": float(np.mean([
                int(row.get("recovery_lineage_failure_count", 0)) > 0
                for row in episode_records
            ])),
            "recovery_attributable_mission_failure_rate": float(np.mean([
                bool(row.get("recovery_attributable_mission_failure", False))
                for row in episode_records
            ])),
            "recovery_lineage_budget_reset_count": int(
                np.sum(guidance_metrics["recovery_lineage_budget_reset_count"])
            ),
            "recovery_duplicate_failure_record_count": int(
                np.sum(guidance_metrics["recovery_duplicate_failure_record_count"])
            ),
            "max_recovery_lineage_steps": int(
                np.max(guidance_metrics["max_recovery_lineage_steps"])
            ),
            "max_rapid_retriggers_per_lineage": int(
                np.max(guidance_metrics["max_rapid_retriggers_per_lineage"])
            ),
            "recovery_candidate_selection_count": int(
                np.sum(guidance_metrics["recovery_candidate_selection_count"])
            ),
            "recovery_candidate_rejection_count": int(
                np.sum(guidance_metrics["recovery_candidate_rejection_count"])
            ),
            "recovery_lineage_failure_emitted_rate": float(np.mean([
                bool(row.get("recovery_lineage_failure_emitted", False))
                for row in episode_records
            ])),
            "release_shadow_check_pass_rate": float(
                np.sum([
                    int(row.get("release_shadow_check_count", 0))
                    - int(row.get("release_shadow_check_failure_count", 0))
                    for row in episode_records
                ])
                / max(np.sum(guidance_metrics["release_shadow_check_count"]), 1)
            ),
            "persistent_safe_recovery_timeout_rate": float(np.mean([
                bool(row.get("persistent_safe_recovery_timeout", False))
                for row in episode_records
            ])),
            "max_safe_recover_streak": int(
                np.max(guidance_metrics["max_safe_recover_streak"])
            ),
            "boundary_recovery_trigger_count": int(
                np.sum(guidance_metrics["boundary_recovery_trigger_count"])
            ),
            "outcome_counts": dict(Counter(row["outcome"] for row in episode_records)),
        }
        rejection_reasons: Counter[str] = Counter()
        for row in episode_records:
            rejection_reasons.update(
                row.get("recovery_candidate_rejection_reason_counts", {})
            )
        result["recovery_candidate_rejection_reason_counts"] = dict(
            rejection_reasons
        )
        action_stage_totals: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        for row in episode_records:
            for stage, values in row.get("action_stage_attribution", {}).items():
                for key in (
                    "step_count", "modified_step_count", "delta_norm_sum",
                    "input_norm_sum", "output_norm_sum",
                    "direction_cosine_sum", "norm_ratio_sum",
                    "agent_count", "material_agent_count", "large_agent_count",
                    "direction_reversal_count",
                ):
                    action_stage_totals[str(stage)][key] += float(
                        values.get(key, 0.0)
                    )
        result["action_stage_attribution"] = {}
        for stage, values in action_stage_totals.items():
            count = max(values["step_count"], 1.0)
            result["action_stage_attribution"][stage] = {
                **dict(values),
                "modification_rate": values["modified_step_count"] / count,
                "material_modification_rate": (
                    values["material_agent_count"] / max(values["agent_count"], 1.0)
                ),
                "large_modification_rate": (
                    values["large_agent_count"] / max(values["agent_count"], 1.0)
                ),
                "direction_reversal_rate": (
                    values["direction_reversal_count"] / max(values["agent_count"], 1.0)
                ),
                "mean_delta_norm": values["delta_norm_sum"] / count,
                "mean_direction_cosine": (
                    values["direction_cosine_sum"] / count
                ),
                "mean_norm_ratio": values["norm_ratio_sum"] / count,
            }
        if intervention_dataset is not None:
            result["intervention_dataset"] = intervention_dataset.finalize(
                dataset_dir
            )
        if counterfactual_dataset is not None:
            result["counterfactual_dataset"] = counterfactual_dataset.finalize(
                counterfactual_dir
            )
        save_json(
            result,
            Path(summary_output_path)
            if summary_output_path is not None
            else self.output_dir / "evaluation_summary.json",
        )
        if episode_records_path is not None:
            records_path = Path(episode_records_path)
            if append_episode_records:
                for row in episode_records:
                    append_jsonl(row, records_path)
            else:
                save_jsonl(episode_records, records_path)
        if audit_dir is not None:
            if len(reward_audit_episodes) != int(episodes):
                raise RuntimeError("reward audit episode count mismatch")
            save_jsonl(reward_audit_episodes, audit_dir / "episodes.jsonl")
            save_jsonl(reward_audit_steps, audit_dir / "steps.jsonl")
        return result

    def _save_best_checkpoints(
        self, evaluation: Mapping[str, Any]
    ) -> None:
        cfms = float(evaluation["cfms"])
        reach_success_rate = float(
            evaluation.get("reach_success_rate", cfms)
        )
        collision_rate = float(evaluation["collision_rate"])
        timeout_rate = float(evaluation.get("timeout_rate", 0.0))
        progress = float(evaluation["mean_goal_progress_ratio"])
        subgoal_reach = float(evaluation.get("subgoal_reach_rate", 0.0))

        if reach_success_rate > 0.0:
            # A genuinely successful checkpoint always outranks a diagnostic
            # zero-success checkpoint. Safety and progress break ties only.
            model_key = (
                1.0,
                cfms,
                reach_success_rate,
                -collision_rate,
                -timeout_rate,
                progress,
                float(evaluation["mean_episode_return"]),
            )
        else:
            # With zero success, do not let progress alone promote a policy
            # that crashes in every episode. Minimise the dominant terminal
            # failure first, then use local reach/progress for diagnosis.
            model_key = (
                0.0,
                -max(collision_rate, timeout_rate),
                subgoal_reach,
                progress,
                -collision_rate,
                -timeout_rate,
                float(evaluation["mean_episode_return"]),
            )
        if self.best_model_key is None or model_key > self.best_model_key:
            self.best_model_key = model_key
            self.save_checkpoint(self.checkpoint_dir / "best.pt")

    # ==================================================================
    # Checkpoint
    # ==================================================================
    def save_checkpoint(self, path: str | Path) -> None:
        """保存模型、优化器和关键结构元数据。"""
        payload = {
            "algorithm": self.algorithm,
            "algorithm_variant": self.algorithm_variant,
            "policy_schema_version": self.POLICY_SCHEMA_VERSION,
            "update": self.current_update,
            "config": self.config,
            "num_agents": self.num_agents,
            "obs_dim": self.obs_dim,
            "global_state_dim": self.global_state_dim,
            "action_dim": self.policy_action_dim,
            "policy_obs_dim": self.policy_obs_dim,
            "policy_rows": self.policy_rows,
            "hidden_dim": self.hidden_dim,
            "task_config": self._task_config_payload(),
            "task_config_fingerprint": self.task_config_fingerprint(),
            "environment_contract_fingerprint": (
                self.environment_contract_fingerprint()
            ),
            "learner_contract_fingerprint": (
                self.learner_contract_fingerprint()
            ),
            "actor_parameter_count": self.actor_parameter_count(),
            "critic_parameter_count": self.critic_parameter_count(),
            "parent_task_config_fingerprint": self.parent_task_config_fingerprint,
            "selector_intervention_used": self.selector_intervention_used,
            "curriculum_history": self.curriculum_history,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None,
            "numpy_rng_state": np.random.get_state(),
            "training_rng_state": self.training_rng.bit_generator.state,
            "capture_steps_cumulative": self.capture_steps_cumulative,
            "best_model_key": self.best_model_key,
        }
        extension = self._checkpoint_extension_payload()
        overlap = set(payload) & set(extension)
        if overlap:
            raise ValueError(
                f"checkpoint extension overrides common fields: {sorted(overlap)}"
            )
        payload.update(extension)
        path = Path(path)
        ensure_dir(path.parent)
        torch.save(payload, path)
        self._loaded_checkpoint_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        append_jsonl(
            {
                "schema_version": "checkpoint_manifest_v1",
                "algorithm": self.algorithm,
                "algorithm_variant": self.algorithm_variant,
                "update": int(self.current_update),
                "checkpoint": str(path.resolve()),
                "checkpoint_sha256": self._loaded_checkpoint_hash,
                "task_config_fingerprint": self.task_config_fingerprint(),
                "environment_contract_fingerprint": (
                    self.environment_contract_fingerprint()
                ),
                "learner_contract_fingerprint": (
                    self.learner_contract_fingerprint()
                ),
                "actor_parameter_count": self.actor_parameter_count(),
                "critic_parameter_count": self.critic_parameter_count(),
            },
            self.output_dir / "checkpoint_manifest.jsonl",
        )

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        allow_task_config_mismatch: bool = False,
        allow_reward_config_mismatch: bool = False,
        allow_selector_config_mismatch: bool = False,
        allow_ood_evaluation_config_mismatch: bool = False,
        allow_safety_ablation_config_mismatch: bool = False,
        allow_recovery_study_evaluation: bool = False,
        allow_curriculum_obstacle_transition: bool = False,
    ) -> None:
        """加载结构完全兼容的checkpoint。"""
        path = Path(path)
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self._loaded_checkpoint_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = {
            "algorithm": self.algorithm,
            "policy_schema_version": self.POLICY_SCHEMA_VERSION,
            "num_agents": self.num_agents,
            "obs_dim": self.obs_dim,
            "global_state_dim": self.global_state_dim,
            "action_dim": self.policy_action_dim,
            "hidden_dim": self.hidden_dim,
        }
        if self.algorithm != "ppo":
            expected["algorithm_variant"] = self.algorithm_variant
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"checkpoint {key} is incompatible")
        checkpoint_fingerprint = payload.get("task_config_fingerprint")
        runtime_fingerprint = self.task_config_fingerprint()
        mismatch = checkpoint_fingerprint != runtime_fingerprint
        checkpoint_task_config = payload.get("task_config", {})
        recovery_study_only = False
        if allow_recovery_study_evaluation:
            canonical_parent = json.dumps(checkpoint_task_config, sort_keys=True, separators=(",", ":"))
            if hashlib.sha256(canonical_parent.encode()).hexdigest() != checkpoint_fingerprint:
                raise ValueError("checkpoint fingerprint does not match stored contract")
            if any((allow_task_config_mismatch, allow_reward_config_mismatch,
                    allow_selector_config_mismatch, allow_ood_evaluation_config_mismatch,
                    allow_safety_ablation_config_mismatch, allow_curriculum_obstacle_transition)):
                raise ValueError("recovery study cannot combine checkpoint overrides")
            parent = deepcopy(checkpoint_task_config)
            child = self._task_config_payload()
            parent_safety = parent.get("safety_execution", {})
            child_safety = child.get("safety_execution", {})
            old = parent_safety.pop("safety_state_machine_version", None)
            new = child_safety.pop("safety_state_machine_version", None)
            old_corridor = parent_safety.get("goal_boundary_corridor", {}).pop("enabled", False)
            new_corridor = child_safety.get("goal_boundary_corridor", {}).pop("enabled", False)
            recovery_study_only = bool(
                old in {"star_scoped_recovery_v4", "star_persistent_recovery_v5_4"}
                and new == "star_persistent_recovery_v5_4"
                and isinstance(old_corridor, bool) and new_corridor is True
                and parent == child
            )
            if not recovery_study_only:
                raise ValueError("recovery study permits only the declared V4-to-V5.4 executor transition")
        self.recovery_study_evaluation_only = bool(allow_recovery_study_evaluation)
        safety_contract_mismatch = (
            checkpoint_task_config.get("safety_execution", {})
            != self._task_config_payload().get("safety_execution", {})
        )
        if (
            self.config.get("research_line") == "ia_mappo_v1"
            and safety_contract_mismatch
            and not allow_safety_ablation_config_mismatch
        ):
            raise ValueError(
                "checkpoint safety_execution contract is incompatible and "
                "cannot be overridden for ia_mappo_v1"
            )
        reward_only_mismatch = False
        selector_only_mismatch = False
        ood_evaluation_only_mismatch = False
        safety_ablation_only_mismatch = False
        safety_ablation_changes: dict[str, dict[str, bool]] = {}
        curriculum_only_mismatch = False
        curriculum_transition: dict[str, Any] | None = None
        if mismatch and allow_reward_config_mismatch:
            checkpoint_without_reward = dict(checkpoint_task_config)
            runtime_without_reward = self._task_config_payload()
            checkpoint_without_reward.pop("reward", None)
            runtime_without_reward.pop("reward", None)
            reward_only_mismatch = checkpoint_without_reward == runtime_without_reward
        if mismatch and allow_selector_config_mismatch:
            checkpoint_without_selector = dict(checkpoint_task_config)
            runtime_without_selector = self._task_config_payload()
            for key in ("subgoal_selector", "llm_subgoal_selector"):
                checkpoint_without_selector.pop(key, None)
                runtime_without_selector.pop(key, None)
            selector_only_mismatch = (
                checkpoint_without_selector == runtime_without_selector
            )
        if mismatch and allow_ood_evaluation_config_mismatch:
            checkpoint_without_ood = deepcopy(checkpoint_task_config)
            runtime_without_ood = self._task_config_payload()
            checkpoint_env = checkpoint_without_ood.setdefault("env", {})
            runtime_env = runtime_without_ood.setdefault("env", {})
            changes: dict[str, dict[str, Any]] = {}
            for key in OOD_EVALUATION_ENV_KEYS:
                checkpoint_value = checkpoint_env.pop(key, None)
                runtime_value = runtime_env.pop(key, None)
                if checkpoint_value != runtime_value:
                    changes[f"env.{key}"] = {
                        "checkpoint": checkpoint_value,
                        "runtime": runtime_value,
                    }
            if allow_selector_config_mismatch:
                for key in ("subgoal_selector", "llm_subgoal_selector"):
                    checkpoint_without_ood.pop(key, None)
                    runtime_without_ood.pop(key, None)
            ood_evaluation_only_mismatch = (
                bool(changes)
                and checkpoint_without_ood == runtime_without_ood
            )
            if ood_evaluation_only_mismatch:
                self.ood_evaluation_changes = changes
        if mismatch and allow_safety_ablation_config_mismatch:
            registered_changes = _registered_safety_ablation_changes(
                checkpoint_task_config,
                self._task_config_payload(),
            )
            safety_ablation_only_mismatch = registered_changes is not None
            if registered_changes is not None:
                safety_ablation_changes = registered_changes
        if mismatch and allow_curriculum_obstacle_transition:
            encoded_parent_contract = json.dumps(
                checkpoint_task_config, sort_keys=True, separators=(",", ":")
            )
            if hashlib.sha256(encoded_parent_contract.encode("utf-8")).hexdigest() != checkpoint_fingerprint:
                raise ValueError("checkpoint task fingerprint does not match its stored contract")
            parent_contract = deepcopy(checkpoint_task_config)
            child_contract = self._task_config_payload()
            parent_env = parent_contract.setdefault("env", {})
            child_env = child_contract.setdefault("env", {})
            old_count = parent_env.pop("static_obstacle_num", None)
            new_count = child_env.pop("static_obstacle_num", None)
            curriculum_only_mismatch = (
                (old_count, new_count) in {(4, 6), (6, 8)}
                and parent_contract == child_contract
                and payload.get("seed", payload.get("config", {}).get("seed")) == self.seed
                and not allow_task_config_mismatch
                and not allow_reward_config_mismatch
                and not allow_selector_config_mismatch
                and not allow_ood_evaluation_config_mismatch
            )
            if curriculum_only_mismatch:
                curriculum_transition = {
                    "schema_version": "static_obstacle_curriculum_transition_v1",
                    "from_static_obstacles": int(old_count),
                    "to_static_obstacles": int(new_count),
                    "parent_checkpoint_sha256": self._loaded_checkpoint_hash,
                    "parent_checkpoint_update": int(payload["update"]),
                    "parent_task_config_fingerprint": str(checkpoint_fingerprint),
                    "child_task_config_fingerprint": runtime_fingerprint,
                    "optimizer_and_rng_restored": True,
                    "best_model_key_reset": True,
                }
        if allow_curriculum_obstacle_transition and not curriculum_only_mismatch:
            raise ValueError("curriculum transition permits only 4->6 or 6->8 static obstacles")
        if mismatch and not (
            allow_task_config_mismatch
            or reward_only_mismatch
            or selector_only_mismatch
            or ood_evaluation_only_mismatch
            or safety_ablation_only_mismatch
            or curriculum_only_mismatch
            or recovery_study_only
        ):
            raise ValueError(
                "checkpoint task configuration is incompatible; "
                "environment, reward, Oracle, selector, shared safety, "
                "normalization, model and PPO contract must match"
            )
        self.loaded_checkpoint_task_config_mismatch = bool(mismatch)
        self.generic_task_config_override_used = bool(
            mismatch
            and allow_task_config_mismatch
            and not reward_only_mismatch
            and not selector_only_mismatch
            and not ood_evaluation_only_mismatch
            and not safety_ablation_only_mismatch
        )
        self.selector_intervention_used = bool(
            selector_only_mismatch
            or (
                ood_evaluation_only_mismatch
                and allow_selector_config_mismatch
                and checkpoint_task_config.get("subgoal_selector")
                != self._task_config_payload().get("subgoal_selector")
            )
        )
        self.ood_evaluation_intervention_used = bool(
            ood_evaluation_only_mismatch
        )
        self.safety_ablation_intervention_used = bool(
            safety_ablation_only_mismatch
        )
        self.safety_ablation_changes = deepcopy(safety_ablation_changes)
        self.parent_task_config_fingerprint = (
            str(checkpoint_fingerprint)
            if (
                selector_only_mismatch
                or ood_evaluation_only_mismatch
                or safety_ablation_only_mismatch
                or curriculum_only_mismatch
                or recovery_study_only
            )
            else payload.get(
                "parent_task_config_fingerprint"
            )
        )
        self._resumed_from_checkpoint = True
        self.curriculum_history = deepcopy(payload.get("curriculum_history", []))
        if curriculum_transition is not None:
            self.curriculum_history.append(curriculum_transition)
        if mismatch:
            self.logger.warning(
                "explicit checkpoint task-config transition | checkpoint=%s runtime=%s selector_only=%s ood_eval_only=%s curriculum_only=%s",
                checkpoint_fingerprint,
                runtime_fingerprint,
                selector_only_mismatch,
                ood_evaluation_only_mismatch,
                curriculum_only_mismatch,
            )

        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        self.current_update = int(payload["update"])

        torch.set_rng_state(payload["torch_rng_state"].cpu())
        if self.device.type == "cuda" and payload.get("cuda_rng_states") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng_states"]])
        np.random.set_state(payload["numpy_rng_state"])
        if "training_rng_state" in payload:
            self.training_rng.bit_generator.state = payload[
                "training_rng_state"
            ]
        if self.capture_schedule is not None and "capture_steps_cumulative" not in payload:
            raise ValueError("scheduled capture resume requires cumulative exposure in checkpoint")
        self.capture_steps_cumulative = int(payload.get("capture_steps_cumulative", 0))
        loaded_best_key = payload.get("best_model_key")
        if loaded_best_key is None:
            loaded_best_key = payload.get("best_safe_return_key")
        self.best_model_key = (
            tuple(float(item) for item in loaded_best_key)
            if isinstance(loaded_best_key, (list, tuple))
            and len(loaded_best_key) in {4, 7}
            else None
        )
        self._load_checkpoint_extension(payload)
        if curriculum_only_mismatch:
            self.best_model_key = None
