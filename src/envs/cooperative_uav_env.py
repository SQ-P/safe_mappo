"""基于 SkyAgent 场景约束的多 UAV 连续二维路径规划环境。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence
from collections import deque
from copy import deepcopy
from dataclasses import replace
from itertools import product
from hashlib import sha256
import json
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.envs.entities import CircularObstacle, UAVState
from src.envs.collision_evidence import sphere_contact_fraction, transition_evidence
from src.planning.oracle_subgoal_planner import OracleSubgoalPlanner
from src.planning.boundary_safety import (
    BoundarySafetyGuardV1,
    JointGuardResult,
    JointSafetyGuardV2,
)
from src.safety.staged_execution import (
    CLOSURE_ACTIVE,
    CLOSURE_PENDING,
    CLOSURE_UNKNOWN,
    CLOSURE_VERIFIED,
    ClosureConsistencyV1,
    ClosureObservationV1,
    ObservableObstacleV1,
    StarRecoveryCommandV1,
    TeamPlanningSnapshotV1,
    assess_obstacle_actions,
    build_star_recovery_candidates,
    build_star_recovery_command,
    deterministic_hold_actions,
    minimum_predicted_obstacle_clearance_m,
    project_goal_boundary_corridor,
    project_obstacle_actions,
    obstacle_track_keys,
    TTCActionCandidateAuditV1,
    TTCActionCandidateEvaluationV1,
    registered_ttc_actions_v1,
    registered_ttc_headroom_actions,
    is_safe_progress_headroom,
    select_ttc_progress_candidate_v2,
)
from src.safety.contracts import (
    CausalSafetySnapshotV1,
    EnvironmentShadowSnapshotV1,
    InterventionLevel,
    StageInterventionRecordV1,
    StageInterventionTraceV1,
    action_modification_level,
)
from src.safety.counterfactual import ShadowSafetyResultV1, ShadowSafetyResultV2


_INCIDENT_RECOVERY_VERSIONS = frozenset({
    "star_incident_recovery_v5_1",
    "star_incident_recovery_v5_2",
    "star_persistent_recovery_v5_3",
    "star_persistent_recovery_v5_4",
    "star_closure_consistency_audit_v1",
})

_PERSISTENT_RECOVERY_VERSIONS = frozenset({
    "star_persistent_recovery_v5_3",
    "star_persistent_recovery_v5_4",
    "star_closure_consistency_audit_v1",
})


class _DisabledSubgoalSelector:
    """No-op placeholder that keeps the frozen Oracle path import-clean."""

    competence_model = None
    pairwise_progress_model = None

    def set_episode_seed(self, seed: int | None) -> None:
        del seed

    def set_planning_event(self, event_id: int) -> None:
        del event_id

    def set_candidate_generation_diagnostics(self, diagnostics: Any) -> None:
        del diagnostics


class _DisabledAttemptRecorder:
    """No-op recorder used when all historical selectors are disabled."""

    def __init__(self) -> None:
        self.records: list[Any] = []
        self.pending = None
        self.next_id = 0

    def reset(self) -> None:
        self.records.clear()
        self.pending = None
        self.next_id = 0

    def drain(self) -> list[Any]:
        result, self.records = self.records, []
        return result

    def drain_support_events(self) -> list[Any]:
        return []

    def observe_actions(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def observe_safe(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(values))
    return np.zeros_like(values) if norm <= 1e-8 else values / norm


class CooperativeUAVEnv(gym.Env):
    """面向中央联合 PPO 的连续双无人机导航环境。"""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 2}

    def __init__(self, config: Mapping[str, Any] | None = None):
        super().__init__()
        cfg = dict(config or {})
        self.config = cfg
        self.collision_mode = str(cfg.get("collision_mode", "rollback"))
        self.boundary_mode = str(cfg.get("boundary_mode", "collision"))
        self.reward_mode = str(cfg.get("reward_mode", "per_agent_v1"))
        self.evidence_window_steps = int(cfg.get("evidence_window_steps", 0))
        runtime_cfg = dict(cfg.get("runtime", {}))
        fastpath_cfg = dict(runtime_cfg.get("serial_fastpath", {}))
        self.runtime_telemetry_level = str(
            fastpath_cfg.get("telemetry_level", "full")
        )
        if self.runtime_telemetry_level not in {"full", "train_minimal"}:
            raise ValueError(
                "runtime telemetry_level must be full or train_minimal"
            )
        if self.collision_mode not in {"rollback", "terminal"}:
            raise ValueError("collision_mode must be rollback or terminal")
        if self.boundary_mode not in {"collision", "clip"}:
            raise ValueError("boundary_mode must be collision or clip")
        if self.reward_mode not in {
            "per_agent_v1", "team_v1", "team_v2_risk", "team_v3_potential",
            "team_v4_subgoal",
        }:
            raise ValueError("unsupported reward_mode")
        if self.evidence_window_steps < 0:
            raise ValueError("evidence_window_steps must be nonnegative")
        self._transition_window = deque(maxlen=max(1, self.evidence_window_steps))
        self.first_failure_window = []

        # 所有任务参数均可由 YAML 覆盖；默认值与主配置保持一致，便于直接实例化。
        self.num_agents = int(cfg.get("num_agents", 2))
        self.map_size = np.asarray(cfg.get("map_size", [100.0, 100.0]), dtype=np.float32)
        self.agent_radius = float(cfg.get("agent_radius", 1.5))
        self.agent_max_speed = float(cfg.get("agent_max_speed", 2.0))
        self.agent_max_acceleration = float(cfg.get("agent_max_acceleration", 1.0))
        self.dt = float(cfg.get("dt", 0.5))
        self.max_episode_steps = int(cfg.get("max_episode_steps", 300))
        self.boundary_recovery_distance = float(
            cfg.get("boundary_recovery_distance", 1.0)
        )

        self.start_x = float(cfg.get("start_x", 5.0))
        self.start_y_positions = np.asarray(cfg.get("start_y_positions", [35.0, 65.0]), dtype=np.float32)
        self.start_y_jitter = float(cfg.get("start_y_jitter", 2.0))
        self.goal_x = float(cfg.get("goal_x", 95.0))
        self.goal_y_positions = np.asarray(cfg.get("goal_y_positions", [35.0, 65.0]), dtype=np.float32)
        self.cross_goals = bool(cfg.get("cross_goals", True))
        self.goal_radius = float(cfg.get("goal_radius", 3.0))

        # 兼容旧测试/调用中的 obstacle_num：显式传入时将其解释为全静态障碍物。
        legacy_obstacle_num = cfg.get("obstacle_num")
        self.static_obstacle_num = int(
            cfg.get("static_obstacle_num", legacy_obstacle_num if legacy_obstacle_num is not None else 8)
        )
        self.dynamic_obstacle_num = int(
            cfg.get("dynamic_obstacle_num", 0 if legacy_obstacle_num is not None else 3)
        )
        self.obstacle_num = self.static_obstacle_num + self.dynamic_obstacle_num
        self.static_obstacles = self.dynamic_obstacle_num == 0
        self.obstacle_radius_range = np.asarray(cfg.get("obstacle_radius_range", [2.0, 5.0]), dtype=np.float32)
        self.dynamic_obstacle_velocity_range = np.asarray(cfg.get("dynamic_obstacle_velocity_range", [-1.0, 1.0]), dtype=np.float32)
        self.obstacle_spawn_clearance = float(cfg.get("obstacle_spawn_clearance", 2.0))
        self.obstacle_spawn_x_range = np.asarray(cfg.get("obstacle_spawn_x_range", [15.0, 85.0]), dtype=np.float32)
        self.obstacle_spawn_y_range = np.asarray(cfg.get("obstacle_spawn_y_range", [8.0, 92.0]), dtype=np.float32)

        self.detection_radius = float(cfg.get("detection_radius", 25.0))
        self.obstacle_slots = int(cfg.get("obstacle_slots", 4))
        self.observation_schema_version = str(
            cfg.get("observation_schema_version", "legacy_obstacle_v1")
        )
        self.obstacle_selection_version = str(
            cfg.get("obstacle_selection_version", "nearest_distance_v1")
        )
        self.obstacle_velocity_estimation_version = str(
            cfg.get("obstacle_velocity_estimation_version", "none_v1")
        )
        self.near_goal_stall_action_threshold = float(
            cfg.get("near_goal_stall_action_threshold", 1e-3)
        )
        self.near_goal_stall_distance_margin = float(
            cfg.get("near_goal_stall_distance_margin", 3.0)
        )
        self.near_goal_stall_patience_steps = int(
            cfg.get("near_goal_stall_patience_steps", 20)
        )
        self.render_mode = cfg.get("render_mode")

        reward_cfg = dict(cfg.get("reward", {}))
        self.progress_weight = float(reward_cfg.get("progress_weight", 1.0))
        self.step_penalty = float(reward_cfg.get("step_penalty", 0.1))
        self.collision_penalty = float(reward_cfg.get("collision_penalty", 50.0))
        self.goal_success_bonus = float(reward_cfg.get("goal_success_bonus", 100.0))
        self.cooperation_bonus = float(reward_cfg.get("cooperation_bonus", 5.0))
        # Optional failure-contract repair.  Keep the legacy default at zero so
        # existing team_v1/team_v2_risk tasks retain their exact semantics.
        self.timeout_penalty = float(reward_cfg.get("timeout_penalty", 0.0))
        # V3 uses final-task potential shaping rather than mutable Oracle subgoals.
        self.potential_progress_scale = float(
            reward_cfg.get("potential_progress_scale", 5.0)
        )
        self.potential_gamma = float(reward_cfg.get("potential_gamma", 0.99))
        self.bottleneck_weight = float(reward_cfg.get("bottleneck_weight", 0.7))
        self.control_effort_weight = float(
            reward_cfg.get("control_effort_weight", 0.0)
        )
        # V4 keeps SkyAgent's dense, subgoal-conditioned credit assignment, but
        # uses signed distance improvement instead of an absolute distance cost.
        # That distinction matters under terminal contact: a persistent negative
        # distance cost would otherwise make an early crash attractive because it
        # stops future costs.  The final-goal term is a bounded episodic record,
        # so progress survives terminal transitions and cannot be earned twice.
        self.subgoal_progress_scale = float(
            reward_cfg.get("subgoal_progress_scale", 0.25)
        )
        self.final_progress_scale = float(
            reward_cfg.get("final_progress_scale", 8.0)
        )
        proximity_risk_cfg = dict(cfg.get("proximity_risk", {}))
        self.proximity_risk_margin = float(proximity_risk_cfg.get("margin", 2.0))
        self.proximity_risk_max_penalty = float(
            proximity_risk_cfg.get("max_penalty", 0.25)
        )

        # Oracle 仅生成 observation 中已有的 2D subgoal，不改变低层网络输入宽度。
        oracle_defaults = {
            "enabled": False,
            "replanning_interval": 20,
            "lookahead_distance": 15.0,
            "sensor_range": 25.0,
            "attractive_gain": 1.0,
            "obstacle_repulsive_gain": 80.0,
            "teammate_repulsive_gain": 60.0,
            "obstacle_safety_margin": 1.5,
            "teammate_safety_distance": 6.0,
            "subgoal_reach_radius": 2.0,
            "min_subgoal_distance": 3.0,
            # 旧 checkpoint 缺少以下字段时保持原 fallback 语义；正式新配置显式开启。
            "recovery_enabled": False,
            "recovery_angle_offsets_degrees": [30.0, -30.0, 60.0, -60.0, 90.0, -90.0],
            "recovery_lookahead_distances": [15.0, 12.0, 9.0, 6.0],
            "recovery_goal_progress_weight": 1.0,
            "recovery_clearance_weight": 0.10,
            "recovery_turn_penalty_weight": 0.02,
            "final_goal_fallback_requires_clear_path": True,
        }
        self.oracle_config = oracle_defaults | dict(cfg.get("oracle_subgoal", {}))
        # 缺少版本的历史 checkpoint 必须保留旧任务语义；新训练 YAML 显式选择 v2。
        self.subgoal_lifecycle_version = self.oracle_config.get("lifecycle_version", "legacy_v1")
        if self.subgoal_lifecycle_version not in ("legacy_v1", "per_agent_v2"):
            raise ValueError("unsupported Oracle subgoal lifecycle version")
        self.subgoal_identity_tolerance = float(self.oracle_config.get("target_identity_tolerance", 1e-5))
        if not np.isfinite(self.subgoal_identity_tolerance) or self.subgoal_identity_tolerance <= 0:
            raise ValueError("target_identity_tolerance must be finite and positive")
        self.oracle_enabled = bool(self.oracle_config["enabled"])
        self.replanning_interval = int(self.oracle_config["replanning_interval"])
        self.subgoal_reach_radius = float(self.oracle_config["subgoal_reach_radius"])
        self.min_subgoal_distance = float(self.oracle_config["min_subgoal_distance"])
        self.oracle_recovery_enabled = bool(self.oracle_config["recovery_enabled"])
        self.oracle_recovery_angles = np.asarray(
            self.oracle_config["recovery_angle_offsets_degrees"], dtype=np.float32
        )
        self.oracle_recovery_distances = np.asarray(
            self.oracle_config["recovery_lookahead_distances"], dtype=np.float32
        )
        self.oracle_recovery_goal_weight = float(
            self.oracle_config["recovery_goal_progress_weight"]
        )
        self.oracle_recovery_clearance_weight = float(
            self.oracle_config["recovery_clearance_weight"]
        )
        self.oracle_recovery_turn_weight = float(
            self.oracle_config["recovery_turn_penalty_weight"]
        )
        self.final_goal_fallback_requires_clear_path = bool(
            self.oracle_config["final_goal_fallback_requires_clear_path"]
        )
        self.oracle_planner = OracleSubgoalPlanner(self.oracle_config)

        incoming_selector = dict(cfg.get("subgoal_selector", {}))
        selector_version = str(incoming_selector.get("version", "disabled_archived_v1"))
        if bool(incoming_selector.get("enabled", False)):
            raise ValueError(
                "historical subgoal selectors are archived; use frozen Oracle subgoals"
            )
        if selector_version.lower().startswith("facs"):
            raise ValueError("FACS/FACS-STAR selector versions are archived")
        if selector_version != "disabled_archived_v1":
            raise ValueError("unsupported disabled subgoal selector version")
        self.subgoal_selector_config = {
            "enabled": False, "version": "disabled_archived_v1",
        }
        safety_execution = dict(cfg.get("safety_execution", {}))
        self.safety_execution_enabled = bool(safety_execution.get("enabled", False))
        self.safety_recovery_enabled = bool(
            safety_execution.get("recovery_enabled", False)
        )
        if self.safety_execution_enabled and str(safety_execution.get("version")) != (
            "staged_joint_guard_ttc_v1"
        ):
            raise ValueError("unsupported shared safety execution version")
        registered_stages = tuple(safety_execution.get("stages", ()))
        if self.safety_execution_enabled and registered_stages != (
            "initial_joint_guard", "obstacle_ttc", "final_joint_guard"
        ):
            raise ValueError("safety_execution stages must use the preregistered order")
        guard_config = dict(safety_execution.get("initial_joint_guard", {}))
        self.boundary_guard = JointSafetyGuardV2(guard_config)
        self.initial_joint_guard_enabled = bool(
            self.safety_execution_enabled
            and self.boundary_guard.enabled
            and self.boundary_guard.action_shield_enabled
        )
        self.final_joint_guard_enabled = bool(
            self.safety_execution_enabled
            and safety_execution.get("final_joint_guard", {}).get(
                "enabled", False
            )
        )
        self.require_primary_segment_clearance = False
        self.stagnation_recovery_enabled = False
        self.stagnation_patience = 2
        self.min_segment_progress_ratio = 0.10
        self.failed_direction_ttl = 2
        self.recovery_hold_cycles = 1
        star_config = dict(safety_execution)
        star_config["obstacle_execution_guard"] = dict(
            safety_execution.get("obstacle_ttc", {})
        )
        star_config.setdefault("replanning_interval", self.replanning_interval)
        self.star_safety_state_machine_version = str(star_config.get(
            "safety_state_machine_version",
            "bounded_recovery_v1"
            if self.safety_recovery_enabled else "legacy_absorbing_v1",
        ))
        self.star_candidate_feasibility_version = str(star_config.get(
            "candidate_feasibility_version", "legacy_soft_v1"
        ))
        if self.star_safety_state_machine_version not in {
            "legacy_absorbing_v1", "star_safe_brake_recovery_v2",
            "bounded_recovery_v1", "star_scoped_recovery_v4",
            "star_adaptive_recovery_v5", "star_incident_recovery_v5_1",
            "star_incident_recovery_v5_2", "star_persistent_recovery_v5_3",
            "star_persistent_recovery_v5_4",
            "star_closure_consistency_audit_v1",
        }:
            raise ValueError("unsupported STAR safety state-machine version")
        if self.star_candidate_feasibility_version not in {
            "legacy_soft_v1", "guard_aligned_observable_v2",
            "guard_aligned_recovery_v3",
        }:
            raise ValueError("unsupported STAR candidate feasibility version")
        self._episode_seed: int | None = None
        self._safe_brake_active = False
        self._safety_mode = "NORMAL"
        self._safe_recovery_target: np.ndarray | None = None
        self._recovery_stable_steps = 0
        recovery_cfg = dict(safety_execution.get("recovery", {}))
        self.safe_recovery_angles = tuple(float(value) for value in recovery_cfg.get(
            "angle_offsets_degrees", [0, 45, -45, 90, -90, 135, -135, 180]
        ))
        self.safe_recovery_distances = tuple(float(value) for value in recovery_cfg.get(
            "distances", [3, 6, 9, 12]
        ))
        self.safe_recovery_beam_width = int(recovery_cfg.get("beam_width", 10))
        self.safe_recovery_exit_stable_steps = int(
            recovery_cfg.get("exit_stable_steps", 3)
        )
        self.safe_recovery_maximum_steps = int(
            recovery_cfg.get("maximum_steps", 20)
        )
        self.safe_recovery_target_distance = float(
            recovery_cfg.get("target_distance", 6.0)
        )
        self.safe_recovery_speed_fraction = float(
            recovery_cfg.get("speed_fraction", 0.5)
        )
        self.safe_recovery_boundary_trigger_margin = float(
            recovery_cfg.get("boundary_trigger_margin", 1.0)
        )
        self.safe_recovery_boundary_stall_patience = int(
            recovery_cfg.get("boundary_stall_patience_steps", 3)
        )
        self.safe_recovery_minimum_displacement = float(
            recovery_cfg.get("minimum_displacement_m", 0.05)
        )
        self.safe_recovery_obstacle_repulsion_margin = float(
            recovery_cfg.get("obstacle_repulsion_margin_m", 4.0)
        )
        self.safe_recovery_attempt_steps = int(
            recovery_cfg.get("attempt_steps", self.safe_recovery_maximum_steps)
        )
        self.safe_recovery_maximum_attempts = int(
            recovery_cfg.get("maximum_attempts", 1)
        )
        self.safe_recovery_exclude_goal_agents = bool(
            recovery_cfg.get("exclude_goal_agents", False)
        )
        self.safe_recovery_progress_patience_steps = int(
            recovery_cfg.get("progress_patience_steps", 3)
        )
        self.safe_recovery_maximum_refreshes = int(
            recovery_cfg.get("maximum_refreshes", 6)
        )
        self.safe_recovery_minimum_margin_improvement = float(
            recovery_cfg.get("minimum_margin_improvement_m", 0.01)
        )
        self.safe_recovery_direction_unavailable_patience = int(
            recovery_cfg.get("direction_unavailable_patience_steps", 3)
        )
        self.safe_recovery_escape_steps = int(
            recovery_cfg.get("escape_steps", 14)
        )
        self.safe_recovery_consolidation_steps = int(
            recovery_cfg.get("consolidation_steps", 6)
        )
        self.safe_recovery_release_hysteresis_m = float(
            recovery_cfg.get("release_hysteresis_m", 0.25)
        )
        self.safe_recovery_rapid_retrigger_window_steps = int(
            recovery_cfg.get("rapid_retrigger_window_steps", 5)
        )
        self.safe_recovery_lineage_max_incidents = int(
            recovery_cfg.get("lineage_max_incidents", 2)
        )
        self.safe_recovery_lineage_max_steps = int(
            recovery_cfg.get("lineage_max_steps", 40)
        )
        self.safe_recovery_shadow_horizon_steps = int(
            recovery_cfg.get("shadow_horizon_steps", 3)
        )
        self.safe_recovery_candidate_fixed_point_patience = int(
            recovery_cfg.get("candidate_fixed_point_patience_steps", 2)
        )
        self.safe_recovery_candidate_minimum_margin_gain = float(
            recovery_cfg.get(
                "candidate_minimum_margin_gain_m",
                self.safe_recovery_minimum_margin_improvement,
            )
        )
        self.closure_clear_window_steps = int(
            recovery_cfg.get("closure_clear_window_steps", 5)
        )
        self.closure_certificate_tail_steps = int(
            recovery_cfg.get("closure_certificate_tail_steps", 3)
        )
        if self.safe_recovery_direction_unavailable_patience <= 0:
            raise ValueError(
                "safe_recovery.direction_unavailable_patience_steps must be positive"
            )
        self.safe_recover_step_count = 0
        self.safe_recover_entry_count = 0
        self.safe_recover_exit_count = 0
        self.safe_recovery_failure_count = 0
        self.safe_recovery_unavailable_count = 0
        self.safe_recovery_retry_count = 0
        self.safe_recovery_abort_count = 0
        self.safe_recovery_hard_abort_count = 0
        self.safe_recovery_refresh_count = 0
        self.safe_recovery_reentry_count = 0
        self.safe_recovery_abort_then_success_count = 0
        self.safe_recovery_episode_failure = False
        self.safe_recovery_affected_agent_count = 0
        self.max_safe_recover_streak = 0
        self.safe_recovery_durations: list[int] = []
        self.safe_recovery_events: list[dict[str, Any]] = []
        self.boundary_recovery_trigger_count = 0
        self._safe_recover_current_streak = 0
        self._safe_recovery_elapsed_steps = 0
        self._safe_recovery_sequence_steps = 0
        self._safe_recovery_attempt_index = 0
        self._safe_recovery_failed = False
        self._safe_recovery_entry_sources: tuple[str, ...] = ()
        self._safe_recovery_previous_margin: float | None = None
        self._safe_recovery_no_improvement_steps = 0
        self._v5_release_after_brake = False
        self._star_recovery_offending_pairs: tuple[tuple[int, int], ...] = ()
        self._star_recovery_affected_mask = np.zeros(self.num_agents, dtype=bool)
        self._boundary_stall_steps = np.zeros(self.num_agents, dtype=np.int32)
        self._safe_recovery_start_positions: np.ndarray | None = None
        self._pending_recovery_pair_infeasible = False
        self._pending_recovery_invariant_violations = 0
        self._last_recovery_reason: str | None = None
        self._last_boundary_guard_result: dict[str, Any] | None = None
        self.shield_intervention_count = 0
        self.joint_guard_intervention_count = 0
        self.initial_joint_guard_intervention_steps: list[int] = []
        self.final_joint_guard_residual_count = 0
        self.final_joint_guard_residual_steps: list[int] = []
        self.obstacle_ttc_projection_count = 0
        self.obstacle_ttc_projection_steps: list[int] = []
        self.obstacle_ttc_fallback_count = 0
        self.obstacle_ttc_post_action_goal_velocity_sum = 0.0
        self.obstacle_ttc_post_action_goal_velocity_count = 0
        self.obstacle_ttc_post_action_min_clearance_m = float("inf")
        self.total_action_modification_count = 0
        self.shield_invariant_violation_count = 0
        self.safe_brake_step_count = 0
        self.safe_brake_entry_count = 0
        self.safe_brake_exit_count = 0
        self._safe_brake_current_streak = 0
        self.max_safe_brake_streak = 0
        self.pair_infeasible_count = 0
        self.pair_infeasible_entry_count = 0
        self._last_pair_infeasible = False
        self.safe_brake_entry_steps: list[int] = []
        self.safe_brake_exit_steps: list[int] = []
        self.safe_brake_durations: list[int] = []
        self.safe_brake_entry_reasons: list[str] = []
        obstacle_audit = dict(star_config.get("obstacle_execution_diagnostics", {}))
        self.obstacle_execution_diagnostics_enabled = bool(
            obstacle_audit.get("enabled", False)
        )
        self.obstacle_ttc_audit_horizon = float(
            obstacle_audit.get("ttc_horizon_seconds", 10.0)
        )
        self.minimum_observed_obstacle_ttc = float("inf")
        self.obstacle_ttc_risk_step_count = 0
        self._first_obstacle_visible_steps: dict[tuple[int, int], int] = {}
        obstacle_guard = dict(star_config.get("obstacle_execution_guard", {}))
        self.obstacle_execution_guard_enabled = bool(
            obstacle_guard.get("enabled", False)
        )
        self.obstacle_execution_guard_version = str(
            obstacle_guard.get("version", "disabled_v1")
        )
        self.obstacle_guard_horizon_seconds = float(
            obstacle_guard.get("horizon_seconds", 10.0)
        )
        self.obstacle_guard_warning_ttc_seconds = float(
            obstacle_guard.get("warning_ttc_seconds", 5.0)
        )
        self.obstacle_guard_emergency_ttc_seconds = float(
            obstacle_guard.get("emergency_ttc_seconds", 2.5)
        )
        headroom_cfg = dict(star_config.get("ttc_headroom_audit", {}))
        self.ttc_headroom_audit_enabled = bool(
            headroom_cfg.get("enabled", False)
        )
        self.ttc_headroom_shadow_steps = int(
            headroom_cfg.get("shadow_steps", 3)
        )
        self.ttc_headroom_minimum_goal_velocity_gain = float(
            headroom_cfg.get("minimum_goal_velocity_gain_mps", 0.10)
        )
        self.ttc_headroom_maximum_extra_actor_deviation = float(
            headroom_cfg.get("maximum_extra_actor_deviation", 0.25)
        )
        self.ttc_headroom_audit_events: list[dict[str, Any]] = []
        self.ttc_headroom_intervention_event_count = 0
        self.ttc_headroom_positive_event_count = 0
        self.ttc_v2_high_progress_selection_count = 0
        self._last_obstacle_guard_result: dict[str, Any] | None = None
        corridor_cfg = dict(star_config.get("goal_boundary_corridor", {}))
        self.goal_boundary_corridor_enabled = bool(
            corridor_cfg.get("enabled", False)
        )
        self.goal_boundary_corridor_version = str(
            corridor_cfg.get("version", "disabled_v1")
        )
        self.goal_boundary_corridor_coordinate = float(
            corridor_cfg.get("corridor_coordinate", 94.0)
        )
        self.goal_corridor_projection_count = 0
        self.goal_corridor_agent_projection_count = 0
        self.goal_corridor_projection_steps: list[int] = []
        self.goal_corridor_tangential_progress_m = 0.0
        self.goal_corridor_nonpositive_progress_count = 0
        self._last_goal_corridor_result: dict[str, Any] | None = None
        trace_cfg = dict(star_config.get("closure_trace_audit", {}))
        self.closure_trace_audit_enabled = bool(
            trace_cfg.get("enabled", False)
        )

        # V5.1 incident state is independent from the historical V5 counters.
        # An incident owns one strict 20-step budget even when new hazard
        # sources are merged while recovery is active.
        self.recovery_incident_count = 0
        self.recovery_incident_merge_count = 0
        self.recovery_incident_reentry_reset_count = 0
        self.recovery_duplicate_hard_abort_count = 0
        self.max_recovery_incident_steps = 0
        self.terminal_recovery_unresolved = False
        self._recovery_incident_counter = 0
        self._recovery_incident_id: int | None = None
        self._recovery_incident_elapsed_steps = 0
        self._recovery_incident_sequence_steps = 0
        self._recovery_incident_refresh_count = 0
        self._recovery_incident_sources: tuple[str, ...] = ()
        self._recovery_incident_hazard_keys: tuple[str, ...] = ()
        self._recovery_incident_hard_abort_emitted = False
        self._recovery_incident_block_reentry_until_step = -1
        self._recovery_incident_start_step: int | None = None
        self._recovery_direction_unavailable_streak = 0
        self._recovery_direction_unavailable_last_step = -1
        self.max_recovery_direction_unavailable_streak = 0
        self.recovery_watchdog_step_count = 0
        self.recovery_incident_close_count = 0
        self.recovery_incident_reopen_count = 0
        self.hazard_retrigger_count = 0
        self.same_hazard_reentry_count = 0
        self.hazard_resolution_steps: list[int] = []
        self.incident_close_reason_counts: dict[str, int] = {}
        self._closed_recovery_hazard_steps: dict[str, int] = {}
        self.terminal_incident_reconciled_count = 0

        # V5.3 lineage state is intentionally separate from the historical
        # V5.1/V5.2 hard-abort counters.  A first bounded exhaustion is a
        # recoverable incident outcome; only lineage-level failure is a
        # mission failure.
        self.recovery_telemetry_schema_version = (
            "shared_closure_consistency_telemetry_v1"
            if self.star_safety_state_machine_version
            == "star_closure_consistency_audit_v1"
            else
            "shared_recovery_telemetry_v1"
            if self.star_safety_state_machine_version
            == "star_persistent_recovery_v5_4"
            else "shared_recovery_telemetry_v1"
        )
        self.recovery_phase = "IDLE"
        self.escape_steps = 0
        self.consolidation_steps = 0
        self.release_shadow_check_passed = False
        self.release_shadow_check_count = 0
        self.release_shadow_check_failure_count = 0
        self.release_margin_m = float("nan")
        self.release_margin_delta_m = float("nan")
        self.bounded_recovery_exhaustion_count = 0
        self.bounded_exhaustion_then_success_count = 0
        self.rapid_same_hazard_retrigger_count = 0
        self.recovery_lineage_failure_count = 0
        self.recovery_attributable_mission_failure = False
        self.recovery_attributable_failure_reason: str | None = None
        self.max_recovery_lineage_steps = 0
        self.recovery_lineage_budget_reset_count = 0
        self.recovery_duplicate_failure_record_count = 0
        self.max_rapid_retriggers_per_lineage = 0
        self._recovery_lineage_counter = 0
        self._recovery_lineage_id: int | None = None
        self._recovery_lineage_hazard_keys: tuple[str, ...] = ()
        self._recovery_lineage_incident_count = 0
        self._recovery_lineage_steps = 0
        self._recovery_lineage_exhaustion_count = 0
        self._recovery_lineage_rapid_retriggers = 0
        self._closed_recovery_lineages: dict[str, dict[str, int]] = {}
        self._failed_recovery_hazard_keys: set[str] = set()
        self._recovery_lineage_failure_emitted = False
        self.recovery_lineage_terminal_state = "IDLE"
        self.recovery_candidate_selection_count = 0
        self.recovery_candidate_rejection_count = 0
        self.recovery_candidate_rejection_reason_counts: dict[str, int] = {}
        self.recovery_candidate_post_projection_norm = float("nan")
        self.recovery_candidate_predicted_margin_gain = float("nan")
        self.recovery_candidate_fixed_point_streak = 0
        self.release_shadow_horizon_min_margin = float("nan")
        self.release_shadow_horizon_min_ttc = float("nan")
        self._closure_tracker = ClosureConsistencyV1(
            clear_window_steps=self.closure_clear_window_steps,
            certificate_tail_steps=self.closure_certificate_tail_steps,
        )
        self.closure_state = CLOSURE_ACTIVE
        self.clearance_pending_steps = 0
        self.pending_clearance_reappearance_count = 0
        self.post_certificate_retrigger_count = 0
        self.close_without_certificate_count = 0
        self.stale_closure_bypass_count = 0
        self.release_certificate_count = 0
        self.release_certificate_id = 0
        self._closure_last_verified_hazard_keys: tuple[str, ...] = ()
        self._closure_last_verified_step = -1
        self._closure_budget_counted_step = -1
        self._current_action_trace: dict[str, list[list[float]]] = {}

        self.action_stage_attribution: dict[str, dict[str, float | int]] = {}
        self._reset_action_stage_attribution()

        self._validate_config()

        # legacy: obstacle=(relative position2, radius, distance), 4 slots -> 26D.
        # tracked v2: obstacle=(relative position2, estimated relative velocity2,
        # radius, slot-valid, velocity-valid), configurable slots.  Simulator
        # truth velocity and static/dynamic identity are deliberately excluded.
        self._base_observation_dim = 10
        self._obstacle_feature_dim = (
            7 if self.observation_schema_version == "tracked_obstacle_v2" else 4
        )
        self.local_observation_dim = self._base_observation_dim + self.obstacle_slots * self._obstacle_feature_dim
        self.global_state_dim = self.num_agents * self.local_observation_dim
        self.obstacle_feature_slice = slice(self._base_observation_dim, self.local_observation_dim)
        self.obstacle_mask_indices = np.asarray([], dtype=np.int64)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.num_agents, 2), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.num_agents, self.local_observation_dim), dtype=np.float32)

        self._rng = np.random.default_rng(int(cfg.get("seed", 0)))
        self.agents: list[UAVState] = []
        self.obstacles: list[CircularObstacle] = []
        self.goals = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.subgoals = self.goals.copy()
        self.goal = self.goals  # 兼容旧回放入口。
        self.start_center = np.zeros(2, dtype=np.float32)
        self.step_count = 0
        self.terminated = False
        self.truncated = False
        self._previous_distances = np.zeros(self.num_agents, dtype=np.float32)
        self._previous_subgoal_distances = np.zeros(self.num_agents, dtype=np.float32)
        self._previous_in_goal = np.zeros(self.num_agents, dtype=bool)
        self._previous_obstacle_clearances = np.full(self.num_agents, np.inf, dtype=np.float32)
        self._episode_start_mean_goal_distance = 1.0
        self._episode_start_goal_distances = np.ones(
            self.num_agents, dtype=np.float32
        )
        self._previous_team_potential = 0.0
        self._best_team_goal_completion = 0.0
        self._ever_in_goal = np.zeros(self.num_agents, dtype=bool)
        self._team_arrival_observed = False
        # 仅用于评估审计：记录每架 UAV 的入目标与离目标时间，不改变成功判定或奖励。
        self._first_goal_entry_steps = np.full(self.num_agents, -1, dtype=np.int32)
        self._last_goal_entry_steps = np.full(self.num_agents, -1, dtype=np.int32)
        self._goal_exit_counts = np.zeros(self.num_agents, dtype=np.int32)
        self._in_goal_step_counts = np.zeros(self.num_agents, dtype=np.int32)
        self._episode_start_max_goal_distance = 0.0
        self._episode_min_obstacle_clearance = np.inf
        self._episode_min_inter_agent_distance = np.inf
        self._max_visible_obstacles_pre_truncation = 0
        self._visible_obstacle_saturated_agent_steps = 0
        self._visible_obstacle_observation_agent_steps = 0
        self._obstacle_track_history: list[dict[int, tuple[np.ndarray, int]]] = [
            {} for _ in range(self.num_agents)
        ]
        self._obstacle_track_cache_step: int | None = None
        self._obstacle_track_cache: list[list[dict[str, Any]]] = [
            [] for _ in range(self.num_agents)
        ]
        self._near_goal_stall_streak = np.zeros(self.num_agents, dtype=np.int32)
        self._max_near_goal_stall_streak = np.zeros(self.num_agents, dtype=np.int32)
        self._near_goal_stall_triggered = np.zeros(self.num_agents, dtype=bool)
        self._near_goal_stall_first_steps = np.full(
            self.num_agents, -1, dtype=np.int32
        )
        # episode 级碰撞统计；碰撞不再终止回合。
        self.collision_count = 0
        self.obstacle_collision_count = 0
        # 将障碍碰撞按静态/动态来源拆分，仅用于评估诊断，不改变碰撞处罚。
        self.static_obstacle_collision_count = 0
        self.dynamic_obstacle_collision_count = 0
        self.inter_agent_collision_count = 0
        self.boundary_collision_count = 0
        self.episode_has_collision = False
        self.first_collision_step: int | None = None
        self.first_collision_type: str | None = None
        self.first_collision_goal_progress_ratio: float | None = None
        self.left_boundary_collision_count = 0
        self.right_boundary_collision_count = 0
        self.top_boundary_collision_count = 0
        self.bottom_boundary_collision_count = 0
        self.boundary_contact_step_count = 0
        self.left_boundary_contact_step_count = 0
        self.right_boundary_contact_step_count = 0
        self.top_boundary_contact_step_count = 0
        self.bottom_boundary_contact_step_count = 0
        self._current_boundary_contact_streak = np.zeros(self.num_agents, dtype=np.int32)
        self.max_boundary_contact_streak = 0
        # 同一接触键持续存在时不重复计数/处罚，脱离后自动解除。
        self._active_collision_keys: set[tuple] = set()
        self._previous_valid_subgoals: np.ndarray | None = None
        self.subgoal_replan_count = 0
        self.early_replan_count = 0
        self._invalid_subgoal_count = 0
        self._subgoal_candidate_count = 0
        self._subgoal_reach_count = 0
        self._subgoal_closed_count = 0
        self._subgoal_timeout_count = 0
        self._subgoal_collision_segment_count = 0
        self._subgoal_steps_sum = 0.0
        self._subgoal_progress_ratio_sum = 0.0
        self._subgoal_distance_sum = 0.0
        self._subgoal_distance_samples = 0
        self._subgoal_segment_start_step = np.zeros(self.num_agents, dtype=np.int32)
        self._subgoal_segment_start_distance = np.zeros(self.num_agents, dtype=np.float32)
        self._subgoal_segment_had_collision = np.zeros(self.num_agents, dtype=bool)
        self._subgoal_finalized = False
        self._episode_max_goal_progress_ratio = 0.0
        self._max_x_per_agent = np.zeros(self.num_agents, dtype=np.float32)
        self._entered_half_map = np.zeros(self.num_agents, dtype=bool)
        self._stagnation_counts = np.zeros(self.num_agents, dtype=np.int32)
        self._force_recovery_mask = np.zeros(self.num_agents, dtype=bool)
        self._failed_direction_buckets: list[dict[int, int]] = [dict() for _ in range(self.num_agents)]
        self._recovery_hold_until_step = np.zeros(self.num_agents, dtype=np.int32)
        self.blocked_primary_segment_count = 0
        self.stagnation_trigger_count = 0
        self.repeated_direction_rejection_count = 0

    @property
    def positions(self) -> np.ndarray:
        return np.stack([agent.position for agent in self.agents]).astype(np.float32)

    @property
    def velocities(self) -> np.ndarray:
        return np.stack([agent.velocity for agent in self.agents]).astype(np.float32)

    def set_runtime_telemetry_level(self, level: str) -> str:
        """Switch side-effect-free telemetry without changing control semantics.

        ``train_minimal`` suppresses expensive evidence and attribution work
        during on-policy collection.  Evaluation restores ``full`` telemetry.
        The previous level is returned to make scoped callers straightforward.
        """

        value = str(level)
        if value not in {"full", "train_minimal"}:
            raise ValueError(
                "runtime telemetry level must be full or train_minimal"
            )
        previous = self.runtime_telemetry_level
        self.runtime_telemetry_level = value
        return previous


    @staticmethod
    def _counterfactual_jsonable(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {
                str(key): CooperativeUAVEnv._counterfactual_jsonable(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [
                CooperativeUAVEnv._counterfactual_jsonable(item)
                for item in value
            ]
        return value

    def export_counterfactual_snapshot(
        self, actor_gru_state: Any = None
    ) -> EnvironmentShadowSnapshotV1:
        """Capture the mutable state required by fixed-candidate branch replay."""
        scalar_names = (
            "step_count", "terminated", "truncated", "collision_count",
            "obstacle_collision_count", "static_obstacle_collision_count",
            "dynamic_obstacle_collision_count", "inter_agent_collision_count",
            "boundary_collision_count", "episode_has_collision",
            "first_collision_step", "first_collision_type",
            "first_collision_goal_progress_ratio", "subgoal_replan_count",
            "early_replan_count", "_safe_brake_active",
            "_safety_mode", "_recovery_stable_steps",
            "shield_intervention_count", "shield_invariant_violation_count",
            "joint_guard_intervention_count", "obstacle_ttc_projection_count",
            "initial_joint_guard_intervention_steps",
            "final_joint_guard_residual_count",
            "final_joint_guard_residual_steps",
            "obstacle_ttc_projection_steps",
            "obstacle_ttc_fallback_count",
            "obstacle_ttc_post_action_goal_velocity_sum",
            "obstacle_ttc_post_action_goal_velocity_count",
            "obstacle_ttc_post_action_min_clearance_m",
            "total_action_modification_count",
            "goal_corridor_projection_count",
            "goal_corridor_agent_projection_count",
            "goal_corridor_projection_steps",
            "goal_corridor_tangential_progress_m",
            "goal_corridor_nonpositive_progress_count",
            "_last_goal_corridor_result",
            "safe_brake_step_count", "safe_brake_entry_count",
            "safe_brake_exit_count", "_safe_brake_current_streak",
            "max_safe_brake_streak", "pair_infeasible_count",
            "pair_infeasible_entry_count", "_last_pair_infeasible",
            "safe_brake_entry_steps", "safe_brake_exit_steps",
            "safe_brake_durations", "safe_brake_entry_reasons",
            "safe_recover_step_count", "safe_recover_entry_count",
            "safe_recover_exit_count", "safe_recovery_failure_count",
            "safe_recovery_unavailable_count", "safe_recovery_retry_count",
            "safe_recovery_abort_count", "safe_recovery_hard_abort_count",
            "safe_recovery_refresh_count", "safe_recovery_reentry_count",
            "safe_recovery_abort_then_success_count",
            "safe_recovery_episode_failure",
            "safe_recovery_affected_agent_count", "max_safe_recover_streak",
            "safe_recovery_durations", "safe_recovery_events",
            "boundary_recovery_trigger_count", "_safe_recover_current_streak",
            "_safe_recovery_elapsed_steps", "_safe_recovery_sequence_steps",
            "_safe_recovery_attempt_index", "_safe_recovery_failed",
            "_safe_recovery_entry_sources", "_safe_recovery_previous_margin",
            "_safe_recovery_no_improvement_steps", "_v5_release_after_brake",
            "recovery_incident_count", "recovery_incident_merge_count",
            "recovery_incident_reentry_reset_count",
            "recovery_duplicate_hard_abort_count",
            "max_recovery_incident_steps", "terminal_recovery_unresolved",
            "_recovery_incident_counter", "_recovery_incident_id",
            "_recovery_incident_elapsed_steps",
            "_recovery_incident_sequence_steps",
            "_recovery_incident_refresh_count", "_recovery_incident_sources",
            "_recovery_incident_hazard_keys",
            "_recovery_incident_hard_abort_emitted",
            "_recovery_incident_block_reentry_until_step",
            "_recovery_incident_start_step",
            "_recovery_direction_unavailable_streak",
            "_recovery_direction_unavailable_last_step",
            "max_recovery_direction_unavailable_streak",
            "recovery_watchdog_step_count", "recovery_incident_close_count",
            "recovery_incident_reopen_count", "hazard_retrigger_count",
            "same_hazard_reentry_count", "hazard_resolution_steps",
            "incident_close_reason_counts", "_closed_recovery_hazard_steps",
            "terminal_incident_reconciled_count", "action_stage_attribution",
            "recovery_telemetry_schema_version", "recovery_phase",
            "escape_steps", "consolidation_steps",
            "release_shadow_check_passed", "release_shadow_check_count",
            "release_shadow_check_failure_count", "release_margin_m",
            "release_margin_delta_m", "bounded_recovery_exhaustion_count",
            "bounded_exhaustion_then_success_count",
            "rapid_same_hazard_retrigger_count",
            "recovery_lineage_failure_count",
            "recovery_attributable_mission_failure",
            "recovery_attributable_failure_reason",
            "max_recovery_lineage_steps",
            "recovery_lineage_budget_reset_count",
            "recovery_duplicate_failure_record_count",
            "max_rapid_retriggers_per_lineage",
            "_recovery_lineage_counter", "_recovery_lineage_id",
            "_recovery_lineage_hazard_keys",
            "_recovery_lineage_incident_count", "_recovery_lineage_steps",
            "_recovery_lineage_exhaustion_count",
            "_recovery_lineage_rapid_retriggers",
            "_closed_recovery_lineages",
            "_failed_recovery_hazard_keys",
            "_recovery_lineage_failure_emitted",
            "recovery_lineage_terminal_state",
            "recovery_candidate_selection_count",
            "recovery_candidate_rejection_count",
            "recovery_candidate_rejection_reason_counts",
            "recovery_candidate_post_projection_norm",
            "recovery_candidate_predicted_margin_gain",
            "recovery_candidate_fixed_point_streak",
            "release_shadow_horizon_min_margin",
            "release_shadow_horizon_min_ttc",
            "_star_recovery_offending_pairs", "_last_recovery_reason",
            "_pending_recovery_pair_infeasible",
            "_pending_recovery_invariant_violations",
            "minimum_observed_obstacle_ttc", "obstacle_ttc_risk_step_count",
        )
        array_names = (
            "goals", "subgoals", "_previous_distances",
            "_previous_subgoal_distances", "_previous_in_goal",
            "_previous_obstacle_clearances", "_ever_in_goal",
            "_first_goal_entry_steps", "_last_goal_entry_steps",
            "_goal_exit_counts", "_in_goal_step_counts", "_stagnation_counts",
            "_force_recovery_mask", "_recovery_hold_until_step",
            "_last_plan_attempt_step", "subgoal_ids",
            "_subgoal_segment_start_step", "_subgoal_segment_start_distance",
            "_subgoal_segment_had_collision", "_subgoal_arrival_seen",
            "_subgoal_started_reached", "_subgoal_unsafe_seen",
            "_star_recovery_affected_mask", "_boundary_stall_steps",
        )
        state: dict[str, Any] = {
            "scalars": {name: deepcopy(getattr(self, name)) for name in scalar_names},
            "arrays": {
                name: np.asarray(getattr(self, name)).copy() for name in array_names
            },
            "agents": [
                {"position": agent.position.copy(), "velocity": agent.velocity.copy()}
                for agent in self.agents
            ],
            "obstacles": [
                {
                    "position": obstacle.position.copy(),
                    "velocity": obstacle.velocity.copy(),
                    "radius": float(obstacle.radius),
                    "dynamic": bool(obstacle.dynamic),
                }
                for obstacle in self.obstacles
            ],
            "rng_state": deepcopy(self._rng.bit_generator.state),
            "active_collision_keys": deepcopy(self._active_collision_keys),
            "previous_valid_subgoals": (
                None if self._previous_valid_subgoals is None
                else self._previous_valid_subgoals.copy()
            ),
            "subgoal_close_reasons": list(self._subgoal_close_reasons),
            "obstacle_track_history": deepcopy(self._obstacle_track_history),
            "safe_recovery_target": (
                None if self._safe_recovery_target is None
                else self._safe_recovery_target.copy()
            ),
            "safe_recovery_start_positions": (
                None if self._safe_recovery_start_positions is None
                else self._safe_recovery_start_positions.copy()
            ),
            "first_obstacle_visible_steps": deepcopy(
                self._first_obstacle_visible_steps
            ),
        }
        canonical = self._counterfactual_jsonable({
            **state,
            "active_collision_keys": sorted(
                [list(item) for item in state["active_collision_keys"]],
                key=str,
            ),
        })
        fingerprint = EnvironmentShadowSnapshotV1.fingerprint(canonical)
        return EnvironmentShadowSnapshotV1(
            environment_state=state,
            environment_rng_state=deepcopy(self._rng.bit_generator.state),
            actor_gru_state=deepcopy(actor_gru_state),
            state_fingerprint=fingerprint,
        )

    def restore_counterfactual_snapshot(
        self, snapshot: EnvironmentShadowSnapshotV1
    ) -> Any:
        if snapshot.schema_version != "EnvironmentShadowSnapshotV1":
            raise ValueError("unsupported counterfactual snapshot")
        state = deepcopy(dict(snapshot.environment_state))
        canonical = self._counterfactual_jsonable({
            **state,
            "active_collision_keys": sorted(
                [list(item) for item in state["active_collision_keys"]],
                key=str,
            ),
        })
        if EnvironmentShadowSnapshotV1.fingerprint(canonical) != snapshot.state_fingerprint:
            raise ValueError("counterfactual snapshot fingerprint mismatch")
        for name, value in state["scalars"].items():
            setattr(self, name, value)
        for name, value in state["arrays"].items():
            setattr(self, name, np.asarray(value).copy())
        for agent, values in zip(self.agents, state["agents"]):
            agent.position = np.asarray(values["position"], dtype=np.float32)
            agent.velocity = np.asarray(values["velocity"], dtype=np.float32)
        self.obstacles = [
            CircularObstacle(
                position=np.asarray(values["position"], dtype=np.float32),
                radius=float(values["radius"]),
                velocity=np.asarray(values["velocity"], dtype=np.float32),
                dynamic=bool(values["dynamic"]),
            )
            for values in state["obstacles"]
        ]
        self._rng.bit_generator.state = deepcopy(state["rng_state"])
        self._active_collision_keys = set(state["active_collision_keys"])
        previous = state["previous_valid_subgoals"]
        self._previous_valid_subgoals = (
            None if previous is None else np.asarray(previous, dtype=np.float32)
        )
        self._subgoal_close_reasons = list(state["subgoal_close_reasons"])
        self._obstacle_track_history = deepcopy(state["obstacle_track_history"])
        recovery_target = state.get("safe_recovery_target")
        self._safe_recovery_target = (
            None if recovery_target is None
            else np.asarray(recovery_target, dtype=np.float32)
        )
        recovery_start = state.get("safe_recovery_start_positions")
        self._safe_recovery_start_positions = (
            None if recovery_start is None
            else np.asarray(recovery_start, dtype=np.float32)
        )
        self._first_obstacle_visible_steps = deepcopy(
            state.get("first_obstacle_visible_steps", {})
        )
        self._obstacle_track_cache = [[] for _ in range(self.num_agents)]
        self._obstacle_track_cache_step = None
        restored = self.export_counterfactual_snapshot(snapshot.actor_gru_state)
        if restored.state_fingerprint != snapshot.state_fingerprint:
            raise RuntimeError("counterfactual state restore was not exact")
        return deepcopy(snapshot.actor_gru_state)

    def _reset_obstacle_tracks(self) -> None:
        self._obstacle_track_history = [{} for _ in range(self.num_agents)]
        self._obstacle_track_cache = [[] for _ in range(self.num_agents)]
        self._obstacle_track_cache_step = None

    def _refresh_obstacle_tracks(self) -> None:
        """Build one causal, per-step local track snapshot.

        Stable simulator obstacle ids are used only for internal data association.
        Neither ids, true velocities nor static/dynamic labels enter the policy
        observation. A track is valid only after consecutive local detections.
        """
        if self._obstacle_track_cache_step == self.step_count:
            return
        next_history: list[dict[int, tuple[np.ndarray, int]]] = []
        cache: list[list[dict[str, Any]]] = []
        horizon = float(self.replanning_interval * self.dt)
        for agent_id, agent in enumerate(self.agents):
            previous = self._obstacle_track_history[agent_id]
            current: dict[int, tuple[np.ndarray, int]] = {}
            records: list[dict[str, Any]] = []
            for obstacle_id, obstacle in enumerate(self.obstacles):
                relative = (obstacle.position - agent.position).astype(np.float32)
                distance = float(np.linalg.norm(relative))
                if distance > self.detection_radius:
                    continue
                velocity = np.zeros(2, dtype=np.float32)
                velocity_valid = False
                previous_track = previous.get(obstacle_id)
                if previous_track is not None and previous_track[1] == self.step_count - 1:
                    velocity = (
                        (relative - previous_track[0]) / max(self.dt, 1e-6)
                    ).astype(np.float32)
                    velocity_valid = bool(np.isfinite(velocity).all())
                    if not velocity_valid:
                        velocity[:] = 0.0
                current[obstacle_id] = (relative.copy(), int(self.step_count))
                combined_radius = self.agent_radius + float(obstacle.radius)
                current_clearance = distance - combined_radius
                predicted_clearance = current_clearance
                if velocity_valid:
                    speed_squared = float(np.dot(velocity, velocity))
                    tau = 0.0 if speed_squared <= 1e-12 else float(
                        np.clip(-np.dot(relative, velocity) / speed_squared, 0.0, horizon)
                    )
                    predicted_clearance = float(
                        np.linalg.norm(relative + tau * velocity) - combined_radius
                    )
                records.append({
                    "obstacle_id": int(obstacle_id),
                    "obstacle": obstacle,
                    "relative_position": relative,
                    "relative_velocity": velocity,
                    "velocity_valid": velocity_valid,
                    "distance": distance,
                    "current_clearance": float(current_clearance),
                    "predicted_clearance": float(predicted_clearance),
                })
            if self.obstacle_selection_version == "predicted_clearance_v1":
                records.sort(key=lambda item: (
                    item["predicted_clearance"], item["current_clearance"],
                    item["distance"], item["obstacle_id"],
                ))
            else:
                records.sort(key=lambda item: (item["distance"], item["obstacle_id"]))
            next_history.append(current)
            cache.append(records[:self.obstacle_slots])
        self._obstacle_track_history = next_history
        self._obstacle_track_cache = cache
        self._obstacle_track_cache_step = int(self.step_count)

    def _tracked_obstacles(self, agent_id: int) -> list[dict[str, Any]]:
        self._refresh_obstacle_tracks()
        return self._obstacle_track_cache[int(agent_id)]

    def _visible_obstacles(self, agent_id: int) -> tuple[VisibleObstacle, ...]:
        if self.observation_schema_version == "tracked_obstacle_v2":
            return tuple(
                VisibleObstacle(
                    obstacle_id=int(item["obstacle_id"]),
                    position=(
                        float(item["obstacle"].position[0]),
                        float(item["obstacle"].position[1]),
                    ),
                    radius=float(item["obstacle"].radius),
                    relative_velocity=(
                        float(item["relative_velocity"][0]),
                        float(item["relative_velocity"][1]),
                    ),
                    velocity_valid=bool(item["velocity_valid"]),
                )
                for item in self._tracked_obstacles(agent_id)
            )
        return tuple(
            VisibleObstacle(
                obstacle_id=int(obstacle_id),
                position=(float(obstacle.position[0]), float(obstacle.position[1])),
                radius=float(obstacle.radius),
            )
            for obstacle_id, obstacle, _ in self.get_nearest_obstacles(int(agent_id))
        )

    def _planning_snapshot(self) -> PlanningSnapshot:
        return PlanningSnapshot(
            positions=self.positions.copy(),
            velocities=self.velocities.copy(),
            final_goals=self.goals.copy(),
            visible_obstacles=tuple(
                self._visible_obstacles(agent_id) for agent_id in range(self.num_agents)
            ),
            map_size=self.map_size.copy(),
            agent_radius=self.agent_radius,
            max_speed=self.agent_max_speed,
            detection_radius=self.detection_radius,
            obstacle_safety_margin=float(self.oracle_config["obstacle_safety_margin"]),
            teammate_safety_distance=float(self.oracle_config["teammate_safety_distance"]),
            replanning_interval=self.replanning_interval,
            stagnation_counts=tuple(int(x) for x in self._stagnation_counts),
            step=int(self.step_count),
            dt=self.dt,
            max_acceleration=self.agent_max_acceleration,
        )

    def _safety_planning_snapshot(self) -> TeamPlanningSnapshotV1:
        """Build STAR state from the same causal tracks exposed to the policy."""
        visible = []
        for agent_id in range(self.num_agents):
            rows = []
            for item in self._tracked_obstacles(agent_id):
                rows.append(ObservableObstacleV1(
                    observer_id=int(agent_id),
                    track_id=int(item["obstacle_id"]),
                    relative_position=tuple(
                        float(x) for x in item["relative_position"]
                    ),
                    relative_velocity=tuple(
                        float(x) for x in item["relative_velocity"]
                    ),
                    radius=float(item["obstacle"].radius),
                    velocity_valid=bool(item["velocity_valid"]),
                ))
            visible.append(tuple(rows))
        return TeamPlanningSnapshotV1.from_arrays(
            positions=self.positions,
            velocities=self.velocities,
            final_goals=self.goals,
            visible_obstacles=visible,
            stagnation_counts=self._stagnation_counts,
            map_size=self.map_size,
            agent_radius=self.agent_radius,
            max_speed=self.agent_max_speed,
            max_acceleration=self.agent_max_acceleration,
            dt=self.dt,
            replanning_interval=self.replanning_interval,
            teammate_safety_distance=float(
                self.oracle_config["teammate_safety_distance"]
            ),
            obstacle_safety_margin=float(
                self.oracle_config["obstacle_safety_margin"]
            ),
            step=self.step_count,
            boundary_surface_clearance=float(
                self.boundary_guard.surface_clearance_m
            ),
            boundary_invariant_buffer=float(
                self.boundary_guard.invariant_buffer_m
            ),
            candidate_feasibility_version=(
                self.star_candidate_feasibility_version
            ),
        )


    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """在左侧生成 UAV、右侧生成交叉目标，并随机生成圆形障碍物。"""
        super().reset(seed=seed)
        del options
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        self._episode_seed = None if seed is None else int(seed)
        self.step_count = 0
        self.terminated = False
        self.truncated = False
        self.collision_count = 0
        self.obstacle_collision_count = 0
        self.static_obstacle_collision_count = 0
        self.dynamic_obstacle_collision_count = 0
        self.inter_agent_collision_count = 0
        self.boundary_collision_count = 0
        self.episode_has_collision = False
        self.first_collision_step = None
        self.first_collision_type = None
        self.first_collision_goal_progress_ratio = None
        self.left_boundary_collision_count = 0
        self.right_boundary_collision_count = 0
        self.top_boundary_collision_count = 0
        self.bottom_boundary_collision_count = 0
        self.boundary_contact_step_count = 0
        self.left_boundary_contact_step_count = 0
        self.right_boundary_contact_step_count = 0
        self.top_boundary_contact_step_count = 0
        self.bottom_boundary_contact_step_count = 0
        self._current_boundary_contact_streak[:] = 0
        self.max_boundary_contact_streak = 0
        self._safe_brake_active = False
        self._safety_mode = "NORMAL"
        self._safe_recovery_target = None
        self._recovery_stable_steps = 0
        self._last_boundary_guard_result = None
        self._last_obstacle_guard_result = None
        self.shield_intervention_count = 0
        self.joint_guard_intervention_count = 0
        self.initial_joint_guard_intervention_steps = []
        self.final_joint_guard_residual_count = 0
        self.final_joint_guard_residual_steps = []
        self.obstacle_ttc_projection_count = 0
        self.obstacle_ttc_projection_steps = []
        self.obstacle_ttc_fallback_count = 0
        self.obstacle_ttc_post_action_goal_velocity_sum = 0.0
        self.obstacle_ttc_post_action_goal_velocity_count = 0
        self.obstacle_ttc_post_action_min_clearance_m = float("inf")
        self.ttc_headroom_audit_events = []
        self.ttc_headroom_intervention_event_count = 0
        self.ttc_headroom_positive_event_count = 0
        self.ttc_v2_high_progress_selection_count = 0
        self.total_action_modification_count = 0
        self.goal_corridor_projection_count = 0
        self.goal_corridor_agent_projection_count = 0
        self.goal_corridor_projection_steps = []
        self.goal_corridor_tangential_progress_m = 0.0
        self.goal_corridor_nonpositive_progress_count = 0
        self._last_goal_corridor_result = None
        self.shield_invariant_violation_count = 0
        self.safe_brake_step_count = 0
        self.safe_brake_entry_count = 0
        self.safe_brake_exit_count = 0
        self._safe_brake_current_streak = 0
        self.max_safe_brake_streak = 0
        self.pair_infeasible_count = 0
        self.pair_infeasible_entry_count = 0
        self._last_pair_infeasible = False
        self.safe_brake_entry_steps = []
        self.safe_brake_exit_steps = []
        self.safe_brake_durations = []
        self.safe_brake_entry_reasons = []
        self.safe_recover_step_count = 0
        self.safe_recover_entry_count = 0
        self.safe_recover_exit_count = 0
        self.safe_recovery_failure_count = 0
        self.safe_recovery_unavailable_count = 0
        self.safe_recovery_retry_count = 0
        self.safe_recovery_abort_count = 0
        self.safe_recovery_hard_abort_count = 0
        self.safe_recovery_refresh_count = 0
        self.safe_recovery_reentry_count = 0
        self.safe_recovery_abort_then_success_count = 0
        self.safe_recovery_episode_failure = False
        self.safe_recovery_affected_agent_count = 0
        self.max_safe_recover_streak = 0
        self.safe_recovery_durations = []
        self.safe_recovery_events = []
        self.boundary_recovery_trigger_count = 0
        self._safe_recover_current_streak = 0
        self._safe_recovery_elapsed_steps = 0
        self._safe_recovery_sequence_steps = 0
        self._safe_recovery_attempt_index = 0
        self._safe_recovery_failed = False
        self._safe_recovery_entry_sources = ()
        self._safe_recovery_previous_margin = None
        self._safe_recovery_no_improvement_steps = 0
        self._v5_release_after_brake = False
        self.recovery_incident_count = 0
        self.recovery_incident_merge_count = 0
        self.recovery_incident_reentry_reset_count = 0
        self.recovery_duplicate_hard_abort_count = 0
        self.max_recovery_incident_steps = 0
        self.terminal_recovery_unresolved = False
        self._recovery_incident_counter = 0
        self._recovery_incident_id = None
        self._recovery_incident_elapsed_steps = 0
        self._recovery_incident_sequence_steps = 0
        self._recovery_incident_refresh_count = 0
        self._recovery_incident_sources = ()
        self._recovery_incident_hazard_keys = ()
        self._recovery_incident_hard_abort_emitted = False
        self._recovery_incident_block_reentry_until_step = -1
        self._recovery_incident_start_step = None
        self._recovery_direction_unavailable_streak = 0
        self._recovery_direction_unavailable_last_step = -1
        self.max_recovery_direction_unavailable_streak = 0
        self.recovery_watchdog_step_count = 0
        self.recovery_incident_close_count = 0
        self.recovery_incident_reopen_count = 0
        self.hazard_retrigger_count = 0
        self.same_hazard_reentry_count = 0
        self.hazard_resolution_steps = []
        self.incident_close_reason_counts = {}
        self._closed_recovery_hazard_steps = {}
        self.terminal_incident_reconciled_count = 0
        self.recovery_telemetry_schema_version = (
            "shared_closure_consistency_telemetry_v1"
            if self.star_safety_state_machine_version
            == "star_closure_consistency_audit_v1"
            else
            "shared_recovery_telemetry_v1"
            if self.star_safety_state_machine_version
            == "star_persistent_recovery_v5_4"
            else "shared_recovery_telemetry_v1"
        )
        self.recovery_phase = "IDLE"
        self.escape_steps = 0
        self.consolidation_steps = 0
        self.release_shadow_check_passed = False
        self.release_shadow_check_count = 0
        self.release_shadow_check_failure_count = 0
        self.release_margin_m = float("nan")
        self.release_margin_delta_m = float("nan")
        self.bounded_recovery_exhaustion_count = 0
        self.bounded_exhaustion_then_success_count = 0
        self.rapid_same_hazard_retrigger_count = 0
        self.recovery_lineage_failure_count = 0
        self.recovery_attributable_mission_failure = False
        self.recovery_attributable_failure_reason = None
        self.max_recovery_lineage_steps = 0
        self.recovery_lineage_budget_reset_count = 0
        self.recovery_duplicate_failure_record_count = 0
        self.max_rapid_retriggers_per_lineage = 0
        self._recovery_lineage_counter = 0
        self._recovery_lineage_id = None
        self._recovery_lineage_hazard_keys = ()
        self._recovery_lineage_incident_count = 0
        self._recovery_lineage_steps = 0
        self._recovery_lineage_exhaustion_count = 0
        self._recovery_lineage_rapid_retriggers = 0
        self._closed_recovery_lineages = {}
        self._failed_recovery_hazard_keys = set()
        self._recovery_lineage_failure_emitted = False
        self.recovery_lineage_terminal_state = "IDLE"
        self.recovery_candidate_selection_count = 0
        self.recovery_candidate_rejection_count = 0
        self.recovery_candidate_rejection_reason_counts = {}
        self.recovery_candidate_post_projection_norm = float("nan")
        self.recovery_candidate_predicted_margin_gain = float("nan")
        self.recovery_candidate_fixed_point_streak = 0
        self.release_shadow_horizon_min_margin = float("nan")
        self.release_shadow_horizon_min_ttc = float("nan")
        self._closure_tracker = ClosureConsistencyV1(
            clear_window_steps=self.closure_clear_window_steps,
            certificate_tail_steps=self.closure_certificate_tail_steps,
        )
        self.closure_state = CLOSURE_ACTIVE
        self.clearance_pending_steps = 0
        self.pending_clearance_reappearance_count = 0
        self.post_certificate_retrigger_count = 0
        self.close_without_certificate_count = 0
        self.stale_closure_bypass_count = 0
        self.release_certificate_count = 0
        self.release_certificate_id = 0
        self._closure_last_verified_hazard_keys = ()
        self._closure_last_verified_step = -1
        self._closure_budget_counted_step = -1
        self._current_action_trace = {}
        self._reset_action_stage_attribution()
        self._star_recovery_offending_pairs = ()
        self._star_recovery_affected_mask[:] = False
        self._boundary_stall_steps[:] = 0
        self._safe_recovery_start_positions = None
        self._pending_recovery_pair_infeasible = False
        self._pending_recovery_invariant_violations = 0
        self._last_recovery_reason = None
        self.minimum_observed_obstacle_ttc = float("inf")
        self.obstacle_ttc_risk_step_count = 0
        self._first_obstacle_visible_steps = {}
        self._active_collision_keys.clear()
        self.agents = self._sample_agents()
        self.goals = self._build_goals()
        self.goal = self.goals
        self.start_center = np.mean(self.positions, axis=0).astype(np.float32)
        self.obstacles = self._sample_obstacles()
        self._reset_obstacle_tracks()
        self._near_goal_stall_streak[:] = 0
        self._max_near_goal_stall_streak[:] = 0
        self._near_goal_stall_triggered[:] = False
        self._near_goal_stall_first_steps[:] = -1
        self._reset_obstacle_visibility_metrics()

        self._reset_subgoal_metrics()
        self.subgoals = self.goals.copy()
        if self.oracle_enabled:
            self.subgoals = self._generate_valid_subgoals()
        self._previous_valid_subgoals = self.subgoals.copy()
        self._previous_subgoal_distances = self._subgoal_distances()
        self._subgoal_segment_start_distance = self._previous_subgoal_distances.copy()
        self._subgoal_started_reached = self._previous_subgoal_distances < self.subgoal_reach_radius
        self._subgoal_arrival_seen = self._subgoal_started_reached.copy()
        self._subgoal_distance_sum = float(np.sum(self._previous_subgoal_distances))
        self._subgoal_distance_samples = self.num_agents

        self._previous_distances = self._goal_distances()
        self._ever_in_goal = self._previous_distances < self.goal_radius
        self._previous_in_goal = self._ever_in_goal.copy()
        self._team_arrival_observed = bool(np.all(self._ever_in_goal))
        self._first_goal_entry_steps = np.where(self._ever_in_goal, 0, -1).astype(np.int32)
        self._last_goal_entry_steps = self._first_goal_entry_steps.copy()
        self._goal_exit_counts = np.zeros(self.num_agents, dtype=np.int32)
        self._in_goal_step_counts = np.zeros(self.num_agents, dtype=np.int32)
        self._episode_start_max_goal_distance = float(np.max(self._previous_distances))
        self._episode_start_mean_goal_distance = float(np.mean(self._previous_distances))
        self._episode_start_goal_distances = self._previous_distances.copy()
        self._previous_team_potential = self._team_goal_potential(
            self._previous_distances
        )
        self._best_team_goal_completion = self._team_goal_completion(
            self._previous_distances
        )
        self._max_x_per_agent = self.positions[:, 0].copy()
        self._entered_half_map = self.positions[:, 0] >= self.map_size[0] * 0.5
        clearances = self._nearest_obstacle_distances()
        self._previous_obstacle_clearances = clearances.copy()
        finite = clearances[np.isfinite(clearances)]
        self._episode_min_obstacle_clearance = float(np.min(finite)) if finite.size else np.inf
        self._episode_min_inter_agent_distance = self._minimum_inter_agent_distance()
        info = {
            "goal": self.goals.copy(), "goals": self.goals.copy(), "subgoals": self.subgoals.copy(),
            "start_center": self.start_center.copy(), "num_agents": self.num_agents,
            "static_obstacles": self.static_obstacles, "fixed_obstacle_layout": False,
            "static_obstacle_num": self.static_obstacle_num, "dynamic_obstacle_num": self.dynamic_obstacle_num,
            "reach_success": False, "safe_success": False,
            "collision_count": 0, "obstacle_collision_count": 0,
            "static_obstacle_collision_count": 0,
            "dynamic_obstacle_collision_count": 0,
            "inter_agent_collision_count": 0, "boundary_collision_count": 0,
            "episode_has_collision": False, "first_collision_step": None,
            "first_collision_type": None,
            "first_collision_goal_progress_ratio": None,
            "remaining_goal_progress_at_first_collision": None,
            **self._boundary_metric_snapshot(),
            "oracle_subgoal_enabled": self.oracle_enabled,
            **self._obstacle_tracking_metric_snapshot(),
            **self._near_goal_stall_metric_snapshot(),
            **self._obstacle_visibility_metric_snapshot(),
            **self._subgoal_metric_snapshot(),
            "seed": seed,
        }
        self._transition_window.clear()
        self.first_failure_window = []
        return self.build_local_observations(), info

    def _update_star_safety_state(self, guard_result: Any, *, brake_applied: bool) -> None:
        """Advance the V2 STAR brake state from execution-level evidence."""
        pair_infeasible = bool(getattr(guard_result, "pair_infeasible", False))
        self.pair_infeasible_count += int(pair_infeasible)
        if pair_infeasible and not self._last_pair_infeasible:
            self.pair_infeasible_entry_count += 1
        self._last_pair_infeasible = pair_infeasible

        if brake_applied:
            self._safe_brake_current_streak += 1
            self.max_safe_brake_streak = max(
                self.max_safe_brake_streak, self._safe_brake_current_streak
            )
        else:
            self._safe_brake_current_streak = 0

        if pair_infeasible:
            if not self._safe_brake_active:
                self.safe_brake_entry_count += 1
                self.safe_brake_entry_steps.append(int(self.step_count))
                self.safe_brake_entry_reasons.append("joint_guard_infeasible")
            self._safety_mode = "SAFE_BRAKE"
            self._safe_brake_active = True
            self._safe_recovery_target = None
            self._recovery_stable_steps = 0
            self._last_recovery_reason = "joint_guard_infeasible"
            return

        stable_reference = bool(
            getattr(guard_result, "reference_safe_action_available", False)
            and getattr(guard_result, "minimum_pair_margin", -np.inf) >= -1e-6
            and getattr(guard_result, "invariant_violations", 1) == 0
        )
        if self._safe_brake_active and brake_applied and stable_reference:
            self._recovery_stable_steps += 1
            if self._recovery_stable_steps >= self.safe_recovery_exit_stable_steps:
                self._safe_brake_active = False
                self._safety_mode = "NORMAL"
                self._recovery_stable_steps = 0
                self.safe_brake_exit_count += 1
                self.safe_brake_exit_steps.append(int(self.step_count))
                self.safe_brake_durations.append(
                    int(self._safe_brake_current_streak)
                )
                self._last_recovery_reason = "joint_guard_recovered"
        elif self._safe_brake_active:
            self._recovery_stable_steps = 0

    def _reset_action_stage_attribution(self) -> None:
        stages = (
            "recovery_overlay", "goal_corridor", "initial_joint_guard",
            "obstacle_ttc", "final_joint_guard", "total",
        )
        self.action_stage_attribution = {
            stage: {
                "step_count": 0,
                "modified_step_count": 0,
                "agent_count": 0,
                "material_agent_count": 0,
                "large_agent_count": 0,
                "direction_reversal_count": 0,
                "delta_norm_sum": 0.0,
                "input_norm_sum": 0.0,
                "output_norm_sum": 0.0,
                "direction_cosine_sum": 0.0,
                "norm_ratio_sum": 0.0,
            }
            for stage in stages
        }
        self._current_action_trace = {}
        self._current_action_stage_actions: dict[str, np.ndarray] = {}

    def _trace_action_stage_snapshot(
        self, stage: str, actions: np.ndarray,
    ) -> None:
        """Record causal risk attribution without changing the action path."""

        if not self.closure_trace_audit_enabled:
            return
        values = np.asarray(actions, dtype=np.float64)
        snapshot = self._safety_planning_snapshot()
        obstacle_risks = assess_obstacle_actions(
            actions=values,
            velocities=self.velocities,
            visible_obstacles=snapshot.visible_obstacles,
            agent_radius=self.agent_radius,
            obstacle_safety_margin=float(
                self.oracle_config["obstacle_safety_margin"]
            ),
            invariant_buffer=self.boundary_guard.invariant_buffer_m,
            max_acceleration=self.agent_max_acceleration,
            dt=self.dt,
            horizon_seconds=self.obstacle_guard_horizon_seconds,
            warning_ttc_seconds=self.obstacle_guard_warning_ttc_seconds,
            emergency_ttc_seconds=self.obstacle_guard_emergency_ttc_seconds,
        )
        minimum_clearance = minimum_predicted_obstacle_clearance_m(
            actions=values,
            velocities=self.velocities,
            visible_obstacles=snapshot.visible_obstacles,
            agent_radius=self.agent_radius,
            obstacle_safety_margin=float(
                self.oracle_config["obstacle_safety_margin"]
            ),
            invariant_buffer=self.boundary_guard.invariant_buffer_m,
            max_acceleration=self.agent_max_acceleration,
            dt=self.dt,
            horizon_seconds=self.obstacle_guard_horizon_seconds,
        )
        guard_kwargs = dict(
            map_size=self.map_size,
            agent_radius=self.agent_radius,
            dt=self.dt,
            max_acceleration=self.agent_max_acceleration,
            max_speed=self.agent_max_speed,
            safe_brake=False,
        )
        if isinstance(self.boundary_guard, JointSafetyGuardV2):
            guard_kwargs["teammate_safety_distance"] = float(
                self.oracle_config["teammate_safety_distance"]
            )
        guard = self.boundary_guard.project_actions(
            self.positions, self.velocities, values, **guard_kwargs
        )
        reasons = []
        if obstacle_risks:
            reasons.append("observable_obstacle_risk")
        if bool(getattr(guard, "recovery_required", False)):
            reasons.append("joint_guard_recovery_required")
        if int(getattr(guard, "invariant_violations", 0)):
            reasons.append("joint_guard_invariant_violation")
        self._current_action_trace[str(stage)] = {
            "actions": values.astype(float).tolist(),
            "obstacle_risk_count": len(obstacle_risks),
            "minimum_predicted_obstacle_clearance_m": (
                float(minimum_clearance)
                if np.isfinite(minimum_clearance) else None
            ),
            "minimum_boundary_surface_clearance_m": float(
                np.min(guard.surface_clearances)
            ),
            "minimum_pair_margin_m": (
                float(guard.minimum_pair_margin)
                if hasattr(guard, "minimum_pair_margin") else None
            ),
            "rejection_reasons": reasons,
        }

    def _record_action_stage(
        self, stage: str, before: np.ndarray, after: np.ndarray,
    ) -> None:
        """Accumulate observable per-stage action attribution."""

        if self.runtime_telemetry_level == "train_minimal":
            self._append_stage_intervention_records(stage, before, after)
            return

        left = np.asarray(before, dtype=np.float64)
        right = np.asarray(after, dtype=np.float64)
        if left.shape != (self.num_agents, 2) or right.shape != left.shape:
            raise ValueError("action attribution matrices have the wrong shape")
        self._current_action_stage_actions[str(stage)] = right.copy()
        self._trace_action_stage_snapshot(stage, right)
        row = self.action_stage_attribution[stage]
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        delta = float(np.linalg.norm(right - left))
        denominator = left_norm * right_norm
        cosine = (
            float(np.sum(left * right) / denominator)
            if denominator > 1e-12 else float(left_norm <= 1e-12 and right_norm <= 1e-12)
        )
        ratio = (
            right_norm / left_norm
            if left_norm > 1e-12
            else float(right_norm <= 1e-12)
        )
        row["step_count"] = int(row["step_count"]) + 1
        row["modified_step_count"] = int(row["modified_step_count"]) + int(
            delta > 1e-7
        )
        for agent_id in range(self.num_agents):
            _, delta_v, agent_cosine = action_modification_level(
                left[agent_id], right[agent_id],
                acceleration_scale=self.agent_max_acceleration, dt=self.dt,
            )
            row["agent_count"] = int(row["agent_count"]) + 1
            row["material_agent_count"] = int(row["material_agent_count"]) + int(
                delta_v >= 0.05
            )
            row["large_agent_count"] = int(row["large_agent_count"]) + int(
                delta_v >= 0.25
            )
            row["direction_reversal_count"] = int(row["direction_reversal_count"]) + int(
                agent_cosine < 0.0
            )
        row["delta_norm_sum"] = float(row["delta_norm_sum"]) + delta
        row["input_norm_sum"] = float(row["input_norm_sum"]) + left_norm
        row["output_norm_sum"] = float(row["output_norm_sum"]) + right_norm
        row["direction_cosine_sum"] = float(
            row["direction_cosine_sum"]
        ) + cosine
        row["norm_ratio_sum"] = float(row["norm_ratio_sum"]) + ratio

        self._append_stage_intervention_records(stage, left, right)

    def _append_stage_intervention_records(
        self, stage: str, before: np.ndarray, after: np.ndarray,
    ) -> None:
        if stage not in {"initial_joint_guard", "obstacle_ttc", "final_joint_guard"}:
            return
        left = np.asarray(before, dtype=np.float64)
        right = np.asarray(after, dtype=np.float64)
        if left.shape != (self.num_agents, 2) or right.shape != left.shape:
            raise ValueError("stage intervention matrices have the wrong shape")
        eligible = not (
            self._safety_mode in {"SAFE_BRAKE", "SAFE_RECOVER", "SAFE_CONSOLIDATE"}
            or self._safe_brake_active
        )
        for agent_id in range(self.num_agents):
            level, delta_v, agent_cosine = action_modification_level(
                left[agent_id], right[agent_id],
                acceleration_scale=self.agent_max_acceleration,
                dt=self.dt,
            )
            self._stage_intervention_records.append(
                StageInterventionRecordV1(
                    uav_id=agent_id,
                    stage=stage,
                    input_action=tuple(float(v) for v in left[agent_id]),
                    output_action=tuple(float(v) for v in right[agent_id]),
                    direction_cosine=agent_cosine,
                    reason="modified_by_safety" if level != InterventionLevel.NONE else "unchanged",
                    safety_margin=None,
                    eligible_for_actor_risk=eligible,
                    modification_level=level,
                    delta_v_mps=delta_v,
                )
            )

    def _causal_safety_snapshot(
        self, requested_actions: np.ndarray,
    ) -> CausalSafetySnapshotV1:
        observable = self._safety_planning_snapshot()
        team_rows = tuple(
            tuple(float(v) for v in np.concatenate((
                self.positions[agent_id], self.velocities[agent_id],
                self.subgoals[agent_id], self.goals[agent_id],
            )))
            for agent_id in range(self.num_agents)
        )
        rng_payload = json.dumps(
            self._rng.bit_generator.state, sort_keys=True, default=str,
            separators=(",", ":"),
        ).encode("utf-8")
        recovery_payload = json.dumps({
            "mode": self._safety_mode,
            "safe_brake": bool(self._safe_brake_active),
            "target": None if self._safe_recovery_target is None else np.asarray(
                self._safe_recovery_target
            ).astype(float).tolist(),
            "stable_steps": int(self._recovery_stable_steps),
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")
        safety_state = (
            float(self.boundary_guard.surface_clearance_m),
            float(self.boundary_guard.invariant_buffer_m),
            float(self.oracle_config["teammate_safety_distance"]),
            float(self.oracle_config["obstacle_safety_margin"]),
            float(
                self._safe_brake_active
                or self._safety_mode
                in {"SAFE_BRAKE", "SAFE_RECOVER", "SAFE_CONSOLIDATE"}
            ),
        )
        return CausalSafetySnapshotV1(
            step=int(self.step_count),
            team_observation=team_rows,
            requested_actions=tuple(
                tuple(float(v) for v in row) for row in requested_actions
            ),
            safety_state=safety_state,
            obstacle_tracks=observable.visible_obstacles,
            rng_state_digest=sha256(rng_payload).hexdigest(),
            recovery_state_digest=sha256(recovery_payload).hexdigest(),
        )

    def _finalize_stage_intervention_trace(
        self, requested_actions: np.ndarray, executed_actions: np.ndarray,
    ) -> None:
        snapshot = self._current_causal_safety_snapshot
        records = list(self._stage_intervention_records)
        ineligible = bool(
            self._safe_brake_active
            or self._safety_mode in {"SAFE_BRAKE", "SAFE_RECOVER", "SAFE_CONSOLIDATE"}
        )
        for stage, actions in (
            ("actor_request", requested_actions),
            ("executed", executed_actions),
        ):
            for agent_id in range(self.num_agents):
                row = tuple(float(v) for v in actions[agent_id])
                records.append(StageInterventionRecordV1(
                    uav_id=agent_id, stage=stage,
                    input_action=row, output_action=row,
                    direction_cosine=1.0, reason="boundary_record",
                    safety_margin=None,
                    eligible_for_actor_risk=not ineligible,
                    modification_level=InterventionLevel.NONE,
                    delta_v_mps=0.0,
                ))
        if ineligible:
            records = [replace(item, eligible_for_actor_risk=False) for item in records]
        self.last_stage_intervention_trace = StageInterventionTraceV1(
            step=int(self.step_count), snapshot_digest=snapshot.digest,
            records=tuple(records),
            recovery_active=self._safety_mode in {"SAFE_RECOVER", "SAFE_CONSOLIDATE"},
            safe_brake_active=bool(self._safe_brake_active),
            terminal_transition=False,
            metadata={"safety_execution_version": "staged_joint_guard_ttc_v1"},
        )

    def _causal_safety_feature_vector(self) -> tuple[np.ndarray, float]:
        snapshot = self._current_causal_safety_snapshot
        if snapshot is None or len(snapshot.safety_state) != 5:
            raise RuntimeError("causal safety context requires the V2 pre-action snapshot")
        positions = np.asarray(
            [row[:2] for row in snapshot.team_observation], dtype=np.float32
        )
        domain_margin = (
            self.agent_radius + float(snapshot.safety_state[0])
            + float(snapshot.safety_state[1])
        )
        boundary_margin = float(np.min(np.concatenate((
            positions - domain_margin,
            self.map_size - domain_margin - positions,
        ))))
        pair_margin = float("inf")
        for left in range(self.num_agents):
            for right in range(left + 1, self.num_agents):
                pair_margin = min(
                    pair_margin,
                    float(np.linalg.norm(positions[left] - positions[right]))
                    - float(snapshot.safety_state[2]),
                )
        obstacle_margin = float("inf")
        for row in snapshot.obstacle_tracks:
            for obstacle in row:
                obstacle_margin = min(
                    obstacle_margin,
                    float(np.linalg.norm(obstacle.relative_position))
                    - self.agent_radius - obstacle.radius
                    - float(snapshot.safety_state[3]),
                )
        finite_fallback = 1e3
        pair_margin = pair_margin if np.isfinite(pair_margin) else finite_fallback
        obstacle_margin = (
            obstacle_margin if np.isfinite(obstacle_margin) else finite_fallback
        )
        current_margin = min(boundary_margin, pair_margin, obstacle_margin)
        context = np.asarray((
            *snapshot.safety_state[:4],
            boundary_margin,
            pair_margin,
            obstacle_margin,
            float(snapshot.safety_state[4]),
        ), dtype=np.float32)
        return context, float(current_margin)

    def _action_stage_attribution_snapshot(self) -> dict[str, dict[str, float | int]]:
        payload: dict[str, dict[str, float | int]] = {}
        for stage, values in self.action_stage_attribution.items():
            count = max(int(values["step_count"]), 1)
            payload[stage] = {
                **values,
                "modification_rate": float(values["modified_step_count"]) / count,
                "material_modification_rate": float(values["material_agent_count"]) / max(int(values["agent_count"]), 1),
                "large_modification_rate": float(values["large_agent_count"]) / max(int(values["agent_count"]), 1),
                "direction_reversal_rate": float(values["direction_reversal_count"]) / max(int(values["agent_count"]), 1),
                "mean_delta_norm": float(values["delta_norm_sum"]) / count,
                "mean_direction_cosine": (
                    float(values["direction_cosine_sum"]) / count
                ),
                "mean_norm_ratio": float(values["norm_ratio_sum"]) / count,
            }
        return payload

    @staticmethod
    def _hazard_agent_ids(hazard_keys: Sequence[str]) -> set[int]:
        agents: set[int] = set()
        for key in hazard_keys:
            parts = str(key).split(":")
            if len(parts) >= 4 and parts[0] == "agent" and parts[2] == "track":
                try:
                    agents.add(int(parts[1]))
                except ValueError:
                    continue
        return agents

    def _closure_tracks_available(self, hazard_keys: Sequence[str]) -> bool:
        """Check only causal slots; pair and boundary geometry is always known."""

        obstacle_keys = {
            str(key) for key in hazard_keys if str(key).startswith("agent:")
        }
        if not obstacle_keys:
            return True
        snapshot = self._safety_planning_snapshot()
        return obstacle_keys <= obstacle_track_keys(snapshot.visible_obstacles)

    def _advance_closure_pending_budget(self) -> None:
        if self._closure_budget_counted_step == int(self.step_count):
            return
        self._closure_budget_counted_step = int(self.step_count)
        self.clearance_pending_steps += 1
        self._recovery_incident_elapsed_steps += 1
        self._recovery_incident_sequence_steps += 1
        self._recovery_lineage_steps += 1
        self.max_recovery_incident_steps = max(
            self.max_recovery_incident_steps,
            self._recovery_incident_sequence_steps,
        )
        self.max_recovery_lineage_steps = max(
            self.max_recovery_lineage_steps,
            self._recovery_lineage_steps,
        )

    def _observe_closure_consistency(
        self, requested_actions: np.ndarray,
    ) -> None:
        """Advance the final closure audit without introducing control actions."""

        if (
            self.star_safety_state_machine_version
            != "star_closure_consistency_audit_v1"
            or self._recovery_incident_id is None
        ):
            return
        stored = set(self._recovery_incident_hazard_keys)
        active = self._currently_active_recovery_hazard_keys()
        same_active = bool(stored.intersection(active))
        track_available = self._closure_tracks_available(
            self._recovery_incident_hazard_keys
        )
        certificate = False
        if not same_active and track_available:
            self.release_shadow_check_count += 1
            certificate = self._v5_4_release_shadow_passes(
                np.asarray(requested_actions, dtype=np.float32)
            )
            self.release_shadow_check_passed = bool(certificate)
            self.release_shadow_check_failure_count += int(not certificate)
        decision = self._closure_tracker.observe(ClosureObservationV1(
            track_available=track_available,
            same_hazard_active=same_active,
            release_certificate_passed=certificate,
        ))
        self.closure_state = decision.state
        self.pending_clearance_reappearance_count = int(
            self._closure_tracker.pending_reappearance_count
        )
        self.safe_recovery_events.append({
            "event": "closure_consistency_observation",
            "step": int(self.step_count),
            "incident_id": int(self._recovery_incident_id),
            "lineage_id": self._recovery_lineage_id,
            "hazard_keys": list(self._recovery_incident_hazard_keys),
            "state": decision.state,
            "track_available": bool(track_available),
            "same_hazard_active": bool(same_active),
            "release_certificate_passed": bool(certificate),
            "clear_streak": int(decision.clear_streak),
            "certificate_tail": list(decision.certificate_tail),
            "pending_reappearance": bool(decision.pending_reappearance),
        })
        if decision.close_allowed:
            keys = tuple(self._recovery_incident_hazard_keys)
            self.release_certificate_id += 1
            self.release_certificate_count += 1
            self._closure_last_verified_hazard_keys = keys
            self._closure_last_verified_step = int(self.step_count)
            self.safe_recovery_events.append({
                "event": "release_certificate",
                "step": int(self.step_count),
                "certificate_id": int(self.release_certificate_id),
                "incident_id": int(self._recovery_incident_id),
                "lineage_id": self._recovery_lineage_id,
                "hazard_keys": list(keys),
                "clear_window_steps": int(self.closure_clear_window_steps),
                "certificate_tail_steps": int(
                    self.closure_certificate_tail_steps
                ),
            })
            self._release_v5_2_recovery_incident(
                "verified_clearance_certificate"
            )
            return
        if decision.state in {CLOSURE_PENDING, CLOSURE_UNKNOWN}:
            # Pending permits the Actor through the unchanged execution stack.
            self._safety_mode = "NORMAL"
            self._safe_brake_active = False
            self.recovery_phase = decision.state
            self._advance_closure_pending_budget()
            if (
                self._recovery_incident_sequence_steps
                >= self.safe_recovery_maximum_steps
            ):
                self._emit_v5_3_bounded_exhaustion(
                    "closure_pending_budget_exhaustion"
                )
        elif decision.pending_reappearance:
            self._safety_mode = "SAFE_RECOVER"
            self._safe_brake_active = True
            self.recovery_phase = "ESCAPE"
            self._safe_recovery_target = None

    def _run_closure_trace_audit(
        self, requested_actions: np.ndarray,
    ) -> None:
        """Score a registered shadow action set without changing execution."""

        if (
            not self.closure_trace_audit_enabled
            or self._recovery_incident_id is None
        ):
            return
        actor = np.asarray(requested_actions, dtype=np.float32)
        affected = self._star_recovery_affected_mask.copy()
        hold = deterministic_hold_actions(
            self.velocities,
            max_acceleration=self.agent_max_acceleration,
            dt=self.dt,
        )
        candidates: list[tuple[str, np.ndarray]] = [
            ("actor", actor.copy()),
            ("hold", hold.copy()),
        ]
        snapshot = self._safety_planning_snapshot()
        sources = tuple(
            value for value in self._safe_recovery_entry_sources
            if value in {"boundary", "pair", "obstacle_guard_fallback"}
        )
        visible_obstacles = snapshot.visible_obstacles
        if "obstacle_guard_fallback" in sources:
            active_keys = set(self._recovery_incident_hazard_keys)
            visible_obstacles = tuple(
                tuple(
                    obstacle for obstacle in tracks
                    if not active_keys
                    or f"agent:{agent_id}:track:{obstacle.track_id}"
                    in active_keys
                )
                for agent_id, tracks in enumerate(snapshot.visible_obstacles)
            )
        if sources:
            production = build_star_recovery_candidates(
                positions=self.positions,
                velocities=self.velocities,
                final_goals=self.goals,
                visible_obstacles=visible_obstacles,
                map_size=self.map_size,
                agent_radius=self.agent_radius,
                surface_clearance=self.boundary_guard.surface_clearance_m,
                invariant_buffer=self.boundary_guard.invariant_buffer_m,
                max_speed=self.agent_max_speed,
                max_acceleration=self.agent_max_acceleration,
                dt=self.dt,
                teammate_safety_distance=float(
                    self.oracle_config["teammate_safety_distance"]
                ),
                obstacle_safety_margin=float(
                    self.oracle_config["obstacle_safety_margin"]
                ),
                offending_pairs=self._star_recovery_offending_pairs,
                affected_agent_mask=affected,
                target_distance=self.safe_recovery_target_distance,
                speed_fraction=self.safe_recovery_speed_fraction,
                boundary_trigger_margin=(
                    self.safe_recovery_boundary_trigger_margin
                ),
                obstacle_repulsion_margin=(
                    self.safe_recovery_obstacle_repulsion_margin
                ),
                entry_sources=sources,
            )
            for candidate in production:
                proposal = actor.copy()
                mask = np.asarray(
                    candidate.affected_agent_mask, dtype=bool
                )
                proposal[mask] = candidate.actions[mask]
                candidates.append((f"production_{candidate.label}", proposal))
        goal_proposal = actor.copy()
        for agent_id in np.flatnonzero(affected):
            delta = self.goals[int(agent_id)] - self.positions[int(agent_id)]
            norm = float(np.linalg.norm(delta))
            if norm <= 1e-12:
                continue
            desired_velocity = (
                self.agent_max_speed * self.safe_recovery_speed_fraction
                * delta / norm
            )
            goal_proposal[int(agent_id)] = np.clip(
                (desired_velocity - self.velocities[int(agent_id)])
                / max(self.agent_max_acceleration * self.dt, 1e-12),
                -1.0,
                1.0,
            )
        candidates.append(("goal_direction", goal_proposal))
        for magnitude in (0.5, 1.0):
            for index in range(16):
                theta = 2.0 * np.pi * float(index) / 16.0
                direction = np.asarray(
                    [np.cos(theta), np.sin(theta)], dtype=np.float32
                ) * float(magnitude)
                proposal = actor.copy()
                proposal[affected] = direction
                candidates.append((f"grid_{index:02d}_m{magnitude:.1f}", proposal))
        baseline = self._v5_4_observable_envelope(
            hold,
            horizon_steps=1,
            active_hazard_keys=self._recovery_incident_hazard_keys,
        )
        baseline_margin = float(baseline["final_margin"])
        rows = []
        for label, proposal in candidates:
            envelope = self._v5_4_observable_envelope(
                proposal,
                horizon_steps=1,
                active_hazard_keys=self._recovery_incident_hazard_keys,
            )
            margin = float(envelope["final_margin"])
            gain = (
                margin - baseline_margin
                if np.isfinite(margin) and np.isfinite(baseline_margin)
                else float("inf") if np.isfinite(margin) else float("-inf")
            )
            rows.append({
                "label": label,
                "feasible": bool(envelope["feasible"]),
                "margin_gain_m": gain,
                "rejection_reason": str(envelope["rejection_reason"]),
            })
        positive = [
            row for row in rows
            if row["feasible"]
            and float(row["margin_gain_m"])
            >= self.safe_recovery_candidate_minimum_margin_gain
        ]
        self.safe_recovery_events.append({
            "event": "closure_registered_action_audit",
            "step": int(self.step_count),
            "incident_id": int(self._recovery_incident_id),
            "lineage_id": self._recovery_lineage_id,
            "hazard_keys": list(self._recovery_incident_hazard_keys),
            "candidate_count": len(rows),
            "feasible_count": sum(bool(row["feasible"]) for row in rows),
            "positive_margin_count": len(positive),
            "no_safe_action_found_in_registered_search": not positive,
            "candidates": rows,
        })

    def _currently_active_recovery_hazard_keys(self) -> set[str]:
        """Return causal hazards that still require an execution response."""

        active: set[str] = set()
        snapshot = self._safety_planning_snapshot()
        hold = deterministic_hold_actions(
            self.velocities,
            max_acceleration=self.agent_max_acceleration,
            dt=self.dt,
        )
        obstacle_risks = assess_obstacle_actions(
            actions=hold,
            velocities=self.velocities,
            visible_obstacles=snapshot.visible_obstacles,
            agent_radius=self.agent_radius,
            obstacle_safety_margin=float(
                self.oracle_config["obstacle_safety_margin"]
            ),
            invariant_buffer=self.boundary_guard.invariant_buffer_m,
            max_acceleration=self.agent_max_acceleration,
            dt=self.dt,
            horizon_seconds=self.obstacle_guard_horizon_seconds,
            warning_ttc_seconds=self.obstacle_guard_warning_ttc_seconds,
            emergency_ttc_seconds=self.obstacle_guard_emergency_ttc_seconds,
        )
        active.update(
            f"agent:{item.agent_id}:track:{item.track_id}"
            for item in obstacle_risks
        )
        teammate_distance = float(
            self.oracle_config["teammate_safety_distance"]
        )
        positions = self.positions
        for left in range(self.num_agents):
            for right in range(left + 1, self.num_agents):
                if float(np.linalg.norm(positions[left] - positions[right])) <= teammate_distance:
                    active.add(f"pair:{left}:{right}")
        low = (
            self.agent_radius + self.boundary_guard.surface_clearance_m
            + self.boundary_guard.invariant_buffer_m
        )
        high = self.map_size - low
        for agent_id, position in enumerate(positions):
            if (
                np.any(position <= low + self.safe_recovery_boundary_trigger_margin)
                or np.any(position >= high - self.safe_recovery_boundary_trigger_margin)
            ):
                active.add(f"boundary:{agent_id}")
        return active

    def _close_v5_2_recovery_incident(self, reason: str) -> None:
        if self._recovery_incident_id is None:
            return
        if (
            self.star_safety_state_machine_version
            == "star_closure_consistency_audit_v1"
            and str(reason) == "closed_as_stale_hazard"
        ):
            self.stale_closure_bypass_count += 1
            self.close_without_certificate_count += 1
            self.safe_recovery_events.append({
                "event": "closure_bypass_blocked",
                "step": int(self.step_count),
                "incident_id": int(self._recovery_incident_id),
                "reason": str(reason),
                "hazard_keys": list(self._recovery_incident_hazard_keys),
            })
            return
        start = (
            int(self.step_count) if self._recovery_incident_start_step is None
            else int(self._recovery_incident_start_step)
        )
        resolution_steps = max(int(self.step_count) - start, 0)
        self.recovery_incident_close_count += 1
        self.hazard_resolution_steps.append(resolution_steps)
        self.incident_close_reason_counts[str(reason)] = (
            self.incident_close_reason_counts.get(str(reason), 0) + 1
        )
        for key in self._recovery_incident_hazard_keys:
            self._closed_recovery_hazard_steps[str(key)] = int(self.step_count)
        self.safe_recovery_events.append({
            "event": "incident_close",
            "step": int(self.step_count),
            "incident_id": int(self._recovery_incident_id),
            "close_reason": str(reason),
            "hazard_keys": list(self._recovery_incident_hazard_keys),
            "resolution_steps": resolution_steps,
        })
        self._clear_v5_1_recovery_incident()

    def _release_v5_2_recovery_incident(self, close_reason: str) -> None:
        self._safety_mode = "NORMAL"
        self._safe_brake_active = False
        self._safe_recovery_target = None
        self._safe_recovery_start_positions = None
        self._safe_recovery_elapsed_steps = 0
        self._safe_recovery_sequence_steps = 0
        self._safe_recovery_attempt_index = 0
        self._safe_recover_current_streak = 0
        self._recovery_stable_steps = 0
        self._star_recovery_offending_pairs = ()
        self._star_recovery_affected_mask[:] = False
        self._safe_recovery_entry_sources = ()
        self._safe_recovery_previous_margin = None
        self._safe_recovery_no_improvement_steps = 0
        self._recovery_direction_unavailable_streak = 0
        if self.star_safety_state_machine_version in {
            "star_persistent_recovery_v5_3",
            "star_persistent_recovery_v5_4",
            "star_closure_consistency_audit_v1",
        }:
            self._close_v5_3_lineage_incident(close_reason)
        else:
            self._close_v5_2_recovery_incident(close_reason)

    def _emit_v5_2_hard_abort(self, reason: str) -> None:
        if self._recovery_incident_hard_abort_emitted:
            self.recovery_duplicate_hard_abort_count += 1
            return
        self.safe_recovery_failure_count += 1
        self.safe_recovery_abort_count += 1
        self.safe_recovery_hard_abort_count += 1
        self.safe_recovery_episode_failure = True
        self._recovery_incident_hard_abort_emitted = True
        self._recovery_incident_block_reentry_until_step = int(self.step_count)
        self.safe_recovery_events.append({
            "event": "hard_abort",
            "step": int(self.step_count),
            "duration": int(self._recovery_incident_sequence_steps),
            "attempt_index": int(self._safe_recovery_attempt_index),
            "entry_sources": list(self._safe_recovery_entry_sources),
            "reason": str(reason),
            "affected_agent_mask": self._star_recovery_affected_mask.tolist(),
        })
        self._safety_mode = "SAFE_BRAKE"
        self._safe_brake_active = True
        self._safe_recovery_failed = True
        self._v5_release_after_brake = True
        self._safe_recovery_target = None
        self._last_recovery_reason = str(reason)

    def _handle_v5_2_recovery_unavailable(self, reason: str) -> None:
        if self._recovery_incident_id is None:
            return
        stored = set(self._recovery_incident_hazard_keys)
        active = self._currently_active_recovery_hazard_keys()
        if stored and not stored.intersection(active):
            if (
                self.star_safety_state_machine_version
                == "star_closure_consistency_audit_v1"
            ):
                # The incident remains open.  The next real Actor steps must
                # earn the five-step release certificate.
                self.closure_state = CLOSURE_PENDING
                self._safety_mode = "NORMAL"
                self._safe_brake_active = False
                self.recovery_phase = CLOSURE_PENDING
                return
            self._release_v5_2_recovery_incident("closed_as_stale_hazard")
            return
        if self._recovery_direction_unavailable_last_step == int(self.step_count):
            return
        self._recovery_direction_unavailable_last_step = int(self.step_count)
        self._recovery_direction_unavailable_streak += 1
        self.max_recovery_direction_unavailable_streak = max(
            self.max_recovery_direction_unavailable_streak,
            self._recovery_direction_unavailable_streak,
        )
        self.recovery_watchdog_step_count += 1
        self._recovery_incident_elapsed_steps += 1
        self._recovery_incident_sequence_steps += 1
        self._safe_recovery_sequence_steps += 1
        self.max_recovery_incident_steps = max(
            self.max_recovery_incident_steps,
            self._recovery_incident_sequence_steps,
        )
        self.safe_recovery_events.append({
            "event": "direction_unavailable_watchdog",
            "step": int(self.step_count),
            "reason": str(reason),
            "streak": int(self._recovery_direction_unavailable_streak),
            "incident_steps": int(self._recovery_incident_sequence_steps),
            "hazard_keys": list(self._recovery_incident_hazard_keys),
            "affected_agent_mask": self._star_recovery_affected_mask.tolist(),
        })
        if (
            self._recovery_direction_unavailable_streak
            >= self.safe_recovery_direction_unavailable_patience
            or self._recovery_incident_sequence_steps
            >= self.safe_recovery_maximum_steps
        ):
            if self.star_safety_state_machine_version in {
                "star_persistent_recovery_v5_3",
                "star_persistent_recovery_v5_4",
                "star_closure_consistency_audit_v1",
            }:
                self._emit_v5_3_bounded_exhaustion(
                    "direction_unavailable_bounded_exhaustion"
                )
            else:
                self._emit_v5_2_hard_abort(
                    "direction_unavailable_bounded_abort"
                )

    def _reconcile_v5_2_terminal_incident(self, episode_reason: str) -> None:
        """Close only causally cleared incidents before terminal auditing."""

        if self._recovery_incident_id is None:
            return
        stored = set(self._recovery_incident_hazard_keys)
        active = self._currently_active_recovery_hazard_keys()
        if stored and stored.intersection(active):
            return
        close_reason = (
            "resolved_by_goal_completion"
            if str(episode_reason) == "team_success"
            else "closed_as_stale_hazard"
        )
        self.terminal_incident_reconciled_count += 1
        self._release_v5_2_recovery_incident(close_reason)

    def _mark_v5_3_lineage_failure(self, reason: str) -> None:
        """Record a V5.3 failure once at lineage scope."""

        if self.star_safety_state_machine_version in {
            "star_persistent_recovery_v5_4",
            "star_closure_consistency_audit_v1",
        }:
            if self._recovery_lineage_failure_emitted:
                return
            self._recovery_lineage_failure_emitted = True
            self.recovery_lineage_terminal_state = "FAILED_LATCHED"
            self.recovery_lineage_failure_count += 1
            self.recovery_attributable_mission_failure = True
            self.recovery_attributable_failure_reason = str(reason)
            self.safe_recovery_episode_failure = True
            self._failed_recovery_hazard_keys.update(
                str(value) for value in self._recovery_lineage_hazard_keys
            )
            self.safe_recovery_events.append({
                "event": "recovery_lineage_failure",
                "step": int(self.step_count),
                "lineage_id": self._recovery_lineage_id,
                "reason": str(reason),
                "lineage_steps": int(self._recovery_lineage_steps),
                "incident_count": int(self._recovery_lineage_incident_count),
            })
            self._safety_mode = "SAFE_BRAKE"
            self._safe_brake_active = True
            self._safe_recovery_failed = True
            self._v5_release_after_brake = True
            self._safe_recovery_target = None
            self._last_recovery_reason = str(reason)
            return
        if self.recovery_attributable_mission_failure:
            self.recovery_duplicate_failure_record_count += 1
            return
        self.recovery_lineage_failure_count += 1
        self.recovery_attributable_mission_failure = True
        self.recovery_attributable_failure_reason = str(reason)
        self.safe_recovery_episode_failure = True
        self.safe_recovery_events.append({
            "event": "recovery_lineage_failure",
            "step": int(self.step_count),
            "lineage_id": self._recovery_lineage_id,
            "reason": str(reason),
            "lineage_steps": int(self._recovery_lineage_steps),
            "incident_count": int(self._recovery_lineage_incident_count),
        })

    def _attach_v5_3_lineage(self, hazard_keys: Sequence[str]) -> None:
        """Attach a new incident to a rapid lineage or create a new lineage."""

        candidates = []
        for key in hazard_keys:
            record = self._closed_recovery_lineages.get(str(key))
            if record is None:
                continue
            gap = int(self.step_count) - int(record["closed_step"])
            if 0 <= gap <= self.safe_recovery_rapid_retrigger_window_steps:
                candidates.append((int(record["closed_step"]), record))
        if candidates:
            _, record = max(candidates, key=lambda item: item[0])
            self._recovery_lineage_id = int(record["lineage_id"])
            self._recovery_lineage_steps = int(record["steps"])
            self._recovery_lineage_incident_count = int(record["incident_count"]) + 1
            self._recovery_lineage_exhaustion_count = int(record["exhaustions"])
            self._recovery_lineage_rapid_retriggers = int(record["rapid"]) + 1
            self._recovery_lineage_failure_emitted = bool(
                record.get("failed", False)
            )
            self.rapid_same_hazard_retrigger_count += 1
            self.max_rapid_retriggers_per_lineage = max(
                self.max_rapid_retriggers_per_lineage,
                self._recovery_lineage_rapid_retriggers,
            )
        else:
            self._recovery_lineage_counter += 1
            self._recovery_lineage_id = self._recovery_lineage_counter
            self._recovery_lineage_steps = 0
            self._recovery_lineage_incident_count = 1
            self._recovery_lineage_exhaustion_count = 0
            self._recovery_lineage_rapid_retriggers = 0
            self._recovery_lineage_failure_emitted = False
        self._recovery_lineage_hazard_keys = tuple(sorted(set(hazard_keys)))
        self.recovery_lineage_terminal_state = "ACTIVE"
        if self._recovery_lineage_incident_count > self.safe_recovery_lineage_max_incidents:
            self._mark_v5_3_lineage_failure("lineage_incident_limit_exceeded")

    def _emit_v5_3_bounded_exhaustion(self, reason: str) -> None:
        """Release after one bounded exhaustion without redefining it as failure."""

        if self._recovery_incident_hard_abort_emitted:
            self.recovery_duplicate_hard_abort_count += 1
            return
        self.safe_recovery_failure_count += 1
        self.safe_recovery_abort_count += 1
        self.safe_recovery_hard_abort_count += 1  # historical compatibility only
        self.bounded_recovery_exhaustion_count += 1
        self._recovery_lineage_exhaustion_count += 1
        self._recovery_incident_hard_abort_emitted = True
        self._recovery_incident_block_reentry_until_step = int(self.step_count)
        if self._recovery_lineage_exhaustion_count >= 2:
            self._mark_v5_3_lineage_failure("second_lineage_budget_exhaustion")
        self.safe_recovery_events.append({
            "event": "bounded_recovery_exhaustion",
            "step": int(self.step_count),
            "lineage_id": self._recovery_lineage_id,
            "incident_id": self._recovery_incident_id,
            "reason": str(reason),
            "incident_steps": int(self._recovery_incident_sequence_steps),
            "lineage_steps": int(self._recovery_lineage_steps),
        })
        self._safety_mode = "SAFE_BRAKE"
        self._safe_brake_active = True
        self._safe_recovery_failed = True
        self._v5_release_after_brake = True
        self._safe_recovery_target = None
        self._last_recovery_reason = str(reason)

    def _close_v5_3_lineage_incident(self, reason: str) -> None:
        keys = tuple(self._recovery_incident_hazard_keys)
        lineage = {
            "lineage_id": int(self._recovery_lineage_id or 0),
            "closed_step": int(self.step_count),
            "steps": int(self._recovery_lineage_steps),
            "incident_count": int(self._recovery_lineage_incident_count),
            "exhaustions": int(self._recovery_lineage_exhaustion_count),
            "rapid": int(self._recovery_lineage_rapid_retriggers),
            "failed": bool(self._recovery_lineage_failure_emitted),
        }
        self._close_v5_2_recovery_incident(reason)
        for key in keys:
            self._closed_recovery_lineages[str(key)] = dict(lineage)
        if self._recovery_lineage_failure_emitted:
            self._failed_recovery_hazard_keys.update(str(key) for key in keys)
            self.recovery_lineage_terminal_state = "FAILED_LATCHED"
        else:
            self.recovery_lineage_terminal_state = "CLOSED"
        self.recovery_phase = "IDLE"

    def _reconcile_v5_3_terminal_incident(self, episode_reason: str) -> None:
        if self._recovery_incident_id is None:
            return
        stored = set(self._recovery_incident_hazard_keys)
        active = self._currently_active_recovery_hazard_keys()
        if stored and stored.intersection(active):
            self._mark_v5_3_lineage_failure("terminal_active_hazard")
            return
        self.terminal_incident_reconciled_count += 1
        self._release_v5_2_recovery_incident(
            "resolved_by_goal_completion"
            if str(episode_reason) == "team_success"
            else "closed_as_stale_hazard"
        )

    def _reconcile_closure_terminal_incident(self, episode_reason: str) -> None:
        if self._recovery_incident_id is None:
            return
        stored = set(self._recovery_incident_hazard_keys)
        active = self._currently_active_recovery_hazard_keys()
        same_active = bool(stored.intersection(active))
        final_feasible = bool(
            self.release_shadow_check_passed
            or self._v5_4_release_shadow_passes(
                deterministic_hold_actions(
                    self.velocities,
                    max_acceleration=self.agent_max_acceleration,
                    dt=self.dt,
                )
            )
        )
        if self._closure_tracker.terminal_certificate(
            team_success=str(episode_reason) == "team_success",
            same_hazard_active=same_active,
            final_stack_feasible=final_feasible,
        ):
            self.release_certificate_id += 1
            self.release_certificate_count += 1
            self.terminal_incident_reconciled_count += 1
            self._release_v5_2_recovery_incident(
                "verified_terminal_goal_certificate"
            )
            return
        self._mark_v5_3_lineage_failure("terminal_unresolved")

    def _clear_v5_1_recovery_incident(self) -> None:
        """Clear the active incident without reopening same-step re-entry."""
        self._recovery_incident_id = None
        self._recovery_incident_elapsed_steps = 0
        self._recovery_incident_sequence_steps = 0
        self._recovery_incident_refresh_count = 0
        self._recovery_incident_sources = ()
        self._recovery_incident_hazard_keys = ()
        self._recovery_incident_hard_abort_emitted = False
        self._recovery_incident_start_step = None
        self._recovery_direction_unavailable_streak = 0
        self._recovery_direction_unavailable_last_step = -1

    def _enter_v5_1_recovery_incident(
        self,
        reason: str,
        *,
        offending_pairs: Sequence[Sequence[int]],
        affected_agent_mask: Sequence[bool] | None,
        entry_sources: Sequence[str],
        hazard_keys: Sequence[str],
    ) -> None:
        """Create or merge one V5.1 incident without resetting its budget."""
        pairs = tuple(sorted({
            (min(int(left), int(right)), max(int(left), int(right)))
            for left, right in offending_pairs
        }))
        if affected_agent_mask is None:
            mask = np.zeros(self.num_agents, dtype=bool)
            for left, right in pairs:
                mask[left] = True
                mask[right] = True
        else:
            mask = np.asarray(affected_agent_mask, dtype=bool).copy()
        if mask.shape != (self.num_agents,):
            raise ValueError("STAR recovery affected mask has the wrong shape")
        raw_mask = mask.copy()
        pair_mask = np.zeros(self.num_agents, dtype=bool)
        for left, right in pairs:
            pair_mask[left] = True
            pair_mask[right] = True
        if entry_sources:
            sources = tuple(sorted({str(value) for value in entry_sources}))
        elif pairs:
            sources = ("pair",)
        elif str(reason).startswith("obstacle"):
            sources = ("obstacle_guard_fallback",)
        elif (
            self.star_safety_state_machine_version
            in _INCIDENT_RECOVERY_VERSIONS
            and not str(reason).startswith("boundary")
        ):
            sources = ("generic",)
        else:
            sources = ("boundary",)
        key_values = {str(value) for value in hazard_keys}
        if self.star_safety_state_machine_version in {
            "star_incident_recovery_v5_2", "star_persistent_recovery_v5_3",
            "star_persistent_recovery_v5_4", "star_closure_consistency_audit_v1",
        }:
            key_values.update(
                f"pair:{left}:{right}" for left, right in pairs
            )
            if "boundary" in sources:
                key_values.update(
                    f"boundary:{agent_id}"
                    for agent_id in np.flatnonzero(raw_mask)
                )
        keys = tuple(sorted(key_values))
        if self.safe_recovery_exclude_goal_agents:
            arrived = self._goal_distances() < self.goal_radius
            safety_endpoint_mask = pair_mask.copy()
            if self.star_safety_state_machine_version in {
                "star_incident_recovery_v5_2", "star_persistent_recovery_v5_3",
                "star_persistent_recovery_v5_4", "star_closure_consistency_audit_v1",
            }:
                for agent_id in self._hazard_agent_ids(keys):
                    if 0 <= agent_id < self.num_agents:
                        safety_endpoint_mask[agent_id] = True
                if "boundary" in sources:
                    low = (
                        self.agent_radius
                        + self.boundary_guard.surface_clearance_m
                        + self.boundary_guard.invariant_buffer_m
                    )
                    high = self.map_size - low
                    positions = self.positions
                    boundary_mask = np.any(
                        (positions <= low + self.safe_recovery_boundary_trigger_margin)
                        | (positions >= high - self.safe_recovery_boundary_trigger_margin),
                        axis=1,
                    )
                    safety_endpoint_mask |= raw_mask & boundary_mask
            mask &= (~arrived | safety_endpoint_mask)

        active = self._recovery_incident_id is not None
        if active:
            previous_mask = self._star_recovery_affected_mask.copy()
            previous_pairs = set(self._star_recovery_offending_pairs)
            previous_sources = set(self._recovery_incident_sources)
            previous_keys = set(self._recovery_incident_hazard_keys)
            self._star_recovery_affected_mask |= mask
            self._star_recovery_offending_pairs = tuple(sorted(
                previous_pairs | set(pairs)
            ))
            self._recovery_incident_sources = tuple(sorted(
                previous_sources | set(sources)
            ))
            self._safe_recovery_entry_sources = self._recovery_incident_sources
            self._recovery_incident_hazard_keys = tuple(sorted(
                previous_keys | set(keys)
            ))
            geometry_changed = bool(
                np.any(previous_mask != self._star_recovery_affected_mask)
                or previous_pairs != set(self._star_recovery_offending_pairs)
                or previous_sources != set(self._recovery_incident_sources)
                or previous_keys != set(self._recovery_incident_hazard_keys)
            )
            if geometry_changed:
                self._safe_recovery_target = None
            self.recovery_incident_merge_count += 1
            self.safe_recovery_events.append({
                "event": "incident_merge",
                "step": int(self.step_count),
                "incident_id": int(self._recovery_incident_id),
                "reason": str(reason),
                "entry_sources": list(self._recovery_incident_sources),
                "hazard_keys": list(self._recovery_incident_hazard_keys),
                "elapsed_steps": int(self._recovery_incident_sequence_steps),
                "affected_agent_mask": self._star_recovery_affected_mask.tolist(),
            })
        else:
            # V5.1 creates incident IDs only from NORMAL.  A hard-abort brake
            # cannot recursively open another incident in its release step.
            if (
                self._safety_mode != "NORMAL"
                or int(self.step_count)
                <= self._recovery_incident_block_reentry_until_step
            ):
                return
            if (
                self.star_safety_state_machine_version in {
                    "star_persistent_recovery_v5_4",
                    "star_closure_consistency_audit_v1",
                }
                and set(keys).intersection(self._failed_recovery_hazard_keys)
            ):
                # A failed lineage is terminal for this episode.  Do not
                # create a fresh incident ID that resets the 40-step cap.
                self._safety_mode = "SAFE_BRAKE"
                self._safe_brake_active = True
                self._safe_recovery_failed = True
                self._last_recovery_reason = "failed_lineage_reentry_blocked"
                return
            self._recovery_incident_counter += 1
            self._recovery_incident_id = self._recovery_incident_counter
            self.recovery_incident_count += 1
            if (
                self.star_safety_state_machine_version
                == "star_closure_consistency_audit_v1"
            ):
                if (
                    set(keys).intersection(
                        self._closure_last_verified_hazard_keys
                    )
                    and 0 <= int(self.step_count)
                    - int(self._closure_last_verified_step)
                    <= self.safe_recovery_rapid_retrigger_window_steps
                ):
                    self.post_certificate_retrigger_count += 1
                self._closure_tracker.reset()
                self.closure_state = CLOSURE_ACTIVE
            if (
                self.star_safety_state_machine_version
                == "star_incident_recovery_v5_2"
            ):
                reopened = set(keys).intersection(
                    self._closed_recovery_hazard_steps
                )
                if reopened:
                    self.recovery_incident_reopen_count += 1
                    self.hazard_retrigger_count += len(reopened)
                    self.same_hazard_reentry_count += len(reopened)
            elif self.star_safety_state_machine_version in {
                "star_persistent_recovery_v5_3",
                "star_persistent_recovery_v5_4",
                "star_closure_consistency_audit_v1",
            }:
                self._attach_v5_3_lineage(keys)
            self._recovery_incident_elapsed_steps = 0
            self._recovery_incident_sequence_steps = 0
            self._recovery_incident_refresh_count = 0
            self._recovery_incident_sources = sources
            self._recovery_incident_hazard_keys = keys
            self._recovery_incident_hard_abort_emitted = False
            self._recovery_incident_start_step = int(self.step_count)
            self._recovery_direction_unavailable_streak = 0
            self._recovery_direction_unavailable_last_step = -1
            self._safe_recovery_elapsed_steps = 0
            self._safe_recovery_sequence_steps = 0
            self._safe_recovery_attempt_index = 0
            self._safe_recovery_failed = False
            self._safe_recovery_target = None
            self._safe_recovery_start_positions = None
            self._safe_recovery_previous_margin = None
            self._safe_recovery_no_improvement_steps = 0
            self._star_recovery_offending_pairs = pairs
            self._star_recovery_affected_mask = mask
            self._safe_recovery_entry_sources = sources
            if self.star_safety_state_machine_version in {
                "star_persistent_recovery_v5_3",
                "star_persistent_recovery_v5_4",
                "star_closure_consistency_audit_v1",
            }:
                self.recovery_phase = "ESCAPE"
            self._safety_mode = "SAFE_BRAKE"
            self._safe_brake_active = True
            self._recovery_stable_steps = 0
            self.safe_brake_entry_count += 1
            self.safe_brake_entry_steps.append(int(self.step_count))
            self.safe_brake_entry_reasons.append(str(reason))
            self.safe_recovery_events.append({
                "event": "incident_entry",
                "step": int(self.step_count),
                "incident_id": int(self._recovery_incident_id),
                "reason": str(reason),
                "entry_sources": list(sources),
                "hazard_keys": list(keys),
                "affected_agent_mask": mask.tolist(),
            })
        self.safe_recovery_affected_agent_count = max(
            self.safe_recovery_affected_agent_count,
            int(np.count_nonzero(self._star_recovery_affected_mask)),
        )
        self._pending_recovery_pair_infeasible = False
        self._pending_recovery_invariant_violations = 0
        self._last_recovery_reason = str(reason)

    def _enter_star_active_recovery_brake(
        self,
        reason: str,
        *,
        offending_pairs: Sequence[Sequence[int]] = (),
        affected_agent_mask: Sequence[bool] | None = None,
        entry_sources: Sequence[str] = (),
        hazard_keys: Sequence[str] = (),
    ) -> None:
        """Enter STAR brake without losing the geometry needed for recovery."""
        if self.star_safety_state_machine_version in _INCIDENT_RECOVERY_VERSIONS:
            self._enter_v5_1_recovery_incident(
                reason,
                offending_pairs=offending_pairs,
                affected_agent_mask=affected_agent_mask,
                entry_sources=entry_sources,
                hazard_keys=hazard_keys,
            )
            return
        new_sequence = not self._safe_brake_active and self._safety_mode != "SAFE_RECOVER"
        if new_sequence:
            self.safe_brake_entry_count += 1
            self.safe_brake_entry_steps.append(int(self.step_count))
            self.safe_brake_entry_reasons.append(str(reason))
        self._safety_mode = "SAFE_BRAKE"
        self._safe_brake_active = True
        self._recovery_stable_steps = 0
        self._safe_recovery_elapsed_steps = 0
        self._safe_recovery_sequence_steps = 0
        self._safe_recovery_attempt_index = 0
        self._safe_recovery_failed = False
        self._safe_recovery_target = None
        self._safe_recovery_start_positions = None
        self._star_recovery_offending_pairs = tuple(
            sorted(
                (min(int(left), int(right)), max(int(left), int(right)))
                for left, right in offending_pairs
            )
        )
        if affected_agent_mask is None:
            mask = np.zeros(self.num_agents, dtype=bool)
            for left, right in self._star_recovery_offending_pairs:
                mask[left] = True
                mask[right] = True
        else:
            mask = np.asarray(affected_agent_mask, dtype=bool).copy()
        if mask.shape != (self.num_agents,):
            raise ValueError("STAR recovery affected mask has the wrong shape")
        if (
            self.star_safety_state_machine_version in {
                "star_scoped_recovery_v4", "star_adaptive_recovery_v5",
                "star_incident_recovery_v5_1", "star_incident_recovery_v5_2",
                "star_persistent_recovery_v5_3",
                "star_persistent_recovery_v5_4",
                "star_closure_consistency_audit_v1",
            }
            and self.safe_recovery_exclude_goal_agents
        ):
            pair_mask = np.zeros(self.num_agents, dtype=bool)
            for left, right in self._star_recovery_offending_pairs:
                pair_mask[left] = True
                pair_mask[right] = True
            arrived = self._goal_distances() < self.goal_radius
            mask &= (~arrived | pair_mask)
        self._star_recovery_affected_mask = mask
        if entry_sources:
            sources = tuple(sorted({str(value) for value in entry_sources}))
        elif self._star_recovery_offending_pairs:
            sources = ("pair",)
        elif str(reason).startswith("obstacle"):
            sources = ("obstacle_guard_fallback",)
        else:
            sources = ("boundary",)
        if self.star_safety_state_machine_version == "star_adaptive_recovery_v5":
            if new_sequence and self.safe_recover_exit_count > 0:
                self.safe_recovery_reentry_count += 1
            self._safe_recovery_entry_sources = sources
            self._safe_recovery_previous_margin = None
            self._safe_recovery_no_improvement_steps = 0
            self.safe_recovery_affected_agent_count = max(
                self.safe_recovery_affected_agent_count,
                int(np.count_nonzero(mask)),
            )
        self._pending_recovery_pair_infeasible = False
        self._pending_recovery_invariant_violations = 0
        self._last_recovery_reason = str(reason)

    def _unprojected_joint_guard_result(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        actions: np.ndarray,
    ) -> JointGuardResult:
        """Return neutral stage plumbing without invoking a Guard projection."""

        positions = np.asarray(positions, dtype=np.float64)
        velocities = np.asarray(velocities, dtype=np.float64)
        requested = np.asarray(actions, dtype=np.float64)
        executed = np.clip(requested, -1.0, 1.0)
        projected = np.clip(
            velocities + executed * self.agent_max_acceleration * self.dt,
            -self.agent_max_speed,
            self.agent_max_speed,
        )
        next_positions = positions + projected * self.dt
        clearances = np.stack([
            self.boundary_guard.surface_clearances(
                position, self.map_size, self.agent_radius
            )
            for position in positions
        ])
        pair_minimum = self.boundary_guard._pair_minimum(
            positions, next_positions
        )
        pair_required = (
            float(self.oracle_config["teammate_safety_distance"])
            + self.boundary_guard.pair_invariant_buffer_m
        )
        return JointGuardResult(
            requested_actions=requested.copy(),
            executed_actions=executed.copy(),
            projected_next_velocities=projected,
            surface_clearances=clearances,
            shielded_axes=tuple((False, False) for _ in range(self.num_agents)),
            reasons=tuple((None, None) for _ in range(self.num_agents)),
            invariant_violations=0,
            safe_brake=False,
            pair_iterations=0,
            requested_pair_min_distance=float(pair_minimum),
            executed_pair_min_distance=float(pair_minimum),
            minimum_pair_margin=float(pair_minimum - pair_required),
            pair_invariant_violations=0,
            pair_infeasible=False,
            recovery_required=False,
            offending_pairs=(),
            affected_agent_mask=tuple(False for _ in range(self.num_agents)),
            reference_safe_actions=executed.copy(),
            reference_safe_action_available=True,
            reference_safe_action_norm=float(np.linalg.norm(executed)),
        )

    def _ttc_candidate_shadow(
        self,
        requested_actions: np.ndarray,
        *,
        actor_reference_actions: np.ndarray,
        agent_id: int,
        horizon_steps: int = 3,
        visible_obstacles: Sequence[Sequence[ObservableObstacleV1]] | None = None,
    ) -> dict[str, Any]:
        """Evaluate a candidate with causal obstacle/guard/residual shadows.

        This method is read-only.  It uses the current observable track state,
        constant-velocity causal propagation and the existing joint guard.  It
        never steps simulator obstacles or reads their identity/true velocity.
        """

        if int(horizon_steps) != 3:
            raise ValueError("TTC headroom audit is frozen to three shadow steps")
        values = np.asarray(requested_actions, dtype=np.float32).copy()
        actor = np.asarray(actor_reference_actions, dtype=np.float32)
        if values.shape != (self.num_agents, 2) or actor.shape != values.shape:
            raise ValueError("TTC shadow actions must be matching Nx2 matrices")
        if not 0 <= int(agent_id) < self.num_agents:
            raise ValueError("TTC shadow agent id is outside the team")

        positions = self.positions.astype(np.float64).copy()
        velocities = self.velocities.astype(np.float64).copy()
        visible = tuple(tuple(row) for row in (
            self._safety_planning_snapshot().visible_obstacles
            if visible_obstacles is None else visible_obstacles
        ))
        minimum_margin = float("inf")
        first_executed = values.copy()
        rejection_reason = "accepted"
        safe = True
        protected = float(self.oracle_config["teammate_safety_distance"])
        low = (
            self.agent_radius + self.boundary_guard.surface_clearance_m
            + self.boundary_guard.invariant_buffer_m
        )
        high = self.map_size.astype(np.float64) - low

        for shadow_index in range(3):
            pre_risks = assess_obstacle_actions(
                actions=values,
                velocities=velocities,
                visible_obstacles=visible,
                agent_radius=self.agent_radius,
                obstacle_safety_margin=float(
                    self.oracle_config["obstacle_safety_margin"]
                ),
                invariant_buffer=self.boundary_guard.invariant_buffer_m,
                max_acceleration=self.agent_max_acceleration,
                dt=self.dt,
                horizon_seconds=self.obstacle_guard_horizon_seconds,
                warning_ttc_seconds=self.obstacle_guard_warning_ttc_seconds,
                emergency_ttc_seconds=self.obstacle_guard_emergency_ttc_seconds,
            )
            guard_kwargs = dict(
                map_size=self.map_size,
                agent_radius=self.agent_radius,
                dt=self.dt,
                max_acceleration=self.agent_max_acceleration,
                max_speed=self.agent_max_speed,
                safe_brake=False,
            )
            if isinstance(self.boundary_guard, JointSafetyGuardV2):
                guard_kwargs["teammate_safety_distance"] = protected
            guard_shadow_enabled = bool(
                self.initial_joint_guard_enabled
                or self.final_joint_guard_enabled
            )
            guarded = (
                self.boundary_guard.project_actions(
                    positions, velocities, values, **guard_kwargs
                )
                if guard_shadow_enabled
                else self._unprojected_joint_guard_result(
                    positions, velocities, values
                )
            )
            executed = guarded.executed_actions.astype(np.float32)
            residual = assess_obstacle_actions(
                actions=executed,
                velocities=velocities,
                visible_obstacles=visible,
                agent_radius=self.agent_radius,
                obstacle_safety_margin=float(
                    self.oracle_config["obstacle_safety_margin"]
                ),
                invariant_buffer=self.boundary_guard.invariant_buffer_m,
                max_acceleration=self.agent_max_acceleration,
                dt=self.dt,
                horizon_seconds=self.obstacle_guard_horizon_seconds,
                warning_ttc_seconds=self.obstacle_guard_warning_ttc_seconds,
                emergency_ttc_seconds=self.obstacle_guard_emergency_ttc_seconds,
            )
            if shadow_index == 0:
                first_executed = executed.copy()
            if pre_risks:
                safe = False
                rejection_reason = "obstacle_feasibility"
            if (
                guard_shadow_enabled
                and bool(getattr(guarded, "recovery_required", False))
            ):
                safe = False
                rejection_reason = "joint_guard_recovery_required"
            if (
                guard_shadow_enabled
                and int(getattr(guarded, "invariant_violations", 0)) != 0
            ):
                safe = False
                rejection_reason = "joint_guard_residual"
            if residual:
                safe = False
                rejection_reason = "obstacle_residual_recheck"

            obstacle_margin = minimum_predicted_obstacle_clearance_m(
                actions=executed,
                velocities=velocities,
                visible_obstacles=visible,
                agent_radius=self.agent_radius,
                obstacle_safety_margin=float(
                    self.oracle_config["obstacle_safety_margin"]
                ),
                invariant_buffer=self.boundary_guard.invariant_buffer_m,
                max_acceleration=self.agent_max_acceleration,
                dt=self.dt,
                horizon_seconds=self.obstacle_guard_horizon_seconds,
            )
            boundary_margin = float(min(
                np.min(positions - low), np.min(high - positions)
            ))
            pair_margin = min(
                (
                    float(np.linalg.norm(positions[left] - positions[right]))
                    - protected
                    for left in range(self.num_agents)
                    for right in range(left + 1, self.num_agents)
                ),
                default=float("inf"),
            )
            minimum_margin = min(
                minimum_margin, obstacle_margin, boundary_margin, pair_margin
            )

            previous_velocities = velocities.copy()
            velocities = np.clip(
                velocities
                + np.clip(executed, -1.0, 1.0)
                * self.agent_max_acceleration * self.dt,
                -self.agent_max_speed,
                self.agent_max_speed,
            )
            positions = positions + velocities * self.dt
            propagated = []
            for observer_id, tracks in enumerate(visible):
                next_tracks = []
                for item in tracks:
                    relative = np.asarray(
                        item.relative_position, dtype=np.float64
                    )
                    relative_velocity = np.asarray(
                        item.relative_velocity, dtype=np.float64
                    )
                    obstacle_velocity = (
                        relative_velocity + previous_velocities[observer_id]
                    )
                    next_relative_velocity = (
                        obstacle_velocity - velocities[observer_id]
                    )
                    next_relative = (
                        relative + next_relative_velocity * self.dt
                    )
                    next_tracks.append(replace(
                        item,
                        relative_position=tuple(
                            float(value) for value in next_relative
                        ),
                        relative_velocity=tuple(
                            float(value) for value in next_relative_velocity
                        ),
                    ))
                propagated.append(tuple(next_tracks))
            visible = tuple(propagated)
            values = executed

        goal_delta = self.goals[int(agent_id)] - self.positions[int(agent_id)]
        goal_norm = float(np.linalg.norm(goal_delta))
        predicted_velocity = np.clip(
            self.velocities[int(agent_id)]
            + first_executed[int(agent_id)]
            * self.agent_max_acceleration * self.dt,
            -self.agent_max_speed,
            self.agent_max_speed,
        )
        goal_velocity = (
            0.0 if goal_norm <= 1e-12 else
            float(np.dot(predicted_velocity, goal_delta / goal_norm))
        )
        finite_margin = (
            float(minimum_margin) if np.isfinite(minimum_margin) else 1e9
        )
        return {
            "safe": bool(safe),
            "rejection_reason": rejection_reason,
            "minimum_safety_margin_m": finite_margin,
            "goal_direction_velocity_mps": goal_velocity,
            "actor_deviation_norm": float(np.linalg.norm(
                first_executed[int(agent_id)] - actor[int(agent_id)]
            )),
            "first_executed_actions": first_executed,
        }

    def _audit_ttc_headroom_and_select_v2(
        self,
        *,
        actor_actions: np.ndarray,
        v1_actions: np.ndarray,
        affected_agent_mask: np.ndarray,
        visible_obstacles: Sequence[Sequence[ObservableObstacleV1]],
    ) -> tuple[dict[str, Any] | None, np.ndarray]:
        """Build one trace audit and, for V2, return its bounded selection."""

        affected_ids = tuple(
            int(value) for value in np.flatnonzero(affected_agent_mask)
        )
        if not affected_ids:
            return None, np.asarray(v1_actions, dtype=np.float32).copy()
        actor = np.asarray(actor_actions, dtype=np.float32)
        baseline_actions = np.asarray(v1_actions, dtype=np.float32).copy()
        v2_actions = baseline_actions.copy()
        all_evaluations: list[TTCActionCandidateEvaluationV1] = []
        baseline_labels: list[str] = []
        any_headroom = False
        best_gain = 0.0
        selected_high_progress_count = 0

        for agent_id in affected_ids:
            tracks = tuple(visible_obstacles[agent_id])
            risk_rows = tuple(
                (
                    np.asarray(item.relative_position, dtype=np.float64),
                    np.asarray(item.relative_velocity, dtype=np.float64),
                )
                for item in tracks if item.velocity_valid
            )
            registered_v1 = registered_ttc_actions_v1(
                raw_action=actor[agent_id],
                velocity=self.velocities[agent_id],
                risk_rows=risk_rows,
                max_acceleration=self.agent_max_acceleration,
                dt=self.dt,
            )
            baseline_label, _ = min(
                registered_v1,
                key=lambda item: (
                    float(np.linalg.norm(item[1] - baseline_actions[agent_id])),
                    item[0],
                ),
            )
            baseline_labels.append(baseline_label)
            baseline_trial = baseline_actions.copy()
            baseline_shadow = self._ttc_candidate_shadow(
                baseline_trial,
                actor_reference_actions=actor,
                agent_id=agent_id,
                horizon_steps=self.ttc_headroom_shadow_steps,
                visible_obstacles=visible_obstacles,
            )
            baseline_eval = TTCActionCandidateEvaluationV1(
                label=f"v1_selected:{baseline_label}",
                agent_id=agent_id,
                action=tuple(float(value) for value in baseline_actions[agent_id]),
                safe_three_step=bool(baseline_shadow["safe"]),
                minimum_safety_margin_m=float(
                    baseline_shadow["minimum_safety_margin_m"]
                ),
                goal_direction_velocity_mps=float(
                    baseline_shadow["goal_direction_velocity_mps"]
                ),
                actor_deviation_norm=float(
                    baseline_shadow["actor_deviation_norm"]
                ),
                rejection_reason=str(baseline_shadow["rejection_reason"]),
            )
            all_evaluations.append(baseline_eval)
            agent_evaluations: list[TTCActionCandidateEvaluationV1] = []
            evaluated_labels: set[str] = set()

            def evaluate_registered(
                label: str, candidate: np.ndarray,
            ) -> TTCActionCandidateEvaluationV1:
                trial = baseline_actions.copy()
                trial[agent_id] = candidate
                shadow = self._ttc_candidate_shadow(
                    trial,
                    actor_reference_actions=actor,
                    agent_id=agent_id,
                    horizon_steps=self.ttc_headroom_shadow_steps,
                    visible_obstacles=visible_obstacles,
                )
                evaluated = TTCActionCandidateEvaluationV1(
                    label=label,
                    agent_id=agent_id,
                    action=tuple(float(value) for value in candidate),
                    safe_three_step=bool(shadow["safe"]),
                    minimum_safety_margin_m=float(
                        shadow["minimum_safety_margin_m"]
                    ),
                    goal_direction_velocity_mps=float(
                        shadow["goal_direction_velocity_mps"]
                    ),
                    actor_deviation_norm=float(shadow["actor_deviation_norm"]),
                    rejection_reason=str(shadow["rejection_reason"]),
                )
                return evaluated

            preliminary = registered_ttc_headroom_actions(
                raw_action=actor[agent_id],
                velocity=self.velocities[agent_id],
                goal_delta=self.goals[agent_id] - self.positions[agent_id],
                risk_rows=risk_rows,
                max_acceleration=self.agent_max_acceleration,
                dt=self.dt,
                safe_v1_labels=(),
            )
            for label, candidate in preliminary:
                evaluated = evaluate_registered(label, candidate)
                agent_evaluations.append(evaluated)
                all_evaluations.append(evaluated)
                evaluated_labels.add(label)
            safe_v1_labels = tuple(
                value.label for value in agent_evaluations
                if value.safe_three_step
                and value.label in {label for label, _ in registered_v1}
            )
            full_registered = registered_ttc_headroom_actions(
                raw_action=actor[agent_id],
                velocity=self.velocities[agent_id],
                goal_delta=self.goals[agent_id] - self.positions[agent_id],
                risk_rows=risk_rows,
                max_acceleration=self.agent_max_acceleration,
                dt=self.dt,
                safe_v1_labels=safe_v1_labels,
            )
            for label, candidate in full_registered:
                if label in evaluated_labels:
                    continue
                evaluated = evaluate_registered(label, candidate)
                agent_evaluations.append(evaluated)
                all_evaluations.append(evaluated)
                evaluated_labels.add(label)
            for evaluated in agent_evaluations:
                if is_safe_progress_headroom(
                    evaluated,
                    baseline_eval,
                    minimum_goal_velocity_gain_mps=(
                        self.ttc_headroom_minimum_goal_velocity_gain
                    ),
                    maximum_extra_actor_deviation=(
                        self.ttc_headroom_maximum_extra_actor_deviation
                    ),
                ):
                    any_headroom = True
                    best_gain = max(
                        best_gain,
                        evaluated.goal_direction_velocity_mps
                        - baseline_eval.goal_direction_velocity_mps,
                    )
            if self.obstacle_execution_guard_version == (
                "causal_ttc_progress_projection_v2"
            ):
                selected = select_ttc_progress_candidate_v2(agent_evaluations)
                if selected is not None:
                    v2_actions[agent_id] = np.asarray(
                        selected.action, dtype=np.float32
                    )
                    if (
                        selected.goal_direction_velocity_mps
                        >= baseline_eval.goal_direction_velocity_mps + 0.10
                    ):
                        selected_high_progress_count += 1

        audit = TTCActionCandidateAuditV1(
            episode_step=int(self.step_count),
            affected_agent_ids=affected_ids,
            v1_selected_labels=tuple(baseline_labels),
            evaluations=tuple(all_evaluations),
            safe_progress_headroom=bool(any_headroom),
            v1_choice_pareto_dominated=bool(any_headroom),
            best_goal_velocity_gain_mps=float(best_gain),
        ).to_dict()
        audit["v2_selected_high_progress_count"] = int(
            selected_high_progress_count
        )
        return audit, v2_actions

    def _v5_4_observable_envelope(
        self,
        requested_actions: np.ndarray,
        *,
        horizon_steps: int,
        active_hazard_keys: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Project a short action envelope without simulator-future leakage."""

        if horizon_steps <= 0:
            raise ValueError("V5.4 shadow horizon must be positive")
        positions = self.positions.astype(np.float64).copy()
        velocities = self.velocities.astype(np.float64).copy()
        actions = np.asarray(requested_actions, dtype=np.float32).copy()
        if actions.shape != (self.num_agents, 2):
            raise ValueError("V5.4 shadow actions must be Nx2")
        snapshot = self._safety_planning_snapshot()
        visible = tuple(tuple(row) for row in snapshot.visible_obstacles)
        active_keys = {str(value) for value in active_hazard_keys}
        affected = self._star_recovery_affected_mask.copy()
        margins: list[float] = []
        minimum_ttc = float("inf")
        first_executed = np.zeros_like(actions)
        feasible = True
        rejection_reason = "ok"
        stage_trace: list[dict[str, Any]] = []
        position_path: list[np.ndarray] = []

        for horizon_index in range(int(horizon_steps)):
            values = actions.copy()
            if self.goal_boundary_corridor_enabled:
                values = project_goal_boundary_corridor(
                    actions=values,
                    positions=positions,
                    velocities=velocities,
                    goals=self.goals,
                    map_size=self.map_size,
                    agent_radius=self.agent_radius,
                    goal_radius=self.goal_radius,
                    max_acceleration=self.agent_max_acceleration,
                    dt=self.dt,
                    corridor_coordinate=self.goal_boundary_corridor_coordinate,
                    boundary_axis=0,
                    boundary_side="high",
                ).actions
            guard_kwargs = dict(
                map_size=self.map_size,
                agent_radius=self.agent_radius,
                dt=self.dt,
                max_acceleration=self.agent_max_acceleration,
                max_speed=self.agent_max_speed,
                safe_brake=False,
            )
            if isinstance(self.boundary_guard, JointSafetyGuardV2):
                guard_kwargs["teammate_safety_distance"] = float(
                    self.oracle_config["teammate_safety_distance"]
                )
            initial = self.boundary_guard.project_actions(
                positions, velocities, values, **guard_kwargs
            )
            obstacle = project_obstacle_actions(
                actions=initial.executed_actions,
                velocities=velocities,
                visible_obstacles=visible,
                agent_radius=self.agent_radius,
                obstacle_safety_margin=float(
                    self.oracle_config["obstacle_safety_margin"]
                ),
                invariant_buffer=self.boundary_guard.invariant_buffer_m,
                max_acceleration=self.agent_max_acceleration,
                dt=self.dt,
                horizon_seconds=self.obstacle_guard_horizon_seconds,
                warning_ttc_seconds=self.obstacle_guard_warning_ttc_seconds,
                emergency_ttc_seconds=self.obstacle_guard_emergency_ttc_seconds,
            )
            final = self.boundary_guard.project_actions(
                positions, velocities, obstacle.actions, **guard_kwargs
            )
            values = final.executed_actions.astype(np.float32)
            remaining = assess_obstacle_actions(
                actions=values,
                velocities=velocities,
                visible_obstacles=visible,
                agent_radius=self.agent_radius,
                obstacle_safety_margin=float(
                    self.oracle_config["obstacle_safety_margin"]
                ),
                invariant_buffer=self.boundary_guard.invariant_buffer_m,
                max_acceleration=self.agent_max_acceleration,
                dt=self.dt,
                horizon_seconds=self.obstacle_guard_horizon_seconds,
                warning_ttc_seconds=self.obstacle_guard_warning_ttc_seconds,
                emergency_ttc_seconds=self.obstacle_guard_emergency_ttc_seconds,
            )
            same_hazard = any(
                f"agent:{item.agent_id}:track:{item.track_id}" in active_keys
                for item in remaining
            )
            if (
                bool(initial.recovery_required)
                or bool(final.recovery_required)
                or int(final.invariant_violations) != 0
                or bool(obstacle.fallback_required)
                or same_hazard
            ):
                feasible = False
                rejection_reason = (
                    "same_hazard_remaining" if same_hazard
                    else "post_projection_infeasible"
                )
            if horizon_index == 0:
                first_executed = values.copy()
            stage_trace.append({
                "shadow_step": int(horizon_index),
                "requested_actions": actions.astype(float).tolist(),
                "initial_guard_actions": np.asarray(
                    initial.executed_actions
                ).astype(float).tolist(),
                "ttc_actions": np.asarray(obstacle.actions).astype(float).tolist(),
                "final_guard_actions": values.astype(float).tolist(),
                "initial_recovery_required": bool(initial.recovery_required),
                "final_recovery_required": bool(final.recovery_required),
                "final_invariant_violations": int(final.invariant_violations),
                "obstacle_fallback_required": bool(obstacle.fallback_required),
                "obstacle_fallback_reason": str(obstacle.reason),
                "obstacle_residual_count": int(len(remaining)),
            })

            previous_velocities = velocities.copy()
            velocities = np.clip(
                velocities
                + np.clip(values, -1.0, 1.0)
                * self.agent_max_acceleration * self.dt,
                -self.agent_max_speed,
                self.agent_max_speed,
            )
            positions = positions + velocities * self.dt

            position_path.append(positions.copy())

            propagated = []
            for agent_id, tracks in enumerate(visible):
                next_tracks = []
                for item in tracks:
                    relative = np.asarray(
                        item.relative_position, dtype=np.float64
                    )
                    relative_velocity = np.asarray(
                        item.relative_velocity, dtype=np.float64
                    )
                    obstacle_velocity = (
                        relative_velocity + previous_velocities[agent_id]
                    )
                    next_relative_velocity = (
                        obstacle_velocity - velocities[agent_id]
                    )
                    next_relative = relative + next_relative_velocity * self.dt
                    next_tracks.append(replace(
                        item,
                        relative_position=tuple(float(v) for v in next_relative),
                        relative_velocity=tuple(
                            float(v) for v in next_relative_velocity
                        ),
                    ))
                propagated.append(tuple(next_tracks))
            visible = tuple(propagated)

            step_margins: list[float] = []
            if "boundary" in self._safe_recovery_entry_sources and np.any(affected):
                low = (
                    self.agent_radius + self.boundary_guard.surface_clearance_m
                    + self.boundary_guard.invariant_buffer_m
                )
                high = self.map_size - low
                step_margins.append(float(min(
                    np.min(positions[affected] - low),
                    np.min(high - positions[affected]),
                )))
            if "pair" in self._safe_recovery_entry_sources:
                protected = float(
                    self.oracle_config["teammate_safety_distance"]
                )
                pair_margins = [
                    float(np.linalg.norm(positions[left] - positions[right]))
                    - protected
                    for left, right in self._star_recovery_offending_pairs
                ]
                if pair_margins:
                    step_margins.append(min(pair_margins))
            if "obstacle_guard_fallback" in self._safe_recovery_entry_sources:
                obstacle_margins = []
                for agent_id in np.flatnonzero(affected):
                    for item in visible[int(agent_id)]:
                        key = f"agent:{int(agent_id)}:track:{item.track_id}"
                        if active_keys and key not in active_keys:
                            continue
                        relative = np.asarray(
                            item.relative_position, dtype=np.float64
                        )
                        relative_velocity = np.asarray(
                            item.relative_velocity, dtype=np.float64
                        )
                        safe_distance = (
                            self.agent_radius + item.radius
                            + float(self.oracle_config["obstacle_safety_margin"])
                            + self.boundary_guard.invariant_buffer_m
                        )
                        obstacle_margins.append(
                            float(np.linalg.norm(relative)) - safe_distance
                        )
                        speed_squared = float(
                            np.dot(relative_velocity, relative_velocity)
                        )
                        if speed_squared > 1e-10:
                            t_star = float(np.clip(
                                -np.dot(relative, relative_velocity)
                                / speed_squared,
                                0.0,
                                self.obstacle_guard_horizon_seconds,
                            ))
                            closest = relative + t_star * relative_velocity
                            if (
                                float(np.dot(relative, relative_velocity)) < 0.0
                                and float(np.linalg.norm(closest)) < safe_distance
                            ):
                                minimum_ttc = min(minimum_ttc, t_star)
                if obstacle_margins:
                    step_margins.append(min(obstacle_margins))
            margins.append(min(step_margins) if step_margins else float("inf"))

        finite_margins = [value for value in margins if np.isfinite(value)]
        return {
            "feasible": bool(feasible),
            "rejection_reason": rejection_reason,
            "first_executed_actions": first_executed,
            "post_projection_norm": (
                float(np.linalg.norm(first_executed[affected]))
                if np.any(affected) else 0.0
            ),
            "positions": positions,
            "margins": margins,
            "minimum_margin": (
                min(finite_margins) if finite_margins else float("inf")
            ),
            "final_margin": (
                finite_margins[-1] if finite_margins else float("inf")
            ),
            "minimum_ttc": minimum_ttc,
            "stage_trace": stage_trace,
            "position_path": position_path,
        }

    def evaluate_shadow_action_v2(
        self,
        snapshot: CausalSafetySnapshotV1,
        requested_actions: np.ndarray,
    ) -> ShadowSafetyResultV2:
        """Run an immutable shadow and retain every UAV's stage cost."""

        candidate = np.asarray(requested_actions, dtype=np.float32)
        if candidate.shape != (self.num_agents, 2):
            raise ValueError("shadow action must have shape [num_agents,2]")
        original_request = np.asarray(
            snapshot.requested_actions, dtype=np.float32
        )
        current = self._causal_safety_snapshot(original_request)
        if current.digest != snapshot.digest:
            raise ValueError("causal safety snapshot does not match current state")
        before = self.export_counterfactual_snapshot(actor_gru_state=None)
        envelope = self._v5_4_observable_envelope(candidate, horizon_steps=1)
        after = self.export_counterfactual_snapshot(actor_gru_state=None)
        if after.state_fingerprint != before.state_fingerprint:
            raise RuntimeError("shadow safety evaluation mutated environment state")

        first = envelope["stage_trace"][0]
        stage_actions = (
            (candidate, np.asarray(first["initial_guard_actions"], dtype=np.float32)),
            (
                np.asarray(first["initial_guard_actions"], dtype=np.float32),
                np.asarray(first["ttc_actions"], dtype=np.float32),
            ),
            (
                np.asarray(first["ttc_actions"], dtype=np.float32),
                np.asarray(first["final_guard_actions"], dtype=np.float32),
            ),
        )
        per_agent_levels: list[list[InterventionLevel]] = [
            [] for _ in range(self.num_agents)
        ]
        per_agent_delta_v: list[list[float]] = [
            [] for _ in range(self.num_agents)
        ]
        for stage_input, stage_output in stage_actions:
            for agent_id in range(self.num_agents):
                level, delta_v, _ = action_modification_level(
                    stage_input[agent_id], stage_output[agent_id],
                    acceleration_scale=self.agent_max_acceleration,
                    dt=self.dt,
                )
                per_agent_levels[agent_id].append(level)
                per_agent_delta_v[agent_id].append(float(delta_v))
        executed = np.asarray(first["final_guard_actions"], dtype=np.float32)
        return ShadowSafetyResultV2(
            requested_actions=tuple(
                tuple(float(value) for value in row) for row in candidate
            ),
            per_agent_stage_delta_v_mps=tuple(
                tuple(values) for values in per_agent_delta_v
            ),
            per_agent_stage_levels=tuple(
                tuple(values) for values in per_agent_levels
            ),
            executed_actions=tuple(
                tuple(float(value) for value in row) for row in executed
            ),
        )

    def evaluate_shadow_action_v1(
        self,
        snapshot: CausalSafetySnapshotV1,
        requested_actions: np.ndarray,
    ) -> ShadowSafetyResultV1:
        """Run one immutable Initial Guard → TTC → Final Guard shadow."""

        detailed = self.evaluate_shadow_action_v2(snapshot, requested_actions)
        levels = np.asarray(detailed.per_agent_stage_levels, dtype=np.int8)
        delta_v = np.asarray(detailed.per_agent_stage_delta_v_mps, dtype=np.float64)
        return ShadowSafetyResultV1(
            requested_actions=detailed.requested_actions,
            stage_delta_v_mps=tuple(float(value) for value in delta_v.max(axis=0)),
            stage_levels=tuple(
                InterventionLevel(int(value)) for value in levels.max(axis=0)
            ),
            executed_actions=detailed.executed_actions,
        )


    def _v5_4_select_recovery_command(
        self, reference_actions: np.ndarray,
    ) -> StarRecoveryCommandV1 | None:
        snapshot = self._safety_planning_snapshot()
        supported_sources = {
            "boundary", "pair", "obstacle_guard_fallback"
        }
        sources = tuple(
            value for value in self._safe_recovery_entry_sources
            if value in supported_sources
        )
        if not sources:
            return None
        visible_obstacles = snapshot.visible_obstacles
        if (
            "obstacle_guard_fallback" in sources
            and self._recovery_incident_hazard_keys
        ):
            active_keys = set(self._recovery_incident_hazard_keys)
            visible_obstacles = tuple(
                tuple(
                    obstacle for obstacle in tracks
                    if f"agent:{agent_id}:track:{obstacle.track_id}"
                    in active_keys
                )
                for agent_id, tracks in enumerate(snapshot.visible_obstacles)
            )
        candidates = build_star_recovery_candidates(
            positions=self.positions,
            velocities=self.velocities,
            final_goals=self.goals,
            visible_obstacles=visible_obstacles,
            map_size=self.map_size,
            agent_radius=self.agent_radius,
            surface_clearance=self.boundary_guard.surface_clearance_m,
            invariant_buffer=self.boundary_guard.invariant_buffer_m,
            max_speed=self.agent_max_speed,
            max_acceleration=self.agent_max_acceleration,
            dt=self.dt,
            teammate_safety_distance=float(
                self.oracle_config["teammate_safety_distance"]
            ),
            obstacle_safety_margin=float(
                self.oracle_config["obstacle_safety_margin"]
            ),
            offending_pairs=self._star_recovery_offending_pairs,
            affected_agent_mask=self._star_recovery_affected_mask,
            target_distance=self.safe_recovery_target_distance,
            speed_fraction=self.safe_recovery_speed_fraction,
            boundary_trigger_margin=self.safe_recovery_boundary_trigger_margin,
            obstacle_repulsion_margin=self.safe_recovery_obstacle_repulsion_margin,
            entry_sources=sources,
        )
        current_margin = self._v5_recovery_margin()
        current_margin = float(current_margin) if current_margin is not None else 0.0
        affected = self._star_recovery_affected_mask.copy()
        scored = []
        study_rows = []
        paired_study_rows = []
        study = getattr(self, "recovery_study", None)
        audit_sink = getattr(self, "recovery_gate_a_sink", None)
        audit_rows = []
        v3_rows = []
        is_v3 = study is not None and study.get("protocol") == "v3"
        if is_v3 and study["arm"] not in {"R0","R1","R2"}:
            raise ValueError("V3 adaptive R3 is deferred pending Fit-only evidence")
        priority = {"normal": 2, "left_tangent": 1, "right_tangent": 0}
        for candidate in candidates:
            combined = np.asarray(reference_actions, dtype=np.float32).copy()
            candidate_mask = np.asarray(
                candidate.affected_agent_mask, dtype=bool
            )
            combined[candidate_mask] = candidate.actions[candidate_mask]
            envelope = self._v5_4_observable_envelope(
                combined,
                horizon_steps=self.safe_recovery_shadow_horizon_steps,
                active_hazard_keys=self._recovery_incident_hazard_keys,
            )
            final_margin = float(envelope["final_margin"])
            margin_gain = (
                final_margin - current_margin
                if np.isfinite(final_margin) else float("inf")
            )
            predicted_positions = np.asarray(
                envelope["positions"], dtype=np.float64
            )
            goal_progress = (
                float(np.sum(
                    np.linalg.norm(
                        self.goals[affected] - self.positions[affected], axis=1
                    )
                    - np.linalg.norm(
                        self.goals[affected] - predicted_positions[affected],
                        axis=1,
                    )
                )) if np.any(affected) else 0.0
            )
            deviation = (
                float(np.linalg.norm(
                    np.asarray(envelope["first_executed_actions"])[affected]
                    - np.asarray(reference_actions)[affected]
                )) if np.any(affected) else 0.0
            )
            accepted = bool(
                envelope["feasible"]
                and float(envelope["post_projection_norm"]) > 1e-6
                and margin_gain
                >= self.safe_recovery_candidate_minimum_margin_gain
            )
            if audit_sink is not None:
                progress = np.linalg.norm(self.goals-self.positions,axis=1)-np.linalg.norm(self.goals-predicted_positions,axis=1)
                audit_rows.append({"candidate":candidate,"envelope":envelope,"legacy_eligible":accepted,
                    "legacy_reason":str(envelope["rejection_reason"]) if not envelope["feasible"] else
                        ("post_projection_zero" if float(envelope["post_projection_norm"]) <= 1e-6 else
                         ("nonpositive_margin_gain" if margin_gain < self.safe_recovery_candidate_minimum_margin_gain else "accepted")),
                    "margin_gain":margin_gain,"affected":affected,"team_progress":float(progress.sum()),
                    "affected_progress":goal_progress,"affected_min_progress":float(progress[affected].min()) if np.any(affected) else 0.,
                    "actor_deviation":deviation})
            if is_v3 and study["arm"] in {"R1","R2"}:
                from src.safety.staged_execution.recovery_v3 import RecoveryResultV3, evaluate_candidate_v3, shadow_consistent_v3
                progress = np.linalg.norm(self.goals-self.positions,axis=1)-np.linalg.norm(self.goals-predicted_positions,axis=1)
                if study["arm"] == "R1":
                    v3_result = RecoveryResultV3(
                        "legacy_feasible" if accepted else "unavailable",
                        "legacy_screen" if accepted else "legacy_candidate_ineligible",
                        margin_gain,float(envelope["minimum_margin"]) if np.isfinite(envelope["minimum_margin"]) else 1e9,
                        float(progress.sum()),float(progress[affected].min()) if np.any(affected) else 0.,deviation)
                else:
                    v3_result = evaluate_candidate_v3(snapshot,envelope["position_path"],envelope["feasible"],
                        lambda key,h:self.recovery_prediction_history.radius_v2(key,h,study["artifact"],False),
                        affected,deviation,legacy_eligible=accepted,legacy_margin_gain=margin_gain,
                        obstacle_speed_bound=study["obstacle_speed_bound_mps"],
                        lost_tracks=self.recovery_prediction_history.lost_reachable(snapshot))
                    if not shadow_consistent_v3(snapshot,envelope):
                        v3_result = replace(v3_result,mode="unavailable",reason="shadow_inconsistent")
                v3_rows.append((v3_result,priority.get(candidate.label,-1),candidate,envelope))
                self.safe_recovery_events.append({"event":"v3_candidate_score","step":int(self.step_count),
                    "arm":study["arm"],"label":candidate.label,**vars(v3_result)})
                if not v3_result.eligible:
                    self.recovery_candidate_rejection_count += 1
                    self.recovery_candidate_rejection_reason_counts[v3_result.reason] = self.recovery_candidate_rejection_reason_counts.get(v3_result.reason,0)+1
                continue
            if study is not None and study["arm"] != "R0":
                from src.safety.staged_execution.robust_recovery_selector import evaluate_candidate
                arm = study["arm"]
                is_v2 = study.get("protocol", "v1") == "v2"
                radius = (
                    (lambda key, horizon: 0.) if arm == "R1" else
                    (lambda key, horizon: (self.recovery_prediction_history.radius_v2 if is_v2 else self.recovery_prediction_history.radius)(
                        key, horizon, study["artifact"], arm == "R3"
                    ))
                )
                if arm == "R1":
                    # Pure ranking ablation: exactly the legacy feasible set.
                    from src.safety.staged_execution.contracts import RobustRecoveryResultV1
                    progress = np.linalg.norm(self.goals - self.positions, axis=1) - np.linalg.norm(self.goals - predicted_positions, axis=1)
                    result = RobustRecoveryResultV1(
                        accepted,
                        float(envelope["minimum_margin"]) if np.isfinite(envelope["minimum_margin"]) else 1e9,
                        float(progress.sum()), float(progress[affected].min()) if np.any(affected) else 0.,
                        deviation, "accepted" if accepted else "legacy_candidate_ineligible",
                    )
                else:
                    robust_kwargs = {}
                    if is_v2:
                        robust_kwargs = {"obstacle_speed_bound": study["obstacle_speed_bound_mps"],
                                         "lost_tracks": self.recovery_prediction_history.lost_reachable(snapshot)}
                    result = evaluate_candidate(
                        snapshot, envelope["position_path"], envelope["feasible"],
                        radius, affected, deviation, **robust_kwargs,
                    )
                    if is_v2 and arm == "R3":
                        probe_started = time.perf_counter()
                        fixed_radius = lambda key, horizon: self.recovery_prediction_history.radius_v2(key, horizon, study["artifact"], False, record=False)
                        fixed_result = evaluate_candidate(snapshot, envelope["position_path"], envelope["feasible"],
                                                          fixed_radius, affected, deviation, **robust_kwargs)
                        if not accepted:
                            fixed_result = replace(fixed_result, safe=False, reason="legacy_candidate_ineligible")
                        paired_study_rows.append((fixed_result, priority.get(candidate.label, -1), candidate, envelope))
                        self._cupjr_probe_seconds = getattr(self, "_cupjr_probe_seconds", 0.) + time.perf_counter() - probe_started
                # Keep V5.4 candidate admission identical; the study adds
                # robust screening and changes ranking, not legacy eligibility.
                if not accepted:
                    result = replace(result, safe=False, reason="legacy_candidate_ineligible")
                if not result.safe:
                    self.recovery_candidate_rejection_count += 1
                    self.recovery_candidate_rejection_reason_counts[result.reason] = (
                        self.recovery_candidate_rejection_reason_counts.get(result.reason, 0) + 1
                    )
                study_rows.append((result, priority.get(candidate.label, -1), candidate, envelope))
                self.safe_recovery_events.append({
                    "event": "robust_candidate_score", "arm": arm,
                    "step": int(self.step_count), "label": candidate.label,
                    **vars(result),
                })
                continue
            if not envelope["feasible"]:
                reason = str(envelope["rejection_reason"])
            elif margin_gain < self.safe_recovery_candidate_minimum_margin_gain:
                reason = "nonpositive_margin_gain"
            elif float(envelope["post_projection_norm"]) <= 1e-6:
                reason = "post_projection_zero"
            else:
                reason = "accepted"
            if not accepted:
                self.recovery_candidate_rejection_count += 1
                self.recovery_candidate_rejection_reason_counts[reason] = (
                    self.recovery_candidate_rejection_reason_counts.get(reason, 0)
                    + 1
                )
            self.safe_recovery_events.append({
                "event": "candidate_score",
                "step": int(self.step_count),
                "label": candidate.label,
                "accepted": accepted,
                "reason": reason,
                "post_projection_norm": float(
                    envelope["post_projection_norm"]
                ),
                "predicted_margin_gain_m": margin_gain,
                "minimum_predicted_margin_m": float(
                    envelope["minimum_margin"]
                ),
                "goal_progress_m": goal_progress,
                "action_deviation_norm": deviation,
            })
            if accepted:
                scored.append((
                    margin_gain,
                    float(envelope["minimum_margin"]),
                    goal_progress,
                    -deviation,
                    priority.get(candidate.label, -1),
                    candidate,
                    envelope,
                ))
        def emit_gate_a(selected=None, envelope=None):
            if audit_sink is None:
                return
            from src.safety.staged_execution.robust_recovery_selector import decision_diagnostics
            started = time.perf_counter()
            audit_sink(decision_diagnostics(snapshot,audit_rows,self.recovery_prediction_history,
                study["artifact"],study["obstacle_speed_bound_mps"],
                (selected.label,envelope["first_executed_actions"]) if selected is not None else None,
                arm=study["arm"],sources=sources,reference_actions=reference_actions,
                tolerance=study["progress_tie_tolerance_m"]))
            self._cupjr_audit_seconds = getattr(self,"_cupjr_audit_seconds",0.)+time.perf_counter()-started

        if is_v3 and study["arm"] in {"R1","R2"}:
            from src.safety.staged_execution.recovery_v3 import select_safety_candidate_v3
            choice = select_safety_candidate_v3(v3_rows)
            if choice is None:
                self.safe_recovery_events.append({"event":"v3_no_feasible_candidate","step":int(self.step_count),"arm":study["arm"]})
                return None
            outcome,_,selected,envelope = choice
            eligible_v3 = [row[0] for row in v3_rows if row[0].eligible and row[0].mode == outcome.mode]
            best_gain = max(r.legacy_margin_gain_m for r in eligible_v3)
            highest_progress = max(eligible_v3,key=lambda r:r.team_progress_m)
            self.safe_recovery_events.append({"event":"v3_candidate_selected","step":int(self.step_count),
                "label":selected.label,"mode":outcome.mode,
                "legacy_safety_gain_sacrifice_m":best_gain-outcome.legacy_margin_gain_m,
                "team_progress_cost_vs_max_m":highest_progress.team_progress_m-outcome.team_progress_m,
                "safety_tie_tolerance_m":0.,"full_robust_screen_passed":outcome.mode == "robust_feasible",
                "full_calibrated_coverage_claim":False,
                "first_executed_actions":np.asarray(envelope["first_executed_actions"]).tolist()})
            self.recovery_candidate_selection_count += 1
            self.recovery_candidate_post_projection_norm = float(envelope["post_projection_norm"])
            self.recovery_candidate_predicted_margin_gain = outcome.legacy_margin_gain_m
            return StarRecoveryCommandV1(available=True,actions=selected.actions.copy(),targets=selected.targets.copy(),
                affected_agent_mask=selected.affected_agent_mask,reason=f"cup_jr_v3_{outcome.mode}_{selected.label}",entry_sources=selected.entry_sources)

        if study is not None and study["arm"] != "R0":
            from src.safety.staged_execution.robust_recovery_selector import select_progress_candidate
            choice = select_progress_candidate(study_rows, study["progress_tie_tolerance_m"])
            if paired_study_rows:
                from src.safety.staged_execution.robust_recovery_selector import selected_actions_differ
                fixed_choice = select_progress_candidate(paired_study_rows, study["progress_tie_tolerance_m"])
                fixed_label = fixed_choice[2].label if fixed_choice else None
                adaptive_label = choice[2].label if choice else None
                self.safe_recovery_events.append({"event": "R2_R3_same_state_choice", "step": int(self.step_count),
                                                  "R2": fixed_label, "R3": adaptive_label,
                                                  "different": selected_actions_differ(fixed_choice, choice),
                                                  "label_different": fixed_label != adaptive_label})
            if choice is None:
                self.safe_recovery_events.append({"event": "robust_no_feasible_candidate", "step": int(self.step_count)})
                emit_gate_a()
                return None
            _, _, selected, envelope = choice
            self.recovery_candidate_selection_count += 1
            self.recovery_candidate_post_projection_norm = float(envelope["post_projection_norm"])
            selected_final_margin = float(envelope["final_margin"])
            self.recovery_candidate_predicted_margin_gain = (
                selected_final_margin - current_margin
                if np.isfinite(selected_final_margin) else float("inf")
            )
            self.safe_recovery_events.append({"event": "robust_candidate_selected", "step": int(self.step_count), "label": selected.label})
            emit_gate_a(selected,envelope)
            return StarRecoveryCommandV1(
                available=True, actions=selected.actions.copy(),
                targets=selected.targets.copy(),
                affected_agent_mask=selected.affected_agent_mask,
                reason=f"cup_jr_{selected.label}", entry_sources=selected.entry_sources,
            )
        if not scored:
            emit_gate_a()
            return None
        selected_row = max(scored, key=lambda item: item[:5])
        selected = selected_row[-2]
        envelope = selected_row[-1]
        self.recovery_candidate_selection_count += 1
        self.recovery_candidate_post_projection_norm = float(
            envelope["post_projection_norm"]
        )
        final_margin = float(envelope["final_margin"])
        self.recovery_candidate_predicted_margin_gain = (
            final_margin - current_margin
            if np.isfinite(final_margin) else float("inf")
        )
        emit_gate_a(selected,envelope)
        return StarRecoveryCommandV1(
            available=True,
            actions=selected.actions.copy(),
            targets=selected.targets.copy(),
            affected_agent_mask=selected.affected_agent_mask,
            reason=f"v5_4_{selected.label}",
            entry_sources=selected.entry_sources,
        )

    def _star_active_recovery_actions(
        self, reference_actions: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Return STAR recovery control, independent of the Actor request."""
        if self._safe_recovery_failed:
            return None
        if (
            self.star_safety_state_machine_version in {
                "star_persistent_recovery_v5_4",
                "star_closure_consistency_audit_v1",
            }
            and self._recovery_lineage_steps
            >= self.safe_recovery_lineage_max_steps
        ):
            self._mark_v5_3_lineage_failure("lineage_step_budget_reached")
            return None
        if (
            self.star_safety_state_machine_version
            in _INCIDENT_RECOVERY_VERSIONS
            and self._safe_recovery_target is not None
        ):
            # A V5.1 target is stable for the complete attempt.  It is
            # replaced only by an explicit 3-step no-improvement refresh or a
            # geometry-changing incident merge.
            delta = self._safe_recovery_target - self.positions
            distance = np.linalg.norm(delta, axis=1, keepdims=True)
            direction = np.divide(
                delta,
                np.maximum(distance, 1e-8),
                out=np.zeros_like(delta),
            )
            desired_velocity = (
                direction * self.agent_max_speed
                * self.safe_recovery_speed_fraction
            )
            desired_velocity[distance[:, 0] <= 1e-6] = 0.0
            actions = np.clip(
                (desired_velocity - self.velocities)
                / (self.agent_max_acceleration * self.dt),
                -1.0,
                1.0,
            ).astype(np.float32)
            actions[~self._star_recovery_affected_mask] = 0.0
            study = getattr(self,"recovery_study",None)
            if study is not None and study.get("protocol") == "v3" and study["arm"] == "R2":
                # A certificate is local to the current observation. Persistent
                # target following must recheck evidence at the current step.
                from src.safety.staged_execution.recovery_v3 import evaluate_candidate_v3, shadow_consistent_v3
                combined = np.zeros_like(actions) if reference_actions is None else np.asarray(reference_actions,dtype=np.float32).copy()
                combined[self._star_recovery_affected_mask] = actions[self._star_recovery_affected_mask]
                envelope = self._v5_4_observable_envelope(combined,horizon_steps=self.safe_recovery_shadow_horizon_steps,
                    active_hazard_keys=self._recovery_incident_hazard_keys)
                current = self._v5_recovery_margin()
                gain = float(envelope["final_margin"])-(float(current) if current is not None else 0.)
                snapshot = self._safety_planning_snapshot()
                outcome = evaluate_candidate_v3(snapshot,envelope["position_path"],envelope["feasible"],
                    lambda key,h:self.recovery_prediction_history.radius_v2(key,h,study["artifact"],False),
                    self._star_recovery_affected_mask,0.,legacy_eligible=(float(envelope["post_projection_norm"])>1e-6
                        and gain >= self.safe_recovery_candidate_minimum_margin_gain),legacy_margin_gain=gain,
                    obstacle_speed_bound=study["obstacle_speed_bound_mps"],lost_tracks=self.recovery_prediction_history.lost_reachable(snapshot))
                if not shadow_consistent_v3(snapshot,envelope):
                    outcome = replace(outcome,mode="unavailable",reason="shadow_inconsistent")
                self.safe_recovery_events.append({"event":"v3_active_certificate","step":int(self.step_count),**vars(outcome)})
                if not outcome.eligible:
                    self.safe_recovery_unavailable_count += 1
                    self._last_recovery_reason = outcome.reason
                    self._handle_v5_2_recovery_unavailable(outcome.reason)
                    return None
            return actions
        if self.star_safety_state_machine_version in {
            "star_persistent_recovery_v5_4",
            "star_closure_consistency_audit_v1",
        }:
            actor_reference = (
                np.zeros((self.num_agents, 2), dtype=np.float32)
                if reference_actions is None
                else np.asarray(reference_actions, dtype=np.float32)
            )
            command = self._v5_4_select_recovery_command(actor_reference)
            if command is None:
                command = StarRecoveryCommandV1(
                    available=False,
                    actions=np.zeros((self.num_agents, 2), dtype=np.float32),
                    targets=self.positions.astype(np.float64).copy(),
                    affected_agent_mask=tuple(
                        bool(value)
                        for value in self._star_recovery_affected_mask
                    ),
                    reason="no_post_projection_progress_candidate",
                    entry_sources=tuple(self._safe_recovery_entry_sources),
                )
        else:
            snapshot = self._safety_planning_snapshot()
            command = build_star_recovery_command(
                positions=self.positions,
                velocities=self.velocities,
                final_goals=self.goals,
                visible_obstacles=snapshot.visible_obstacles,
                map_size=self.map_size,
                agent_radius=self.agent_radius,
                surface_clearance=self.boundary_guard.surface_clearance_m,
                invariant_buffer=self.boundary_guard.invariant_buffer_m,
                max_speed=self.agent_max_speed,
                max_acceleration=self.agent_max_acceleration,
                dt=self.dt,
                teammate_safety_distance=float(
                    self.oracle_config["teammate_safety_distance"]
                ),
                obstacle_safety_margin=float(
                    self.oracle_config["obstacle_safety_margin"]
                ),
                offending_pairs=self._star_recovery_offending_pairs,
                affected_agent_mask=self._star_recovery_affected_mask,
                target_distance=self.safe_recovery_target_distance,
                speed_fraction=self.safe_recovery_speed_fraction,
                boundary_trigger_margin=(
                    self.safe_recovery_boundary_trigger_margin
                ),
                obstacle_repulsion_margin=(
                    self.safe_recovery_obstacle_repulsion_margin
                ),
                strict_affected_scope=(
                    self.star_safety_state_machine_version
                    in {
                        "star_scoped_recovery_v4", "star_adaptive_recovery_v5",
                        "star_incident_recovery_v5_1",
                        "star_incident_recovery_v5_2",
                        "star_persistent_recovery_v5_3",
                    }
                ),
                attempt_index=self._safe_recovery_attempt_index,
                entry_sources=(
                    self._safe_recovery_entry_sources
                    if self.star_safety_state_machine_version
                    in {
                        "star_adaptive_recovery_v5",
                        "star_incident_recovery_v5_1",
                        "star_incident_recovery_v5_2",
                        "star_persistent_recovery_v5_3",
                    }
                    else ()
                ),
                persistent_margin_mode=(
                    self.star_safety_state_machine_version
                    == "star_persistent_recovery_v5_3"
                ),
            )
        if not command.available:
            self.safe_recovery_unavailable_count += 1
            self._last_recovery_reason = command.reason
            if (
                not self.safe_recovery_events
                or self.safe_recovery_events[-1].get("event") != "unavailable"
                or self.safe_recovery_events[-1].get("reason") != command.reason
            ):
                self.safe_recovery_events.append({
                    "event": "unavailable",
                    "step": int(self.step_count),
                    "reason": command.reason,
                    "offending_pairs": [
                        list(pair) for pair in self._star_recovery_offending_pairs
                    ],
                    "affected_agent_mask": self._star_recovery_affected_mask.tolist(),
                })
            if self.star_safety_state_machine_version in {
                "star_incident_recovery_v5_2", "star_persistent_recovery_v5_3",
                "star_persistent_recovery_v5_4",
                "star_closure_consistency_audit_v1",
            }:
                self._handle_v5_2_recovery_unavailable(command.reason)
            return None
        if self.star_safety_state_machine_version in {
            "star_incident_recovery_v5_2", "star_persistent_recovery_v5_3",
            "star_persistent_recovery_v5_4",
            "star_closure_consistency_audit_v1",
        }:
            self._recovery_direction_unavailable_streak = 0
        if self._safety_mode != "SAFE_RECOVER":
            self.safe_recover_entry_count += 1
            self._safe_recovery_start_positions = self.positions.copy()
            self.safe_recovery_events.append({
                "event": "entry",
                "step": int(self.step_count),
                "reason": command.reason,
                "offending_pairs": [
                    list(pair) for pair in self._star_recovery_offending_pairs
                ],
                "affected_agent_mask": list(command.affected_agent_mask),
                "targets": command.targets.astype(float).tolist(),
                "reference_action_norm": float(np.linalg.norm(command.actions)),
                "attempt_index": int(self._safe_recovery_attempt_index),
                "entry_sources": list(command.entry_sources),
            })
        self._safety_mode = "SAFE_RECOVER"
        self._safe_brake_active = False
        self._safe_recovery_target = command.targets.copy()
        self._star_recovery_affected_mask = np.asarray(
            command.affected_agent_mask, dtype=bool
        )
        self._last_recovery_reason = command.reason
        return command.actions.copy()

    def _update_star_active_recovery_state(
        self,
        guard_result: Any,
        *,
        brake_applied: bool,
        recovery_control_applied: bool,
        control_requested_actions: np.ndarray,
    ) -> None:
        """Advance V3 brake/recover states from execution-level evidence."""
        pair_infeasible = bool(getattr(guard_result, "pair_infeasible", False))
        self.pair_infeasible_count += int(pair_infeasible)
        if pair_infeasible and not self._last_pair_infeasible:
            self.pair_infeasible_entry_count += 1
        self._last_pair_infeasible = pair_infeasible

        if brake_applied:
            self._safe_brake_current_streak += 1
            self.max_safe_brake_streak = max(
                self.max_safe_brake_streak, self._safe_brake_current_streak
            )
        else:
            self._safe_brake_current_streak = 0

        if recovery_control_applied:
            self.safe_recover_step_count += 1
            self._safe_recover_current_streak += 1
            self._safe_recovery_elapsed_steps += 1
            self.max_safe_recover_streak = max(
                self.max_safe_recover_streak,
                self._safe_recover_current_streak,
            )
            if pair_infeasible:
                self._star_recovery_offending_pairs = tuple(
                    getattr(guard_result, "offending_pairs", ())
                )
                self._star_recovery_affected_mask |= np.asarray(
                    getattr(
                        guard_result,
                        "affected_agent_mask",
                        [False] * self.num_agents,
                    ),
                    dtype=bool,
                )
                self.safe_recovery_events.append({
                    "event": "infeasible_step",
                    "step": int(self.step_count),
                    "offending_pairs": [
                        list(pair)
                        for pair in self._star_recovery_offending_pairs
                    ],
                    "reference_action_norm": float(np.linalg.norm(
                        control_requested_actions
                    )),
                    "guard_reasons": [
                        list(row) for row in getattr(
                            guard_result, "reasons", ()
                        )
                    ],
                })
            projected_velocity = np.asarray(
                getattr(
                    guard_result,
                    "projected_next_velocities",
                    np.zeros((self.num_agents, 2)),
                ),
                dtype=np.float64,
            )
            affected = self._star_recovery_affected_mask
            displacement = (
                float(np.max(np.linalg.norm(projected_velocity[affected], axis=1)))
                * self.dt
                if np.any(affected) else 0.0
            )
            stable = bool(
                not pair_infeasible
                and getattr(guard_result, "invariant_violations", 1) == 0
                and displacement >= self.safe_recovery_minimum_displacement
            )
            self._recovery_stable_steps = (
                self._recovery_stable_steps + 1 if stable else 0
            )
            if self._recovery_stable_steps >= self.safe_recovery_exit_stable_steps:
                self.safe_recover_exit_count += 1
                self.safe_brake_exit_count += 1
                self.safe_brake_exit_steps.append(int(self.step_count))
                self.safe_recovery_durations.append(
                    int(self._safe_recovery_elapsed_steps)
                )
                self.safe_brake_durations.append(
                    int(self._safe_recovery_elapsed_steps)
                )
                total_displacement = 0.0
                if self._safe_recovery_start_positions is not None:
                    total_displacement = float(np.max(np.linalg.norm(
                        self.positions - self._safe_recovery_start_positions,
                        axis=1,
                    )))
                self.safe_recovery_events.append({
                    "event": "exit",
                    "step": int(self.step_count),
                    "duration": int(self._safe_recovery_elapsed_steps),
                    "maximum_agent_displacement_m": total_displacement,
                })
                self._safety_mode = "NORMAL"
                self._safe_brake_active = False
                self._safe_recovery_target = None
                self._safe_recovery_start_positions = None
                self._safe_recovery_elapsed_steps = 0
                self._safe_recover_current_streak = 0
                self._recovery_stable_steps = 0
                self._star_recovery_offending_pairs = ()
                self._star_recovery_affected_mask[:] = False
                self._last_recovery_reason = "active_recovery_complete"
                return
            if self._safe_recovery_elapsed_steps >= self.safe_recovery_maximum_steps:
                self.safe_recovery_failure_count += 1
                self._safe_recovery_failed = True
                self._safety_mode = "SAFE_BRAKE"
                self._safe_brake_active = True
                self.safe_recovery_events.append({
                    "event": "failure",
                    "step": int(self.step_count),
                    "duration": int(self._safe_recovery_elapsed_steps),
                    "reason": "maximum_steps_exceeded",
                })
                self._last_recovery_reason = "maximum_steps_exceeded"
            return

        self._safe_recover_current_streak = 0
        if pair_infeasible:
            self._enter_star_active_recovery_brake(
                "joint_guard_infeasible",
                offending_pairs=getattr(guard_result, "offending_pairs", ()),
                affected_agent_mask=getattr(
                    guard_result, "affected_agent_mask", None
                ),
            )
            return

        if self._safety_mode != "NORMAL":
            return
        positions = self.positions
        low = (
            self.agent_radius
            + self.boundary_guard.surface_clearance_m
            + self.boundary_guard.invariant_buffer_m
        )
        high = self.map_size - low
        projected_velocity = np.asarray(
            getattr(guard_result, "projected_next_velocities", self.velocities),
            dtype=np.float64,
        )
        for agent_id in range(self.num_agents):
            outward = False
            for axis in range(2):
                outward |= bool(
                    positions[agent_id, axis]
                    <= low + self.safe_recovery_boundary_trigger_margin
                    and control_requested_actions[agent_id, axis] < -1e-5
                )
                outward |= bool(
                    positions[agent_id, axis]
                    >= high[axis] - self.safe_recovery_boundary_trigger_margin
                    and control_requested_actions[agent_id, axis] > 1e-5
                )
            displacement = float(np.linalg.norm(projected_velocity[agent_id])) * self.dt
            if outward and displacement < self.safe_recovery_minimum_displacement:
                self._boundary_stall_steps[agent_id] += 1
            else:
                self._boundary_stall_steps[agent_id] = 0
        triggered = (
            self._boundary_stall_steps
            >= self.safe_recovery_boundary_stall_patience
        )
        if np.any(triggered):
            self.boundary_recovery_trigger_count += 1
            self._boundary_stall_steps[triggered] = 0
            self._enter_star_active_recovery_brake(
                "boundary_stall",
                affected_agent_mask=triggered,
            )

    def _update_star_scoped_recovery_state(
        self,
        guard_result: Any,
        *,
        brake_applied: bool,
        recovery_control_applied: bool,
        control_requested_actions: np.ndarray,
    ) -> None:
        """Advance V4 state before dynamics; exit is decided post-dynamics."""
        pair_infeasible = bool(getattr(guard_result, "pair_infeasible", False))
        self.pair_infeasible_count += int(pair_infeasible)
        if pair_infeasible and not self._last_pair_infeasible:
            self.pair_infeasible_entry_count += 1
        self._last_pair_infeasible = pair_infeasible

        if brake_applied:
            self._safe_brake_current_streak += 1
            self.max_safe_brake_streak = max(
                self.max_safe_brake_streak, self._safe_brake_current_streak
            )
        else:
            self._safe_brake_current_streak = 0

        if recovery_control_applied:
            self.safe_recover_step_count += 1
            self._safe_recover_current_streak += 1
            self._safe_recovery_elapsed_steps += 1
            self._safe_recovery_sequence_steps += 1
            if (
                self.star_safety_state_machine_version
                in _INCIDENT_RECOVERY_VERSIONS
                and self._recovery_incident_id is not None
            ):
                self._recovery_incident_elapsed_steps += 1
                self._recovery_incident_sequence_steps += 1
                self.max_recovery_incident_steps = max(
                    self.max_recovery_incident_steps,
                    self._recovery_incident_sequence_steps,
                )
                if self.star_safety_state_machine_version in {
                    "star_persistent_recovery_v5_3",
                    "star_persistent_recovery_v5_4",
                    "star_closure_consistency_audit_v1",
                }:
                    if not self._recovery_lineage_failure_emitted:
                        self._recovery_lineage_steps = min(
                            self._recovery_lineage_steps + 1,
                            self.safe_recovery_lineage_max_steps,
                        )
                    self.max_recovery_lineage_steps = max(
                        self.max_recovery_lineage_steps,
                        self._recovery_lineage_steps,
                    )
                    if self.recovery_phase == "CONSOLIDATE":
                        self.consolidation_steps += 1
                    else:
                        self.escape_steps += 1
                    if (
                        self.star_safety_state_machine_version in {
                            "star_persistent_recovery_v5_4",
                            "star_closure_consistency_audit_v1",
                        }
                        and self._recovery_lineage_steps
                        >= self.safe_recovery_lineage_max_steps
                    ):
                        self._mark_v5_3_lineage_failure(
                            "lineage_step_budget_reached"
                        )
                    elif (
                        self._recovery_lineage_steps
                        > self.safe_recovery_lineage_max_steps
                    ):
                        self._mark_v5_3_lineage_failure(
                            "lineage_step_budget_exceeded"
                        )
            self.max_safe_recover_streak = max(
                self.max_safe_recover_streak,
                self._safe_recover_current_streak,
            )
            self._pending_recovery_pair_infeasible = pair_infeasible
            self._pending_recovery_invariant_violations = int(
                getattr(guard_result, "invariant_violations", 1)
            )
            if pair_infeasible:
                pairs = tuple(getattr(guard_result, "offending_pairs", ()))
                self._star_recovery_offending_pairs = pairs
                pair_mask = np.zeros(self.num_agents, dtype=bool)
                for left, right in pairs:
                    pair_mask[int(left)] = True
                    pair_mask[int(right)] = True
                self._star_recovery_affected_mask |= pair_mask
                self.safe_recovery_events.append({
                    "event": "infeasible_step",
                    "step": int(self.step_count),
                    "attempt_index": int(self._safe_recovery_attempt_index),
                    "offending_pairs": [list(pair) for pair in pairs],
                    "affected_agent_mask": (
                        self._star_recovery_affected_mask.tolist()
                    ),
                    "reference_action_norm": float(np.linalg.norm(
                        control_requested_actions
                    )),
                    "guard_reasons": [
                        list(row) for row in getattr(guard_result, "reasons", ())
                    ],
                })
            return

        self._safe_recover_current_streak = 0
        self._pending_recovery_pair_infeasible = False
        self._pending_recovery_invariant_violations = 0
        if pair_infeasible:
            self._enter_star_active_recovery_brake(
                "joint_guard_infeasible",
                offending_pairs=getattr(guard_result, "offending_pairs", ()),
                affected_agent_mask=getattr(
                    guard_result, "affected_agent_mask", None
                ),
            )
            return

        if self._safety_mode != "NORMAL":
            return
        positions = self.positions
        low = (
            self.agent_radius
            + self.boundary_guard.surface_clearance_m
            + self.boundary_guard.invariant_buffer_m
        )
        high = self.map_size - low
        projected_velocity = np.asarray(
            getattr(guard_result, "projected_next_velocities", self.velocities),
            dtype=np.float64,
        )
        for agent_id in range(self.num_agents):
            outward = False
            for axis in range(2):
                outward |= bool(
                    positions[agent_id, axis]
                    <= low + self.safe_recovery_boundary_trigger_margin
                    and control_requested_actions[agent_id, axis] < -1e-5
                )
                outward |= bool(
                    positions[agent_id, axis]
                    >= high[axis] - self.safe_recovery_boundary_trigger_margin
                    and control_requested_actions[agent_id, axis] > 1e-5
                )
            displacement = (
                float(np.linalg.norm(projected_velocity[agent_id])) * self.dt
            )
            if outward and displacement < self.safe_recovery_minimum_displacement:
                self._boundary_stall_steps[agent_id] += 1
            else:
                self._boundary_stall_steps[agent_id] = 0
        triggered = (
            self._boundary_stall_steps
            >= self.safe_recovery_boundary_stall_patience
        )
        if self.safe_recovery_exclude_goal_agents:
            triggered &= self._goal_distances() >= self.goal_radius
        if np.any(triggered):
            self.boundary_recovery_trigger_count += 1
            self._boundary_stall_steps[triggered] = 0
            self._enter_star_active_recovery_brake(
                "boundary_stall",
                affected_agent_mask=triggered,
            )

    def _finalize_star_scoped_recovery_step(
        self,
        *,
        previous_positions: np.ndarray,
        recovery_control_applied: bool,
    ) -> None:
        """Use post-dynamics motion to exit, retry, or release V4 recovery."""
        if not recovery_control_applied or self._safety_mode != "SAFE_RECOVER":
            return
        affected = self._star_recovery_affected_mask.copy()
        step_displacements = np.linalg.norm(
            self.positions - np.asarray(previous_positions, dtype=np.float64),
            axis=1,
        )
        low = (
            self.agent_radius
            + self.boundary_guard.surface_clearance_m
            + self.boundary_guard.invariant_buffer_m
        )
        high = self.map_size - low
        boundary_safe = bool(
            np.any(affected)
            and np.all(self.positions[affected] >= low - 1e-6)
            and np.all(self.positions[affected] <= high + 1e-6)
        )
        actual_min_displacement = (
            float(np.min(step_displacements[affected]))
            if np.any(affected) else 0.0
        )
        recovery_margin = self._v5_recovery_margin() if (
            self.star_safety_state_machine_version
            in {
                "star_adaptive_recovery_v5", "star_incident_recovery_v5_1",
                "star_incident_recovery_v5_2",
                "star_persistent_recovery_v5_3",
                "star_persistent_recovery_v5_4",
                "star_closure_consistency_audit_v1",
            }
        ) else None
        if (
            self.star_safety_state_machine_version
            in _PERSISTENT_RECOVERY_VERSIONS
            and recovery_margin is not None
        ):
            previous_release_margin = self.release_margin_m
            self.release_margin_m = float(recovery_margin)
            self.release_margin_delta_m = (
                float("nan") if not np.isfinite(previous_release_margin)
                else float(recovery_margin - previous_release_margin)
            )
            if (
                self.recovery_phase == "ESCAPE"
                and (
                    self._recovery_incident_sequence_steps
                    >= self.safe_recovery_escape_steps
                    or recovery_margin >= self.safe_recovery_release_hysteresis_m
                )
            ):
                self.recovery_phase = "CONSOLIDATE"
                self._recovery_stable_steps = 0
        stable = bool(
            not self._pending_recovery_pair_infeasible
            and self._pending_recovery_invariant_violations == 0
            and boundary_safe
            and actual_min_displacement
            >= self.safe_recovery_minimum_displacement
            and (recovery_margin is None or recovery_margin >= -1e-6)
        )
        if (
            self.star_safety_state_machine_version
            in _PERSISTENT_RECOVERY_VERSIONS
        ):
            stable = bool(
                stable
                and self.recovery_phase == "CONSOLIDATE"
                and recovery_margin is not None
                and recovery_margin >= self.safe_recovery_release_hysteresis_m
                and (
                    not np.isfinite(self.release_margin_delta_m)
                    or self.release_margin_delta_m >= -1e-6
                )
            )
        self._recovery_stable_steps = (
            self._recovery_stable_steps + 1 if stable else 0
        )
        if self._recovery_stable_steps >= self.safe_recovery_exit_stable_steps:
            if (
                self.star_safety_state_machine_version
                in _PERSISTENT_RECOVERY_VERSIONS
            ):
                self._safety_mode = "SAFE_CONSOLIDATE"
                self._safe_brake_active = False
                self._safe_recovery_target = None
                self._last_recovery_reason = "awaiting_actor_shadow_check"
                return
            self.safe_recover_exit_count += 1
            self.safe_brake_exit_count += 1
            self.safe_brake_exit_steps.append(int(self.step_count))
            self.safe_recovery_durations.append(
                int(self._safe_recovery_sequence_steps)
            )
            self.safe_brake_durations.append(
                int(self._safe_recovery_sequence_steps)
            )
            total_displacement = 0.0
            if self._safe_recovery_start_positions is not None and np.any(affected):
                total_displacement = float(np.max(np.linalg.norm(
                    self.positions[affected]
                    - self._safe_recovery_start_positions[affected],
                    axis=1,
                )))
            self.safe_recovery_events.append({
                "event": "exit",
                "step": int(self.step_count),
                "duration": int(self._safe_recovery_sequence_steps),
                "attempt_index": int(self._safe_recovery_attempt_index),
                "actual_min_step_displacement_m": actual_min_displacement,
                "maximum_agent_displacement_m": total_displacement,
            })
            self._safety_mode = "NORMAL"
            self._safe_brake_active = False
            self._safe_recovery_target = None
            self._safe_recovery_start_positions = None
            self._safe_recovery_elapsed_steps = 0
            self._safe_recovery_sequence_steps = 0
            self._safe_recovery_attempt_index = 0
            self._safe_recover_current_streak = 0
            self._recovery_stable_steps = 0
            self._star_recovery_offending_pairs = ()
            self._star_recovery_affected_mask[:] = False
            self._safe_recovery_entry_sources = ()
            self._safe_recovery_previous_margin = None
            self._safe_recovery_no_improvement_steps = 0
            self._last_recovery_reason = "scoped_recovery_complete"
            if (
                self.star_safety_state_machine_version
                in _INCIDENT_RECOVERY_VERSIONS
            ):
                if (
                    self.star_safety_state_machine_version
                    == "star_incident_recovery_v5_2"
                ):
                    self._close_v5_2_recovery_incident(
                        "resolved_by_safety_clearance"
                    )
                else:
                    self._clear_v5_1_recovery_incident()
            return

        if self.star_safety_state_machine_version in {
            "star_adaptive_recovery_v5", "star_incident_recovery_v5_1",
            "star_incident_recovery_v5_2",
            "star_persistent_recovery_v5_3",
            "star_persistent_recovery_v5_4",
            "star_closure_consistency_audit_v1",
        }:
            if (
                self.star_safety_state_machine_version in {
                    "star_persistent_recovery_v5_4",
                    "star_closure_consistency_audit_v1",
                }
            ):
                if (
                    actual_min_displacement
                    < self.safe_recovery_minimum_displacement
                    or (
                        np.isfinite(self.recovery_candidate_post_projection_norm)
                        and self.recovery_candidate_post_projection_norm <= 1e-6
                    )
                ):
                    self.recovery_candidate_fixed_point_streak += 1
                else:
                    self.recovery_candidate_fixed_point_streak = 0
            if recovery_margin is not None:
                previous_margin = self._safe_recovery_previous_margin
                improved = bool(
                    previous_margin is None
                    or recovery_margin - previous_margin
                    >= self.safe_recovery_minimum_margin_improvement
                )
                self._safe_recovery_no_improvement_steps = (
                    0 if improved else self._safe_recovery_no_improvement_steps + 1
                )
                self.safe_recovery_events.append({
                    "event": "margin",
                    "step": int(self.step_count),
                    "entry_sources": list(self._safe_recovery_entry_sources),
                    "margin_m": float(recovery_margin),
                    "margin_delta_m": (
                        None if previous_margin is None
                        else float(recovery_margin - previous_margin)
                    ),
                    "no_improvement_steps": int(
                        self._safe_recovery_no_improvement_steps
                    ),
                })
                self._safe_recovery_previous_margin = float(recovery_margin)
            if (
                self.star_safety_state_machine_version in {
                    "star_persistent_recovery_v5_4",
                    "star_closure_consistency_audit_v1",
                }
                and self.recovery_candidate_fixed_point_streak
                >= self.safe_recovery_candidate_fixed_point_patience
            ):
                self._safe_recovery_no_improvement_steps = max(
                    self._safe_recovery_no_improvement_steps,
                    self.safe_recovery_progress_patience_steps,
                )
            refresh_count = (
                self._recovery_incident_refresh_count
                if self.star_safety_state_machine_version
                in _INCIDENT_RECOVERY_VERSIONS
                else self.safe_recovery_refresh_count
            )
            should_refresh = bool(
                self._safe_recovery_no_improvement_steps
                >= self.safe_recovery_progress_patience_steps
                and refresh_count < self.safe_recovery_maximum_refreshes
                and self._safe_recovery_sequence_steps
                < self.safe_recovery_maximum_steps
                and (
                    self.star_safety_state_machine_version
                    not in {
                        "star_persistent_recovery_v5_3",
                        "star_persistent_recovery_v5_4",
                        "star_closure_consistency_audit_v1",
                    }
                    or self.recovery_phase == "ESCAPE"
                )
            )
            if should_refresh:
                self.safe_recovery_refresh_count += 1
                if (
                    self.star_safety_state_machine_version
                    in _INCIDENT_RECOVERY_VERSIONS
                ):
                    self._recovery_incident_refresh_count += 1
                self.safe_recovery_retry_count += 1
                self._safe_recovery_attempt_index += 1
                self.safe_recovery_events.append({
                    "event": "target_refresh",
                    "step": int(self.step_count),
                    "attempt_refresh_count": int(
                        self._recovery_incident_refresh_count
                        if self.star_safety_state_machine_version
                        in _INCIDENT_RECOVERY_VERSIONS
                        else self.safe_recovery_refresh_count
                    ),
                    "attempt_index": int(self._safe_recovery_attempt_index),
                    "entry_sources": list(self._safe_recovery_entry_sources),
                    "affected_agent_mask": affected.tolist(),
                })
                self._safety_mode = "SAFE_BRAKE"
                self._safe_brake_active = True
                self._safe_recovery_target = None
                self._safe_recovery_start_positions = self.positions.copy()
                self._safe_recovery_elapsed_steps = 0
                self._safe_recover_current_streak = 0
                self._recovery_stable_steps = 0
                self._safe_recovery_no_improvement_steps = 0
                self._last_recovery_reason = "adaptive_target_refresh"
                return
            if self._safe_recovery_sequence_steps >= self.safe_recovery_maximum_steps:
                if (
                    self.star_safety_state_machine_version
                    in _PERSISTENT_RECOVERY_VERSIONS
                ):
                    self._emit_v5_3_bounded_exhaustion(
                        "maximum_steps_exceeded_one_step_brake"
                    )
                    return
                if (
                    self.star_safety_state_machine_version
                    == "star_incident_recovery_v5_2"
                ):
                    self._emit_v5_2_hard_abort(
                        "maximum_steps_exceeded_one_step_brake"
                    )
                    return
                if (
                    self.star_safety_state_machine_version
                    in _INCIDENT_RECOVERY_VERSIONS
                    and self._recovery_incident_hard_abort_emitted
                ):
                    self.recovery_duplicate_hard_abort_count += 1
                    return
                self.safe_recovery_failure_count += 1
                self.safe_recovery_abort_count += 1
                self.safe_recovery_hard_abort_count += 1
                self.safe_recovery_episode_failure = True
                if (
                    self.star_safety_state_machine_version
                    in _INCIDENT_RECOVERY_VERSIONS
                ):
                    self._recovery_incident_hard_abort_emitted = True
                    self._recovery_incident_block_reentry_until_step = int(
                        self.step_count
                    )
                self.safe_recovery_events.append({
                    "event": "hard_abort",
                    "step": int(self.step_count),
                    "duration": int(self._safe_recovery_sequence_steps),
                    "attempt_index": int(self._safe_recovery_attempt_index),
                    "entry_sources": list(self._safe_recovery_entry_sources),
                    "reason": "maximum_steps_exceeded_one_step_brake",
                    "affected_agent_mask": affected.tolist(),
                })
                self._safety_mode = "SAFE_BRAKE"
                self._safe_brake_active = True
                self._safe_recovery_failed = True
                self._v5_release_after_brake = True
                self._safe_recovery_target = None
                self._last_recovery_reason = (
                    "maximum_steps_exceeded_one_step_brake"
                )
                return
            return

        attempt_timed_out = (
            self._safe_recovery_elapsed_steps
            >= self.safe_recovery_attempt_steps
        )
        sequence_exhausted = (
            self._safe_recovery_sequence_steps
            >= self.safe_recovery_maximum_steps
        )
        if not attempt_timed_out and not sequence_exhausted:
            return
        can_retry = bool(
            not sequence_exhausted
            and self._safe_recovery_attempt_index + 1
            < self.safe_recovery_maximum_attempts
        )
        if can_retry:
            self.safe_recovery_retry_count += 1
            self._safe_recovery_attempt_index += 1
            self.safe_recovery_events.append({
                "event": "retry",
                "step": int(self.step_count),
                "attempt_index": int(self._safe_recovery_attempt_index),
                "completed_attempt_steps": int(
                    self._safe_recovery_elapsed_steps
                ),
                "actual_min_step_displacement_m": actual_min_displacement,
                "affected_agent_mask": affected.tolist(),
            })
            self._safety_mode = "SAFE_BRAKE"
            self._safe_brake_active = True
            self._safe_recovery_target = None
            self._safe_recovery_start_positions = self.positions.copy()
            self._safe_recovery_elapsed_steps = 0
            self._safe_recover_current_streak = 0
            self._recovery_stable_steps = 0
            self._last_recovery_reason = "retry_alternative_tangent"
            return

        # The final fallback is deliberately non-absorbing.  Record a hard
        # recovery failure, release the Actor, and let JointSafetyGuardV2
        # re-enter only if the next requested action remains infeasible.
        self.safe_recovery_failure_count += 1
        self.safe_recovery_abort_count += 1
        self.safe_recovery_events.append({
            "event": "failure",
            "step": int(self.step_count),
            "duration": int(self._safe_recovery_sequence_steps),
            "attempt_index": int(self._safe_recovery_attempt_index),
            "reason": "attempt_budget_exhausted_release",
            "actual_min_step_displacement_m": actual_min_displacement,
            "affected_agent_mask": affected.tolist(),
        })
        self._safety_mode = "NORMAL"
        self._safe_brake_active = False
        self._safe_recovery_failed = False
        self._safe_recovery_target = None
        self._safe_recovery_start_positions = None
        self._safe_recovery_elapsed_steps = 0
        self._safe_recovery_sequence_steps = 0
        self._safe_recovery_attempt_index = 0
        self._safe_recover_current_streak = 0
        self._recovery_stable_steps = 0
        self._star_recovery_offending_pairs = ()
        self._star_recovery_affected_mask[:] = False
        self._last_recovery_reason = "attempt_budget_exhausted_release"

    def _v5_recovery_margin(self) -> float | None:
        """Return the worst causal safety margin for the active V5 source."""

        affected = self._star_recovery_affected_mask
        if not np.any(affected):
            return None
        margins: list[float] = []
        if "boundary" in self._safe_recovery_entry_sources:
            low = (
                self.agent_radius + self.boundary_guard.surface_clearance_m
                + self.boundary_guard.invariant_buffer_m
            )
            high = self.map_size - low
            margins.append(float(min(
                np.min(self.positions[affected] - low),
                np.min(high - self.positions[affected]),
            )))
        if "pair" in self._safe_recovery_entry_sources:
            protected = float(self.oracle_config["teammate_safety_distance"])
            pair_values = [
                float(np.linalg.norm(self.positions[left] - self.positions[right]))
                - protected
                for left, right in self._star_recovery_offending_pairs
            ]
            if pair_values:
                margins.append(float(min(pair_values)))
        if "obstacle_guard_fallback" in self._safe_recovery_entry_sources:
            snapshot = self._safety_planning_snapshot()
            obstacle_values = []
            for agent_id in np.flatnonzero(affected):
                for obstacle in snapshot.visible_obstacles[int(agent_id)]:
                    predicted_relative = (
                        np.asarray(obstacle.relative_position, dtype=np.float64)
                        + self.dt * np.asarray(
                            obstacle.relative_velocity, dtype=np.float64
                        )
                    )
                    obstacle_values.append(
                        float(np.linalg.norm(predicted_relative))
                        - self.agent_radius - obstacle.radius
                        - float(self.oracle_config["obstacle_safety_margin"])
                        - self.boundary_guard.invariant_buffer_m
                    )
            if obstacle_values:
                margins.append(float(min(obstacle_values)))
        return min(margins) if margins else None

    def _update_obstacle_execution_diagnostics(self) -> None:
        """Record causal visibility and TTC without changing control."""
        if (
            not self.obstacle_execution_diagnostics_enabled
            or self.runtime_telemetry_level == "train_minimal"
        ):
            return
        risky = False
        for agent_id in range(self.num_agents):
            for item in self._tracked_obstacles(agent_id):
                obstacle_id = int(item["obstacle_id"])
                self._first_obstacle_visible_steps.setdefault(
                    (agent_id, obstacle_id), int(self.step_count)
                )
                if not bool(item["velocity_valid"]):
                    continue
                relative = np.asarray(item["relative_position"], dtype=np.float64)
                velocity = np.asarray(item["relative_velocity"], dtype=np.float64)
                radius = (
                    self.agent_radius
                    + float(item["obstacle"].radius)
                    + float(self.oracle_config["obstacle_safety_margin"])
                )
                a = float(np.dot(velocity, velocity))
                b = 2.0 * float(np.dot(relative, velocity))
                c = float(np.dot(relative, relative) - radius * radius)
                if a <= 1e-12:
                    continue
                discriminant = b * b - 4.0 * a * c
                if discriminant < 0.0:
                    continue
                roots = [
                    value for value in (
                        (-b - np.sqrt(discriminant)) / (2.0 * a),
                        (-b + np.sqrt(discriminant)) / (2.0 * a),
                    ) if value >= 0.0
                ]
                if not roots:
                    continue
                ttc = float(min(roots))
                self.minimum_observed_obstacle_ttc = min(
                    self.minimum_observed_obstacle_ttc, ttc
                )
                risky |= ttc <= self.obstacle_ttc_audit_horizon
        self.obstacle_ttc_risk_step_count += int(risky)

    def _v5_3_release_shadow_passes(self, actor_actions: np.ndarray) -> bool:
        """Run one observable-only Actor shadow action through the frozen stack."""

        values = np.asarray(actor_actions, dtype=np.float32).copy()
        if self.goal_boundary_corridor_enabled:
            values = project_goal_boundary_corridor(
                actions=values,
                positions=self.positions,
                velocities=self.velocities,
                goals=self.goals,
                map_size=self.map_size,
                agent_radius=self.agent_radius,
                goal_radius=self.goal_radius,
                max_acceleration=self.agent_max_acceleration,
                dt=self.dt,
                corridor_coordinate=self.goal_boundary_corridor_coordinate,
                boundary_axis=0,
                boundary_side="high",
            ).actions
        guard_kwargs = dict(
            map_size=self.map_size,
            agent_radius=self.agent_radius,
            dt=self.dt,
            max_acceleration=self.agent_max_acceleration,
            max_speed=self.agent_max_speed,
            safe_brake=False,
        )
        if isinstance(self.boundary_guard, JointSafetyGuardV2):
            guard_kwargs["teammate_safety_distance"] = float(
                self.oracle_config["teammate_safety_distance"]
            )
        initial = self.boundary_guard.project_actions(
            self.positions, self.velocities, values, **guard_kwargs
        )
        values = initial.executed_actions
        snapshot = self._safety_planning_snapshot()
        obstacle = project_obstacle_actions(
            actions=values,
            velocities=self.velocities,
            visible_obstacles=snapshot.visible_obstacles,
            agent_radius=self.agent_radius,
            obstacle_safety_margin=float(self.oracle_config["obstacle_safety_margin"]),
            invariant_buffer=self.boundary_guard.invariant_buffer_m,
            max_acceleration=self.agent_max_acceleration,
            dt=self.dt,
            horizon_seconds=self.obstacle_guard_horizon_seconds,
            warning_ttc_seconds=self.obstacle_guard_warning_ttc_seconds,
            emergency_ttc_seconds=self.obstacle_guard_emergency_ttc_seconds,
        )
        final = self.boundary_guard.project_actions(
            self.positions, self.velocities, obstacle.actions, **guard_kwargs
        )
        remaining = assess_obstacle_actions(
            actions=final.executed_actions,
            velocities=self.velocities,
            visible_obstacles=snapshot.visible_obstacles,
            agent_radius=self.agent_radius,
            obstacle_safety_margin=float(self.oracle_config["obstacle_safety_margin"]),
            invariant_buffer=self.boundary_guard.invariant_buffer_m,
            max_acceleration=self.agent_max_acceleration,
            dt=self.dt,
            horizon_seconds=self.obstacle_guard_horizon_seconds,
            warning_ttc_seconds=self.obstacle_guard_warning_ttc_seconds,
            emergency_ttc_seconds=self.obstacle_guard_emergency_ttc_seconds,
        )
        active_keys = set(self._recovery_incident_hazard_keys)
        same_hazard = any(
            f"agent:{item.agent_id}:track:{item.track_id}" in active_keys
            for item in remaining
        )
        return bool(
            not initial.recovery_required
            and not final.recovery_required
            and int(final.invariant_violations) == 0
            and not obstacle.fallback_required
            and not same_hazard
        )

    def _v5_4_release_shadow_passes(self, actor_actions: np.ndarray) -> bool:
        """Require a three-step observable-only release certificate."""

        envelope = self._v5_4_observable_envelope(
            actor_actions,
            horizon_steps=self.safe_recovery_shadow_horizon_steps,
            active_hazard_keys=self._recovery_incident_hazard_keys,
        )
        margins = [
            float(value) for value in envelope["margins"]
            if np.isfinite(value)
        ]
        self.release_shadow_horizon_min_margin = float(
            envelope["minimum_margin"]
        )
        self.release_shadow_horizon_min_ttc = float(
            envelope["minimum_ttc"]
        )
        nondecreasing = bool(
            not margins
            or all(
                right >= left - 1e-6
                for left, right in zip(margins, margins[1:])
            )
        )
        final_margin = float(envelope["final_margin"])
        ttc = float(envelope["minimum_ttc"])
        return bool(
            envelope["feasible"]
            and nondecreasing
            and final_margin >= self.safe_recovery_release_hysteresis_m
            and (
                not np.isfinite(ttc)
                or ttc >= self.obstacle_guard_warning_ttc_seconds
            )
        )

    def step(self, actions):
        """执行一步；terminal 模式首次碰撞即失败，rollback 保留历史语义。"""
        if self.terminated or self.truncated:
            raise RuntimeError("reset must be called after an episode ends")
        # Never mutate the caller's Actor buffer while applying HOLD/recovery
        # overlays.  Requested-action provenance must remain an immutable input.
        values = np.asarray(actions, dtype=np.float32).copy()
        expected = (self.num_agents, 2)
        if values.shape != expected or not np.isfinite(values).all():
            raise ValueError(f"actions must be a finite matrix with shape {expected}")

        # 动作执行前保存安全状态；碰撞参与者只回滚自己的位置并清零速度。
        requested_actions = values.copy()
        obstacle_fallback_applied = False
        step_final_guard_residual = False
        trace_incident_id = self._recovery_incident_id
        trace_hazard_keys = tuple(self._recovery_incident_hazard_keys)
        self._current_action_trace = {}
        self._current_action_stage_actions = {}
        self._stage_intervention_records: list[StageInterventionRecordV1] = []
        self._current_causal_safety_snapshot = self._causal_safety_snapshot(
            requested_actions
        )
        if self.runtime_telemetry_level != "train_minimal":
            self._current_action_stage_actions["actor_raw"] = (
                requested_actions.astype(np.float64).copy()
            )
        self._trace_action_stage_snapshot("actor_raw", requested_actions)
        self._run_closure_trace_audit(requested_actions)
        self._observe_closure_consistency(requested_actions)
        if (
            self.star_safety_state_machine_version
            in {"star_persistent_recovery_v5_3", "star_persistent_recovery_v5_4"}
            and self._safety_mode == "SAFE_CONSOLIDATE"
        ):
            self.release_shadow_check_count += 1
            if (
                self.star_safety_state_machine_version
                == "star_persistent_recovery_v5_4"
            ):
                self.release_shadow_check_passed = (
                    self._v5_4_release_shadow_passes(requested_actions)
                )
            else:
                self.release_shadow_check_passed = (
                    self._v5_3_release_shadow_passes(requested_actions)
                )
            self.safe_recovery_events.append({
                "event": "release_shadow_check",
                "step": int(self.step_count),
                "lineage_id": self._recovery_lineage_id,
                "passed": bool(self.release_shadow_check_passed),
                "horizon_steps": int(
                    self.safe_recovery_shadow_horizon_steps
                    if self.star_safety_state_machine_version
                    == "star_persistent_recovery_v5_4" else 1
                ),
            })
            if self.release_shadow_check_passed:
                self.safe_recover_exit_count += 1
                self.safe_brake_exit_count += 1
                self.safe_brake_exit_steps.append(int(self.step_count))
                self.safe_recovery_durations.append(
                    int(self._recovery_incident_sequence_steps)
                )
                self._release_v5_2_recovery_incident(
                    "resolved_by_shadow_clearance"
                )
                self._last_recovery_reason = "persistent_recovery_complete"
            else:
                self.release_shadow_check_failure_count += 1
                self._safety_mode = "SAFE_RECOVER"
                self.recovery_phase = "ESCAPE"
                self._safe_recovery_target = None
                self._recovery_stable_steps = 0
        self._update_obstacle_execution_diagnostics()
        recovery_control_applied = False
        v5_release_brake_step = bool(
            self.star_safety_state_machine_version
            in {
                "star_adaptive_recovery_v5", "star_incident_recovery_v5_1",
                "star_incident_recovery_v5_2",
                "star_persistent_recovery_v5_3",
                "star_persistent_recovery_v5_4",
                "star_closure_consistency_audit_v1",
            }
            and self._v5_release_after_brake
        )
        if (
            self.safety_recovery_enabled
            and self.star_safety_state_machine_version
            in {
                "bounded_recovery_v1", "star_scoped_recovery_v4",
                "star_adaptive_recovery_v5", "star_incident_recovery_v5_1",
                "star_incident_recovery_v5_2",
                "star_persistent_recovery_v5_3",
                "star_persistent_recovery_v5_4",
                "star_closure_consistency_audit_v1",
            }
            and self._safety_mode in {"SAFE_BRAKE", "SAFE_RECOVER"}
        ):
            recovery_actions = self._star_active_recovery_actions(
                requested_actions
            )
            if recovery_actions is not None:
                if (
                    self.star_safety_state_machine_version
                    in {
                        "star_scoped_recovery_v4", "star_adaptive_recovery_v5",
                        "star_incident_recovery_v5_1", "star_incident_recovery_v5_2",
                        "star_persistent_recovery_v5_3",
                        "star_persistent_recovery_v5_4",
                        "star_closure_consistency_audit_v1",
                    }
                ):
                    # V4 recovers only implicated agents; unaffected Actor
                    # actions continue through the same joint guard.
                    affected = self._star_recovery_affected_mask
                    values[affected] = recovery_actions[affected]
                else:
                    values = recovery_actions
                recovery_control_applied = True
        self._record_action_stage(
            "recovery_overlay", requested_actions, values
        )
        corridor_input_actions = values.copy()
        goal_corridor_decision = None
        if self.goal_boundary_corridor_enabled:
            goal_corridor_decision = project_goal_boundary_corridor(
                actions=values,
                positions=self.positions,
                velocities=self.velocities,
                goals=self.goals,
                map_size=self.map_size,
                agent_radius=self.agent_radius,
                goal_radius=self.goal_radius,
                max_acceleration=self.agent_max_acceleration,
                dt=self.dt,
                corridor_coordinate=self.goal_boundary_corridor_coordinate,
                boundary_axis=0,
                boundary_side="high",
            )
            values = goal_corridor_decision.actions.copy()
            if goal_corridor_decision.intervention_count:
                self.goal_corridor_projection_count += 1
                self.goal_corridor_agent_projection_count += int(
                    goal_corridor_decision.intervention_count
                )
                self.goal_corridor_projection_steps.append(int(self.step_count))
            self._last_goal_corridor_result = {
                "version": self.goal_boundary_corridor_version,
                "reason": goal_corridor_decision.reason,
                "intervention_count": int(
                    goal_corridor_decision.intervention_count
                ),
                "affected_agent_mask": list(
                    goal_corridor_decision.affected_agent_mask
                ),
                "affected_axes": [
                    list(row) for row in goal_corridor_decision.affected_axes
                ],
            }
        self._record_action_stage(
            "goal_corridor", corridor_input_actions, values
        )
        control_requested_actions = values.copy()
        previous_positions = self.positions.copy()
        previous_velocities = self.velocities.copy()
        previous_obstacle_positions = [item.position.copy() for item in self.obstacles]
        evidence = (
            transition_evidence(self, values)
            if self.evidence_window_steps
            and self.runtime_telemetry_level == "full"
            else None
        )
        previous_distances = self._previous_distances.copy()
        previous_subgoal_distances = (
            previous_distances.copy()
            if np.array_equal(self.subgoals, self.goals)
            else self._previous_subgoal_distances.copy()
        )
        causal_obstacle_snapshot = (
            self._safety_planning_snapshot()
            if self.safety_execution_enabled and self.obstacle_execution_guard_enabled
            else None
        )
        for obstacle in self.obstacles:
            obstacle.step(self.dt, tuple(self.map_size))
        if self.safety_execution_enabled and (
            self.initial_joint_guard_enabled
            or causal_obstacle_snapshot is not None
            or self.final_joint_guard_enabled
        ):
            final_guard_intervened = False
            brake_applied = bool(self._safe_brake_active)
            guard_kwargs = dict(
                map_size=self.map_size,
                agent_radius=self.agent_radius,
                dt=self.dt,
                max_acceleration=self.agent_max_acceleration,
                max_speed=self.agent_max_speed,
                safe_brake=brake_applied,
            )
            if isinstance(self.boundary_guard, JointSafetyGuardV2):
                guard_kwargs["teammate_safety_distance"] = float(
                    self.oracle_config["teammate_safety_distance"]
                )
            initial_guard_result = self._unprojected_joint_guard_result(
                previous_positions, previous_velocities, values
            )
            if self.initial_joint_guard_enabled:
                initial_guard_result = self.boundary_guard.project_actions(
                    previous_positions, previous_velocities, values, **guard_kwargs
                )
                values = initial_guard_result.executed_actions
            self._record_action_stage(
                "initial_joint_guard", control_requested_actions, values
            )
            joint_intervened = bool(np.any(
                np.abs(values - control_requested_actions) > 1e-7
            ))
            self.joint_guard_intervention_count += int(joint_intervened)
            if joint_intervened:
                self.initial_joint_guard_intervention_steps.append(
                    int(self.step_count)
                )

            if causal_obstacle_snapshot is not None:
                obstacle_ttc_input_actions = values.copy()
                obstacle_decision = project_obstacle_actions(
                    actions=values,
                    velocities=previous_velocities,
                    visible_obstacles=causal_obstacle_snapshot.visible_obstacles,
                    agent_radius=self.agent_radius,
                    obstacle_safety_margin=float(
                        self.oracle_config["obstacle_safety_margin"]
                    ),
                    invariant_buffer=self.boundary_guard.invariant_buffer_m,
                    max_acceleration=self.agent_max_acceleration,
                    dt=self.dt,
                    horizon_seconds=self.obstacle_guard_horizon_seconds,
                    warning_ttc_seconds=(
                        self.obstacle_guard_warning_ttc_seconds
                    ),
                    emergency_ttc_seconds=(
                        self.obstacle_guard_emergency_ttc_seconds
                    ),
                )
                affected = np.asarray(
                    obstacle_decision.affected_agent_mask, dtype=bool
                )
                if (
                    np.any(affected)
                    and not obstacle_decision.fallback_required
                    and (
                        self.ttc_headroom_audit_enabled
                        or self.obstacle_execution_guard_version
                        == "causal_ttc_progress_projection_v2"
                    )
                ):
                    audit, v2_actions = self._audit_ttc_headroom_and_select_v2(
                        actor_actions=obstacle_ttc_input_actions,
                        v1_actions=obstacle_decision.actions,
                        affected_agent_mask=affected,
                        visible_obstacles=(
                            causal_obstacle_snapshot.visible_obstacles
                        ),
                    )
                    if audit is not None:
                        self.ttc_headroom_intervention_event_count += 1
                        self.ttc_headroom_positive_event_count += int(
                            audit["safe_progress_headroom"]
                        )
                        if self.ttc_headroom_audit_enabled:
                            self.ttc_headroom_audit_events.append(audit)
                    if self.obstacle_execution_guard_version == (
                        "causal_ttc_progress_projection_v2"
                    ):
                        combined_shadow = self._ttc_candidate_shadow(
                            v2_actions,
                            actor_reference_actions=obstacle_ttc_input_actions,
                            agent_id=int(np.flatnonzero(affected)[0]),
                            horizon_steps=3,
                            visible_obstacles=(
                                causal_obstacle_snapshot.visible_obstacles
                            ),
                        )
                        if combined_shadow["safe"]:
                            obstacle_decision = replace(
                                obstacle_decision,
                                actions=np.asarray(v2_actions, dtype=np.float32),
                                intervention_count=int(np.count_nonzero(
                                    np.any(
                                        np.abs(
                                            np.asarray(v2_actions)
                                            - obstacle_ttc_input_actions
                                        ) > 1e-7,
                                        axis=1,
                                    )
                                )),
                                reason="progress_projected_v2",
                            )
                            self.ttc_v2_high_progress_selection_count += int(
                                audit.get("v2_selected_high_progress_count", 0)
                                if audit is not None else 0
                            )
                self.obstacle_ttc_projection_count += int(
                    obstacle_decision.intervention_count
                )
                if obstacle_decision.intervention_count:
                    self.obstacle_ttc_projection_steps.append(
                        int(self.step_count)
                    )
                affected = np.asarray(
                    obstacle_decision.affected_agent_mask, dtype=bool
                )
                values = obstacle_decision.actions.copy()
                if obstacle_decision.fallback_required:
                    obstacle_fallback_applied = True
                    self.obstacle_ttc_fallback_count += 1
                    recovery_actions = None
                    if self.safety_recovery_enabled:
                        self._enter_star_active_recovery_brake(
                            "obstacle_guard_fallback",
                            affected_agent_mask=affected,
                            entry_sources=("obstacle_guard_fallback",),
                            hazard_keys=tuple(
                                f"agent:{item.agent_id}:track:{item.track_id}"
                                for item in obstacle_decision.risks
                            ),
                        )
                        recovery_actions = self._star_active_recovery_actions(
                            requested_actions
                        )
                    if recovery_actions is not None:
                        values[affected] = recovery_actions[affected]
                        recovery_control_applied = True
                    else:
                        hold = deterministic_hold_actions(
                            previous_velocities,
                            max_acceleration=self.agent_max_acceleration,
                            dt=self.dt,
                        )
                        values[affected] = hold[affected]

                self._record_action_stage(
                    "obstacle_ttc", obstacle_ttc_input_actions, values
                )
                final_guard_input_actions = values.copy()

                # Final pair/boundary validation.  This is the sole bounded
                # revalidation pass after obstacle projection/recovery.
                final_guard_kwargs = dict(guard_kwargs)
                final_guard_kwargs["safe_brake"] = bool(
                    self._safe_brake_active
                )
                pre_final_guard_actions = values.copy()
                guard_result = initial_guard_result
                if self.final_joint_guard_enabled:
                    guard_result = self.boundary_guard.project_actions(
                        previous_positions, previous_velocities, values,
                        **final_guard_kwargs,
                    )
                    values = guard_result.executed_actions
                final_guard_intervened |= bool(np.any(
                    np.abs(values - pre_final_guard_actions) > 1e-7
                ))
                remaining_risks = assess_obstacle_actions(
                    actions=values,
                    velocities=previous_velocities,
                    visible_obstacles=causal_obstacle_snapshot.visible_obstacles,
                    agent_radius=self.agent_radius,
                    obstacle_safety_margin=float(
                        self.oracle_config["obstacle_safety_margin"]
                    ),
                    invariant_buffer=self.boundary_guard.invariant_buffer_m,
                    max_acceleration=self.agent_max_acceleration,
                    dt=self.dt,
                    horizon_seconds=self.obstacle_guard_horizon_seconds,
                    warning_ttc_seconds=(
                        self.obstacle_guard_warning_ttc_seconds
                    ),
                    emergency_ttc_seconds=(
                        self.obstacle_guard_emergency_ttc_seconds
                    ),
                )
                if remaining_risks:
                    obstacle_fallback_applied = True
                    step_final_guard_residual = True
                    unresolved = np.zeros(self.num_agents, dtype=bool)
                    for risk in remaining_risks:
                        unresolved[risk.agent_id] = True
                    if not obstacle_decision.fallback_required:
                        self.obstacle_ttc_fallback_count += 1
                        if self.safety_recovery_enabled:
                            self._enter_star_active_recovery_brake(
                                "obstacle_guard_fallback_after_joint_revalidation",
                                affected_agent_mask=unresolved,
                                entry_sources=("obstacle_guard_fallback",),
                                hazard_keys=tuple(
                                    f"agent:{item.agent_id}:track:{item.track_id}"
                                    for item in remaining_risks
                                ),
                            )
                    hold = deterministic_hold_actions(
                        previous_velocities,
                        max_acceleration=self.agent_max_acceleration,
                        dt=self.dt,
                    )
                    values[unresolved] = hold[unresolved]
                    pre_final_guard_actions = values.copy()
                    if self.final_joint_guard_enabled:
                        guard_result = self.boundary_guard.project_actions(
                            previous_positions, previous_velocities, values,
                            **final_guard_kwargs,
                        )
                        values = guard_result.executed_actions
                    final_guard_intervened |= bool(np.any(
                        np.abs(values - pre_final_guard_actions) > 1e-7
                    ))
                self._record_action_stage(
                    "final_joint_guard", final_guard_input_actions, values
                )
                if obstacle_decision.risks:
                    scoped_tracks = tuple(
                        tuple(tracks) if affected[index] else ()
                        for index, tracks in enumerate(
                            causal_obstacle_snapshot.visible_obstacles
                        )
                    )
                    predicted_velocity = np.clip(
                        previous_velocities
                        + np.clip(values, -1.0, 1.0)
                        * self.agent_max_acceleration * self.dt,
                        -self.agent_max_speed,
                        self.agent_max_speed,
                    )
                    for agent_id in np.flatnonzero(affected):
                        goal_delta = (
                            self.goals[int(agent_id)]
                            - previous_positions[int(agent_id)]
                        )
                        goal_norm = float(np.linalg.norm(goal_delta))
                        if goal_norm <= 1e-12:
                            continue
                        self.obstacle_ttc_post_action_goal_velocity_sum += float(
                            np.dot(
                                predicted_velocity[int(agent_id)],
                                goal_delta / goal_norm,
                            )
                        )
                        self.obstacle_ttc_post_action_goal_velocity_count += 1
                    clearance = minimum_predicted_obstacle_clearance_m(
                        actions=values,
                        velocities=previous_velocities,
                        visible_obstacles=scoped_tracks,
                        agent_radius=self.agent_radius,
                        obstacle_safety_margin=float(
                            self.oracle_config["obstacle_safety_margin"]
                        ),
                        invariant_buffer=self.boundary_guard.invariant_buffer_m,
                        max_acceleration=self.agent_max_acceleration,
                        dt=self.dt,
                        horizon_seconds=self.obstacle_guard_horizon_seconds,
                    )
                    if np.isfinite(clearance):
                        self.obstacle_ttc_post_action_min_clearance_m = min(
                            self.obstacle_ttc_post_action_min_clearance_m,
                            float(clearance),
                        )
                self._last_obstacle_guard_result = {
                    "version": self.obstacle_execution_guard_version,
                    "reason": obstacle_decision.reason,
                    "intervention_count": int(
                        obstacle_decision.intervention_count
                    ),
                    "fallback_required": bool(
                        obstacle_decision.fallback_required or remaining_risks
                    ),
                    "affected_agent_mask": affected.tolist(),
                    "risks": [
                        {
                            "agent_id": int(item.agent_id),
                            "track_id": int(item.track_id),
                            "ttc_seconds": float(item.ttc_seconds),
                            "predicted_clearance_m": float(
                                item.predicted_clearance_m
                            ),
                            "emergency": bool(item.emergency),
                        }
                        for item in obstacle_decision.risks
                    ],
                }
            else:
                guard_result = initial_guard_result
                self._record_action_stage("obstacle_ttc", values, values)
                final_guard_input_actions = values.copy()
                if self.final_joint_guard_enabled:
                    guard_result = self.boundary_guard.project_actions(
                        previous_positions, previous_velocities, values,
                        **guard_kwargs,
                    )
                    values = guard_result.executed_actions
                    final_guard_intervened = bool(np.any(
                        np.abs(values - final_guard_input_actions) > 1e-7
                    ))
                self._record_action_stage(
                    "final_joint_guard", final_guard_input_actions, values
                )
            if final_guard_intervened:
                self.final_joint_guard_residual_count += 1
                self.final_joint_guard_residual_steps.append(int(self.step_count))
            self._last_boundary_guard_result = guard_result.summary()
            self._last_boundary_guard_result["policy_requested_actions"] = (
                requested_actions.astype(float).tolist()
            )
            self._last_boundary_guard_result["control_requested_actions"] = (
                control_requested_actions.astype(float).tolist()
            )
            intervened = bool(np.any(np.abs(values - requested_actions) > 1e-7))
            self.shield_intervention_count += int(intervened)
            self.total_action_modification_count += int(intervened)
            self.shield_invariant_violation_count += int(
                guard_result.invariant_violations
            )
            self.safe_brake_step_count += int(brake_applied)
            if v5_release_brake_step:
                self._v5_release_after_brake = False
                self._safe_recovery_failed = False
                self._safe_brake_active = False
                self._safety_mode = "NORMAL"
                self._safe_recovery_target = None
                self._safe_recovery_start_positions = None
                self._safe_recovery_elapsed_steps = 0
                self._safe_recovery_sequence_steps = 0
                self._safe_recovery_attempt_index = 0
                self._safe_recover_current_streak = 0
                self._recovery_stable_steps = 0
                self._star_recovery_offending_pairs = ()
                self._star_recovery_affected_mask[:] = False
                self._safe_recovery_entry_sources = ()
                self._safe_recovery_previous_margin = None
                self._safe_recovery_no_improvement_steps = 0
                if (
                    self.star_safety_state_machine_version
                    in _INCIDENT_RECOVERY_VERSIONS
                ):
                    if (
                        self.star_safety_state_machine_version
                        in {
                            "star_incident_recovery_v5_2",
                            "star_persistent_recovery_v5_3",
                            "star_persistent_recovery_v5_4",
                            "star_closure_consistency_audit_v1",
                        }
                    ):
                        if self.star_safety_state_machine_version in {
                            "star_persistent_recovery_v5_3",
                            "star_persistent_recovery_v5_4",
                            "star_closure_consistency_audit_v1",
                        }:
                            self._close_v5_3_lineage_incident(
                                "closed_after_bounded_exhaustion"
                            )
                        else:
                            self._close_v5_2_recovery_incident(
                                "closed_after_bounded_failure"
                            )
                    else:
                        self._clear_v5_1_recovery_incident()
            elif (
                self.safety_recovery_enabled
                and self.star_safety_state_machine_version
                == "star_safe_brake_recovery_v2"
            ):
                self._update_star_safety_state(
                    guard_result, brake_applied=brake_applied
                )
            elif (
                self.safety_recovery_enabled
                and self.star_safety_state_machine_version
                == "bounded_recovery_v1"
            ):
                self._update_star_active_recovery_state(
                    guard_result,
                    brake_applied=brake_applied,
                    recovery_control_applied=recovery_control_applied,
                    control_requested_actions=control_requested_actions,
                )
            elif (
                self.safety_recovery_enabled
                and self.star_safety_state_machine_version
                in {
                    "star_scoped_recovery_v4", "star_adaptive_recovery_v5",
                    "star_incident_recovery_v5_1", "star_incident_recovery_v5_2",
                    "star_persistent_recovery_v5_3",
                    "star_persistent_recovery_v5_4",
                    "star_closure_consistency_audit_v1",
                }
            ):
                self._update_star_scoped_recovery_state(
                    guard_result,
                    brake_applied=brake_applied,
                    recovery_control_applied=recovery_control_applied,
                    control_requested_actions=control_requested_actions,
                )
            elif (
                self.safety_recovery_enabled
                and bool(getattr(guard_result, "recovery_required", False))
            ):
                self._safety_mode = "SAFE_BRAKE"
                self._safe_brake_active = True
                self._safe_recovery_target = None
                self._recovery_stable_steps = 0
                self._last_recovery_reason = "joint_guard_infeasible"
        else:
            self._last_boundary_guard_result = {
                "requested_actions": requested_actions.astype(float).tolist(),
                "executed_actions": values.astype(float).tolist(),
                "shielded_axes": [[False, False] for _ in range(self.num_agents)],
                "reasons": [[None, None] for _ in range(self.num_agents)],
                "invariant_violations": 0,
                "safe_brake": False,
                "policy_requested_actions": requested_actions.astype(float).tolist(),
                "control_requested_actions": control_requested_actions.astype(float).tolist(),
            }
            self._record_action_stage(
                "initial_joint_guard", control_requested_actions, values
            )
            self._record_action_stage("obstacle_ttc", values, values)
            self._record_action_stage("final_joint_guard", values, values)
        self._record_action_stage("total", requested_actions, values)
        self._finalize_stage_intervention_trace(requested_actions, values)
        if self.runtime_telemetry_level != "train_minimal":
            self._current_action_stage_actions["executed"] = (
                values.astype(np.float64).copy()
            )
        self._trace_action_stage_snapshot("executed", values)
        if self.closure_trace_audit_enabled and trace_incident_id is not None:
            self.safe_recovery_events.append({
                "event": "closure_action_stack_trace",
                "step": int(self.step_count),
                "incident_id": int(trace_incident_id),
                "lineage_id": self._recovery_lineage_id,
                "hazard_keys": list(trace_hazard_keys),
                "closure_state": self.closure_state,
                "stages": deepcopy(self._current_action_trace),
            })
        # Diagnostic-only capture on an isolated PPCSA clone. In normal runs
        # this hook is absent; the existing execution path is unchanged.
        ppcsa_capture = getattr(self, "_ppcsa_capture_control", None)
        if ppcsa_capture is not None:
            ppcsa_capture(values.copy(), self, {
                "initial_guard": initial_guard_result.summary(),
                "final_guard": deepcopy(self._last_boundary_guard_result),
                "ttc_fallback": bool(obstacle_fallback_applied),
                "safe_brake_applied": bool(brake_applied),
                "recovery_overlay": bool(recovery_control_applied),
            })
        attempted_velocities = np.clip(
            previous_velocities
            + np.clip(values, -1.0, 1.0) * self.agent_max_acceleration * self.dt,
            -self.agent_max_speed,
            self.agent_max_speed,
        )
        attempted_positions = previous_positions + attempted_velocities * self.dt
        boundary_sides = [
            self._boundary_sides(attempted_positions[agent_id])
            for agent_id in range(self.num_agents)
        ]
        boundary_hits = np.zeros(self.num_agents, dtype=bool)
        for agent_id, (agent, action) in enumerate(zip(self.agents, values)):
            boundary_hits[agent_id] = agent.apply_acceleration(
                action,
                self.dt,
                tuple(self.map_size),
                max_acceleration=self.agent_max_acceleration,
            )

        if (
            self.safety_recovery_enabled
            and self.star_safety_state_machine_version
            in {
                "star_scoped_recovery_v4", "star_adaptive_recovery_v5",
                "star_incident_recovery_v5_1", "star_incident_recovery_v5_2",
                "star_persistent_recovery_v5_3",
                "star_persistent_recovery_v5_4",
                "star_closure_consistency_audit_v1",
            }
        ):
            self._finalize_star_scoped_recovery_step(
                previous_positions=previous_positions,
                recovery_control_applied=recovery_control_applied,
            )

        if self.boundary_mode == "clip":
            boundary_hits[:] = False
        if self.collision_mode == "terminal" and self.boundary_mode == "collision":
            # Collision geometry must use the un-clipped attempted segment.
            for agent_id, agent in enumerate(self.agents):
                agent.position = attempted_positions[agent_id].copy()

        collision_events = (
            self._terminal_contacts(previous_positions, previous_obstacle_positions)
            if self.collision_mode == "terminal" else self._detect_collision_events(
                boundary_hits=boundary_hits,
                previous_positions=previous_positions,
                previous_obstacle_positions=previous_obstacle_positions,
            )
        )
        if self.collision_mode == "terminal":
            if collision_events:
                fraction = collision_events[0]["contact_fraction"]
                for agent_id, agent in enumerate(self.agents):
                    agent.position = (previous_positions[agent_id] + fraction *
                                      (agent.position - previous_positions[agent_id])).astype(np.float32)
                for oid, obstacle in enumerate(self.obstacles):
                    obstacle.position = (previous_obstacle_positions[oid] + fraction *
                                         (obstacle.position - previous_obstacle_positions[oid])).astype(np.float32)
                self.terminated = True
                boundary_hits[:] = False
                boundary_sides = [[] for _ in range(self.num_agents)]
                for event in collision_events:
                    if event["type"] == "boundary_collision":
                        boundary_hits[event["agent_ids"]] = True
                        for agent_id in event["agent_ids"]:
                            boundary_sides[agent_id] = list(event["boundary_sides"])
        if evidence is not None:
            evidence.update(attempted_positions=attempted_positions.tolist(),
                            end_positions=self.positions.tolist(),
                            collision_events=deepcopy(collision_events))
            self._transition_window.append(evidence)
            if collision_events and not self.first_failure_window:
                self.first_failure_window = deepcopy(list(self._transition_window))
        detected_keys = {self._collision_event_key(event) for event in collision_events}
        new_collision_events = [
            event
            for event in collision_events
            if self._collision_event_key(event) not in self._active_collision_keys
        ]
        # 只有先产生一个无该接触的 step，事件键才会解除；持续顶住同一物体不重复罚。
        self._active_collision_keys = detected_keys

        colliding_agent_ids = sorted(
            {
                int(agent_id)
                for event in collision_events
                for agent_id in event.get("agent_ids", [])
            }
        )
        rollback_ids = colliding_agent_ids if self.collision_mode == "rollback" else []
        for agent_id in rollback_ids:
            self.agents[agent_id].position = previous_positions[agent_id].copy()
            self.agents[agent_id].velocity = np.zeros(2, dtype=np.float32)

        # 边界碰撞后退回一个可配置的内部安全距离，避免策略在同一墙面连续数百步回滚。
        low = np.full(2, self.agent_radius, dtype=np.float32)
        high = self.map_size - self.agent_radius
        for agent_id in (np.flatnonzero(boundary_hits) if self.collision_mode == "rollback" else []):
            point = self.agents[int(agent_id)].position.copy()
            for side in boundary_sides[int(agent_id)]:
                if side == "left":
                    point[0] = max(point[0], low[0] + self.boundary_recovery_distance)
                elif side == "right":
                    point[0] = min(point[0], high[0] - self.boundary_recovery_distance)
                elif side == "bottom":
                    point[1] = max(point[1], low[1] + self.boundary_recovery_distance)
                elif side == "top":
                    point[1] = min(point[1], high[1] - self.boundary_recovery_distance)
            self.agents[int(agent_id)].position = point.astype(np.float32)

        self._update_boundary_contact_metrics(boundary_hits, boundary_sides)

        # 极少数情况下动态障碍物会在本步移动到 UAV 的旧安全点；同时回滚该障碍物，
        # 保证 UAV 回滚后仍不发生 penetration。
        rolled_back_obstacles: set[int] = set()
        for event in collision_events:
            if self.collision_mode == "terminal":
                break
            if event["type"] != "obstacle_collision":
                continue
            obstacle_id = int(event["obstacle_id"])
            obstacle = self.obstacles[obstacle_id]
            if any(
                np.linalg.norm(self.agents[agent_id].position - obstacle.position)
                < self.agent_radius + obstacle.radius
                for agent_id in event["agent_ids"]
            ):
                obstacle.position = previous_obstacle_positions[obstacle_id].copy()
                rolled_back_obstacles.add(obstacle_id)

        event_step = self.step_count + 1
        for event in new_collision_events:
            self.collision_count += 1
            if event["type"] == "obstacle_collision":
                self.obstacle_collision_count += 1
                obstacle = self.obstacles[int(event["obstacle_id"])]
                if obstacle.dynamic:
                    self.dynamic_obstacle_collision_count += 1
                else:
                    self.static_obstacle_collision_count += 1
            elif event["type"] == "agent_collision":
                self.inter_agent_collision_count += 1
            else:
                self.boundary_collision_count += 1
                for agent_id in event["agent_ids"]:
                    for side in boundary_sides[int(agent_id)]:
                        attribute = f"{side}_boundary_collision_count"
                        setattr(self, attribute, int(getattr(self, attribute)) + 1)
        if new_collision_events:
            self.episode_has_collision = True
            for agent_id in {
                int(item)
                for event in new_collision_events
                for item in event.get("agent_ids", [])
            }:
                self._subgoal_segment_had_collision[agent_id] = True
            if self.first_collision_step is None:
                self.first_collision_step = event_step
                first_event = new_collision_events[0]
                if first_event["type"] == "obstacle_collision":
                    obstacle = self.obstacles[int(first_event["obstacle_id"])]
                    self.first_collision_type = (
                        "dynamic_obstacle_collision"
                        if obstacle.dynamic
                        else "static_obstacle_collision"
                    )
                else:
                    self.first_collision_type = str(first_event["type"])
                # 记录首次碰撞时任务已经完成的比例，便于区分“早期规划失败”
                # 和“接近目标时控制失误”，不改变任何奖励或终止逻辑。
                collision_required_distance = float(
                    np.max(self._goal_distances())
                )
                collision_start_distance = max(
                    self._episode_start_max_goal_distance, 1e-6
                )
                self.first_collision_goal_progress_ratio = float(
                    np.clip(
                        (
                            self._episode_start_max_goal_distance
                            - collision_required_distance
                        )
                        / collision_start_distance,
                        -1.0,
                        1.0,
                    )
                )

        # 必须在回滚后重新计算进展，非法穿透位移不能获得 progress reward。
        distances = self._goal_distances()
        subgoal_distances = self._subgoal_distances()
        in_goal = distances < self.goal_radius
        # 先记录本 step 的终点进入/离开事件，再更新 previous_in_goal。
        # 这些字段用于识别“单机先到后漂出、两机未同步”的失败，不参与控制逻辑。
        entered_goal = in_goal & ~self._previous_in_goal
        exited_goal = ~in_goal & self._previous_in_goal
        first_entries = entered_goal & (self._first_goal_entry_steps < 0)
        self._first_goal_entry_steps[first_entries] = event_step
        self._last_goal_entry_steps[entered_goal] = event_step
        self._goal_exit_counts += exited_goal.astype(np.int32)
        self._in_goal_step_counts += in_goal.astype(np.int32)
        self._ever_in_goal |= in_goal
        reach_success = self.check_success() and not (
            self.collision_mode == "terminal" and collision_events
        )
        safe_success = reach_success and self.collision_count == 0
        self._team_arrival_observed |= reach_success

        team_potential = self._team_goal_potential(distances)
        will_timeout = (
            not self.terminated
            and self.step_count + 1 >= self.max_episode_steps
        )
        terminal_transition = bool(collision_events) or reach_success or will_timeout
        if terminal_transition and self.last_stage_intervention_trace is not None:
            self.last_stage_intervention_trace = replace(
                self.last_stage_intervention_trace,
                records=tuple(
                    replace(record, eligible_for_actor_risk=False)
                    for record in self.last_stage_intervention_trace.records
                ),
                terminal_transition=True,
            )
        shaping_next_potential = 0.0 if terminal_transition else team_potential
        potential_progress_reward = self.potential_progress_scale * (
            self.potential_gamma * shaping_next_potential
            - self._previous_team_potential
        )
        control_effort_penalty = self.control_effort_weight * float(
            np.mean(np.square(values))
        )

        # 旧奖励模式对当前局部目标的接近量给予 progress reward；
        # V3 在下方替换为只依赖最终目标的团队势函数差。
        progress = (previous_subgoal_distances - subgoal_distances).astype(np.float32)
        progress_reward = self.progress_weight * progress
        max_step_distance = max(
            np.sqrt(2.0) * self.agent_max_speed * self.dt, 1e-6
        )
        normalized_subgoal_progress = np.clip(
            progress / max_step_distance, -1.0, 1.0
        ).astype(np.float32)
        team_subgoal_progress = float(
            self.bottleneck_weight * np.min(normalized_subgoal_progress)
            + (1.0 - self.bottleneck_weight)
            * np.mean(normalized_subgoal_progress)
        )
        subgoal_progress_reward = (
            self.subgoal_progress_scale * team_subgoal_progress
        )
        team_goal_completion = self._team_goal_completion(distances)
        new_best_goal_completion = max(
            0.0, team_goal_completion - self._best_team_goal_completion
        )
        historical_final_progress_reward = (
            self.final_progress_scale * new_best_goal_completion
        )
        self._best_team_goal_completion = max(
            self._best_team_goal_completion, team_goal_completion
        )
        goal_bonus = np.zeros(self.num_agents, dtype=np.float32)
        # 协作奖励使用全队最慢成员的正向进展，避免退化为成功奖励的重复副本。
        # 只有所有 UAV 同时朝各自目标前进时该项才为正，并按初始任务距离归一化，
        # 防止默认 cooperation_bonus=5 在每一步压过导航奖励。
        normalized_positive_progress = np.clip(
            progress / max(self._episode_start_max_goal_distance, 1e-6), 0.0, 1.0
        )
        cooperation_value = self.cooperation_bonus * float(
            np.min(normalized_positive_progress)
        )
        cooperation = np.full(
            self.num_agents, cooperation_value, dtype=np.float32
        )
        if self.reward_mode == "team_v3_potential":
            progress_reward[:] = potential_progress_reward
            cooperation_value = 0.0
            cooperation[:] = 0.0
        elif self.reward_mode == "team_v4_subgoal":
            progress_reward[:] = (
                subgoal_progress_reward + historical_final_progress_reward
            )
            cooperation_value = 0.0
            cooperation[:] = 0.0
        proximity_risk_penalty = 0.0
        proximity_risk = self._proximity_risk_snapshot()
        if self.reward_mode in {
            "team_v2_risk", "team_v3_potential", "team_v4_subgoal"
        }:
            proximity_risk_penalty = self.proximity_risk_max_penalty * float(
                proximity_risk["team_risk"]
            )
        obstacle_penalty = np.zeros(self.num_agents, dtype=np.float32)
        inter_agent_penalty = np.zeros(self.num_agents, dtype=np.float32)
        boundary_penalty = np.zeros(self.num_agents, dtype=np.float32)
        reason = "ongoing"
        if collision_events:
            cooperation[:] = 0.0
            reason = str(collision_events[0]["type"])
        # 处罚以新事件为单位累加；持续接触仍会回滚，但不会连续重复扣分。
        for event in new_collision_events:
            ids = np.asarray(event["agent_ids"], dtype=int)
            if event["type"] == "obstacle_collision":
                obstacle_penalty[ids] += self.collision_penalty
            elif event["type"] == "agent_collision":
                inter_agent_penalty[ids] += self.collision_penalty
            else:
                boundary_penalty[ids] += self.collision_penalty

        if reach_success:
            reason = "team_success"
            goal_bonus[:] = self.goal_success_bonus
            self.terminated = True
            if (
                self.safe_recovery_episode_failure
                and self.safe_recovery_abort_then_success_count == 0
            ):
                self.safe_recovery_abort_then_success_count = 1

        rewards = (progress_reward + goal_bonus + cooperation - obstacle_penalty - inter_agent_penalty - boundary_penalty - self.step_penalty).astype(np.float32)
        if self.reward_mode in {
            "team_v1", "team_v2_risk", "team_v3_potential", "team_v4_subgoal"
        }:
            # Navigation is averaged; each collision TYPE is charged once to the team.
            # Broadcast only for the legacy multi-agent environment interface.
            team_collision_penalty = self.collision_penalty * len(
                {event["type"] for event in new_collision_events}
            )
            team_reward = float(np.mean(progress_reward) + np.mean(goal_bonus) +
                                cooperation_value * (not bool(collision_events)) -
                                team_collision_penalty - proximity_risk_penalty -
                                self.step_penalty)
            if self.reward_mode == "team_v3_potential":
                team_reward -= control_effort_penalty
            rewards[:] = team_reward
        if (
            goal_corridor_decision is not None
            and goal_corridor_decision.intervention_count
        ):
            for agent_id, tangential_axis in enumerate(
                goal_corridor_decision.tangential_axes
            ):
                if tangential_axis is None:
                    continue
                before = abs(float(
                    self.goals[agent_id, tangential_axis]
                    - previous_positions[agent_id, tangential_axis]
                ))
                after = abs(float(
                    self.goals[agent_id, tangential_axis]
                    - self.positions[agent_id, tangential_axis]
                ))
                tangential_progress = before - after
                self.goal_corridor_tangential_progress_m += tangential_progress
                self.goal_corridor_nonpositive_progress_count += int(
                    tangential_progress <= 0.0
                )
        self.step_count += 1
        self._obstacle_track_cache_step = None
        self._update_near_goal_stall_metrics(distances, values)
        self._update_obstacle_visibility_metrics()
        timeout_failure_penalty = 0.0
        if not self.terminated and self.step_count >= self.max_episode_steps:
            self.truncated = True
            reason = "timeout"
            timeout_failure_penalty = self.timeout_penalty
            rewards -= timeout_failure_penalty
        if self.terminated or self.truncated:
            if (
                self.star_safety_state_machine_version
                == "star_closure_consistency_audit_v1"
            ):
                self._reconcile_closure_terminal_incident(reason)
            elif self.star_safety_state_machine_version in {
                "star_persistent_recovery_v5_3",
                "star_persistent_recovery_v5_4",
            }:
                self._reconcile_v5_3_terminal_incident(reason)
            elif self.star_safety_state_machine_version == "star_incident_recovery_v5_2":
                self._reconcile_v5_2_terminal_incident(reason)
            self.terminal_recovery_unresolved = bool(
                self.star_safety_state_machine_version
                in _INCIDENT_RECOVERY_VERSIONS
                and (
                    self._recovery_incident_id is not None
                    or self._safety_mode != "NORMAL"
                )
            )
            if self.star_safety_state_machine_version in {
                "star_persistent_recovery_v5_3",
                "star_persistent_recovery_v5_4",
                "star_closure_consistency_audit_v1",
            }:
                recent_lineage = any(
                    0 <= int(self.step_count) - int(item["closed_step"])
                    <= self.safe_recovery_rapid_retrigger_window_steps
                    for item in self._closed_recovery_lineages.values()
                )
                if self.terminal_recovery_unresolved:
                    self._mark_v5_3_lineage_failure("terminal_unresolved")
                elif reason == "collision" and recent_lineage:
                    self._mark_v5_3_lineage_failure(
                        "collision_during_or_after_recovery_lineage"
                    )
                elif reason == "timeout" and recent_lineage:
                    self._mark_v5_3_lineage_failure(
                        "timeout_after_rapid_lineage_retrigger"
                    )
                if reason == "team_success" and not self.recovery_attributable_mission_failure:
                    self.bounded_exhaustion_then_success_count = int(
                        self.bounded_recovery_exhaustion_count
                    )
        self._previous_distances = distances.copy()
        self._previous_in_goal = in_goal.copy()
        self._previous_team_potential = team_potential

        self._update_task_phase_metrics(distances)
        self._subgoal_distance_sum += float(np.sum(subgoal_distances))
        self._subgoal_distance_samples += self.num_agents

        if not self.terminated and not self.truncated:
            replanned = self._maybe_replan_subgoals(subgoal_distances)
            if replanned:
                subgoal_distances = self._subgoal_distances()
        else:
            reached_subgoals = subgoal_distances < self.subgoal_reach_radius
            self._close_subgoal_segments(
                reached=reached_subgoals,
                # episode timeout 时尚未到达的当前 subgoal 也属于一次超时段。
                timed_out=np.full(self.num_agents, self.truncated, dtype=bool)
                & ~reached_subgoals,
            )
            self._subgoal_finalized = True
        self._previous_subgoal_distances = subgoal_distances.copy()

        clearances = self._nearest_obstacle_distances()
        finite = clearances[np.isfinite(clearances)]
        if finite.size:
            self._episode_min_obstacle_clearance = min(self._episode_min_obstacle_clearance, float(np.min(finite)))
        self._episode_min_inter_agent_distance = min(self._episode_min_inter_agent_distance, self._minimum_inter_agent_distance())
        self._previous_obstacle_clearances = clearances.copy()
        required_distance = float(np.max(distances))
        start_distance = max(self._episode_start_max_goal_distance, 1e-6)
        goal_progress_ratio = float(np.clip((self._episode_start_max_goal_distance - required_distance) / start_distance, -1.0, 1.0))
        reward_components = {
            "normalized_progress": progress.copy(), "individual_navigation_reward": progress_reward.copy(),
            "navigation_reward": progress_reward.copy(), "progress_reward": progress_reward.copy(),
            "goal_bonus": goal_bonus.copy(), "cooperation_bonus": cooperation.copy(),
            "cooperation_reward": cooperation.copy(),
            "obstacle_collision_penalty": obstacle_penalty.copy(),
            "inter_agent_collision_penalty": inter_agent_penalty.copy(),
            "boundary_collision_penalty": boundary_penalty.copy(),
            # 保留诊断聚合器已使用的模块名，值对应当前简化奖励中的实际贡献。
            "obstacle_reward": -obstacle_penalty.copy(),
            "inter_agent_reward": -inter_agent_penalty.copy(),
            "boundary_reward": -boundary_penalty.copy(),
            "time_reward": np.full(self.num_agents, -self.step_penalty, dtype=np.float32),
            "timeout_penalty": np.full(
                self.num_agents, timeout_failure_penalty, dtype=np.float32
            ),
            "timeout_reward": np.full(
                self.num_agents, -timeout_failure_penalty, dtype=np.float32
            ),
            "proximity_risk_penalty": np.full(
                self.num_agents, proximity_risk_penalty, dtype=np.float32
            ),
            "proximity_risk_reward": np.full(
                self.num_agents, -proximity_risk_penalty, dtype=np.float32
            ),
            "team_goal_potential": np.full(
                self.num_agents, team_potential, dtype=np.float32
            ),
            "team_goal_completion": np.full(
                self.num_agents, team_goal_completion, dtype=np.float32
            ),
            "normalized_subgoal_progress": normalized_subgoal_progress.copy(),
            "subgoal_progress_reward": np.full(
                self.num_agents,
                subgoal_progress_reward
                if self.reward_mode == "team_v4_subgoal" else 0.0,
                dtype=np.float32,
            ),
            "historical_final_progress_reward": np.full(
                self.num_agents,
                historical_final_progress_reward
                if self.reward_mode == "team_v4_subgoal" else 0.0,
                dtype=np.float32,
            ),
            "potential_progress_reward": np.full(
                self.num_agents,
                potential_progress_reward if self.reward_mode == "team_v3_potential" else 0.0,
                dtype=np.float32,
            ),
            "control_effort_penalty": np.full(
                self.num_agents,
                control_effort_penalty if self.reward_mode == "team_v3_potential" else 0.0,
                dtype=np.float32,
            ),
            "control_effort_reward": np.full(
                self.num_agents,
                -control_effort_penalty if self.reward_mode == "team_v3_potential" else 0.0,
                dtype=np.float32,
            ),
            "terminal_bonus": float(goal_bonus[0]) if reach_success else 0.0,
            "terminal_penalty": float(
                np.mean(obstacle_penalty + inter_agent_penalty + boundary_penalty)
            ) + timeout_failure_penalty,
            "terminal_reward": goal_bonus.copy(),
            "total_reward": rewards.copy(),
        }
        execution_intervention_mask = np.any(
            np.abs(values - requested_actions) > 1e-7, axis=1
        )
        execution_target_eligible = np.full(
            self.num_agents,
            bool(
                not self.terminated
                and not self.truncated
                and not obstacle_fallback_applied
                and not recovery_control_applied
                and not self._safe_brake_active
                and not step_final_guard_residual
            ),
            dtype=bool,
        ) & execution_intervention_mask
        if obstacle_fallback_applied:
            execution_stage = "obstacle_fallback"
        elif recovery_control_applied:
            execution_stage = "recovery"
        elif self._safe_brake_active:
            execution_stage = "safe_brake"
        elif self.obstacle_execution_guard_enabled and (
            self.step_count - 1 in self.obstacle_ttc_projection_steps
        ):
            execution_stage = "ttc_projection"
        elif np.any(execution_intervention_mask):
            execution_stage = "joint_guard"
        else:
            execution_stage = "actor"
        causal_safety_context, current_safety_margin = (
            self._causal_safety_feature_vector()
        )
        info = {
            "reason": reason, "success": reach_success,
            "reach_success": reach_success, "safe_success": safe_success,
            "step": self.step_count,
            "goal": self.goals.copy(), "goals": self.goals.copy(), "subgoals": self.subgoals.copy(),
            "start_center": self.start_center.copy(), "num_agents": self.num_agents,
            "goal_distances": distances.copy(), "required_goal_distance": required_distance,
            "subgoal_distances": subgoal_distances.copy(),
            "agents_in_goal": int(np.count_nonzero(in_goal)), "required_goal_agents": self.num_agents,
            "all_agents_in_goal": bool(np.all(in_goal)), "agents_ever_in_goal": self._ever_in_goal.copy(),
            "first_goal_entry_steps": self._first_goal_entry_steps.copy(),
            "last_goal_entry_steps": self._last_goal_entry_steps.copy(),
            "goal_exit_counts": self._goal_exit_counts.copy(),
            "in_goal_step_counts": self._in_goal_step_counts.copy(),
            "team_arrival_observed": bool(self._team_arrival_observed), "success_hold_count": int(reach_success),
            "goal_hold_steps": 1, "goal_progress_ratio": goal_progress_ratio,
            "episode_start_required_goal_distance": self._episode_start_max_goal_distance,
            "min_obstacle_clearance": float(self._episode_min_obstacle_clearance),
            "min_inter_agent_distance": float(self._episode_min_inter_agent_distance),
            "progress": progress.copy(),
            "is_active": np.full(self.num_agents, not (self.terminated or self.truncated), dtype=bool),
            "boundary_hits": boundary_hits.copy(),
            "collision": collision_events[0] if collision_events else None,
            "collision_events": collision_events,
            "new_collision_events": new_collision_events,
            "collision_count": self.collision_count,
            "obstacle_collision_count": self.obstacle_collision_count,
            "static_obstacle_collision_count": self.static_obstacle_collision_count,
            "dynamic_obstacle_collision_count": self.dynamic_obstacle_collision_count,
            "inter_agent_collision_count": self.inter_agent_collision_count,
            "boundary_collision_count": self.boundary_collision_count,
            "episode_has_collision": self.episode_has_collision,
            "first_collision_step": self.first_collision_step,
            "first_collision_type": self.first_collision_type,
            "first_collision_goal_progress_ratio": self.first_collision_goal_progress_ratio,
            "remaining_goal_progress_at_first_collision": (
                None
                if self.first_collision_goal_progress_ratio is None
                else 1.0 - self.first_collision_goal_progress_ratio
            ),
            "boundary_collision_sides": boundary_sides,
            **self._boundary_metric_snapshot(),
            "oracle_subgoal_enabled": self.oracle_enabled,
            **self._obstacle_tracking_metric_snapshot(),
            **self._near_goal_stall_metric_snapshot(),
            **self._subgoal_metric_snapshot(),
            "rolled_back_agent_ids": rollback_ids,
            "rolled_back_obstacle_ids": sorted(rolled_back_obstacles),
            "static_obstacles": self.static_obstacles, "fixed_obstacle_layout": False,
            "static_obstacle_num": self.static_obstacle_num, "dynamic_obstacle_num": self.dynamic_obstacle_num,
            "nearest_obstacle_distances": clearances.copy(), "visible_obstacle_ids": self._visible_obstacle_ids(),
            **self._obstacle_visibility_metric_snapshot(),
            "proximity_risk": proximity_risk,
            "reward_components": reward_components,
            "requested_actions": requested_actions.copy(),
            "executed_actions": values.copy(),
            "action_stage_actions": {
                stage: stage_actions.copy()
                for stage, stage_actions
                in self._current_action_stage_actions.items()
            },
            "projection_delta_norm": np.linalg.norm(
                values - requested_actions, axis=1
            ).astype(np.float32),
            "intervention_mask": execution_intervention_mask.copy(),
            "execution_target_eligible": execution_target_eligible.copy(),
            "execution_stage": execution_stage,
            "causal_safety_context": causal_safety_context.copy(),
            "current_safety_margin": float(current_safety_margin),
            "stage_intervention_trace": self.last_stage_intervention_trace,
            "obstacle_ttc_fallback_applied": bool(
                obstacle_fallback_applied
            ),
            "recovery_control_applied": bool(recovery_control_applied),
            "safe_brake_active": bool(self._safe_brake_active),
            "final_joint_guard_residual_count_step": int(
                step_final_guard_residual
            ),
            "boundary_guard": deepcopy(self._last_boundary_guard_result),
            "shield_intervention_count": int(self.shield_intervention_count),
            "joint_guard_intervention_count": int(
                self.joint_guard_intervention_count
            ),
            "initial_joint_guard_intervention_steps": list(
                self.initial_joint_guard_intervention_steps
            ),
            "final_joint_guard_residual_count": int(
                self.final_joint_guard_residual_count
            ),
            "final_joint_guard_residual_steps": list(
                self.final_joint_guard_residual_steps
            ),
            "obstacle_ttc_projection_count": int(
                self.obstacle_ttc_projection_count
            ),
            "obstacle_ttc_projection_steps": list(
                self.obstacle_ttc_projection_steps
            ),
            "obstacle_ttc_fallback_count": int(
                self.obstacle_ttc_fallback_count
            ),
            "obstacle_ttc_post_action_goal_velocity_sum": float(
                self.obstacle_ttc_post_action_goal_velocity_sum
            ),
            "obstacle_ttc_post_action_goal_velocity_count": int(
                self.obstacle_ttc_post_action_goal_velocity_count
            ),
            "obstacle_ttc_post_action_min_clearance_m": (
                float(self.obstacle_ttc_post_action_min_clearance_m)
                if np.isfinite(self.obstacle_ttc_post_action_min_clearance_m)
                else None
            ),
            "ttc_headroom_audit_events": list(
                self.ttc_headroom_audit_events
            ),
            "ttc_headroom_intervention_event_count": int(
                self.ttc_headroom_intervention_event_count
            ),
            "ttc_headroom_positive_event_count": int(
                self.ttc_headroom_positive_event_count
            ),
            "ttc_v2_high_progress_selection_count": int(
                self.ttc_v2_high_progress_selection_count
            ),
            "total_action_modification_count": int(
                self.total_action_modification_count
            ),
            "obstacle_execution_guard": deepcopy(
                self._last_obstacle_guard_result
            ),
            "goal_boundary_corridor": deepcopy(
                self._last_goal_corridor_result
            ),
            "goal_corridor_projection_count": int(
                self.goal_corridor_projection_count
            ),
            "goal_corridor_agent_projection_count": int(
                self.goal_corridor_agent_projection_count
            ),
            "goal_corridor_projection_steps": list(
                self.goal_corridor_projection_steps
            ),
            "goal_corridor_tangential_progress_m": float(
                self.goal_corridor_tangential_progress_m
            ),
            "goal_corridor_nonpositive_progress_count": int(
                self.goal_corridor_nonpositive_progress_count
            ),
            "shield_invariant_violation_count": int(
                self.shield_invariant_violation_count
            ),
            "safe_brake_active": bool(self._safe_brake_active),
            "safety_mode": self._safety_mode,
            "safe_brake_step_count": int(self.safe_brake_step_count),
            "safe_brake_entry_count": int(self.safe_brake_entry_count),
            "safe_brake_exit_count": int(self.safe_brake_exit_count),
            "max_safe_brake_streak": int(self.max_safe_brake_streak),
            "pair_infeasible_count": int(self.pair_infeasible_count),
            "pair_infeasible_entry_count": int(
                self.pair_infeasible_entry_count
            ),
            "safe_brake_entry_steps": list(self.safe_brake_entry_steps),
            "safe_brake_exit_steps": list(self.safe_brake_exit_steps),
            "safe_brake_durations": list(self.safe_brake_durations),
            "safe_brake_entry_reasons": list(
                self.safe_brake_entry_reasons
            ),
            "star_safety_state_machine_version": (
                self.star_safety_state_machine_version
            ),
            "star_candidate_feasibility_version": (
                self.star_candidate_feasibility_version
            ),
            "safe_recover_step_count": int(self.safe_recover_step_count),
            "safe_recover_entry_count": int(self.safe_recover_entry_count),
            "safe_recover_exit_count": int(self.safe_recover_exit_count),
            "safe_recovery_failure_count": int(
                self.safe_recovery_failure_count
            ),
            "safe_recovery_unavailable_count": int(
                self.safe_recovery_unavailable_count
            ),
            "safe_recovery_retry_count": int(
                self.safe_recovery_retry_count
            ),
            "safe_recovery_abort_count": int(
                self.safe_recovery_abort_count
            ),
            "safe_recovery_hard_abort_count": int(
                self.safe_recovery_hard_abort_count
            ),
            "safe_recovery_refresh_count": int(
                self.safe_recovery_refresh_count
            ),
            "safe_recovery_reentry_count": int(
                self.safe_recovery_reentry_count
            ),
            "safe_recovery_abort_then_success_count": int(
                self.safe_recovery_abort_then_success_count
            ),
            "safe_recovery_episode_failure": bool(
                self.safe_recovery_episode_failure
            ),
            "safe_recovery_affected_agent_count": int(
                self.safe_recovery_affected_agent_count
            ),
            "safe_recovery_entry_sources": list(
                self._safe_recovery_entry_sources
            ),
            "max_safe_recover_streak": int(self.max_safe_recover_streak),
            "boundary_recovery_trigger_count": int(
                self.boundary_recovery_trigger_count
            ),
            "safe_recovery_reason": self._last_recovery_reason,
            "safe_recovery_durations": list(self.safe_recovery_durations),
            "safe_recovery_events": deepcopy(self.safe_recovery_events),
            "recovery_incident_count": int(self.recovery_incident_count),
            "recovery_incident_merge_count": int(
                self.recovery_incident_merge_count
            ),
            "recovery_incident_reentry_reset_count": int(
                self.recovery_incident_reentry_reset_count
            ),
            "recovery_duplicate_hard_abort_count": int(
                self.recovery_duplicate_hard_abort_count
            ),
            "max_recovery_incident_steps": int(
                self.max_recovery_incident_steps
            ),
            "active_recovery_incident_id": self._recovery_incident_id,
            "active_recovery_incident_refresh_count": int(
                self._recovery_incident_refresh_count
            ),
            "terminal_recovery_unresolved": bool(
                self.terminal_recovery_unresolved
            ),
            "recovery_watchdog_step_count": int(
                self.recovery_watchdog_step_count
            ),
            "max_recovery_direction_unavailable_streak": int(
                self.max_recovery_direction_unavailable_streak
            ),
            "recovery_incident_close_count": int(
                self.recovery_incident_close_count
            ),
            "recovery_incident_reopen_count": int(
                self.recovery_incident_reopen_count
            ),
            "hazard_retrigger_count": int(self.hazard_retrigger_count),
            "same_hazard_reentry_count": int(
                self.same_hazard_reentry_count
            ),
            "hazard_resolution_steps": list(self.hazard_resolution_steps),
            "incident_close_reason_counts": dict(
                self.incident_close_reason_counts
            ),
            "terminal_incident_reconciled_count": int(
                self.terminal_incident_reconciled_count
            ),
            "recovery_telemetry_schema_version": self.recovery_telemetry_schema_version,
            "closure_state": self.closure_state,
            "clearance_pending_steps": int(self.clearance_pending_steps),
            "pending_clearance_reappearance_count": int(
                self.pending_clearance_reappearance_count
            ),
            "post_certificate_retrigger_count": int(
                self.post_certificate_retrigger_count
            ),
            "close_without_certificate_count": int(
                self.close_without_certificate_count
            ),
            "stale_closure_bypass_count": int(
                self.stale_closure_bypass_count
            ),
            "release_certificate_count": int(self.release_certificate_count),
            "release_certificate_id": int(self.release_certificate_id),
            "recovery_lineage_id": self._recovery_lineage_id,
            "recovery_phase": self.recovery_phase,
            "escape_steps": int(self.escape_steps),
            "consolidation_steps": int(self.consolidation_steps),
            "release_shadow_check_passed": bool(self.release_shadow_check_passed),
            "release_shadow_check_count": int(self.release_shadow_check_count),
            "release_shadow_check_failure_count": int(
                self.release_shadow_check_failure_count
            ),
            "release_margin_m": (
                None if not np.isfinite(self.release_margin_m)
                else float(self.release_margin_m)
            ),
            "release_margin_delta_m": (
                None if not np.isfinite(self.release_margin_delta_m)
                else float(self.release_margin_delta_m)
            ),
            "bounded_recovery_exhaustion_count": int(
                self.bounded_recovery_exhaustion_count
            ),
            "bounded_exhaustion_then_success_count": int(
                self.bounded_exhaustion_then_success_count
            ),
            "rapid_same_hazard_retrigger_count": int(
                self.rapid_same_hazard_retrigger_count
            ),
            "recovery_lineage_failure_count": int(
                self.recovery_lineage_failure_count
            ),
            "recovery_attributable_mission_failure": bool(
                self.recovery_attributable_mission_failure
            ),
            "recovery_attributable_failure_reason": (
                self.recovery_attributable_failure_reason
            ),
            "max_recovery_lineage_steps": int(self.max_recovery_lineage_steps),
            "recovery_lineage_budget_reset_count": int(
                self.recovery_lineage_budget_reset_count
            ),
            "recovery_duplicate_failure_record_count": int(
                self.recovery_duplicate_failure_record_count
            ),
            "max_rapid_retriggers_per_lineage": int(
                self.max_rapid_retriggers_per_lineage
            ),
            "recovery_lineage_failure_emitted": bool(
                self._recovery_lineage_failure_emitted
            ),
            "recovery_lineage_terminal_state": self.recovery_lineage_terminal_state,
            "recovery_candidate_selection_count": int(
                self.recovery_candidate_selection_count
            ),
            "recovery_candidate_rejection_count": int(
                self.recovery_candidate_rejection_count
            ),
            "recovery_candidate_rejection_reason_counts": dict(
                self.recovery_candidate_rejection_reason_counts
            ),
            "recovery_candidate_post_projection_norm": (
                None if not np.isfinite(self.recovery_candidate_post_projection_norm)
                else float(self.recovery_candidate_post_projection_norm)
            ),
            "recovery_candidate_predicted_margin_gain": (
                None if not np.isfinite(self.recovery_candidate_predicted_margin_gain)
                else float(self.recovery_candidate_predicted_margin_gain)
            ),
            "recovery_candidate_fixed_point_streak": int(
                self.recovery_candidate_fixed_point_streak
            ),
            "release_shadow_horizon_min_margin": (
                None if not np.isfinite(self.release_shadow_horizon_min_margin)
                else float(self.release_shadow_horizon_min_margin)
            ),
            "release_shadow_horizon_min_ttc": (
                None if not np.isfinite(self.release_shadow_horizon_min_ttc)
                else float(self.release_shadow_horizon_min_ttc)
            ),
            "action_stage_attribution": (
                self._action_stage_attribution_snapshot()
            ),
            "minimum_observed_obstacle_ttc": (
                None if not np.isfinite(self.minimum_observed_obstacle_ttc)
                else float(self.minimum_observed_obstacle_ttc)
            ),
            "obstacle_ttc_risk_step_count": int(
                self.obstacle_ttc_risk_step_count
            ),
            "first_obstacle_visible_steps": [
                {
                    "agent_id": int(agent_id),
                    "obstacle_id": int(obstacle_id),
                    "step": int(step),
                }
                for (agent_id, obstacle_id), step in sorted(
                    self._first_obstacle_visible_steps.items()
                )
            ],
        }
        info["collision_mode"] = self.collision_mode
        info["reward_mode"] = self.reward_mode
        info["runtime_telemetry_level"] = self.runtime_telemetry_level
        info["team_reward"] = float(np.mean(rewards))
        if self.reward_mode in {
            "team_v1", "team_v2_risk", "team_v3_potential", "team_v4_subgoal"
        }:
            for kind, component in (("obstacle_collision", "obstacle"),
                                    ("agent_collision", "inter_agent"),
                                    ("boundary_collision", "boundary")):
                penalty = self.collision_penalty * any(e["type"] == kind for e in new_collision_events)
                reward_components[f"{component}_collision_penalty"] = np.full(self.num_agents, penalty)
                reward_components[f"{component}_reward"] = np.full(self.num_agents, -penalty)
            reward_components["terminal_penalty"] = (
                team_collision_penalty + timeout_failure_penalty
            )
        info["outcome"] = self._outcome(reason, collision_events)
        if (self.terminated or self.truncated) and self.evidence_window_steps:
            info["first_failure_window"] = deepcopy(self.first_failure_window)
            info["terminal_window"] = deepcopy(list(self._transition_window))
        return self.build_local_observations(), rewards, self.terminated, self.truncated, info

    def _outcome(self, reason, events):
        if self.collision_mode == "terminal" and events:
            names = {"obstacle_collision": "FAIL_OBSTACLE", "agent_collision": "FAIL_INTER_UAV",
                     "boundary_collision": "FAIL_BOUNDARY"}
            kinds = {names[event["type"]] for event in events}
            return next(iter(kinds)) if len(kinds) == 1 else "FAIL_MULTIPLE_COLLISIONS"
        if reason == "team_success":
            return "SUCCESS" if not self.episode_has_collision else "UNSAFE_SUCCESS_LEGACY"
        return "FAIL_TIMEOUT" if self.truncated else "ONGOING"

    def _terminal_contacts(self, starts, obstacle_starts):
        """Retain only simultaneous earliest contacts, rather than loop ordering."""
        events = []
        for i, agent in enumerate(self.agents):
            for oid, obstacle in enumerate(self.obstacles):
                fraction = sphere_contact_fraction(
                    starts[i] - obstacle_starts[oid], agent.position - obstacle.position,
                    agent.radius + obstacle.radius)
                if fraction is not None:
                    events.append({"type": "obstacle_collision", "agent_ids": [i],
                                   "obstacle_id": oid, "obstacle_dynamic": bool(obstacle.dynamic),
                                   "obstacle_kind": "dynamic" if obstacle.dynamic else "static",
                                   "contact_fraction": fraction})
            if self.boundary_mode == "collision":
                delta = agent.position - starts[i]
                wall_contacts = []
                for axis in range(2):
                    low, high = self.agent_radius, self.map_size[axis] - self.agent_radius
                    for wall, side, outward in (
                        (low, ("left", "bottom")[axis], delta[axis] < -1e-12),
                        (high, ("right", "top")[axis], delta[axis] > 1e-12),
                    ):
                        outside = starts[i, axis] < low if wall == low else starts[i, axis] > high
                        if outside:
                            wall_contacts.append((0.0, side))
                        elif outward:
                            f = (wall - starts[i, axis]) / delta[axis]
                            if 0 <= f <= 1:
                                wall_contacts.append((float(f), side))
                if wall_contacts:
                    first_wall = min(f for f, _ in wall_contacts)
                    events.append({"type": "boundary_collision", "agent_ids": [i],
                                   "contact_fraction": first_wall,
                                   "boundary_sides": [side for f, side in wall_contacts if abs(f - first_wall) <= 1e-7]})
            for j in range(i + 1, self.num_agents):
                fraction = sphere_contact_fraction(starts[i] - starts[j],
                    agent.position - self.agents[j].position, 2 * self.agent_radius)
                if fraction is not None:
                    events.append({"type": "agent_collision", "agent_ids": [i, j],
                                   "contact_fraction": fraction})
        if not events:
            return []
        first = min(event["contact_fraction"] for event in events)
        return [event for event in events if abs(event["contact_fraction"] - first) <= 1e-7]

    def get_nearest_obstacles(self, agent_id: int) -> list[tuple[int, CircularObstacle, float]]:
        """返回感知范围内最近的 K 个障碍物及其中心距离。"""
        if not 0 <= int(agent_id) < self.num_agents:
            raise IndexError("agent_id out of range")
        agent = self.agents[int(agent_id)]
        candidates = [(oid, obstacle, float(np.linalg.norm(obstacle.position - agent.position))) for oid, obstacle in enumerate(self.obstacles)]
        return sorted((item for item in candidates if item[2] <= self.detection_radius), key=lambda item: item[2])[:self.obstacle_slots]

    def _visible_obstacle_counts_pre_truncation(self) -> list[int]:
        """Count sensed obstacles before the fixed four-slot observation truncation."""
        return [
            sum(
                float(np.linalg.norm(obstacle.position - agent.position))
                <= self.detection_radius
                for obstacle in self.obstacles
            )
            for agent in self.agents
        ]

    def _reset_obstacle_visibility_metrics(self) -> None:
        self._max_visible_obstacles_pre_truncation = 0
        self._visible_obstacle_saturated_agent_steps = 0
        self._visible_obstacle_observation_agent_steps = 0
        self._selected_track_count = 0
        self._selected_velocity_valid_count = 0
        self._visible_dynamic_track_count = 0
        self._selected_dynamic_track_count = 0
        self._update_obstacle_visibility_metrics()

    def _update_obstacle_visibility_metrics(self) -> None:
        counts = self._visible_obstacle_counts_pre_truncation()
        if counts:
            self._max_visible_obstacles_pre_truncation = max(
                self._max_visible_obstacles_pre_truncation, max(counts)
            )
        self._visible_obstacle_saturated_agent_steps += sum(
            count > self.obstacle_slots for count in counts
        )
        self._visible_obstacle_observation_agent_steps += len(counts)
        if self.observation_schema_version == "tracked_obstacle_v2":
            selected = [
                self._tracked_obstacles(agent_id)
                for agent_id in range(self.num_agents)
            ]
            for agent_id, agent in enumerate(self.agents):
                self._selected_track_count += len(selected[agent_id])
                self._selected_velocity_valid_count += sum(
                    bool(item["velocity_valid"]) for item in selected[agent_id]
                )
                visible_dynamic_ids = {
                    obstacle_id for obstacle_id, obstacle in enumerate(self.obstacles)
                    if obstacle.dynamic
                    and float(np.linalg.norm(obstacle.position - agent.position))
                    <= self.detection_radius
                }
                self._visible_dynamic_track_count += len(visible_dynamic_ids)
                self._selected_dynamic_track_count += sum(
                    int(item["obstacle_id"] in visible_dynamic_ids)
                    for item in selected[agent_id]
                )

    def _obstacle_visibility_metric_snapshot(self) -> dict[str, Any]:
        counts = self._visible_obstacle_counts_pre_truncation()
        denominator = max(self._visible_obstacle_observation_agent_steps, 1)
        return {
            "visible_obstacle_counts_pre_truncation": counts,
            "visible_obstacle_truncation_flags": [
                count > self.obstacle_slots for count in counts
            ],
            "max_visible_obstacles_pre_truncation": int(
                self._max_visible_obstacles_pre_truncation
            ),
            "visible_obstacle_truncation_agent_step_rate": float(
                self._visible_obstacle_saturated_agent_steps / denominator
            ),
        }

    def _obstacle_tracking_metric_snapshot(self) -> dict[str, Any]:
        if self.observation_schema_version != "tracked_obstacle_v2":
            return {
                "observation_schema_version": self.observation_schema_version,
                "obstacle_selection_version": self.obstacle_selection_version,
                "obstacle_velocity_estimation_version": self.obstacle_velocity_estimation_version,
                "obstacle_slots": int(self.obstacle_slots),
                "selected_obstacle_counts": [
                    len(self.get_nearest_obstacles(agent_id))
                    for agent_id in range(self.num_agents)
                ],
                "selected_velocity_valid_rate": 0.0,
                "dynamic_obstacle_slot_coverage": None,
            }
        selected = [self._tracked_obstacles(agent_id) for agent_id in range(self.num_agents)]
        flat = [item for rows in selected for item in rows]
        dynamic_visible = []
        dynamic_selected = []
        for agent_id, agent in enumerate(self.agents):
            visible_ids = {
                obstacle_id for obstacle_id, obstacle in enumerate(self.obstacles)
                if obstacle.dynamic
                and float(np.linalg.norm(obstacle.position - agent.position))
                <= self.detection_radius
            }
            dynamic_visible.append(len(visible_ids))
            dynamic_selected.append(sum(
                int(item["obstacle_id"] in visible_ids) for item in selected[agent_id]
            ))
        total_dynamic_visible = sum(dynamic_visible)
        return {
            "observation_schema_version": self.observation_schema_version,
            "obstacle_selection_version": self.obstacle_selection_version,
            "obstacle_velocity_estimation_version": self.obstacle_velocity_estimation_version,
            "obstacle_slots": int(self.obstacle_slots),
            "selected_obstacle_counts": [len(rows) for rows in selected],
            "selected_velocity_valid_rate": float(
                sum(bool(item["velocity_valid"]) for item in flat) / max(len(flat), 1)
            ),
            "visible_dynamic_obstacle_counts": dynamic_visible,
            "selected_dynamic_obstacle_counts": dynamic_selected,
            "dynamic_obstacle_slot_coverage": (
                float(sum(dynamic_selected) / total_dynamic_visible)
                if total_dynamic_visible else None
            ),
            "episode_selected_velocity_valid_rate": float(
                self._selected_velocity_valid_count / max(self._selected_track_count, 1)
            ),
            "episode_dynamic_obstacle_slot_coverage": (
                float(
                    self._selected_dynamic_track_count
                    / self._visible_dynamic_track_count
                )
                if self._visible_dynamic_track_count else None
            ),
        }

    def _update_near_goal_stall_metrics(
        self, distances: np.ndarray, actions: np.ndarray
    ) -> None:
        action_norms = np.linalg.norm(np.asarray(actions, dtype=np.float32), axis=1)
        in_band = (
            (np.asarray(distances) >= self.goal_radius)
            & (np.asarray(distances) <= self.goal_radius + self.near_goal_stall_distance_margin)
            & (action_norms <= self.near_goal_stall_action_threshold)
        )
        self._near_goal_stall_streak = np.where(
            in_band, self._near_goal_stall_streak + 1, 0
        ).astype(np.int32)
        self._max_near_goal_stall_streak = np.maximum(
            self._max_near_goal_stall_streak, self._near_goal_stall_streak
        )
        new_triggers = (
            ~self._near_goal_stall_triggered
            & (self._near_goal_stall_streak >= self.near_goal_stall_patience_steps)
        )
        self._near_goal_stall_first_steps[new_triggers] = int(self.step_count)
        self._near_goal_stall_triggered |= new_triggers

    def _near_goal_stall_metric_snapshot(self) -> dict[str, Any]:
        return {
            "near_goal_stall_triggered": self._near_goal_stall_triggered.copy(),
            "near_goal_stall_first_steps": self._near_goal_stall_first_steps.copy(),
            "near_goal_stall_max_streaks": self._max_near_goal_stall_streak.copy(),
            "near_goal_stall_action_threshold": self.near_goal_stall_action_threshold,
            "near_goal_stall_distance_margin": self.near_goal_stall_distance_margin,
            "near_goal_stall_patience_steps": self.near_goal_stall_patience_steps,
        }

    def check_collision(self, boundary_hits: np.ndarray | None = None) -> dict | None:
        """兼容旧调用：返回当前状态检测到的首个碰撞事件。"""
        events = self._detect_collision_events(boundary_hits=boundary_hits)
        return events[0] if events else None

    def _boundary_sides(self, candidate: np.ndarray) -> list[str]:
        """根据未裁剪候选位置返回实际越界方向。"""
        point = np.asarray(candidate, dtype=np.float32)
        low = np.full(2, self.agent_radius, dtype=np.float32)
        high = self.map_size - self.agent_radius
        sides: list[str] = []
        if point[0] < low[0]:
            sides.append("left")
        if point[0] > high[0]:
            sides.append("right")
        if point[1] < low[1]:
            sides.append("bottom")
        if point[1] > high[1]:
            sides.append("top")
        return sides

    def _update_boundary_contact_metrics(
        self, boundary_hits: np.ndarray, boundary_sides: list[list[str]]
    ) -> None:
        """按接触 step 统计方向和最长连续接触，不受事件去重影响。"""
        hits = np.asarray(boundary_hits, dtype=bool)
        self.boundary_contact_step_count += int(np.count_nonzero(hits))
        for agent_id in range(self.num_agents):
            if hits[agent_id]:
                self._current_boundary_contact_streak[agent_id] += 1
                self.max_boundary_contact_streak = max(
                    self.max_boundary_contact_streak,
                    int(self._current_boundary_contact_streak[agent_id]),
                )
                for side in boundary_sides[agent_id]:
                    attribute = f"{side}_boundary_contact_step_count"
                    setattr(self, attribute, int(getattr(self, attribute)) + 1)
            else:
                self._current_boundary_contact_streak[agent_id] = 0

    def _boundary_metric_snapshot(self) -> dict[str, int]:
        return {
            "left_boundary_collision_count": self.left_boundary_collision_count,
            "right_boundary_collision_count": self.right_boundary_collision_count,
            "top_boundary_collision_count": self.top_boundary_collision_count,
            "bottom_boundary_collision_count": self.bottom_boundary_collision_count,
            "boundary_contact_step_count": self.boundary_contact_step_count,
            "left_boundary_contact_step_count": self.left_boundary_contact_step_count,
            "right_boundary_contact_step_count": self.right_boundary_contact_step_count,
            "top_boundary_contact_step_count": self.top_boundary_contact_step_count,
            "bottom_boundary_contact_step_count": self.bottom_boundary_contact_step_count,
            "max_boundary_contact_streak": self.max_boundary_contact_streak,
        }

    def _detect_collision_events(
        self,
        boundary_hits: np.ndarray | None = None,
        previous_positions: np.ndarray | None = None,
        previous_obstacle_positions: list[np.ndarray] | None = None,
    ) -> list[dict]:
        """返回本步所有碰撞事件，并用扫掠线段阻止高速穿透。"""
        current_positions = self.positions
        starts = (
            current_positions
            if previous_positions is None
            else np.asarray(previous_positions, dtype=np.float32)
        )
        obstacle_starts = (
            [item.position for item in self.obstacles]
            if previous_obstacle_positions is None
            else previous_obstacle_positions
        )
        events: list[dict] = []

        if boundary_hits is not None:
            for agent_id in np.flatnonzero(np.asarray(boundary_hits, dtype=bool)):
                events.append(
                    {"type": "boundary_collision", "agent_ids": [int(agent_id)]}
                )

        # 在相对运动坐标系中检测 UAV 与圆形障碍物的扫掠碰撞。
        for agent_id, agent in enumerate(self.agents):
            for obstacle_id, obstacle in enumerate(self.obstacles):
                relative_start = starts[agent_id] - obstacle_starts[obstacle_id]
                relative_end = agent.position - obstacle.position
                if self._segment_origin_distance(relative_start, relative_end) < (
                    agent.radius + obstacle.radius
                ):
                    events.append(
                        {
                            "type": "obstacle_collision",
                            "agent_ids": [agent_id],
                            "obstacle_id": obstacle_id,
                            "obstacle_dynamic": bool(obstacle.dynamic),
                            "obstacle_kind": "dynamic" if obstacle.dynamic else "static",
                        }
                    )

        # 两架 UAV 采用相对位移线段检测，交叉穿越也会被捕获。
        for left in range(self.num_agents):
            for right in range(left + 1, self.num_agents):
                relative_start = starts[left] - starts[right]
                relative_end = current_positions[left] - current_positions[right]
                if self._segment_origin_distance(relative_start, relative_end) < (
                    2.0 * self.agent_radius
                ):
                    events.append(
                        {
                            "type": "agent_collision",
                            "agent_ids": [left, right],
                        }
                    )
        return events

    @staticmethod
    def _segment_origin_distance(start: np.ndarray, end: np.ndarray) -> float:
        """计算原点到二维线段的最短距离。"""
        start = np.asarray(start, dtype=np.float32)
        end = np.asarray(end, dtype=np.float32)
        delta = end - start
        squared = float(np.dot(delta, delta))
        if squared <= 1e-12:
            return float(np.linalg.norm(start))
        fraction = float(np.clip(-np.dot(start, delta) / squared, 0.0, 1.0))
        return float(np.linalg.norm(start + fraction * delta))

    @staticmethod
    def _collision_event_key(event: Mapping[str, Any]) -> tuple:
        """构造连续接触去重键。"""
        event_type = str(event["type"])
        agent_ids = tuple(sorted(int(item) for item in event.get("agent_ids", [])))
        if event_type == "obstacle_collision":
            return event_type, agent_ids, int(event["obstacle_id"])
        return event_type, agent_ids

    def check_success(self) -> bool:
        """仅当所有 UAV 距离各自最终目标小于 goal_radius 时成功。"""
        return bool(np.all(self._goal_distances() < self.goal_radius))

    def _reset_subgoal_metrics(self) -> None:
        """清零 Oracle 与任务阶段的 episode 级统计。"""
        self._previous_valid_subgoals = None
        self.subgoal_replan_count = 0
        self.early_replan_count = 0
        self._invalid_subgoal_count = 0
        self._subgoal_candidate_count = 0
        self._subgoal_reach_count = 0
        self._subgoal_closed_count = 0
        self._subgoal_timeout_count = 0
        self._subgoal_collision_segment_count = 0
        self._subgoal_steps_sum = 0.0
        self._subgoal_progress_ratio_sum = 0.0
        self._subgoal_distance_sum = 0.0
        self._subgoal_distance_samples = 0
        self._subgoal_segment_start_step = np.zeros(self.num_agents, dtype=np.int32)
        self._subgoal_segment_start_distance = np.zeros(self.num_agents, dtype=np.float32)
        self._subgoal_segment_had_collision = np.zeros(self.num_agents, dtype=bool)
        self._subgoal_finalized = False
        self._episode_max_goal_progress_ratio = 0.0
        self.subgoal_ids = np.zeros(self.num_agents, dtype=np.int32)
        self._last_plan_attempt_step = np.zeros(self.num_agents, dtype=np.int32)
        self._subgoal_arrival_seen = np.zeros(self.num_agents, dtype=bool)
        self._subgoal_started_reached = np.zeros(self.num_agents, dtype=bool)
        self._subgoal_unsafe_seen = np.zeros(self.num_agents, dtype=bool)
        self._final_goal_guidance = np.zeros(self.num_agents, dtype=bool)
        self._subgoal_close_reasons = ["episode_end"] * self.num_agents
        self.subgoal_arrival_event_count = 0
        self.subgoal_assignment_count = self.num_agents
        self.subgoal_plan_attempt_count = 0
        self.oracle_recovery_attempt_count = 0
        self.oracle_recovery_candidate_count = 0
        self.oracle_recovery_success_count = 0
        self.previous_target_hold_count = 0
        self.blocked_final_goal_fallback_count = 0
        self.blocked_primary_segment_count = 0
        self.stagnation_trigger_count = 0
        self.repeated_direction_rejection_count = 0
        self._stagnation_counts = np.zeros(self.num_agents, dtype=np.int32)
        self._force_recovery_mask = np.zeros(self.num_agents, dtype=bool)
        self._failed_direction_buckets = [dict() for _ in range(self.num_agents)]
        self._recovery_hold_until_step = np.zeros(self.num_agents, dtype=np.int32)

    def _subgoal_distances(self) -> np.ndarray:
        return np.linalg.norm(self.positions - self.subgoals, axis=1).astype(np.float32)

    def _is_subgoal_valid(
        self,
        agent_id: int,
        subgoal: np.ndarray,
        *,
        dynamic_only: bool = False,
        enforce_min_distance: bool = True,
    ) -> bool:
        """保留布尔接口；拒绝原因与实际判定共用一份实现，避免审计口径漂移。"""
        return not self._subgoal_rejection_reasons(
            agent_id, subgoal, dynamic_only=dynamic_only,
            enforce_min_distance=enforce_min_distance,
        )

    def _subgoal_rejection_reasons(
        self, agent_id: int, subgoal: np.ndarray, *,
        dynamic_only: bool = False, enforce_min_distance: bool = True,
        observable_only: bool = False,
    ) -> list[str]:
        """只读几何检查：多原因可同时出现，不改变原来的合法性规则。"""
        point = np.asarray(subgoal, dtype=np.float32)
        if point.shape != (2,) or not np.isfinite(point).all():
            return ["nonfinite_or_shape"]
        reasons = []
        low = np.full(2, self.agent_radius, dtype=np.float32)
        high = self.map_size - self.agent_radius
        if np.any(point < low) or np.any(point > high):
            reasons.append("boundary_margin")
        if enforce_min_distance:
            distance = float(np.linalg.norm(point - self.agents[agent_id].position))
            is_final_goal = float(np.linalg.norm(point - self.goals[agent_id])) <= 1e-5
            if distance < self.min_subgoal_distance and not is_final_goal:
                reasons.append("min_subgoal_distance")
        margin = float(self.oracle_config["obstacle_safety_margin"])
        obstacles = (
            [item[1] for item in self.get_nearest_obstacles(int(agent_id))]
            if observable_only else self.obstacles
        )
        for obstacle in obstacles:
            if dynamic_only and not obstacle.dynamic:
                continue
            required = self.agent_radius + obstacle.radius + margin
            if float(np.linalg.norm(point - obstacle.position)) < required:
                reason = "dynamic_obstacle_margin" if obstacle.dynamic else "static_obstacle_margin"
                if reason not in reasons:
                    reasons.append(reason)
        return reasons

    def _subgoals_separated(self, subgoals: np.ndarray) -> bool:
        minimum = float(self.oracle_config["teammate_safety_distance"])
        return all(
            float(np.linalg.norm(subgoals[left] - subgoals[right])) >= minimum
            for left in range(self.num_agents)
            for right in range(left + 1, self.num_agents)
        )

    def _segment_obstacle_clearance(
        self, agent_id: int, target: np.ndarray
    ) -> float:
        """返回当前位置到候选端点的最小安全净空；负值表示线段穿过安全区。"""
        start = self.positions[int(agent_id)]
        margin = float(self.oracle_config["obstacle_safety_margin"])
        clearance = float("inf")
        for obstacle in self.obstacles:
            center_distance = self._segment_origin_distance(
                start - obstacle.position,
                np.asarray(target, dtype=np.float32) - obstacle.position,
            )
            clearance = min(
                clearance,
                center_distance - obstacle.radius - self.agent_radius - margin,
            )
        return clearance

    def _visible_segment_obstacle_clearance(
        self, agent_id: int, target: np.ndarray
    ) -> float:
        """D1 path check restricted to the current 25 m observable obstacle set."""
        start = self.positions[int(agent_id)]
        margin = float(self.oracle_config["obstacle_safety_margin"])
        clearance = float("inf")
        for _, obstacle, _ in self.get_nearest_obstacles(int(agent_id)):
            center_distance = self._segment_origin_distance(
                start - obstacle.position,
                np.asarray(target, dtype=np.float32) - obstacle.position,
            )
            clearance = min(
                clearance,
                center_distance - obstacle.radius - self.agent_radius - margin,
            )
        return clearance

    def _subgoal_pair_valid_for_agent(
        self, agent_id: int, candidate: np.ndarray, selected: np.ndarray
    ) -> bool:
        """检查单个替代端点与当前其他 agent 引导目标的间距。"""
        minimum = float(self.oracle_config["teammate_safety_distance"])
        return all(
            other_id == int(agent_id)
            or float(np.linalg.norm(candidate - selected[other_id])) >= minimum
            for other_id in range(self.num_agents)
        )

    def _search_recovery_subgoal(
        self, agent_id: int, selected: np.ndarray
    ) -> dict[str, Any] | None:
        """主候选无效时，在最终目标方向周围搜索有限个安全替代端点。

        搜索空间、排序权重全部来自 YAML。这里只恢复局部引导，不修改势场公式。
        """
        position = self.positions[int(agent_id)]
        goal_delta = self.goals[int(agent_id)] - position
        goal_distance = float(np.linalg.norm(goal_delta))
        if goal_distance <= 1e-6:
            return None
        goal_direction = goal_delta / goal_distance
        options: list[dict[str, Any]] = []
        for angle_degrees in self.oracle_recovery_angles:
            angle_bucket = int(round(float(angle_degrees) / 30.0) * 30)
            if self._failed_direction_buckets[int(agent_id)].get(angle_bucket, 0) > 0:
                self.repeated_direction_rejection_count += 1
                continue
            radians = np.deg2rad(float(angle_degrees))
            rotation = np.asarray(
                [[np.cos(radians), -np.sin(radians)],
                 [np.sin(radians), np.cos(radians)]],
                dtype=np.float32,
            )
            direction = rotation @ goal_direction
            for configured_distance in self.oracle_recovery_distances:
                self.oracle_recovery_candidate_count += 1
                travel = min(float(configured_distance), goal_distance)
                candidate = (position + travel * direction).astype(np.float32)
                if self._subgoal_rejection_reasons(
                    int(agent_id), candidate,
                    observable_only=self.require_primary_segment_clearance,
                ):
                    continue
                if not self._subgoal_pair_valid_for_agent(
                    int(agent_id), candidate, selected
                ):
                    continue
                clearance = (
                    self._visible_segment_obstacle_clearance(int(agent_id), candidate)
                    if self.require_primary_segment_clearance
                    else self._segment_obstacle_clearance(int(agent_id), candidate)
                )
                if clearance < 0.0:
                    continue
                final_distance = float(
                    np.linalg.norm(candidate - self.goals[int(agent_id)])
                )
                goal_progress = goal_distance - final_distance
                finite_clearance = (
                    min(clearance, self.oracle_planner.sensor_range)
                    if np.isfinite(clearance)
                    else self.oracle_planner.sensor_range
                )
                score = (
                    self.oracle_recovery_goal_weight * goal_progress
                    + self.oracle_recovery_clearance_weight * finite_clearance
                    - self.oracle_recovery_turn_weight * abs(float(angle_degrees))
                )
                options.append(
                    {
                        "target": candidate,
                        "angle_degrees": float(angle_degrees),
                        "lookahead_distance": travel,
                        "clearance": float(clearance),
                        "goal_progress": goal_progress,
                        "score": float(score),
                    }
                )
        return max(options, key=lambda item: item["score"]) if options else None

    def _v3_1_recovery_pair(self) -> np.ndarray | None:
        """Find a deterministic full-direction recovery pair.

        This search is deliberately separate from the normal goal-biased Rule
        candidates.  It is used only after the normal guarded set is empty.
        """
        if self.num_agents != 2:
            return None
        per_agent: list[list[tuple[float, str, np.ndarray]]] = []
        for agent_id in range(self.num_agents):
            position = self.positions[agent_id]
            goal_delta = self.goals[agent_id] - position
            goal_distance = float(np.linalg.norm(goal_delta))
            base_direction = (
                np.asarray([1.0, 0.0], dtype=np.float32)
                if goal_distance <= 1e-8 else goal_delta / goal_distance
            )
            options: list[tuple[float, str, np.ndarray]] = []
            for angle_degrees in self.safe_recovery_angles:
                radians = np.deg2rad(angle_degrees)
                rotation = np.asarray([
                    [np.cos(radians), -np.sin(radians)],
                    [np.sin(radians), np.cos(radians)],
                ], dtype=np.float32)
                direction = rotation @ base_direction
                for distance in self.safe_recovery_distances:
                    target = (position + float(distance) * direction).astype(np.float32)
                    if not self._v3_1_recovery_path_safe(agent_id, target):
                        continue
                    start_surface = self.boundary_guard.surface_clearances(
                        position, self.map_size, self.agent_radius
                    )
                    endpoint_surface = self.boundary_guard.surface_clearances(
                        target, self.map_size, self.agent_radius
                    )
                    endpoint_clearance = float(np.min(endpoint_surface))
                    segment_clearance = float(
                        min(np.min(start_surface), endpoint_clearance)
                    )
                    obstacle_clearance = self._visible_segment_obstacle_clearance(
                        agent_id, target
                    )
                    progress = goal_distance - float(
                        np.linalg.norm(self.goals[agent_id] - target)
                    )
                    finite_obstacle = (
                        self.detection_radius
                        if not np.isfinite(obstacle_clearance)
                        else min(obstacle_clearance, self.detection_radius)
                    )
                    score = (
                        min(endpoint_clearance, segment_clearance)
                        + 0.25 * finite_obstacle
                        + 0.05 * progress
                    )
                    key = f"a{agent_id}:ang{angle_degrees:g}:d{distance:g}"
                    options.append((float(score), key, target))
            options.sort(key=lambda row: (-row[0], row[1]))
            per_agent.append(options[: self.safe_recovery_beam_width])
        if any(not rows for rows in per_agent):
            return None

        choices: list[tuple[float, str, np.ndarray]] = []
        safety_distance = float(self.oracle_config["teammate_safety_distance"])
        for left, right in product(per_agent[0], per_agent[1]):
            targets = np.stack([left[2], right[2]]).astype(np.float32)
            endpoint_distance = float(np.linalg.norm(targets[0] - targets[1]))
            synchronized = synchronized_minimum_distance(
                self.positions[0], targets[0], self.positions[1], targets[1]
            )
            if min(endpoint_distance, synchronized) < safety_distance:
                continue
            identifier = f"{left[1]}|{right[1]}"
            choices.append((
                min(left[0], right[0]) + 0.01 * (left[0] + right[0]),
                identifier,
                targets,
            ))
        if not choices:
            return None
        choices.sort(key=lambda row: (-row[0], row[1]))
        return choices[0][2]

    def _v3_1_recovery_path_safe(
        self, agent_id: int, target: np.ndarray
    ) -> bool:
        """Allow monotone escape from an already-invalid recovery start.

        Ordinary admission requires the complete segment, including its start,
        to be safe.  That is correct in NORMAL mode but makes recovery from an
        existing obstacle/boundary margin violation mathematically impossible.
        Recovery therefore admits only directions that strictly increase every
        currently violated clearance and end inside the normal safe set.
        """
        start = np.asarray(self.positions[agent_id], dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        direction = target - start
        if float(np.linalg.norm(direction)) < self.min_subgoal_distance - 1e-5:
            return False

        start_surface = self.boundary_guard.surface_clearances(
            start, self.map_size, self.agent_radius
        )
        endpoint_surface = self.boundary_guard.surface_clearances(
            target, self.map_size, self.agent_radius
        )
        required_wall = self.boundary_guard.surface_clearance_m
        if float(np.min(endpoint_surface)) < required_wall:
            return False
        for before, after in zip(start_surface, endpoint_surface):
            if before < required_wall and after <= before + 1e-5:
                return False

        margin = float(self.oracle_config["obstacle_safety_margin"])
        for _, obstacle, _ in self.get_nearest_obstacles(agent_id):
            required = obstacle.radius + self.agent_radius + margin
            start_distance = float(np.linalg.norm(start - obstacle.position)) - required
            endpoint_distance = float(np.linalg.norm(target - obstacle.position)) - required
            if start_distance < 0.0:
                away = float(np.dot(direction, start - obstacle.position))
                if endpoint_distance < 0.0 or away <= 1e-6:
                    return False
            else:
                segment_distance = self._segment_origin_distance(
                    start - obstacle.position, target - obstacle.position
                ) - required
                if segment_distance < 0.0:
                    return False
        return True

    def _recovery_invariant_ready(self) -> bool:
        """Whether current velocity can be stopped inside boundary/pair margins."""
        for agent_id in range(self.num_agents):
            clearances = self.boundary_guard.surface_clearances(
                self.positions[agent_id], self.map_size, self.agent_radius
            )
            for axis in range(2):
                velocity = float(self.velocities[agent_id, axis])
                distance = clearances[2 * axis + (1 if velocity > 0.0 else 0)]
                toward = abs(velocity)
                available = (
                    distance - self.boundary_guard.surface_clearance_m
                    - self.boundary_guard.invariant_buffer_m
                )
                if available + 1e-9 < self.boundary_guard.stopping_distance(
                    toward, dt=self.dt,
                    max_acceleration=self.agent_max_acceleration,
                ):
                    return False
        protected = (
            float(self.oracle_config["teammate_safety_distance"])
            + float(getattr(self.boundary_guard, "pair_invariant_buffer_m", 0.0))
        )
        for left in range(self.num_agents):
            for right in range(left + 1, self.num_agents):
                relative = self.positions[left] - self.positions[right]
                distance = float(np.linalg.norm(relative))
                if distance <= 1e-9:
                    return False
                normal = relative / distance
                relative_velocity = self.velocities[left] - self.velocities[right]
                closing = max(0.0, -float(np.dot(relative_velocity, normal)))
                braking = self.boundary_guard.stopping_distance(
                    closing, dt=self.dt,
                    max_acceleration=2.0 * self.agent_max_acceleration,
                )
                if distance - braking < protected - 1e-9:
                    return False
        return True

    def _safe_recovery_actions(self) -> np.ndarray:
        if self._safe_recovery_target is None:
            return np.zeros((self.num_agents, 2), dtype=np.float32)
        actions = np.zeros((self.num_agents, 2), dtype=np.float32)
        for agent_id in range(self.num_agents):
            delta = self._safe_recovery_target[agent_id] - self.positions[agent_id]
            distance = float(np.linalg.norm(delta))
            direction = np.zeros(2, dtype=np.float32) if distance <= 1e-8 else delta / distance
            desired_speed = self.agent_max_speed * min(1.0, distance / 3.0)
            desired_velocity = desired_speed * direction
            acceleration = (desired_velocity - self.velocities[agent_id]) / self.dt
            actions[agent_id] = np.clip(
                acceleration / self.agent_max_acceleration, -1.0, 1.0
            )
        return actions

    def _final_goal_fallback_is_safe(self, agent_id: int) -> bool:
        """远距离 final-goal fallback 仅在线段不穿越障碍安全区时允许。"""
        if not self._is_subgoal_valid(
            int(agent_id), self.goals[int(agent_id)], enforce_min_distance=False
        ):
            return False
        if not self.final_goal_fallback_requires_clear_path:
            return True
        return self._segment_obstacle_clearance(
            int(agent_id), self.goals[int(agent_id)]
        ) >= 0.0


    def _generate_valid_subgoals(self, agent_mask=None) -> np.ndarray:
        """规划并验证候选；无效时按 previous-valid、final-goal 顺序回退。"""
        if self.subgoal_lifecycle_version == "per_agent_v2":
            return self._generate_individual_subgoals(agent_mask)
        candidates = self.oracle_planner.plan(
            positions=self.positions,
            velocities=self.velocities,
            final_goals=self.goals,
            obstacles=self.obstacles,
            teammate_positions=self.positions,
            world_boundary=self.map_size,
        )
        if candidates.shape != (self.num_agents, 2):
            candidates = np.full((self.num_agents, 2), np.nan, dtype=np.float32)
        reasons = [self._subgoal_rejection_reasons(i, candidates[i]) for i in range(self.num_agents)]
        valid = np.asarray([not item for item in reasons], dtype=bool)
        if valid.all() and not self._subgoals_separated(candidates):
            valid[:] = False
            for item in reasons:
                item.append("candidate_pair_separation")

        self._subgoal_candidate_count += self.num_agents
        self._invalid_subgoal_count += int(np.count_nonzero(~valid))
        selected = candidates.copy()
        sources = ["oracle_candidate"] * self.num_agents
        for agent_id in np.flatnonzero(~valid):
            fallback = None
            if self._previous_valid_subgoals is not None:
                previous = self._previous_valid_subgoals[agent_id]
                if self._is_subgoal_valid(
                    int(agent_id), previous, enforce_min_distance=False
                ):
                    fallback = previous
            selected[agent_id] = (
                self.goals[agent_id] if fallback is None else fallback
            )
            sources[agent_id] = "final_goal_fallback" if fallback is None else "previous_valid_fallback"

        # 独立 fallback 后仍冲突时统一回到天然分离的最终目标，不终止 episode。
        pair_fallback = not self._subgoals_separated(selected)
        if pair_fallback:
            selected = self.goals.copy()
            sources = ["pair_final_goal_fallback"] * self.num_agents
        # 仅保留最近一次选择元数据；逐段文件由独立审计入口生成，训练不写大日志。
        self.last_subgoal_selection = {
            "candidates": candidates.copy(), "rejection_reasons": reasons,
            "candidate_valid": valid.copy(), "selected_sources": sources,
            "pair_fallback": pair_fallback,
        }
        return selected.astype(np.float32)

    def _generate_individual_subgoals(self, agent_mask=None) -> np.ndarray:
        """只应用请求更新的 agent；联合检查新目标与未更新队友目标的间距。"""
        requested = np.ones(self.num_agents, dtype=bool) if agent_mask is None else np.asarray(agent_mask, dtype=bool)
        candidates = np.asarray(self.oracle_planner.plan(
            positions=self.positions, velocities=self.velocities, final_goals=self.goals,
            obstacles=self.obstacles, teammate_positions=self.positions, world_boundary=self.map_size,
        ), dtype=np.float32)
        if candidates.shape != (self.num_agents, 2):
            candidates = np.full((self.num_agents, 2), np.nan, dtype=np.float32)
        self._final_goal_guidance |= self._goal_distances() < self.goal_radius
        # 到过终点范围后引导目标固定为 final goal；不冻结动作/速度，不改变成功判定。
        candidates[self._final_goal_guidance] = self.goals[self._final_goal_guidance]
        selected = self.subgoals.copy()
        reasons = [[] for _ in range(self.num_agents)]
        valid = np.zeros(self.num_agents, dtype=bool)
        sources = ["unchanged_peer"] * self.num_agents
        recovery_details: list[dict[str, Any] | None] = [None] * self.num_agents
        for i in np.flatnonzero(requested):
            reasons[i] = self._subgoal_rejection_reasons(
                int(i), candidates[i],
                observable_only=self.require_primary_segment_clearance,
            )
            if (
                not reasons[i]
                and self.require_primary_segment_clearance
                and self._visible_segment_obstacle_clearance(int(i), candidates[i]) < 0.0
            ):
                reasons[i].append("blocked_primary_segment")
                self.blocked_primary_segment_count += 1
            valid[i] = not reasons[i]
            force_recovery = bool(self._force_recovery_mask[i])
            if valid[i] and not force_recovery:
                selected[i] = candidates[i]
                sources[i] = "final_goal_guidance" if self._final_goal_guidance[i] else "oracle_candidate"
            else:
                recovery = None
                if self.oracle_recovery_enabled and not self._final_goal_guidance[i]:
                    self.oracle_recovery_attempt_count += 1
                    recovery = self._search_recovery_subgoal(int(i), selected)
                if recovery is not None:
                    selected[i] = recovery["target"]
                    sources[i] = "oracle_recovery"
                    recovery_details[i] = {
                        key: value for key, value in recovery.items() if key != "target"
                    }
                    self.oracle_recovery_success_count += 1
                    self._recovery_hold_until_step[i] = (
                        self.step_count + self.recovery_hold_cycles * self.replanning_interval
                    )
                    continue
                previous = None if self._previous_valid_subgoals is None else self._previous_valid_subgoals[i]
                if previous is not None and self._is_subgoal_valid(int(i), previous, enforce_min_distance=False):
                    selected[i], sources[i] = previous, "previous_valid_fallback"
                elif self._final_goal_fallback_is_safe(int(i)):
                    selected[i], sources[i] = self.goals[i], "final_goal_fallback"
                else:
                    if (
                        self.final_goal_fallback_requires_clear_path
                        and self._is_subgoal_valid(
                            int(i), self.goals[i], enforce_min_distance=False
                        )
                    ):
                        self.blocked_final_goal_fallback_count += 1
                    # reset 时旧目标就是 final goal；若其直线被挡，则使用当前位置安全等待，
                    # 避免以 hold 名义重新引入被禁止的远距离 fallback。
                    if self._previous_valid_subgoals is None:
                        selected[i] = self.positions[i]
                        sources[i] = "safe_position_hold"
                    else:
                        sources[i] = "no_valid_fallback_hold"
        pair_fallback = False
        # 回退可能改变与第三架的距离，循环到没有可撤销更新；不偷偷全队换成 final goal。
        for _ in range(self.num_agents + 1):
            conflicts = set()
            for left in range(self.num_agents):
                for right in range(left + 1, self.num_agents):
                    if np.linalg.norm(selected[left] - selected[right]) < float(self.oracle_config["teammate_safety_distance"]):
                        conflicts.update((left, right))
            rollback = [i for i in conflicts if requested[i] and not np.array_equal(selected[i], self.subgoals[i])]
            if not rollback:
                break
            pair_fallback = True
            for i in rollback:
                if "candidate_pair_separation" not in reasons[i]:
                    reasons[i].append("candidate_pair_separation")
                valid[i] = False
                if sources[i] == "oracle_recovery":
                    self.oracle_recovery_success_count -= 1
                    recovery_details[i] = None
                selected[i] = self.subgoals[i]
                sources[i] = "pair_previous_target_hold"
        self.previous_target_hold_count += sum(
            sources[i] in (
                "previous_valid_fallback",
                "no_valid_fallback_hold",
                "pair_previous_target_hold",
            )
            for i in np.flatnonzero(requested)
        )
        self._subgoal_candidate_count += int(np.count_nonzero(requested))
        self.subgoal_plan_attempt_count += int(np.count_nonzero(requested))
        self._invalid_subgoal_count += int(np.count_nonzero(requested & ~valid))
        self.last_subgoal_selection = {
            "candidates": candidates.copy(), "rejection_reasons": reasons,
            "candidate_valid": valid, "selected_sources": sources,
            "pair_fallback": pair_fallback, "requested_agents": requested.copy(),
            "recovery_details": recovery_details,
        }
        return selected.astype(np.float32)

    def _close_subgoal_segments(
        self, reached: np.ndarray, timed_out: bool | np.ndarray, agent_mask=None
    ) -> None:
        """结算当前每个 agent 的 subgoal 段，供 episode 指标聚合。"""
        if self._subgoal_finalized:
            return
        reached = np.asarray(reached, dtype=bool)
        timed = np.broadcast_to(np.asarray(timed_out, dtype=bool), (self.num_agents,))
        mask = np.ones(self.num_agents, dtype=bool) if agent_mask is None else np.asarray(agent_mask, dtype=bool)
        if self.subgoal_lifecycle_version == "per_agent_v2":
            if agent_mask is None:
                # terminal step 不经过 maybe_replan，最后一步到达也必须只记录一次。
                self.subgoal_arrival_event_count += int(np.count_nonzero(reached & ~self._subgoal_arrival_seen))
                self._subgoal_arrival_seen |= reached
            reached = (reached | self._subgoal_arrival_seen) & ~self._subgoal_started_reached
            # 曾经到达的唯一目标不能因之后漂离而变为超时；holding段也不算跟踪超时。
            timed = timed & ~self._subgoal_arrival_seen & ~self._subgoal_started_reached
        end_distances = self._subgoal_distances()
        durations = self.step_count - self._subgoal_segment_start_step
        ratios = (
            self._subgoal_segment_start_distance - end_distances
        ) / np.maximum(self._subgoal_segment_start_distance, 1e-6)
        self._subgoal_reach_count += int(np.count_nonzero(reached & mask))
        self._subgoal_timeout_count += int(np.count_nonzero(timed & mask))
        self._subgoal_collision_segment_count += int(
            np.count_nonzero(self._subgoal_segment_had_collision & mask)
        )
        self._subgoal_closed_count += int(np.count_nonzero(mask))
        self._subgoal_steps_sum += float(np.sum(durations[mask]))
        self._subgoal_progress_ratio_sum += float(np.sum(np.clip(ratios[mask], -1.0, 1.0)))

    def _maybe_replan_subgoals(self, subgoal_distances: np.ndarray) -> bool:
        """按固定周期重规划，并允许到达或动态障碍风险触发提前重规划。"""
        if not self.oracle_enabled:
            return False
        if self.subgoal_lifecycle_version == "per_agent_v2":
            return self._maybe_replan_individual_subgoals(subgoal_distances)
        reached = np.asarray(subgoal_distances) < self.subgoal_reach_radius
        unsafe = np.asarray(
            [
                not self._is_subgoal_valid(
                    agent_id,
                    self.subgoals[agent_id],
                    dynamic_only=True,
                    enforce_min_distance=False,
                )
                for agent_id in range(self.num_agents)
            ],
            dtype=bool,
        )
        scheduled = self.step_count % self.replanning_interval == 0
        if not scheduled and not reached.any() and not unsafe.any():
            return False

        self._close_subgoal_segments(
            reached=reached,
            timed_out=np.full(self.num_agents, scheduled, dtype=bool) & ~reached,
        )
        self.subgoal_replan_count += 1
        if not scheduled and (reached.any() or unsafe.any()):
            self.early_replan_count += 1
        self.subgoals = self._generate_valid_subgoals()
        self._previous_valid_subgoals = self.subgoals.copy()
        self._subgoal_segment_start_step[:] = self.step_count
        self._subgoal_segment_start_distance = self._subgoal_distances()
        self._subgoal_segment_had_collision[:] = False
        return True

    def _maybe_replan_individual_subgoals(self, distances):
        """一次到达事件、独立尝试时钟、仅在目标真正改变时关闭该agent的逻辑段。"""
        reached = np.asarray(distances) < self.subgoal_reach_radius
        newly_reached = reached & ~self._subgoal_arrival_seen
        self.subgoal_arrival_event_count += int(np.count_nonzero(newly_reached))
        self._subgoal_arrival_seen |= reached
        unsafe = np.asarray([not self._is_subgoal_valid(i, self.subgoals[i], dynamic_only=True,
                                                     enforce_min_distance=False) for i in range(self.num_agents)])
        new_unsafe = unsafe & ~self._subgoal_unsafe_seen
        entered_final = (self._goal_distances() < self.goal_radius) & ~self._final_goal_guidance
        recovery_hold_active = self.step_count < self._recovery_hold_until_step
        newly_reached &= ~recovery_hold_active
        entered_final &= ~recovery_hold_active
        scheduled = self.step_count - self._last_plan_attempt_step >= self.replanning_interval
        scheduled &= self.step_count >= self._recovery_hold_until_step
        if self._safe_brake_active or self._safety_mode == "SAFE_RECOVER":
            # Safety recovery is reconsidered every environment step.
            scheduled[:] = True
        stagnation_triggered = np.zeros(self.num_agents, dtype=bool)
        if self.stagnation_recovery_enabled:
            start = np.maximum(self._subgoal_segment_start_distance, 1e-6)
            ratios = (start - np.asarray(distances, dtype=np.float32)) / start
            for agent_id in range(self.num_agents):
                if not scheduled[agent_id]:
                    continue
                if ratios[agent_id] < self.min_segment_progress_ratio and not reached[agent_id]:
                    self._stagnation_counts[agent_id] += 1
                else:
                    self._stagnation_counts[agent_id] = 0
                if self._stagnation_counts[agent_id] >= self.stagnation_patience:
                    stagnation_triggered[agent_id] = True
                    self.stagnation_trigger_count += 1
                    goal_direction = _unit_vector(
                        self.goals[agent_id] - self.positions[agent_id]
                    )
                    target_direction = _unit_vector(
                        self.subgoals[agent_id] - self.positions[agent_id]
                    )
                    cross = float(
                        goal_direction[0] * target_direction[1]
                        - goal_direction[1] * target_direction[0]
                    )
                    dot = float(np.clip(np.dot(goal_direction, target_direction), -1.0, 1.0))
                    bucket = int(round(np.degrees(np.arctan2(cross, dot)) / 30.0) * 30)
                    self._failed_direction_buckets[agent_id][bucket] = (
                        self.failed_direction_ttl
                    )
                    self._stagnation_counts[agent_id] = 0
        requested = scheduled | newly_reached | new_unsafe | entered_final | stagnation_triggered
        self._subgoal_unsafe_seen = unsafe.copy()
        if not requested.any():
            return False
        for agent_id in np.flatnonzero(requested):
            active = self._failed_direction_buckets[int(agent_id)]
            for bucket in list(active):
                if not stagnation_triggered[int(agent_id)]:
                    active[bucket] -= 1
                if active[bucket] <= 0:
                    del active[bucket]
        self._force_recovery_mask = stagnation_triggered.copy()
        selected = self._generate_valid_subgoals(requested)
        self._force_recovery_mask[:] = False
        self._last_plan_attempt_step[requested] = self.step_count
        self.subgoal_replan_count += 1
        if np.any(requested & ~scheduled):
            self.early_replan_count += 1
        changed = requested & (np.max(np.abs(selected - self.subgoals), axis=1) > self.subgoal_identity_tolerance)
        # 忽略浮点微差时也保留原坐标，使“未更新”真正意味着 observation 不变。
        selected[~changed] = self.subgoals[~changed]
        self._subgoal_close_reasons = [
            "final_goal_guidance" if entered_final[i] else "own_unsafe" if new_unsafe[i]
            else "own_reached" if newly_reached[i]
            else "stagnation_recovery" if stagnation_triggered[i]
            else "scheduled_replan"
            for i in range(self.num_agents)
        ]
        if changed.any():

            self._close_subgoal_segments(reached, scheduled & ~self._subgoal_arrival_seen, agent_mask=changed)
            self.subgoals = selected
            self.subgoal_ids[changed] += 1
            self.subgoal_assignment_count += int(np.count_nonzero(changed))
            self._subgoal_segment_start_step[changed] = self.step_count
            self._subgoal_segment_start_distance[changed] = self._subgoal_distances()[changed]
            self._subgoal_segment_had_collision[changed] = False
            self._subgoal_started_reached[changed] = self._subgoal_distances()[changed] < self.subgoal_reach_radius
            self._subgoal_arrival_seen[changed] = self._subgoal_started_reached[changed]
            self._subgoal_unsafe_seen[changed] = [not self._is_subgoal_valid(int(i), self.subgoals[i],
                                                    dynamic_only=True, enforce_min_distance=False) for i in np.flatnonzero(changed)]

        self._previous_valid_subgoals = self.subgoals.copy()
        return bool(changed.any())

    def _update_task_phase_metrics(self, distances: np.ndarray) -> None:
        required = float(np.max(distances))
        start = max(self._episode_start_max_goal_distance, 1e-6)
        ratio = float(np.clip((self._episode_start_max_goal_distance - required) / start, -1.0, 1.0))
        self._episode_max_goal_progress_ratio = max(
            self._episode_max_goal_progress_ratio, ratio
        )
        self._max_x_per_agent = np.maximum(self._max_x_per_agent, self.positions[:, 0])
        self._entered_half_map |= self.positions[:, 0] >= self.map_size[0] * 0.5

    def _subgoal_metric_snapshot(self) -> dict[str, Any]:
        closed = max(self._subgoal_closed_count, 1)
        samples = max(self._subgoal_distance_samples, 1)
        candidates = max(self._subgoal_candidate_count, 1)
        return {
            "subgoal_reach_rate": self._subgoal_reach_count / closed,
            "subgoal_progress_ratio": self._subgoal_progress_ratio_sum / closed,
            "mean_subgoal_distance": self._subgoal_distance_sum / samples,
            "mean_steps_per_subgoal": self._subgoal_steps_sum / closed,
            "subgoal_timeout_rate": self._subgoal_timeout_count / closed,
            "invalid_subgoal_rate": self._invalid_subgoal_count / candidates,
            "subgoal_replan_count": self.subgoal_replan_count,
            "early_replan_count": self.early_replan_count,
            "subgoal_collision_rate": self._subgoal_collision_segment_count / closed,
            "shield_intervention_count": int(self.shield_intervention_count),
            "joint_guard_intervention_count": int(
                self.joint_guard_intervention_count
            ),
            "initial_joint_guard_intervention_steps": list(
                self.initial_joint_guard_intervention_steps
            ),
            "final_joint_guard_residual_count": int(
                self.final_joint_guard_residual_count
            ),
            "final_joint_guard_residual_steps": list(
                self.final_joint_guard_residual_steps
            ),
            "obstacle_ttc_projection_count": int(
                self.obstacle_ttc_projection_count
            ),
            "obstacle_ttc_projection_steps": list(
                self.obstacle_ttc_projection_steps
            ),
            "obstacle_ttc_fallback_count": int(
                self.obstacle_ttc_fallback_count
            ),
            "obstacle_ttc_post_action_goal_velocity_sum": float(
                self.obstacle_ttc_post_action_goal_velocity_sum
            ),
            "obstacle_ttc_post_action_goal_velocity_count": int(
                self.obstacle_ttc_post_action_goal_velocity_count
            ),
            "obstacle_ttc_post_action_min_clearance_m": (
                float(self.obstacle_ttc_post_action_min_clearance_m)
                if np.isfinite(self.obstacle_ttc_post_action_min_clearance_m)
                else None
            ),
            "total_action_modification_count": int(
                self.total_action_modification_count
            ),
            "goal_corridor_projection_count": int(
                self.goal_corridor_projection_count
            ),
            "goal_corridor_agent_projection_count": int(
                self.goal_corridor_agent_projection_count
            ),
            "goal_corridor_projection_steps": list(
                self.goal_corridor_projection_steps
            ),
            "goal_corridor_tangential_progress_m": float(
                self.goal_corridor_tangential_progress_m
            ),
            "goal_corridor_nonpositive_progress_count": int(
                self.goal_corridor_nonpositive_progress_count
            ),
            "shield_invariant_violation_count": int(
                self.shield_invariant_violation_count
            ),
            "safe_brake_step_count": int(self.safe_brake_step_count),
            "safe_brake_entry_count": int(self.safe_brake_entry_count),
            "safe_brake_exit_count": int(self.safe_brake_exit_count),
            "max_safe_brake_streak": int(self.max_safe_brake_streak),
            "pair_infeasible_count": int(self.pair_infeasible_count),
            "pair_infeasible_entry_count": int(
                self.pair_infeasible_entry_count
            ),
            "safe_recover_step_count": int(self.safe_recover_step_count),
            "safe_recover_entry_count": int(self.safe_recover_entry_count),
            "safe_recover_exit_count": int(self.safe_recover_exit_count),
            "safe_recovery_failure_count": int(
                self.safe_recovery_failure_count
            ),
            "safe_recovery_unavailable_count": int(
                self.safe_recovery_unavailable_count
            ),
            "safe_recovery_retry_count": int(
                self.safe_recovery_retry_count
            ),
            "safe_recovery_abort_count": int(
                self.safe_recovery_abort_count
            ),
            "safe_recovery_hard_abort_count": int(
                self.safe_recovery_hard_abort_count
            ),
            "safe_recovery_refresh_count": int(
                self.safe_recovery_refresh_count
            ),
            "safe_recovery_reentry_count": int(
                self.safe_recovery_reentry_count
            ),
            "safe_recovery_abort_then_success_count": int(
                self.safe_recovery_abort_then_success_count
            ),
            "safe_recovery_episode_failure": bool(
                self.safe_recovery_episode_failure
            ),
            "safe_recovery_affected_agent_count": int(
                self.safe_recovery_affected_agent_count
            ),
            "recovery_incident_count": int(self.recovery_incident_count),
            "recovery_incident_merge_count": int(
                self.recovery_incident_merge_count
            ),
            "recovery_incident_reentry_reset_count": int(
                self.recovery_incident_reentry_reset_count
            ),
            "recovery_duplicate_hard_abort_count": int(
                self.recovery_duplicate_hard_abort_count
            ),
            "max_recovery_incident_steps": int(
                self.max_recovery_incident_steps
            ),
            "terminal_recovery_unresolved": bool(
                self.terminal_recovery_unresolved
            ),
            "recovery_watchdog_step_count": int(
                self.recovery_watchdog_step_count
            ),
            "max_recovery_direction_unavailable_streak": int(
                self.max_recovery_direction_unavailable_streak
            ),
            "recovery_incident_close_count": int(
                self.recovery_incident_close_count
            ),
            "recovery_incident_reopen_count": int(
                self.recovery_incident_reopen_count
            ),
            "hazard_retrigger_count": int(self.hazard_retrigger_count),
            "same_hazard_reentry_count": int(
                self.same_hazard_reentry_count
            ),
            "hazard_resolution_steps": list(self.hazard_resolution_steps),
            "incident_close_reason_counts": dict(
                self.incident_close_reason_counts
            ),
            "terminal_incident_reconciled_count": int(
                self.terminal_incident_reconciled_count
            ),
            "action_stage_attribution": (
                self._action_stage_attribution_snapshot()
            ),
            "recovery_telemetry_schema_version": self.recovery_telemetry_schema_version,
            "closure_state": self.closure_state,
            "clearance_pending_steps": int(self.clearance_pending_steps),
            "pending_clearance_reappearance_count": int(self.pending_clearance_reappearance_count),
            "post_certificate_retrigger_count": int(self.post_certificate_retrigger_count),
            "close_without_certificate_count": int(self.close_without_certificate_count),
            "stale_closure_bypass_count": int(self.stale_closure_bypass_count),
            "release_certificate_count": int(self.release_certificate_count),
            "release_certificate_id": int(self.release_certificate_id),
            "recovery_lineage_id": self._recovery_lineage_id,
            "recovery_phase": self.recovery_phase,
            "escape_steps": int(self.escape_steps),
            "consolidation_steps": int(self.consolidation_steps),
            "release_shadow_check_passed": bool(self.release_shadow_check_passed),
            "release_shadow_check_count": int(self.release_shadow_check_count),
            "release_shadow_check_failure_count": int(self.release_shadow_check_failure_count),
            "release_margin_m": None if not np.isfinite(self.release_margin_m) else float(self.release_margin_m),
            "release_margin_delta_m": None if not np.isfinite(self.release_margin_delta_m) else float(self.release_margin_delta_m),
            "bounded_recovery_exhaustion_count": int(self.bounded_recovery_exhaustion_count),
            "bounded_exhaustion_then_success_count": int(self.bounded_exhaustion_then_success_count),
            "rapid_same_hazard_retrigger_count": int(self.rapid_same_hazard_retrigger_count),
            "recovery_lineage_failure_count": int(self.recovery_lineage_failure_count),
            "recovery_attributable_mission_failure": bool(self.recovery_attributable_mission_failure),
            "recovery_attributable_failure_reason": self.recovery_attributable_failure_reason,
            "max_recovery_lineage_steps": int(self.max_recovery_lineage_steps),
            "recovery_lineage_budget_reset_count": int(self.recovery_lineage_budget_reset_count),
            "recovery_duplicate_failure_record_count": int(self.recovery_duplicate_failure_record_count),
            "max_rapid_retriggers_per_lineage": int(self.max_rapid_retriggers_per_lineage),
            "recovery_lineage_failure_emitted": bool(self._recovery_lineage_failure_emitted),
            "recovery_lineage_terminal_state": self.recovery_lineage_terminal_state,
            "recovery_candidate_selection_count": int(self.recovery_candidate_selection_count),
            "recovery_candidate_rejection_count": int(self.recovery_candidate_rejection_count),
            "recovery_candidate_rejection_reason_counts": dict(self.recovery_candidate_rejection_reason_counts),
            "recovery_candidate_post_projection_norm": None if not np.isfinite(self.recovery_candidate_post_projection_norm) else float(self.recovery_candidate_post_projection_norm),
            "recovery_candidate_predicted_margin_gain": None if not np.isfinite(self.recovery_candidate_predicted_margin_gain) else float(self.recovery_candidate_predicted_margin_gain),
            "recovery_candidate_fixed_point_streak": int(self.recovery_candidate_fixed_point_streak),
            "release_shadow_horizon_min_margin": None if not np.isfinite(self.release_shadow_horizon_min_margin) else float(self.release_shadow_horizon_min_margin),
            "release_shadow_horizon_min_ttc": None if not np.isfinite(self.release_shadow_horizon_min_ttc) else float(self.release_shadow_horizon_min_ttc),
            "max_safe_recover_streak": int(self.max_safe_recover_streak),
            "boundary_recovery_trigger_count": int(
                self.boundary_recovery_trigger_count
            ),
            "safety_mode": self._safety_mode,
            "subgoal_lifecycle_version": self.subgoal_lifecycle_version,
            "subgoal_ids": self.subgoal_ids.copy(),
            "subgoal_arrival_event_count": self.subgoal_arrival_event_count,
            "subgoal_assignment_count": self.subgoal_assignment_count,
            "subgoal_plan_attempt_count": self.subgoal_plan_attempt_count,
            "oracle_recovery_attempt_count": self.oracle_recovery_attempt_count,
            "oracle_recovery_candidate_count": self.oracle_recovery_candidate_count,
            "oracle_recovery_success_count": self.oracle_recovery_success_count,
            "oracle_recovery_success_rate": self.oracle_recovery_success_count
            / max(self.oracle_recovery_attempt_count, 1),
            "previous_target_hold_count": self.previous_target_hold_count,
            "previous_target_hold_rate": self.previous_target_hold_count
            / max(self.subgoal_plan_attempt_count, 1),
            "blocked_final_goal_fallback_count": self.blocked_final_goal_fallback_count,
            "blocked_primary_segment_count": self.blocked_primary_segment_count,
            "blocked_primary_segment_rate": self.blocked_primary_segment_count
            / max(self.subgoal_plan_attempt_count, 1),
            "blocked_primary_candidate_rate": self.blocked_primary_segment_count
            / max(self.subgoal_plan_attempt_count, 1),
            "recovery_activation_rate": self.oracle_recovery_attempt_count
            / max(self.subgoal_plan_attempt_count, 1),
            "stagnation_trigger_count": self.stagnation_trigger_count,
            "stagnation_trigger_rate": self.stagnation_trigger_count
            / max(self.subgoal_plan_attempt_count, 1),
            "repeated_direction_rejection_count": self.repeated_direction_rejection_count,
            "repeated_direction_rate": min(
                self.repeated_direction_rejection_count
                / max(self.subgoal_plan_attempt_count, 1),
                1.0,
            ),
            "max_goal_progress_ratio": self._episode_max_goal_progress_ratio,
            "halfway_reach_rate": float(self._episode_max_goal_progress_ratio >= 0.5),
            "near_goal_reach_rate": float(self._episode_max_goal_progress_ratio >= 0.8),
            "mean_max_x": float(np.mean(self._max_x_per_agent)),
            # 以 agent 为单位计算，另保留 any/all 标记以避免“任一/每个”的语义歧义。
            "half_map_entry_rate": float(np.mean(self._entered_half_map)),
            "any_half_map_entry": bool(np.any(self._entered_half_map)),
            "all_half_map_entry": bool(np.all(self._entered_half_map)),
            "agents_entered_half_map": self._entered_half_map.copy(),
        }

    # 旧私有接口作为薄适配层保留，避免训练/外部脚本断裂。
    def _detect_collision(self, *_args, boundary_hits=None, **_kwargs):
        if boundary_hits is None and _args:
            candidate = np.asarray(_args[-1])
            if candidate.shape == (self.num_agents,) and candidate.dtype == bool:
                boundary_hits = candidate
        return self.check_collision(boundary_hits=boundary_hits)

    def _success_metrics(self, distances: np.ndarray) -> tuple[int, bool]:
        inside = np.asarray(distances) < self.goal_radius
        count = int(np.count_nonzero(inside))
        return count, bool(count == self.num_agents)

    def _normalize_progress(self, previous_distance, current_distance) -> np.ndarray:
        return (np.asarray(previous_distance, dtype=np.float32) - np.asarray(current_distance, dtype=np.float32)) / max(self.agent_max_speed * self.dt, 1e-6)


    def build_local_observations(
        self, subgoal_override: np.ndarray | None = None,
    ) -> np.ndarray:
        """Construct the versioned local observation without simulator truth."""
        targets = self.subgoals if subgoal_override is None else np.asarray(
            subgoal_override, dtype=np.float32
        )
        if targets.shape != (self.num_agents, 2) or not np.isfinite(targets).all():
            raise ValueError("subgoal override must be a finite [num_agents,2] array")
        rows: list[np.ndarray] = []
        for agent_id, agent in enumerate(self.agents):
            obstacle_features: list[float] = []
            if self.observation_schema_version == "tracked_obstacle_v2":
                for item in self._tracked_obstacles(agent_id):
                    relative = item["relative_position"]
                    velocity = item["relative_velocity"]
                    obstacle_features.extend([
                        float(relative[0]), float(relative[1]),
                        float(velocity[0]), float(velocity[1]),
                        float(item["obstacle"].radius), 1.0,
                        float(bool(item["velocity_valid"])),
                    ])
            else:
                for _, obstacle, distance in self.get_nearest_obstacles(agent_id):
                    relative = obstacle.position - agent.position
                    obstacle_features.extend([float(relative[0]), float(relative[1]), obstacle.radius, distance])
            obstacle_features.extend([0.0] * (self.obstacle_slots * self._obstacle_feature_dim - len(obstacle_features)))
            row = np.concatenate([
                agent.position, agent.velocity, self.goals[agent_id], targets[agent_id],
                self._nearest_teammate_relative(agent_id), np.asarray(obstacle_features, dtype=np.float32),
            ]).astype(np.float32)
            if row.shape != (self.local_observation_dim,) or not np.isfinite(row).all():
                raise RuntimeError("local observation width or values are invalid")
            rows.append(row)
        return np.stack(rows)

    def get_global_state(self) -> np.ndarray:
        """Central critic state: concatenate the versioned local observations."""
        state = self.build_local_observations().reshape(-1).astype(np.float32)
        if state.shape != (self.global_state_dim,):
            raise RuntimeError("global state width is invalid")
        return state

    def _nearest_teammate_relative(self, agent_id: int) -> np.ndarray:
        candidates = [other.position - self.agents[agent_id].position for other_id, other in enumerate(self.agents) if other_id != agent_id]
        if not candidates:
            return np.zeros(2, dtype=np.float32)
        return min(candidates, key=lambda item: float(np.linalg.norm(item))).astype(np.float32)

    def _sample_agents(self) -> list[UAVState]:
        base_y = self._resample_axis_values(self.start_y_positions, self.num_agents)
        jitter = self._rng.uniform(-self.start_y_jitter, self.start_y_jitter, size=self.num_agents)
        return [
            UAVState(np.array([self.start_x, y], dtype=np.float32), np.zeros(2, dtype=np.float32), self.agent_radius, self.agent_max_speed)
            for y in base_y + jitter
        ]

    def _build_goals(self) -> np.ndarray:
        y_values = self._resample_axis_values(self.goal_y_positions, self.num_agents)
        if self.cross_goals:
            y_values = y_values[::-1]
        return np.column_stack([np.full(self.num_agents, self.goal_x, dtype=np.float32), y_values]).astype(np.float32)

    @staticmethod
    def _resample_axis_values(values: np.ndarray, count: int) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.size == count:
            return values.copy()
        return np.linspace(float(values.min()), float(values.max()), count, dtype=np.float32)

    def _sample_obstacles(self) -> list[CircularObstacle]:
        result: list[CircularObstacle] = []
        for obstacle_id in range(self.obstacle_num):
            dynamic = obstacle_id >= self.static_obstacle_num
            radius = float(self._rng.uniform(*self.obstacle_radius_range))
            for _ in range(5000):
                center = np.array([self._rng.uniform(*self.obstacle_spawn_x_range), self._rng.uniform(*self.obstacle_spawn_y_range)], dtype=np.float32)
                if np.any(center - radius < 0.0) or np.any(center + radius > self.map_size):
                    continue
                safe_agents = all(np.linalg.norm(center - agent.position) >= radius + agent.radius + self.obstacle_spawn_clearance for agent in self.agents)
                safe_goals = all(np.linalg.norm(center - goal) >= radius + self.goal_radius + self.obstacle_spawn_clearance for goal in self.goals)
                safe_obstacles = all(np.linalg.norm(center - other.position) >= radius + other.radius + self.obstacle_spawn_clearance for other in result)
                if safe_agents and safe_goals and safe_obstacles:
                    break
            else:
                raise RuntimeError("failed to sample non-overlapping circular obstacles")
            velocity = self._rng.uniform(*self.dynamic_obstacle_velocity_range, size=2).astype(np.float32) if dynamic else np.zeros(2, dtype=np.float32)
            result.append(CircularObstacle(center, radius, velocity, dynamic))
        return result

    def _goal_distances(self) -> np.ndarray:
        return np.linalg.norm(self.positions - self.goals, axis=1).astype(np.float32)

    def _team_goal_potential(self, distances: np.ndarray) -> float:
        """Normalized cooperative completion potential in [-1, 1].

        The bottleneck term focuses learning on the slower UAV while the mean
        term still credits joint progress.  It is based on final goals only, so
        Oracle replanning cannot create an artificial reward discontinuity.
        """
        values = np.asarray(distances, dtype=np.float32)
        max_ratio = float(np.max(values)) / max(
            self._episode_start_max_goal_distance, 1e-6
        )
        mean_ratio = float(np.mean(values)) / max(
            self._episode_start_mean_goal_distance, 1e-6
        )
        residual = (
            self.bottleneck_weight * max_ratio
            + (1.0 - self.bottleneck_weight) * mean_ratio
        )
        return float(np.clip(1.0 - residual, -1.0, 1.0))

    def _team_goal_completion(self, distances: np.ndarray) -> float:
        """Return bottleneck-aware completion relative to each UAV's own start.

        Unlike the V3 potential this value is not forced to zero at episode end.
        V4 only pays increases in its historical maximum, bounding the total
        final-goal shaping credit and preventing backtracking/re-approach loops.
        """
        starts = np.maximum(self._episode_start_goal_distances, 1e-6)
        individual = np.clip(
            (starts - np.asarray(distances, dtype=np.float32)) / starts,
            -1.0,
            1.0,
        )
        completion = (
            self.bottleneck_weight * float(np.min(individual))
            + (1.0 - self.bottleneck_weight) * float(np.mean(individual))
        )
        return float(np.clip(completion, -1.0, 1.0))

    def _nearest_obstacle_distances(self) -> np.ndarray:
        result = np.full(self.num_agents, np.inf, dtype=np.float32)
        for agent_id, agent in enumerate(self.agents):
            if self.obstacles:
                result[agent_id] = min(float(np.linalg.norm(agent.position - obstacle.position)) - agent.radius - obstacle.radius for obstacle in self.obstacles)
        return result

    def _boundary_clearances(self) -> np.ndarray:
        """Surface clearance to the closest wall for each UAV."""
        positions = self.positions
        low = positions - self.agent_radius
        high = self.map_size - self.agent_radius - positions
        return np.min(np.concatenate([low, high], axis=1), axis=1).astype(np.float32)

    def _inter_agent_clearances(self) -> np.ndarray:
        """Surface clearance to the closest teammate for each UAV."""
        result = np.full(self.num_agents, np.inf, dtype=np.float32)
        for left in range(self.num_agents):
            for right in range(left + 1, self.num_agents):
                clearance = float(
                    np.linalg.norm(self.agents[left].position - self.agents[right].position)
                    - 2.0 * self.agent_radius
                )
                result[left] = min(result[left], clearance)
                result[right] = min(result[right], clearance)
        return result

    def _clearance_risk(self, clearances: np.ndarray) -> np.ndarray:
        """Quadratic risk in [0, 1], active only inside the configured margin."""
        normalized = np.clip(
            (self.proximity_risk_margin - np.asarray(clearances, dtype=np.float32))
            / self.proximity_risk_margin,
            0.0,
            1.0,
        )
        return np.square(normalized).astype(np.float32)

    def _proximity_risk_snapshot(self) -> dict[str, Any]:
        """Return bounded pre-contact risk without exposing simulator-only future state."""
        obstacle = self._clearance_risk(self._nearest_obstacle_distances())
        boundary = self._clearance_risk(self._boundary_clearances())
        inter_agent = self._clearance_risk(self._inter_agent_clearances())
        team_risk = float(
            max(
                np.max(obstacle, initial=0.0),
                np.max(boundary, initial=0.0),
                np.max(inter_agent, initial=0.0),
            )
        )
        return {
            "margin": self.proximity_risk_margin,
            "max_penalty": self.proximity_risk_max_penalty,
            "team_risk": team_risk,
            "obstacle_risk": obstacle,
            "boundary_risk": boundary,
            "inter_agent_risk": inter_agent,
        }

    def _minimum_inter_agent_distance(self) -> float:
        return min(float(np.linalg.norm(self.agents[left].position - self.agents[right].position)) for left in range(self.num_agents) for right in range(left + 1, self.num_agents))

    def _visible_obstacle_ids(self) -> list[list[int]]:
        return [[obstacle_id for obstacle_id, _, _ in self.get_nearest_obstacles(agent_id)] for agent_id in range(self.num_agents)]

    def render(self):
        """返回轻量 RGB 俯视图，保持环境可独立渲染。"""
        width = int(self.config.get("render_width", 600))
        height = int(self.config.get("render_height", 600))
        image = np.full((height, width, 3), 245, dtype=np.uint8)

        def pixel(point):
            x = int(np.clip(point[0] / self.map_size[0] * (width - 1), 0, width - 1))
            y = int(np.clip((1.0 - point[1] / self.map_size[1]) * (height - 1), 0, height - 1))
            return x, y

        def disk(center, radius, color):
            cx, cy = pixel(center)
            rx = max(1, int(radius / self.map_size[0] * width))
            ry = max(1, int(radius / self.map_size[1] * height))
            yy, xx = np.ogrid[:height, :width]
            mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
            image[mask] = color

        for goal in self.goals:
            disk(goal, self.goal_radius, (80, 190, 100))
        for obstacle in self.obstacles:
            disk(obstacle.position, obstacle.radius, (210, 120, 60) if obstacle.dynamic else (90, 90, 90))
        for agent in self.agents:
            disk(agent.position, self.agent_radius, (40, 110, 220))
        return image

    def _validate_config(self) -> None:
        if self.num_agents < 2:
            raise ValueError("num_agents must be at least 2")
        if self.map_size.shape != (2,) or np.any(self.map_size <= 0.0):
            raise ValueError("map_size must contain two positive values")
        if (
            self.dt <= 0.0 or self.agent_radius <= 0.0
            or self.agent_max_speed <= 0.0 or self.agent_max_acceleration <= 0.0
        ):
            raise ValueError(
                "dt, agent_radius, agent_max_speed and agent_max_acceleration must be positive"
            )
        if self.max_episode_steps <= 0 or self.goal_radius <= 0.0:
            raise ValueError("episode length and goal radius must be positive")
        if not np.isfinite(self.proximity_risk_margin) or self.proximity_risk_margin <= 0.0:
            raise ValueError("proximity_risk.margin must be finite and positive")
        if (
            not np.isfinite(self.proximity_risk_max_penalty)
            or self.proximity_risk_max_penalty < 0.0
        ):
            raise ValueError("proximity_risk.max_penalty must be finite and nonnegative")
        if not np.isfinite(self.timeout_penalty) or self.timeout_penalty < 0.0:
            raise ValueError("reward.timeout_penalty must be finite and nonnegative")
        if (
            not np.isfinite(self.potential_progress_scale)
            or self.potential_progress_scale < 0.0
        ):
            raise ValueError("reward.potential_progress_scale must be finite and nonnegative")
        if not 0.0 <= self.potential_gamma <= 1.0:
            raise ValueError("reward.potential_gamma must be in [0, 1]")
        if not 0.0 <= self.bottleneck_weight <= 1.0:
            raise ValueError("reward.bottleneck_weight must be in [0, 1]")
        if (
            not np.isfinite(self.control_effort_weight)
            or self.control_effort_weight < 0.0
        ):
            raise ValueError("reward.control_effort_weight must be finite and nonnegative")
        if (
            not np.isfinite(self.subgoal_progress_scale)
            or self.subgoal_progress_scale < 0.0
        ):
            raise ValueError("reward.subgoal_progress_scale must be finite and nonnegative")
        if (
            not np.isfinite(self.final_progress_scale)
            or self.final_progress_scale < 0.0
        ):
            raise ValueError("reward.final_progress_scale must be finite and nonnegative")
        available_half_span = float(np.min(self.map_size - 2.0 * self.agent_radius)) * 0.5
        if not 0.0 < self.boundary_recovery_distance < available_half_span:
            raise ValueError("boundary_recovery_distance must fit inside the map")
        if self.static_obstacle_num < 0 or self.dynamic_obstacle_num < 0:
            raise ValueError("obstacle counts must be nonnegative")
        if self.observation_schema_version not in {
            "legacy_obstacle_v1", "tracked_obstacle_v2"
        }:
            raise ValueError("unsupported observation_schema_version")
        if self.obstacle_selection_version not in {
            "nearest_distance_v1", "predicted_clearance_v1"
        }:
            raise ValueError("unsupported obstacle_selection_version")
        if self.obstacle_velocity_estimation_version not in {
            "none_v1", "causal_track_difference_v1"
        }:
            raise ValueError("unsupported obstacle_velocity_estimation_version")
        if self.observation_schema_version == "legacy_obstacle_v1":
            if self.obstacle_slots != 4:
                raise ValueError("legacy_obstacle_v1 requires 4 slots for the 26D observation")
            if self.obstacle_selection_version != "nearest_distance_v1":
                raise ValueError("legacy_obstacle_v1 requires nearest_distance_v1")
        else:
            if self.obstacle_slots <= 0:
                raise ValueError("tracked_obstacle_v2 requires positive obstacle_slots")
            if self.obstacle_velocity_estimation_version != "causal_track_difference_v1":
                raise ValueError(
                    "tracked_obstacle_v2 requires causal_track_difference_v1"
                )
        if (
            self.near_goal_stall_action_threshold < 0.0
            or self.near_goal_stall_distance_margin <= 0.0
            or self.near_goal_stall_patience_steps <= 0
        ):
            raise ValueError("near-goal stall diagnostic thresholds are invalid")
        if self.detection_radius <= 0.0 or self.start_y_jitter < 0.0:
            raise ValueError("detection radius must be positive and jitter nonnegative")
        if self.replanning_interval <= 0:
            raise ValueError("oracle replanning_interval must be positive")
        if self.subgoal_reach_radius <= 0.0 or self.min_subgoal_distance <= 0.0:
            raise ValueError("oracle subgoal distances must be positive")
        if self.oracle_recovery_enabled:
            if (
                self.oracle_recovery_angles.ndim != 1
                or self.oracle_recovery_angles.size == 0
                or not np.isfinite(self.oracle_recovery_angles).all()
            ):
                raise ValueError("oracle recovery angles must be a finite non-empty list")
            if (
                self.oracle_recovery_distances.ndim != 1
                or self.oracle_recovery_distances.size == 0
                or not np.isfinite(self.oracle_recovery_distances).all()
                or np.any(self.oracle_recovery_distances <= 0.0)
            ):
                raise ValueError("oracle recovery distances must be finite and positive")
            if min(
                self.oracle_recovery_goal_weight,
                self.oracle_recovery_clearance_weight,
                self.oracle_recovery_turn_weight,
            ) < 0.0:
                raise ValueError("oracle recovery scoring weights must be nonnegative")

        if (
            self.safety_recovery_enabled
            and self.star_safety_state_machine_version
            in {
                "bounded_recovery_v1", "star_scoped_recovery_v4",
                "star_adaptive_recovery_v5", "star_incident_recovery_v5_1",
                "star_incident_recovery_v5_2",
            }
            and (
                self.safe_recovery_exit_stable_steps <= 0
                or self.safe_recovery_maximum_steps
                < self.safe_recovery_exit_stable_steps
                or self.safe_recovery_target_distance <= 0.0
                or not 0.0 < self.safe_recovery_speed_fraction <= 1.0
                or self.safe_recovery_boundary_trigger_margin < 0.0
                or self.safe_recovery_boundary_stall_patience <= 0
                or self.safe_recovery_minimum_displacement <= 0.0
                or self.safe_recovery_obstacle_repulsion_margin < 0.0
            )
        ):
            raise ValueError("invalid shared active-recovery contract")
        if (
            self.safety_recovery_enabled
            and self.star_safety_state_machine_version
            == "star_scoped_recovery_v4"
            and (
                self.safe_recovery_attempt_steps <= 0
                or self.safe_recovery_maximum_attempts <= 0
                or self.safe_recovery_attempt_steps
                * self.safe_recovery_maximum_attempts
                > self.safe_recovery_maximum_steps
                or not self.safe_recovery_exclude_goal_agents
            )
        ):
            raise ValueError("invalid shared scoped-recovery contract")
        if (
            self.safety_recovery_enabled
            and self.star_safety_state_machine_version
            in {
                "star_adaptive_recovery_v5", "star_incident_recovery_v5_1",
                "star_incident_recovery_v5_2",
            }
            and (
                not self.safe_recovery_exclude_goal_agents
                or self.safe_recovery_progress_patience_steps != 3
                or self.safe_recovery_maximum_steps != 20
                or self.safe_recovery_maximum_refreshes != 6
                or self.safe_recovery_minimum_margin_improvement <= 0.0
            )
        ):
            raise ValueError("invalid shared adaptive-recovery contract")
        if (
            self.safety_recovery_enabled
            and self.star_safety_state_machine_version
            in _PERSISTENT_RECOVERY_VERSIONS
            and (
                not self.safe_recovery_exclude_goal_agents
                or self.safe_recovery_progress_patience_steps != 3
                or self.safe_recovery_maximum_steps != 20
                or self.safe_recovery_escape_steps != 14
                or self.safe_recovery_consolidation_steps != 6
                or self.safe_recovery_escape_steps
                + self.safe_recovery_consolidation_steps != 20
                or self.safe_recovery_maximum_refreshes != 4
                or self.safe_recovery_release_hysteresis_m != 0.25
                or self.safe_recovery_rapid_retrigger_window_steps != 5
                or self.safe_recovery_lineage_max_incidents != 2
                or self.safe_recovery_lineage_max_steps != 40
                or (
                    self.star_safety_state_machine_version in {
                        "star_persistent_recovery_v5_4",
                        "star_closure_consistency_audit_v1",
                    }
                    and (
                        self.safe_recovery_shadow_horizon_steps != 3
                        or self.safe_recovery_candidate_fixed_point_patience != 2
                        or self.safe_recovery_candidate_minimum_margin_gain <= 0.0
                    )
                )
                or (
                    self.star_safety_state_machine_version
                    == "star_closure_consistency_audit_v1"
                    and (
                        self.closure_clear_window_steps != 5
                        or self.closure_certificate_tail_steps != 3
                    )
                )
            )
        ):
            raise ValueError("invalid shared persistent-recovery contract")
        if (
            self.obstacle_execution_diagnostics_enabled
            and self.obstacle_ttc_audit_horizon <= 0.0
        ):
            raise ValueError("obstacle TTC audit horizon must be positive")
        if self.obstacle_execution_guard_enabled:
            if (
                self.obstacle_execution_guard_version
                not in {
                    "causal_ttc_projection_v1",
                    "causal_ttc_progress_projection_v2",
                }
                or self.observation_schema_version != "tracked_obstacle_v2"
                or self.obstacle_velocity_estimation_version
                != "causal_track_difference_v1"
                or not (
                    0.0 < self.obstacle_guard_emergency_ttc_seconds
                    < self.obstacle_guard_warning_ttc_seconds
                    < self.obstacle_guard_horizon_seconds
                )
            ):
                raise ValueError("invalid shared obstacle guard contract")
        if self.ttc_headroom_audit_enabled and (
            not self.obstacle_execution_guard_enabled
            or self.ttc_headroom_shadow_steps != 3
            or self.ttc_headroom_minimum_goal_velocity_gain <= 0.0
            or self.ttc_headroom_maximum_extra_actor_deviation < 0.0
        ):
            raise ValueError("invalid TTC headroom audit contract")
        if self.goal_boundary_corridor_enabled:
            if (
                self.goal_boundary_corridor_version
                != "goal_boundary_corridor_v1"
                or not 0.0 < self.goal_boundary_corridor_coordinate
                < float(self.map_size[0])
            ):
                raise ValueError("invalid shared goal corridor contract")
        if (
            self.star_safety_state_machine_version
            in _INCIDENT_RECOVERY_VERSIONS
            and (
                not self.goal_boundary_corridor_enabled
                or self.goal_boundary_corridor_coordinate != 94.0
                or not self.obstacle_execution_guard_enabled
                or self.star_candidate_feasibility_version
                != "guard_aligned_observable_v2"
            )
        ):
            raise ValueError("invalid shared safety integration contract")
        if self.stagnation_patience < 1 or not 0.0 <= self.min_segment_progress_ratio <= 1.0:
            raise ValueError("stagnation patience/progress threshold is invalid")
        if self.failed_direction_ttl < 1 or self.recovery_hold_cycles < 1:
            raise ValueError("failed direction TTL and recovery hold cycles must be positive")
        for name, values in (("obstacle_radius_range", self.obstacle_radius_range), ("dynamic_obstacle_velocity_range", self.dynamic_obstacle_velocity_range), ("obstacle_spawn_x_range", self.obstacle_spawn_x_range), ("obstacle_spawn_y_range", self.obstacle_spawn_y_range)):
            if values.shape != (2,) or values[0] > values[1]:
                raise ValueError(f"{name} must be an ordered pair")
        if self.obstacle_radius_range[0] <= 0.0:
            raise ValueError("obstacle radii must be positive")
        if self.start_y_positions.size < 2 or self.goal_y_positions.size < 2:
            raise ValueError("start_y_positions and goal_y_positions need at least two values")
        if not (self.agent_radius <= self.start_x <= self.map_size[0] - self.agent_radius):
            raise ValueError("start_x lies outside the map")
        if not (self.goal_radius <= self.goal_x <= self.map_size[0] - self.goal_radius):
            raise ValueError("goal_x lies outside the map")
