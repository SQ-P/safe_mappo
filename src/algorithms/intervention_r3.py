"""R3 immutable-shadow dataset, twin critic training, and certification."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.algorithms.intervention_critic import (
    TwinInterventionCriticV1,
    fit_stage_temperatures,
    intervention_critic_loss,
)
from src.algorithms.intervention_data import (
    TRAINABLE_STAGES,
    _causal_context_from_snapshot,
)
from src.algorithms.intervention_team_data import (
    ACTION_TTC_VERSION,
    action_conditioned_obstacle_features,
    per_agent_snapshot_margins,
)
from src.algorithms.intervention_evaluation import (
    evaluate_team_learnability_gate_v2,
)
from src.safety.counterfactual import (
    evaluate_avoidable_intervention_v2,
    registered_shadow_actions,
)
from src.utils.io_utils import save_json


R3_DATASET_SCHEMA = "IAMAPPOR3ShadowDatasetV3"
R3_REPORT_SCHEMA = "IAMAPPOR3TwinCriticCertificationV3"
R3_ARTIFACT_SCHEMA = "R3TwinInterventionCriticArtifactV3"
R3_SPLIT_SEED = 20261109
R3_FEATURE_CONTRACT = (
    "normalized_global_state+raw_joint_mean_action+causal_safety_context+"
    "current_safety_margin+per_agent_margins+observable_raw_action_ttc_v1"
)
R4_ACTOR_GRADIENT_CONTRACT = (
    "direct_raw_joint_action_with_detached_action_conditioned_ttc_context_v1"
)
R4_1_ACTOR_GRADIENT_CONTRACT = (
    "straight_through_rollout_action+per_uav_goal_progress_projected_"
    "risk_gradient+smooth_clearance_backward_v1"
)
R4_2_ACTOR_GRADIENT_CONTRACT = (
    "warmup50+in_support_straight_through_risk+parameter_pcgrad_"
    "ppo_and_per_uav_oracle_subgoal_progress_v1"
)
R4_2_1_ACTOR_GRADIENT_CONTRACT = (
    "warmup100+in_support_straight_through_risk+parameter_pcgrad_"
    "ppo_and_per_uav_oracle_subgoal_progress_v1"
)
R4_3_1_ACTOR_GRADIENT_CONTRACT = (
    "warmup100+team_step_actor_risk_eligibility+in_support_sampled_"
    "straight_through_risk+parameter_pcgrad_ppo_and_per_uav_oracle_"
    "subgoal_progress_v1"
)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_r3_entry_evidence(
    *,
    combined_gate_report: str | Path,
    primary_r2r_report: str | Path,
    checkpoint: str | Path,
) -> dict[str, Any]:
    """Certify the R2R→dataset→checkpoint provenance chain before R3."""

    combined_path = Path(combined_gate_report).resolve()
    primary_path = Path(primary_r2r_report).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    for path, name in (
        (combined_path, "combined GateFix report"),
        (primary_path, "primary GateFix report"),
        (checkpoint_path, "Standard checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")
    combined = json.loads(combined_path.read_text(encoding="utf-8"))
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    if combined.get("schema_version") != "IAMAPPOR2RCombinedDecisionV2":
        raise ValueError("R3 requires IAMAPPOR2RCombinedDecisionV2")
    if combined.get("primary_decision") != "advance_r3_flat_and_agent":
        raise RuntimeError("R3 requires the frozen Flat+Agent primary decision")
    primary_pass = dict(combined.get("primary_model_pass", {}))
    if not primary_pass.get("flat_global_mlp") or not (
        primary_pass.get("agent_encoder_mean")
        or primary_pass.get("agent_encoder_meanmax_context")
    ):
        raise RuntimeError("combined GateFix model flags disagree with its decision")
    if (
        primary.get("schema_version") != "IAMAPPOR2RTeamLearnabilityReportV2"
        or int(primary.get("policy_seed", -1)) != 41
        or primary.get("primary_decision") != combined["primary_decision"]
        or dict(primary.get("model_pass", {})) != primary_pass
    ):
        raise ValueError("primary GateFix report does not match combined decision")

    team_dataset = Path(primary["dataset"]).resolve()
    team_manifest_path = team_dataset / "manifest.json"
    if (
        not team_manifest_path.is_file()
        or _file_sha256(team_manifest_path)
        != primary.get("dataset_manifest_sha256")
    ):
        raise ValueError("primary R2R TeamDatasetV3 manifest hash mismatch")
    team_manifest = json.loads(team_manifest_path.read_text(encoding="utf-8"))
    source_dataset = Path(team_manifest["source_dataset"]).resolve()
    source_manifest_path = source_dataset / "manifest.json"
    if (
        not source_manifest_path.is_file()
        or _file_sha256(source_manifest_path)
        != team_manifest.get("source_manifest_sha256")
        or team_manifest.get("source_manifest_sha256")
        != primary.get("source_v2_manifest_sha256")
    ):
        raise ValueError("R2R source DatasetV2 manifest hash mismatch")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    checkpoint_sha = _file_sha256(checkpoint_path)
    checkpoint_payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    fingerprints = {
        str(primary.get("task_config_fingerprint")),
        str(team_manifest.get("task_config_fingerprint")),
        str(source_manifest.get("task_config_fingerprint")),
        str(checkpoint_payload.get("task_config_fingerprint")),
    }
    if len(fingerprints) != 1:
        raise ValueError("R3 evidence chain has mismatched task fingerprints")
    if (
        int(source_manifest.get("policy_seed", -1)) != 41
        or int(source_manifest.get("checkpoint_update", -1)) != 200
        or source_manifest.get("checkpoint_sha256") != checkpoint_sha
        or int(checkpoint_payload.get("update", -1)) != 200
        or int(checkpoint_payload.get("config", {}).get("seed", -1)) != 41
    ):
        raise ValueError("R3 checkpoint does not match the passed R2R evidence")
    return {
        "combined_gate_report_sha256": _file_sha256(combined_path),
        "primary_r2r_report_sha256": _file_sha256(primary_path),
        "team_dataset_manifest_sha256": _file_sha256(team_manifest_path),
        "source_dataset_manifest_sha256": _file_sha256(source_manifest_path),
        "checkpoint_sha256": checkpoint_sha,
        "task_config_fingerprint": fingerprints.pop(),
        "policy_seed": 41,
        "checkpoint_update": 200,
        "primary_decision": combined["primary_decision"],
    }


def _array_digest(value: Any) -> str:
    array = np.asarray(value)
    return sha256(array.tobytes(order="C")).hexdigest()


def _split_layouts(
    layout_ids: Sequence[int], *, seed: int = R3_SPLIT_SEED,
) -> dict[str, list[int]]:
    unique = np.asarray(sorted(set(map(int, layout_ids))), dtype=np.int64)
    if len(unique) < 10:
        raise ValueError("R3.2 requires at least ten independent layouts")
    shuffled = unique[np.random.default_rng(int(seed)).permutation(len(unique))]
    train_end = int(round(0.60 * len(unique)))
    calibration_end = train_end + int(round(0.10 * len(unique)))
    threshold_end = calibration_end + int(round(0.10 * len(unique)))
    result = {
        "train": sorted(map(int, shuffled[:train_end])),
        "calibration": sorted(map(int, shuffled[train_end:calibration_end])),
        "threshold_selection": sorted(map(int, shuffled[calibration_end:threshold_end])),
        "test": sorted(map(int, shuffled[threshold_end:])),
    }
    groups = [set(values) for values in result.values()]
    if not all(groups) or any(
        left & right
        for index, left in enumerate(groups)
        for right in groups[index + 1:]
    ):
        raise RuntimeError("R3 layout split is empty or overlapping")
    return result


def evaluate_r3_shadow_step(
    env,
    *,
    raw_joint_action: np.ndarray,
    global_state_normalized: np.ndarray,
    actor_hidden: Any,
    sobol_seed: int = R3_SPLIT_SEED,
    maximum_progress_loss_mps: float = 0.10,
) -> dict[str, Any]:
    """Evaluate all registered candidates before real environment execution."""

    raw = np.asarray(raw_joint_action, dtype=np.float32)
    if raw.shape != (env.num_agents, 2):
        raise ValueError("R3 raw joint action must have shape [UAV,2]")
    snapshot = env._causal_safety_snapshot(raw)
    snapshot_payload = snapshot.canonical_payload()
    team_observation = np.asarray(snapshot.team_observation, dtype=np.float32)
    per_agent_margins = per_agent_snapshot_margins(
        snapshot_payload,
        agent_radius=float(env.agent_radius),
        map_size=tuple(float(value) for value in env.map_size),
    )
    def candidate_ttc_features(action: np.ndarray) -> tuple[np.ndarray, ...]:
        action_ttc = []
        ttc_warning = []
        ttc_emergency = []
        predicted_clearance = []
        for agent in range(env.num_agents):
            ttc, warning, emergency, clearance = action_conditioned_obstacle_features(
                current_velocity=team_observation[agent, 2:4],
                raw_action=action[agent],
                obstacle_tracks=snapshot_payload["obstacle_tracks"][agent],
                agent_radius=float(env.agent_radius),
                obstacle_safety_margin=float(snapshot.safety_state[3]),
                invariant_buffer=float(env.boundary_guard.invariant_buffer_m),
                max_acceleration=float(env.agent_max_acceleration),
                max_speed=float(env.agent_max_speed),
                dt=float(env.dt),
                horizon_seconds=float(env.obstacle_guard_horizon_seconds),
                warning_ttc_seconds=float(env.obstacle_guard_warning_ttc_seconds),
                emergency_ttc_seconds=float(env.obstacle_guard_emergency_ttc_seconds),
            )
            action_ttc.append(ttc)
            ttc_warning.append(warning)
            ttc_emergency.append(emergency)
            predicted_clearance.append(clearance)
        return (
            np.asarray(action_ttc, dtype=np.float32),
            np.asarray(ttc_warning, dtype=bool),
            np.asarray(ttc_emergency, dtype=bool),
            np.asarray(predicted_clearance, dtype=np.float32),
        )

    action_ttc, ttc_warning, ttc_emergency, predicted_clearance = (
        candidate_ttc_features(raw)
    )
    hidden_host = np.asarray(
        actor_hidden.detach().cpu().numpy()
        if isinstance(actor_hidden, torch.Tensor) else actor_hidden
    )
    hidden_digest_before = _array_digest(hidden_host)
    environment_before = env.export_counterfactual_snapshot(hidden_host)
    goal_delta = np.asarray(env.goals - env.positions, dtype=np.float32)
    norms = np.linalg.norm(goal_delta, axis=1, keepdims=True)
    goal_actions = np.divide(
        goal_delta, norms, out=np.zeros_like(goal_delta), where=norms > 1e-8
    )
    reached = np.asarray(env._goal_distances() < env.goal_radius, dtype=bool)
    shadows = registered_shadow_actions(
        raw, goal_actions, sobol_seed=int(sobol_seed)
    )
    label = evaluate_avoidable_intervention_v2(
        snapshot,
        shadows,
        env.evaluate_shadow_action_v2,
        goal_directions=goal_delta,
        current_velocities=np.asarray(
            [row[2:4] for row in snapshot.team_observation], dtype=np.float32
        ),
        reached_mask=reached,
        acceleration_scale=float(env.agent_max_acceleration),
        dt=float(env.dt),
        maximum_speed=float(env.agent_max_speed),
        maximum_progress_loss_mps=float(maximum_progress_loss_mps),
    )
    # R4.4 stores the deterministic online reference bank (HOLD, brake and
    # four goal-direction scales).  The 24 Sobol candidates remain label-only.
    reference_actions = np.asarray(shadows[1:7], dtype=np.float32)
    reference_features = [candidate_ttc_features(action) for action in reference_actions]
    reference_stage_costs = np.asarray(label.shadow_stage_costs[1:7], dtype=np.float32)
    reference_total_costs = np.asarray(label.shadow_total_costs[1:7], dtype=np.float32)
    reference_valid = np.asarray(label.valid_shadow_mask[1:7], dtype=bool)
    environment_after = env.export_counterfactual_snapshot(hidden_host)
    hidden_digest_after = _array_digest(hidden_host)
    snapshot_after = env._causal_safety_snapshot(raw)
    if environment_after.state_fingerprint != environment_before.state_fingerprint:
        raise RuntimeError("R3 shadow panel mutated environment/RNG/recovery state")
    if hidden_digest_after != hidden_digest_before:
        raise RuntimeError("R3 shadow panel mutated Actor GRU state")
    if snapshot_after.digest != snapshot.digest:
        raise RuntimeError("R3 shadow panel mutated causal obstacle tracks")
    context, margin = _causal_context_from_snapshot(
        snapshot,
        agent_radius=float(env.agent_radius),
        map_size=tuple(float(value) for value in env.map_size),
    )
    return {
        "snapshot_digest_before": snapshot.digest,
        "snapshot_digest_after": snapshot_after.digest,
        "environment_fingerprint_before": environment_before.state_fingerprint,
        "environment_fingerprint_after": environment_after.state_fingerprint,
        "actor_hidden_digest_before": hidden_digest_before,
        "actor_hidden_digest_after": hidden_digest_after,
        "global_state_normalized": np.asarray(
            global_state_normalized, dtype=np.float32
        ).copy(),
        "raw_joint_mean_action": raw.copy(),
        "causal_safety_context": context,
        "current_safety_margin": np.float32(margin),
        "per_agent_margins": np.asarray(per_agent_margins, dtype=np.float32),
        "action_conditioned_ttc_s": action_ttc,
        "ttc_warning": ttc_warning,
        "ttc_emergency": ttc_emergency,
        "minimum_predicted_clearance_m": predicted_clearance,
        "reference_joint_actions": reference_actions,
        "reference_action_conditioned_ttc_s": np.asarray(
            [item[0] for item in reference_features], dtype=np.float32
        ),
        "reference_ttc_warning": np.asarray(
            [item[1] for item in reference_features], dtype=bool
        ),
        "reference_ttc_emergency": np.asarray(
            [item[2] for item in reference_features], dtype=bool
        ),
        "reference_minimum_predicted_clearance_m": np.asarray(
            [item[3] for item in reference_features], dtype=np.float32
        ),
        "reference_stage_costs": reference_stage_costs,
        "reference_total_costs": reference_total_costs,
        "reference_valid": reference_valid,
        "raw_stage_costs": np.asarray(label.raw_stage_costs, dtype=np.float32),
        "best_shadow_stage_costs": np.asarray(
            label.best_valid_shadow_stage_costs, dtype=np.float32
        ),
        "stage_avoidable_costs": np.asarray(
            label.stage_avoidable_costs, dtype=np.float32
        ),
        "raw_total_cost": np.float32(label.raw_cost),
        "best_shadow_total_cost": np.float32(label.best_valid_shadow_cost),
        "avoidable_total_cost": np.float32(label.avoidable_cost),
        "best_shadow_index": np.int16(
            -1 if label.best_shadow_index is None else label.best_shadow_index
        ),
        "valid_shadow_count": np.int16(label.valid_shadow_count),
        "shadow_candidate_count": np.int16(len(shadows)),
    }


class R3ShadowDatasetAccumulatorV1:
    def __init__(
        self,
        *,
        policy_seed: int,
        checkpoint_update: int,
        checkpoint_sha256: str,
        task_config_fingerprint: str,
        r2r_gate_report_sha256: str,
        expected_layout_count: int = 200,
        action_query_kind: str = "bounded_actor_mean",
    ) -> None:
        self.policy_seed = int(policy_seed)
        self.checkpoint_update = int(checkpoint_update)
        self.checkpoint_sha256 = str(checkpoint_sha256)
        self.task_config_fingerprint = str(task_config_fingerprint)
        self.r2r_gate_report_sha256 = str(r2r_gate_report_sha256)
        self.expected_layout_count = int(expected_layout_count)
        self.action_query_kind = str(action_query_kind)
        if self.action_query_kind not in {
            "bounded_actor_mean", "sampled_bounded_rollout_action"
        }:
            raise ValueError("unsupported R3 shadow action query kind")
        self.rows: list[dict[str, Any]] = []
        self.episodes: list[dict[str, Any]] = []

    def append_step(
        self,
        record: Mapping[str, Any],
        *,
        layout_id: int,
        episode_index: int,
        step: int,
        trace,
        terminated: bool,
        truncated: bool,
    ) -> None:
        by_stage: dict[str, list[bool]] = {
            stage: [] for stage in TRAINABLE_STAGES
        }
        for item in trace.records:
            if item.stage in by_stage:
                by_stage[item.stage].append(bool(item.eligible_for_actor_risk))
        agent_count = len({int(item.uav_id) for item in trace.records})
        stage_eligible = []
        for stage in TRAINABLE_STAGES:
            values = by_stage[stage]
            if len(values) != agent_count or len(set(values)) != 1:
                raise ValueError("R3 team-stage eligibility differs across UAVs")
            stage_eligible.append(values[0])
        terminal = bool(terminated or truncated or trace.terminal_transition)
        excluded = bool(trace.recovery_active or trace.safe_brake_active or terminal)
        if excluded:
            stage_eligible = [False, False, False]
        row = dict(record)
        row.update({
            "layout_id": np.int64(layout_id),
            "episode_index": np.int32(episode_index),
            "step": np.int32(step),
            "stage_eligible": np.asarray(stage_eligible, dtype=bool),
            "recovery_active": bool(trace.recovery_active),
            "safe_brake_active": bool(trace.safe_brake_active),
            "terminal_transition": terminal,
        })
        self.rows.append(row)

    def record_episode(
        self, *, layout_id: int, episode_index: int, outcome: Mapping[str, Any]
    ) -> None:
        self.episodes.append({
            "layout_id": int(layout_id),
            "episode_index": int(episode_index),
            **dict(outcome),
        })

    def finalize(self, output_dir: str | Path) -> dict[str, Any]:
        output = Path(output_dir)
        if output.exists():
            raise FileExistsError(f"R3 dataset output already exists: {output}")
        if not self.rows or not self.episodes:
            raise RuntimeError("cannot finalize an empty R3 dataset")
        layouts = sorted({int(row["layout_id"]) for row in self.rows})
        if len(layouts) != self.expected_layout_count:
            raise ValueError(
                f"R3 requires {self.expected_layout_count} layouts, got {len(layouts)}"
            )
        output.mkdir(parents=True, exist_ok=False)
        arrays = {
            key: np.asarray([row[key] for row in self.rows])
            for key in self.rows[0]
        }
        events_path = output / "shadow_events.npz"
        np.savez_compressed(events_path, **arrays)
        splits = _split_layouts(layouts)
        splits_payload = {
            "schema_version": "IAMAPPOR3LayoutSplitV2",
            "seed": R3_SPLIT_SEED,
            "partitions": splits,
            "partition_sizes": {key: len(value) for key, value in splits.items()},
        }
        splits_path = output / "layout_splits.json"
        save_json(splits_payload, splits_path)
        labels = arrays["stage_avoidable_costs"] > 1e-8
        eligible = arrays["stage_eligible"].astype(bool)
        layout_values = arrays["layout_id"].astype(np.int64)
        support: dict[str, Any] = {}
        for head, stage in enumerate(TRAINABLE_STAGES):
            partitions = {}
            for name, partition_layouts in splits.items():
                mask = np.isin(layout_values, partition_layouts) & eligible[:, head]
                partitions[name] = {
                    "positive": int(np.count_nonzero(labels[mask, head])),
                    "negative": int(np.count_nonzero(~labels[mask, head])),
                }
            enough = bool(
                min(partitions["train"].values()) >= 100
                and min(partitions["calibration"].values()) >= 20
                and min(partitions["threshold_selection"].values()) >= 20
                and min(partitions["test"].values()) >= 20
            )
            # R2R pre-registered Final Guard as unsupported for V1.
            supported = enough and stage != "final_joint_guard"
            support[stage] = {
                "partitions": partitions,
                "event_support_sufficient": enough,
                "supported": supported,
                "reason": (
                    None if supported else
                    "r2r_final_head_frozen_unsupported"
                    if stage == "final_joint_guard" else
                    "r3_event_support_insufficient"
                ),
            }
        support_path = output / "support.json"
        save_json({
            "schema_version": "IAMAPPOR3HeadSupportV2",
            "requirements": {
                "train_minimum_per_class": 100,
                "calibration_minimum_per_class": 20,
                "threshold_selection_minimum_per_class": 20,
                "test_minimum_per_class": 20,
            },
            "heads": support,
        }, support_path)
        episodes_path = output / "episodes.json"
        save_json({
            "schema_version": "IAMAPPOR3EpisodesV2",
            "episodes": self.episodes,
        }, episodes_path)
        invariance = {
            "snapshot_digest_equal": bool(np.all(
                arrays["snapshot_digest_before"]
                == arrays["snapshot_digest_after"]
            )),
            "environment_fingerprint_equal": bool(np.all(
                arrays["environment_fingerprint_before"]
                == arrays["environment_fingerprint_after"]
            )),
            "actor_hidden_digest_equal": bool(np.all(
                arrays["actor_hidden_digest_before"]
                == arrays["actor_hidden_digest_after"]
            )),
        }
        if not all(invariance.values()):
            raise RuntimeError("R3 shadow invariance certification failed")
        artifacts = (events_path, splits_path, support_path, episodes_path)
        manifest = {
            "schema_version": R3_DATASET_SCHEMA,
            "policy_seed": self.policy_seed,
            "checkpoint_update": self.checkpoint_update,
            "checkpoint_sha256": self.checkpoint_sha256,
            "task_config_fingerprint": self.task_config_fingerprint,
            "r2r_gate_report_sha256": self.r2r_gate_report_sha256,
            "layout_count": len(layouts),
            "team_step_count": len(self.rows),
            "shadow_candidate_count": 31,
            "action_query_kind": self.action_query_kind,
            "cost_contract": "team_mean_per_agent_stage_cost_v1",
            "stage_attribution_contract": "positive_improvement_conserves_total_v1",
            "feature_contract": R3_FEATURE_CONTRACT,
            "action_conditioned_ttc_version": ACTION_TTC_VERSION,
            "reference_bank_contract": (
                "hold_brake_goal_scales_025_050_075_100_v1"
                if "reference_joint_actions" in arrays else None
            ),
            "invariance": invariance,
            "supported_heads": {
                stage: bool(payload["supported"])
                for stage, payload in support.items()
            },
            "artifacts": {
                path.name: {
                    "sha256": _file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in artifacts
            },
        }
        save_json(manifest, output / "manifest.json")
        validate_r3_shadow_dataset(output)
        return manifest


def validate_r3_shadow_dataset(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != R3_DATASET_SCHEMA:
        raise ValueError("unsupported R3 dataset schema")
    if (
        manifest.get("feature_contract") != R3_FEATURE_CONTRACT
        or manifest.get("action_conditioned_ttc_version") != ACTION_TTC_VERSION
    ):
        raise ValueError("R3.2 feature contract/version mismatch")
    for filename, metadata in manifest["artifacts"].items():
        path = output / filename
        if not path.is_file() or _file_sha256(path) != metadata["sha256"]:
            raise ValueError(f"R3 artifact hash mismatch: {filename}")
    with np.load(output / "shadow_events.npz", allow_pickle=False) as data:
        required = {
            "layout_id", "global_state_normalized", "raw_joint_mean_action",
            "causal_safety_context", "current_safety_margin",
            "per_agent_margins", "action_conditioned_ttc_s", "ttc_warning",
            "ttc_emergency", "minimum_predicted_clearance_m",
            "stage_avoidable_costs", "avoidable_total_cost", "stage_eligible",
            "snapshot_digest_before", "snapshot_digest_after",
            "environment_fingerprint_before", "environment_fingerprint_after",
            "actor_hidden_digest_before", "actor_hidden_digest_after",
        }
        if not required.issubset(data.files):
            raise ValueError("R3 dataset is missing required arrays")
        rows = len(data["layout_id"])
        agent_count = data["raw_joint_mean_action"].shape[1]
        if data["per_agent_margins"].shape != (rows, agent_count, 3):
            raise ValueError("R3 per-agent safety margin shape is invalid")
        for name in (
            "action_conditioned_ttc_s", "ttc_warning", "ttc_emergency",
            "minimum_predicted_clearance_m",
        ):
            if data[name].shape != (rows, agent_count):
                raise ValueError(f"R3 action-conditioned feature shape is invalid: {name}")
        for name in (
            "per_agent_margins", "action_conditioned_ttc_s",
            "minimum_predicted_clearance_m",
        ):
            if not np.isfinite(data[name]).all():
                raise ValueError(f"R3 action-conditioned feature is non-finite: {name}")
        reference_fields = {
            "reference_joint_actions",
            "reference_action_conditioned_ttc_s",
            "reference_ttc_warning",
            "reference_ttc_emergency",
            "reference_minimum_predicted_clearance_m",
            "reference_stage_costs",
            "reference_total_costs",
            "reference_valid",
        }
        present_reference_fields = reference_fields.intersection(data.files)
        if present_reference_fields and present_reference_fields != reference_fields:
            raise ValueError("R3 reference bank is only partially recorded")
        if present_reference_fields:
            reference_count = 6
            if data["reference_joint_actions"].shape != (
                rows, reference_count, agent_count, 2
            ):
                raise ValueError("R3 reference action bank shape is invalid")
            for name in (
                "reference_action_conditioned_ttc_s",
                "reference_ttc_warning", "reference_ttc_emergency",
                "reference_minimum_predicted_clearance_m",
            ):
                if data[name].shape != (rows, reference_count, agent_count):
                    raise ValueError(f"R3 reference feature shape is invalid: {name}")
            if data["reference_stage_costs"].shape != (rows, reference_count, 3):
                raise ValueError("R3 reference stage cost shape is invalid")
            if data["reference_total_costs"].shape != (rows, reference_count):
                raise ValueError("R3 reference total cost shape is invalid")
            if data["reference_valid"].shape != (rows, reference_count):
                raise ValueError("R3 reference validity shape is invalid")
            for name in (
                "reference_joint_actions", "reference_action_conditioned_ttc_s",
                "reference_minimum_predicted_clearance_m",
                "reference_stage_costs", "reference_total_costs",
            ):
                if not np.isfinite(data[name]).all():
                    raise ValueError(f"R3 reference field is non-finite: {name}")
        if data["stage_avoidable_costs"].shape != (rows, 3):
            raise ValueError("R3 stage avoidable target shape is invalid")
        if not np.allclose(
            data["stage_avoidable_costs"].mean(axis=1),
            data["avoidable_total_cost"], atol=1e-6,
        ):
            raise ValueError("R3 stage attribution does not conserve total cost")
        if np.any(data["stage_avoidable_costs"] < -1e-8):
            raise ValueError("R3 avoidable costs must be nonnegative")
        for before, after in (
            ("snapshot_digest_before", "snapshot_digest_after"),
            ("environment_fingerprint_before", "environment_fingerprint_after"),
            ("actor_hidden_digest_before", "actor_hidden_digest_after"),
        ):
            if not np.array_equal(data[before], data[after]):
                raise ValueError(f"R3 invariance mismatch: {before}")
    return manifest


def _critic_features(data: Mapping[str, np.ndarray]) -> np.ndarray:
    rows = len(data["layout_id"])
    return np.concatenate((
        np.asarray(data["global_state_normalized"], dtype=np.float32),
        np.asarray(data["raw_joint_mean_action"], dtype=np.float32).reshape(
            rows, -1
        ),
        np.asarray(data["causal_safety_context"], dtype=np.float32),
        np.asarray(data["current_safety_margin"], dtype=np.float32).reshape(-1, 1),
        np.asarray(data["per_agent_margins"], dtype=np.float32).reshape(rows, -1),
        np.asarray(data["action_conditioned_ttc_s"], dtype=np.float32),
        np.asarray(data["ttc_warning"], dtype=np.float32),
        np.asarray(data["ttc_emergency"], dtype=np.float32),
        np.asarray(
            data["minimum_predicted_clearance_m"], dtype=np.float32
        ),
    ), axis=1).astype(np.float32)


def _binary_operating_point(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float,
) -> dict[str, float | int]:
    target = np.asarray(labels, dtype=bool)
    scores = np.asarray(probabilities, dtype=np.float64)
    predicted = scores >= float(threshold)
    true_positive = int(np.count_nonzero(predicted & target))
    false_positive = int(np.count_nonzero(predicted & ~target))
    false_negative = int(np.count_nonzero(~predicted & target))
    true_negative = int(np.count_nonzero(~predicted & ~target))
    return {
        "threshold": float(threshold),
        "recall": true_positive / max(true_positive + false_negative, 1),
        "precision": true_positive / max(true_positive + false_positive, 1),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }


def select_validation_operating_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    layout_ids: np.ndarray,
    *,
    point_minimum_recall: float = 0.85,
    lower_minimum_recall: float = 0.80,
    lower_minimum_precision: float = 0.50,
    bootstrap_replicates: int = 2000,
    seed: int = R3_SPLIT_SEED,
) -> dict[str, Any]:
    """Select a robust threshold on an isolated layout partition.

    R3.2 requires a five-point recall margin and layout-bootstrap lower bounds.
    Temperature calibration uses different layouts, and test layouts remain
    untouched until this threshold has been frozen.
    """

    target = np.asarray(labels, dtype=bool)
    scores = np.asarray(probabilities, dtype=np.float64)
    layouts = np.asarray(layout_ids, dtype=np.int64)
    if target.ndim != 1 or scores.shape != target.shape or layouts.shape != target.shape:
        raise ValueError(
            "threshold selection requires equal one-dimensional arrays"
        )
    if not np.isfinite(scores).all() or not np.any(target) or np.all(target):
        raise ValueError("threshold selection requires finite two-class validation data")
    unique_layouts = np.unique(layouts)
    if len(unique_layouts) < 10 or int(bootstrap_replicates) <= 0:
        raise ValueError("robust threshold selection requires at least ten layouts")

    candidates = np.linspace(0.0, 1.0, 201, dtype=np.float64)
    predicted = scores[:, None] >= candidates[None, :]
    layout_tp = []
    layout_fp = []
    layout_fn = []
    for layout in unique_layouts:
        mask = layouts == layout
        local_target = target[mask, None]
        local_predicted = predicted[mask]
        layout_tp.append(np.count_nonzero(local_predicted & local_target, axis=0))
        layout_fp.append(np.count_nonzero(local_predicted & ~local_target, axis=0))
        layout_fn.append(np.count_nonzero(~local_predicted & local_target, axis=0))
    layout_tp = np.asarray(layout_tp, dtype=np.float64)
    layout_fp = np.asarray(layout_fp, dtype=np.float64)
    layout_fn = np.asarray(layout_fn, dtype=np.float64)

    rng = np.random.default_rng(int(seed))
    sampled = rng.integers(
        0, len(unique_layouts),
        size=(int(bootstrap_replicates), len(unique_layouts)),
    )
    weights = np.eye(len(unique_layouts), dtype=np.float64)[sampled].sum(axis=1)
    boot_tp = weights @ layout_tp
    boot_fp = weights @ layout_fp
    boot_fn = weights @ layout_fn
    boot_recall = np.divide(
        boot_tp, boot_tp + boot_fn,
        out=np.zeros_like(boot_tp), where=(boot_tp + boot_fn) > 0,
    )
    boot_precision = np.divide(
        boot_tp, boot_tp + boot_fp,
        out=np.zeros_like(boot_tp), where=(boot_tp + boot_fp) > 0,
    )
    recall_lower = np.quantile(boot_recall, 0.025, axis=0)
    precision_lower = np.quantile(boot_precision, 0.025, axis=0)
    points = []
    for index, threshold in enumerate(candidates):
        point = _binary_operating_point(target, scores, float(threshold))
        point["recall_lower_95_layout_bootstrap"] = float(recall_lower[index])
        point["precision_lower_95_layout_bootstrap"] = float(
            precision_lower[index]
        )
        points.append(point)
    feasible = [
        point for point in points
        if point["recall"] >= float(point_minimum_recall)
        and point["recall_lower_95_layout_bootstrap"]
        >= float(lower_minimum_recall)
        and point["precision_lower_95_layout_bootstrap"]
        >= float(lower_minimum_precision)
    ]
    if feasible:
        selected = max(
            feasible,
            key=lambda point: (
                point["precision"], point["threshold"],
                point["recall_lower_95_layout_bootstrap"],
            ),
        )
        passed = True
    else:
        precision_feasible = [
            point for point in points
            if point["precision_lower_95_layout_bootstrap"]
            >= float(lower_minimum_precision)
        ]
        pool = precision_feasible or points
        selected = max(
            pool,
            key=lambda point: (
                point["recall_lower_95_layout_bootstrap"],
                point["precision_lower_95_layout_bootstrap"],
                point["threshold"],
            ),
        )
        passed = False
    return {
        "schema_version": "ValidationOperatingThresholdV2",
        "selection_partition": "threshold_selection_layouts_only",
        "selection_rule": (
            "max_precision_subject_to_point_recall_ge_0.85_and_layout_"
            "bootstrap_recall_lower_95_ge_0.80_and_precision_lower_95_ge_0.50;"
            "ties_higher_threshold_then_recall_lower"
        ),
        "point_minimum_recall": float(point_minimum_recall),
        "lower_minimum_recall": float(lower_minimum_recall),
        "lower_minimum_precision": float(lower_minimum_precision),
        "bootstrap_unit": "layout",
        "bootstrap_replicates": int(bootstrap_replicates),
        "layout_count": int(len(unique_layouts)),
        "selection_passed": bool(passed),
        "candidate_count": len(points),
        "selected": selected,
        "curve_201": points,
    }


def _severity_baseline_diagnostic(
    *,
    train_targets: np.ndarray,
    test_targets: np.ndarray,
    test_predictions: np.ndarray,
    test_layout_ids: np.ndarray,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Require severity regression to beat a train-median constant baseline."""

    train = np.asarray(train_targets, dtype=np.float64)
    target = np.asarray(test_targets, dtype=np.float64)
    prediction = np.asarray(test_predictions, dtype=np.float64)
    layouts = np.asarray(test_layout_ids, dtype=np.int64)
    if not len(train) or target.shape != prediction.shape or target.shape != layouts.shape:
        raise ValueError("invalid severity baseline diagnostic arrays")
    baseline_value = float(np.median(train))
    model_error = np.abs(prediction - target)
    baseline_error = np.abs(baseline_value - target)
    improvement = baseline_error - model_error
    unique_layouts = np.unique(layouts)
    if len(unique_layouts) < 2 or int(bootstrap_replicates) <= 0:
        raise ValueError("severity diagnostic requires multiple layouts")
    rng = np.random.default_rng(int(seed))
    samples = []
    rows_by_layout = {
        layout: np.flatnonzero(layouts == layout) for layout in unique_layouts
    }
    for _ in range(int(bootstrap_replicates)):
        sampled = rng.choice(unique_layouts, len(unique_layouts), replace=True)
        rows = np.concatenate([rows_by_layout[item] for item in sampled])
        samples.append(float(np.mean(improvement[rows])))
    model_mae = float(np.mean(model_error))
    baseline_mae = float(np.mean(baseline_error))
    improvement_mean = baseline_mae - model_mae
    improvement_lower = float(np.quantile(samples, 0.025))
    return {
        "schema_version": "SeverityBaselineDiagnosticV1",
        "baseline": "train_positive_target_median",
        "baseline_value": baseline_value,
        "model_mae": model_mae,
        "baseline_mae": baseline_mae,
        "mae_improvement": improvement_mean,
        "relative_mae_improvement": (
            improvement_mean / baseline_mae if baseline_mae > 0.0 else 0.0
        ),
        "mae_improvement_lower_95_layout_bootstrap": improvement_lower,
        "passed": bool(improvement_lower > 0.0),
    }


def train_and_certify_r3_twin_critic(
    *,
    dataset_dir: str | Path,
    output_dir: str | Path,
    device: str = "cuda",
    epochs: int = 40,
    batch_size: int = 1024,
    learning_rate: float = 3e-4,
    bootstrap_replicates: int = 2000,
    seed: int = R3_SPLIT_SEED,
) -> dict[str, Any]:
    dataset = Path(dataset_dir)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"R3 critic output already exists: {output}")
    manifest = validate_r3_shadow_dataset(dataset)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for R3 critic training but is unavailable")
    torch_device = torch.device(device)
    with np.load(dataset / "shadow_events.npz", allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    splits = json.loads(
        (dataset / "layout_splits.json").read_text(encoding="utf-8")
    )["partitions"]
    support = json.loads(
        (dataset / "support.json").read_text(encoding="utf-8")
    )["heads"]
    enabled = tuple(bool(support[stage]["supported"]) for stage in TRAINABLE_STAGES)
    if not enabled[0] or not enabled[1] or enabled[2]:
        raise RuntimeError("R3.2 requires supported Initial/TTC and disabled Final")
    layouts = np.asarray(data["layout_id"], dtype=np.int64)
    eligible = np.asarray(data["stage_eligible"], dtype=bool)
    common_eligible = eligible[:, 0]
    if not np.all(eligible[:, np.asarray(enabled)] == common_eligible[:, None]):
        raise ValueError("R3 enabled heads have different eligibility rows")
    indices = {
        name: np.flatnonzero(np.isin(layouts, ids) & common_eligible)
        for name, ids in splits.items()
    }
    features = _critic_features(data)
    severity_targets = np.asarray(data["stage_avoidable_costs"], dtype=np.float32)
    material_targets = (severity_targets > 1e-8).astype(np.float32)
    input_dim = int(features.shape[1])
    train_feature_mean = features[indices["train"]].mean(
        axis=0, dtype=np.float64
    ).astype(np.float32)
    train_feature_std = features[indices["train"]].std(
        axis=0, dtype=np.float64
    ).astype(np.float32)
    train_feature_std[train_feature_std < 1e-6] = 1.0
    normalized_features = (
        features - train_feature_mean
    ) / train_feature_std
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    twin = TwinInterventionCriticV1(input_dim).to(torch_device)
    twin.feature_mean.copy_(torch.as_tensor(
        train_feature_mean, device=torch_device
    ))
    twin.feature_std.copy_(torch.as_tensor(
        train_feature_std, device=torch_device
    ))
    twin.head_enabled.copy_(torch.as_tensor(enabled, device=torch_device))
    temperatures = []
    prior_logit_offsets = []
    training_history: list[dict[str, float]] = []
    for critic_index, critic in enumerate(twin.critics):
        local_seed = int(seed) + 10_000 * critic_index
        torch.manual_seed(local_seed)
        np.random.seed(local_seed)
        optimizer = torch.optim.Adam(critic.parameters(), lr=float(learning_rate))
        train_index = indices["train"]
        y_train = torch.as_tensor(
            material_targets[train_index], device=torch_device
        )
        z_train = torch.as_tensor(
            severity_targets[train_index], device=torch_device
        )
        x_train = torch.as_tensor(
            normalized_features[train_index], device=torch_device
        )
        enabled_tensor = torch.as_tensor(enabled, dtype=torch.bool, device=torch_device)
        positive = y_train[:, enabled_tensor].sum(dim=0)
        negative = y_train.shape[0] - positive
        weights_enabled = negative / positive.clamp_min(1.0)
        positive_weights = torch.ones(3, device=torch_device)
        positive_weights[enabled_tensor] = weights_enabled
        prior_logit_offsets.append(torch.log(positive_weights))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(local_seed)
        critic.train()
        for epoch in range(int(epochs)):
            order = torch.randperm(len(train_index), generator=generator)
            epoch_losses = []
            for start in range(0, len(order), int(batch_size)):
                batch = order[start:start + int(batch_size)].to(torch_device)
                optimizer.zero_grad(set_to_none=True)
                logits, severity = critic(x_train[batch])
                loss = intervention_critic_loss(
                    logits, severity, y_train[batch], z_train[batch],
                    positive_weights=positive_weights.tolist(),
                    head_enabled=enabled,
                )
                loss.total.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
                optimizer.step()
                epoch_losses.append(float(loss.total.detach().cpu()))
            training_history.append({
                "critic": float(critic_index),
                "epoch": float(epoch + 1),
                "loss": float(np.mean(epoch_losses)),
            })
        critic.eval()
        calibration_index = indices["calibration"]
        with torch.no_grad():
            calibration_logits, _ = critic(torch.as_tensor(
                normalized_features[calibration_index], device=torch_device
            ))
        corrected_calibration_logits = (
            calibration_logits - prior_logit_offsets[-1]
        )
        temperature = fit_stage_temperatures(
            corrected_calibration_logits,
            torch.as_tensor(
                material_targets[calibration_index], device=torch_device
            ),
            head_enabled=enabled,
        )
        temperatures.append(temperature)
    twin.temperatures.copy_(torch.stack(temperatures))
    twin.prior_logit_offsets.copy_(torch.stack(prior_logit_offsets))

    threshold_index = indices["threshold_selection"]
    x_threshold = torch.as_tensor(
        normalized_features[threshold_index], device=torch_device
    )
    test_index = indices["test"]
    x_test = torch.as_tensor(
        normalized_features[test_index], device=torch_device
    )
    threshold_probabilities = []
    probabilities = []
    severities = []
    with torch.no_grad():
        for critic_index, critic in enumerate(twin.critics):
            threshold_logits, _ = critic(x_threshold)
            threshold_probabilities.append(torch.sigmoid(
                (threshold_logits - twin.prior_logit_offsets[critic_index])
                / twin.temperatures[critic_index].clamp_min(1e-3)
            ))
            logits, severity = critic(x_test)
            probabilities.append(torch.sigmoid(
                (logits - twin.prior_logit_offsets[critic_index])
                / twin.temperatures[critic_index].clamp_min(1e-3)
            ))
            severities.append(severity)
    conservative_threshold_probability = torch.stack(
        threshold_probabilities
    ).max(dim=0).values.cpu().numpy()
    conservative_probability = torch.stack(probabilities).max(dim=0).values.cpu().numpy()
    conservative_severity = torch.stack(severities).max(dim=0).values.cpu().numpy()
    heads: dict[str, Any] = {}
    all_pass = True
    decision_thresholds = np.full(3, np.nan, dtype=np.float32)
    for head, stage in enumerate(TRAINABLE_STAGES):
        if not enabled[head]:
            heads[stage] = {
                "status": "unsupported",
                "reason": support[stage]["reason"],
            }
            continue
        threshold_selection = select_validation_operating_threshold(
            material_targets[threshold_index, head],
            conservative_threshold_probability[:, head],
            layouts[threshold_index],
            bootstrap_replicates=bootstrap_replicates,
            seed=int(seed) + 100 + head,
        )
        threshold = float(threshold_selection["selected"]["threshold"])
        decision_thresholds[head] = threshold
        gate = evaluate_team_learnability_gate_v2(
            train_labels=material_targets[indices["train"], head],
            heldout_labels=material_targets[test_index, head],
            heldout_probabilities=conservative_probability[:, head],
            heldout_layout_ids=layouts[test_index],
            bootstrap_replicates=bootstrap_replicates,
            seed=int(seed) + head,
            threshold=threshold,
        )
        positive = material_targets[test_index, head] > 0.5
        train_positive = material_targets[indices["train"], head] > 0.5
        severity_diagnostic = _severity_baseline_diagnostic(
            train_targets=severity_targets[indices["train"], head][train_positive],
            test_targets=severity_targets[test_index, head][positive],
            test_predictions=conservative_severity[positive, head],
            test_layout_ids=layouts[test_index][positive],
            bootstrap_replicates=bootstrap_replicates,
            seed=int(seed) + 200 + head,
        )
        heads[stage] = {
            "status": "evaluated",
            "gate": gate.to_dict(),
            "threshold_selection": threshold_selection,
            "frozen_test_threshold": threshold,
            "severity_baseline_diagnostic": severity_diagnostic,
        }
        all_pass = (
            all_pass
            and bool(threshold_selection["selection_passed"])
            and gate.passed
            and bool(severity_diagnostic["passed"])
        )
    output.mkdir(parents=True, exist_ok=False)
    prediction_audit_path = output / "prediction_audit.npz"
    np.savez_compressed(
        prediction_audit_path,
        threshold_selection_layout_id=layouts[threshold_index],
        threshold_selection_labels=material_targets[threshold_index],
        threshold_selection_probabilities=conservative_threshold_probability,
        test_layout_id=layouts[test_index],
        test_labels=material_targets[test_index],
        test_probabilities=conservative_probability,
        test_severity_targets=severity_targets[test_index],
        test_severity_predictions=conservative_severity,
    )
    dataset_manifest_hash = _file_sha256(dataset / "manifest.json")
    report = {
        "schema_version": R3_REPORT_SCHEMA,
        "dataset": str(dataset.resolve()),
        "dataset_manifest_sha256": dataset_manifest_hash,
        "input_dim": input_dim,
        "feature_contract": R3_FEATURE_CONTRACT,
        "feature_normalization": "train_layouts_mean_std_v1",
        "prediction_audit": {
            "path": str(prediction_audit_path.resolve()),
            "sha256": _file_sha256(prediction_audit_path),
            "size_bytes": prediction_audit_path.stat().st_size,
        },
        "enabled_heads": {
            stage: bool(enabled[index])
            for index, stage in enumerate(TRAINABLE_STAGES)
        },
        "heads": heads,
        "snapshot_invariance": manifest["invariance"],
        "layout_partitions": {
            name: len(values) for name, values in splits.items()
        },
        "all_enabled_heads_pass": bool(all_pass),
        "certified_for_r4_lambda_calibration": bool(all_pass),
        "certified_for_actor_training": False,
        "actor_training_block_reason": (
            "lambda_gradient_ratio_not_calibrated"
            if all_pass else "r3_2_certification_failed"
        ),
        "protocol": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "bootstrap_replicates": int(bootstrap_replicates),
            "classification_loss": "frequency_weighted_bce",
            "classification_calibration": (
                "weighted_bce_prior_correction_then_temperature_on_"
                "isolated_calibration_layouts"
            ),
            "decision_threshold_selection": (
                "isolated_threshold_layouts_point_recall_ge_0.85_and_"
                "layout_bootstrap_recall_lower_95_ge_0.80_and_"
                "precision_lower_95_ge_0.50"
            ),
            "test_threshold_policy": (
                "frozen_isolated_threshold_selection_value_per_head"
            ),
            "severity_loss": "positive_stage_avoidable_cost_huber",
            "severity_gate": (
                "test_model_mae_beats_train_positive_median_baseline_with_"
                "layout_bootstrap_95pct_lower_improvement_above_zero"
            ),
            "twin_test_aggregation": "per_head_conservative_max",
        },
    }
    report_path = output / "certification.json"
    save_json(report, report_path)
    save_json({
        "schema_version": "IAMAPPOR3TrainingHistoryV3",
        "rows": training_history,
    }, output / "training_history.json")
    if all_pass:
        torch.save({
            "schema_version": R3_ARTIFACT_SCHEMA,
            "input_dim": input_dim,
            "state_dict": twin.state_dict(),
            "temperatures": twin.temperatures.detach().cpu(),
            "prior_logit_offsets": twin.prior_logit_offsets.detach().cpu(),
            "feature_mean": twin.feature_mean.detach().cpu(),
            "feature_std": twin.feature_std.detach().cpu(),
            "head_enabled": twin.head_enabled.detach().cpu(),
            "decision_thresholds": torch.as_tensor(decision_thresholds),
            "dataset_manifest_sha256": dataset_manifest_hash,
            "certification_report_sha256": _file_sha256(report_path),
            "certified_for_actor_training": False,
            "requires_lambda_gradient_calibration": True,
        }, output / "r3_twin_critic.pt")
    return report
