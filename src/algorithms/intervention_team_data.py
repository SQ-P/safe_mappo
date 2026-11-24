"""Offline derivation of one-row-per-team-step IA-MAPPO R2R data."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.algorithms.intervention_data import (
    TRAINABLE_STAGES,
    grouped_layout_folds,
    stage_support,
    validate_intervention_dataset,
)
from src.utils.io_utils import load_yaml, read_jsonl, save_json


TEAM_DATASET_SCHEMA = "IAMAPPOR2TeamDatasetV3"
ACTION_TTC_VERSION = "observable_raw_action_ttc_v1"
TEAM_AGGREGATION_VERSION = "any_classification_mean_severity_v1"


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def action_conditioned_obstacle_features(
    *,
    current_velocity: Sequence[float],
    raw_action: Sequence[float],
    obstacle_tracks: Sequence[Mapping[str, Any]],
    agent_radius: float,
    obstacle_safety_margin: float,
    invariant_buffer: float,
    max_acceleration: float,
    max_speed: float,
    dt: float,
    horizon_seconds: float,
    warning_ttc_seconds: float,
    emergency_ttc_seconds: float,
    no_risk_ttc: float = 1_000.0,
    no_obstacle_clearance: float = 1_000.0,
) -> tuple[float, bool, bool, float]:
    """Causal raw-action TTC using the same track semantics as the TTC guard."""

    velocity = np.asarray(current_velocity, dtype=np.float64)
    action = np.clip(np.asarray(raw_action, dtype=np.float64), -1.0, 1.0)
    if velocity.shape != (2,) or action.shape != (2,):
        raise ValueError("velocity and raw action must be two-dimensional")
    acceleration_delta = action * float(max_acceleration) * float(dt)
    next_velocity = np.clip(
        velocity + acceleration_delta, -float(max_speed), float(max_speed)
    )
    effective_delta = next_velocity - velocity
    minimum_ttc = float(no_risk_ttc)
    minimum_clearance = float(no_obstacle_clearance)
    for obstacle in obstacle_tracks:
        if not bool(obstacle.get("velocity_valid", False)):
            continue
        relative_position = np.asarray(
            obstacle["relative_position"], dtype=np.float64
        )
        stored_relative_velocity = np.asarray(
            obstacle["relative_velocity"], dtype=np.float64
        )
        relative_velocity = stored_relative_velocity - effective_delta
        protected_radius = (
            float(agent_radius) + float(obstacle["radius"])
            + float(obstacle_safety_margin) + float(invariant_buffer)
        )
        speed_squared = float(np.dot(relative_velocity, relative_velocity))
        closing = float(np.dot(relative_position, relative_velocity)) < 0.0
        if speed_squared <= 1e-12:
            closest_time = 0.0
        else:
            closest_time = float(np.clip(
                -np.dot(relative_position, relative_velocity) / speed_squared,
                0.0,
                float(horizon_seconds),
            ))
        closest_clearance = float(
            np.linalg.norm(
                relative_position + closest_time * relative_velocity
            ) - protected_radius
        )
        minimum_clearance = min(minimum_clearance, closest_clearance)
        if not closing:
            continue
        a = speed_squared
        b = 2.0 * float(np.dot(relative_position, relative_velocity))
        c = float(
            np.dot(relative_position, relative_position)
            - protected_radius * protected_radius
        )
        if c <= 0.0:
            minimum_ttc = 0.0
            minimum_clearance = min(
                minimum_clearance,
                float(np.linalg.norm(relative_position) - protected_radius),
            )
            continue
        discriminant = b * b - 4.0 * a * c
        if a <= 1e-12 or discriminant < 0.0:
            continue
        roots = [
            root for root in (
                (-b - np.sqrt(discriminant)) / (2.0 * a),
                (-b + np.sqrt(discriminant)) / (2.0 * a),
            )
            if root >= 0.0
        ]
        if roots:
            minimum_ttc = min(minimum_ttc, float(min(roots)))
    warning = bool(
        minimum_ttc <= float(warning_ttc_seconds)
        and minimum_clearance < 0.0
    )
    emergency = bool(
        minimum_ttc <= float(emergency_ttc_seconds)
        and minimum_clearance < 0.0
    )
    return minimum_ttc, warning, emergency, minimum_clearance


def padded_snapshot_obstacle_tracks(
    snapshot: Mapping[str, Any], *, obstacle_slots: int,
) -> np.ndarray:
    """Encode causal observable tracks as [UAV, slot, 6] without simulator truth."""

    tracks = snapshot["obstacle_tracks"]
    result = np.zeros((len(tracks), int(obstacle_slots), 6), dtype=np.float32)
    for agent, rows in enumerate(tracks):
        if len(rows) > int(obstacle_slots):
            raise ValueError("causal snapshot exceeds configured obstacle slots")
        for slot, obstacle in enumerate(rows):
            result[agent, slot, :2] = obstacle["relative_position"]
            result[agent, slot, 2:4] = obstacle["relative_velocity"]
            result[agent, slot, 4] = float(obstacle["radius"])
            result[agent, slot, 5] = float(bool(obstacle["velocity_valid"]))
    return result


def torch_action_conditioned_obstacle_features(
    *,
    current_velocity: torch.Tensor,
    raw_action: torch.Tensor,
    obstacle_tracks: torch.Tensor,
    agent_radius: float,
    obstacle_safety_margin: float,
    invariant_buffer: float,
    max_acceleration: float,
    max_speed: float,
    dt: float,
    horizon_seconds: float,
    warning_ttc_seconds: float,
    emergency_ttc_seconds: float,
    no_risk_ttc: float = 1_000.0,
    no_obstacle_clearance: float = 1_000.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable batched equivalent of the registered V1 TTC features."""

    if current_velocity.shape != raw_action.shape or current_velocity.shape[-1] != 2:
        raise ValueError("velocity and raw_action must have equal [..., UAV, 2] shapes")
    if obstacle_tracks.shape[:-2] != current_velocity.shape[:-1] or obstacle_tracks.shape[-1] != 6:
        raise ValueError("obstacle_tracks must have shape [..., UAV, slots, 6]")
    action = raw_action.clamp(-1.0, 1.0)
    next_velocity = (
        current_velocity + action * float(max_acceleration) * float(dt)
    ).clamp(-float(max_speed), float(max_speed))
    relative_position = obstacle_tracks[..., :2]
    relative_velocity = obstacle_tracks[..., 2:4] - (
        next_velocity - current_velocity
    ).unsqueeze(-2)
    radius = obstacle_tracks[..., 4]
    valid = obstacle_tracks[..., 5] > 0.5
    protected_radius = (
        radius + float(agent_radius) + float(obstacle_safety_margin)
        + float(invariant_buffer)
    )
    speed_squared = (relative_velocity * relative_velocity).sum(dim=-1)
    dot = (relative_position * relative_velocity).sum(dim=-1)
    closest_time = (-dot / speed_squared.clamp_min(1e-12)).clamp(
        0.0, float(horizon_seconds)
    )
    closest_clearance = torch.linalg.vector_norm(
        relative_position + closest_time.unsqueeze(-1) * relative_velocity,
        dim=-1,
    ) - protected_radius
    clearance_fill = torch.full_like(closest_clearance, float(no_obstacle_clearance))
    minimum_clearance = torch.where(valid, closest_clearance, clearance_fill).min(dim=-1).values

    c = (relative_position * relative_position).sum(dim=-1) - protected_radius.square()
    b = 2.0 * dot
    discriminant = b.square() - 4.0 * speed_squared * c
    # sqrt has an infinite derivative at exactly zero.  A tangential contact
    # is common enough online that the mathematically valid forward value can
    # otherwise produce 0*inf=NaN through torch.where during backpropagation.
    sqrt_discriminant = discriminant.clamp_min(1e-8).sqrt()
    denominator = (2.0 * speed_squared).clamp_min(1e-12)
    root_near = (-b - sqrt_discriminant) / denominator
    root_far = (-b + sqrt_discriminant) / denominator
    infinity = torch.full_like(root_near, float(no_risk_ttc))
    roots = torch.stack((
        torch.where(root_near >= 0.0, root_near, infinity),
        torch.where(root_far >= 0.0, root_far, infinity),
    ), dim=-1).min(dim=-1).values
    collision_root = torch.where(c <= 0.0, torch.zeros_like(roots), roots)
    collision_valid = (
        valid & (dot < 0.0) & (speed_squared > 1e-12)
        & ((c <= 0.0) | (discriminant >= 0.0))
    )
    per_track_ttc = torch.where(collision_valid, collision_root, infinity)
    minimum_ttc = per_track_ttc.min(dim=-1).values
    warning = (
        (minimum_ttc <= float(warning_ttc_seconds)) & (minimum_clearance < 0.0)
    ).to(raw_action.dtype)
    emergency = (
        (minimum_ttc <= float(emergency_ttc_seconds)) & (minimum_clearance < 0.0)
    ).to(raw_action.dtype)
    return minimum_ttc, warning, emergency, minimum_clearance


def torch_smooth_minimum_obstacle_clearance(
    *,
    current_velocity: torch.Tensor,
    raw_action: torch.Tensor,
    obstacle_tracks: torch.Tensor,
    agent_radius: float,
    obstacle_safety_margin: float,
    invariant_buffer: float,
    max_acceleration: float,
    max_speed: float,
    dt: float,
    horizon_seconds: float,
    time_samples: int = 33,
    softmin_temperature_m: float = 0.25,
    no_obstacle_clearance: float = 1_000.0,
) -> torch.Tensor:
    """Return a bounded-gradient clearance surrogate for Actor optimization.

    The registered R3 feature keeps using the exact closest-approach clearance
    in its forward pass.  R4.1 uses this time-grid soft minimum only as its
    backward surrogate, avoiding the tangent-root singularity that previously
    produced NaNs while retaining a causal, action-conditioned obstacle path.
    """

    if current_velocity.shape != raw_action.shape or current_velocity.shape[-1] != 2:
        raise ValueError("velocity and raw_action must have equal [..., UAV, 2] shapes")
    if obstacle_tracks.shape[:-2] != current_velocity.shape[:-1] or obstacle_tracks.shape[-1] != 6:
        raise ValueError("obstacle_tracks must have shape [..., UAV, slots, 6]")
    if int(time_samples) < 2:
        raise ValueError("time_samples must be at least two")
    if float(softmin_temperature_m) <= 0.0:
        raise ValueError("softmin_temperature_m must be positive")

    action = raw_action.clamp(-1.0, 1.0)
    next_velocity = (
        current_velocity + action * float(max_acceleration) * float(dt)
    ).clamp(-float(max_speed), float(max_speed))
    relative_position = obstacle_tracks[..., :2]
    relative_velocity = obstacle_tracks[..., 2:4] - (
        next_velocity - current_velocity
    ).unsqueeze(-2)
    protected_radius = (
        obstacle_tracks[..., 4] + float(agent_radius)
        + float(obstacle_safety_margin) + float(invariant_buffer)
    )
    valid = obstacle_tracks[..., 5] > 0.5
    times = torch.linspace(
        0.0, float(horizon_seconds), int(time_samples),
        dtype=raw_action.dtype, device=raw_action.device,
    )
    predicted = (
        relative_position.unsqueeze(-2)
        + times.reshape(*([1] * (relative_velocity.ndim - 1)), -1, 1)
        * relative_velocity.unsqueeze(-2)
    )
    clearance = torch.sqrt(
        predicted.square().sum(dim=-1).clamp_min(1e-6)
    ) - protected_radius.unsqueeze(-1)
    fill = torch.full_like(clearance, float(no_obstacle_clearance))
    clearance = torch.where(valid.unsqueeze(-1), clearance, fill)
    flattened = clearance.flatten(start_dim=-2)
    temperature = float(softmin_temperature_m)
    smooth = -temperature * torch.logsumexp(
        -flattened / temperature, dim=-1
    )
    has_obstacle = valid.any(dim=-1)
    return torch.where(
        has_obstacle,
        smooth,
        torch.full_like(smooth, float(no_obstacle_clearance)),
    )


def per_agent_snapshot_margins(
    snapshot: Mapping[str, Any],
    *,
    agent_radius: float,
    map_size: Sequence[float],
) -> np.ndarray:
    safety_state = tuple(map(float, snapshot["safety_state"]))
    if len(safety_state) != 5:
        raise ValueError("team V3 derivation requires a V2 causal snapshot")
    positions = np.asarray(
        [row[:2] for row in snapshot["team_observation"]], dtype=np.float64
    )
    map_values = np.asarray(map_size, dtype=np.float64)
    domain_margin = float(agent_radius) + safety_state[0] + safety_state[1]
    margins = np.full((len(positions), 3), 1_000.0, dtype=np.float32)
    for agent in range(len(positions)):
        margins[agent, 0] = float(np.min(np.concatenate((
            positions[agent] - domain_margin,
            map_values - domain_margin - positions[agent],
        ))))
        teammate_values = [
            float(np.linalg.norm(positions[agent] - positions[other]))
            - safety_state[2]
            for other in range(len(positions)) if other != agent
        ]
        if teammate_values:
            margins[agent, 1] = min(teammate_values)
        obstacle_values = [
            float(np.linalg.norm(obstacle["relative_position"]))
            - float(agent_radius) - float(obstacle["radius"]) - safety_state[3]
            for obstacle in snapshot["obstacle_tracks"][agent]
        ]
        if obstacle_values:
            margins[agent, 2] = min(obstacle_values)
    return margins


def aggregate_team_stage_costs(
    delta_v_mps: np.ndarray,
    *,
    material_delta_v_mps: float = 0.05,
    severity_scale_mps: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    delta = np.asarray(delta_v_mps, dtype=np.float32)
    if delta.ndim != 2 or delta.shape[1] != len(TRAINABLE_STAGES):
        raise ValueError("delta_v_mps must have shape [agents, stages]")
    material = delta >= float(material_delta_v_mps)
    per_agent_cost = material.astype(np.float32) * (
        1.0 + np.clip(delta / float(severity_scale_mps), 0.0, 1.0)
    )
    team_labels = np.any(material, axis=0)
    team_severity = np.mean(per_agent_cost, axis=0, dtype=np.float32)
    total_cost = float(np.mean(team_severity))
    return per_agent_cost, team_labels, team_severity, total_cost


def _all_equal(values: np.ndarray, *, atol: float = 1e-6) -> bool:
    return bool(np.allclose(values, values[0], atol=atol, equal_nan=True))


def derive_team_dataset_v3(
    *,
    source_dataset_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    source = Path(source_dataset_dir)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"team dataset output already exists: {destination}")
    source_manifest = validate_intervention_dataset(source)
    if source_manifest.get("schema_version") != "IAMAPPOR2DatasetV2":
        raise ValueError("TeamDatasetV3 requires an IAMAPPOR2DatasetV2 source")
    config = load_yaml(config_path)
    env = dict(config["env"])
    safety = dict(config["safety_execution"])
    initial_guard = dict(safety["initial_joint_guard"])
    obstacle_ttc = dict(safety["obstacle_ttc"])
    num_agents = int(env["num_agents"])
    if num_agents != 3:
        raise ValueError("the registered TeamDatasetV3 is frozen to three UAVs")
    dynamics = {
        "agent_radius": float(env.get("agent_radius", 1.5)),
        "map_size": tuple(map(float, env.get("map_size", (100.0, 100.0)))),
        "max_acceleration": float(env.get("agent_max_acceleration", 1.0)),
        "max_speed": float(env.get("agent_max_speed", 2.0)),
        "dt": float(env.get("dt", 0.5)),
        "invariant_buffer": float(
            initial_guard.get("invariant_buffer_m", 0.25)
        ),
        "horizon_seconds": float(obstacle_ttc.get("horizon_seconds", 10.0)),
        "warning_ttc_seconds": float(
            obstacle_ttc.get("warning_ttc_seconds", 5.0)
        ),
        "emergency_ttc_seconds": float(
            obstacle_ttc.get("emergency_ttc_seconds", 2.5)
        ),
    }
    with np.load(source / "events.npz", allow_pickle=False) as loaded:
        events = {key: loaded[key] for key in loaded.files}
    snapshots = read_jsonl(source / "snapshots.jsonl")
    if len(snapshots) != int(source_manifest["team_step_count"]):
        raise ValueError("source snapshot count does not match its manifest")
    rows: dict[str, list[Any]] = {}

    def append(key: str, value: Any) -> None:
        rows.setdefault(key, []).append(value)

    unique_snapshot_indices = np.unique(events["snapshot_index"])
    if len(unique_snapshot_indices) != len(snapshots):
        raise ValueError("source events do not map one-to-one to snapshots")
    thresholds = dict(source_manifest["thresholds"])
    for expected_index, snapshot_row in enumerate(snapshots):
        source_rows = np.flatnonzero(events["snapshot_index"] == expected_index)
        if len(source_rows) != num_agents:
            raise ValueError("each source snapshot must have exactly three agent rows")
        order = np.argsort(events["agent_id"][source_rows])
        source_rows = source_rows[order]
        if not np.array_equal(events["agent_id"][source_rows], np.arange(num_agents)):
            raise ValueError("source snapshot is missing a unique UAV row")
        snapshot = snapshot_row["snapshot"]
        if snapshot_row["snapshot_digest"] != events["snapshot_digest"][source_rows[0]]:
            raise ValueError("snapshot digest and event rows disagree")
        repeated_keys = (
            "layout_id", "policy_seed", "episode_index", "step",
            "global_state_raw", "global_state_normalized",
            "joint_bounded_mean_action", "causal_safety_context",
            "current_safety_margin", "recovery_active", "safe_brake_active",
            "terminal_transition",
        )
        for key in repeated_keys:
            if not _all_equal(np.asarray(events[key][source_rows])):
                raise ValueError(f"source team field differs across agent rows: {key}")
        raw_joint_action = np.asarray(
            events["joint_bounded_mean_action"][source_rows[0]],
            dtype=np.float32,
        )
        requested = np.asarray(snapshot["requested_actions"], dtype=np.float32)
        if not np.allclose(raw_joint_action, requested, atol=1e-6):
            raise ValueError("source raw joint action and snapshot disagree")
        local_raw = np.asarray(
            events["local_observation_raw"][source_rows], dtype=np.float32
        )
        local_normalized = np.asarray(
            events["local_observation_normalized"][source_rows],
            dtype=np.float32,
        )
        delta_v = np.asarray(
            events["stage_delta_v_mps"][source_rows], dtype=np.float32
        )
        per_agent_cost, team_labels, team_severity, total_cost = (
            aggregate_team_stage_costs(
                delta_v,
                material_delta_v_mps=float(
                    thresholds["material_delta_v_mps"]
                ),
            )
        )
        stage_eligibility = np.asarray(
            events["stage_eligible"][source_rows], dtype=bool
        )
        if not np.all(stage_eligibility == stage_eligibility[0]):
            raise ValueError("team-stage eligibility differs across UAV rows")
        margins = per_agent_snapshot_margins(
            snapshot,
            agent_radius=dynamics["agent_radius"],
            map_size=dynamics["map_size"],
        )
        team_observation = np.asarray(
            snapshot["team_observation"], dtype=np.float32
        )
        ttc_values = []
        warning_values = []
        emergency_values = []
        clearance_values = []
        for agent in range(num_agents):
            ttc, warning, emergency, clearance = (
                action_conditioned_obstacle_features(
                    current_velocity=team_observation[agent, 2:4],
                    raw_action=raw_joint_action[agent],
                    obstacle_tracks=snapshot["obstacle_tracks"][agent],
                    agent_radius=dynamics["agent_radius"],
                    obstacle_safety_margin=float(snapshot["safety_state"][3]),
                    invariant_buffer=dynamics["invariant_buffer"],
                    max_acceleration=dynamics["max_acceleration"],
                    max_speed=dynamics["max_speed"],
                    dt=dynamics["dt"],
                    horizon_seconds=dynamics["horizon_seconds"],
                    warning_ttc_seconds=dynamics["warning_ttc_seconds"],
                    emergency_ttc_seconds=dynamics["emergency_ttc_seconds"],
                )
            )
            ttc_values.append(ttc)
            warning_values.append(warning)
            emergency_values.append(emergency)
            clearance_values.append(clearance)
        append("layout_id", events["layout_id"][source_rows[0]])
        append("policy_seed", events["policy_seed"][source_rows[0]])
        append("episode_index", events["episode_index"][source_rows[0]])
        append("step", events["step"][source_rows[0]])
        append("snapshot_index", expected_index)
        append("snapshot_digest", events["snapshot_digest"][source_rows[0]])
        append("local_observation_raw", local_raw)
        append("local_observation_normalized", local_normalized)
        append("global_state_raw", events["global_state_raw"][source_rows[0]])
        append(
            "global_state_normalized",
            events["global_state_normalized"][source_rows[0]],
        )
        append("raw_joint_mean_action", raw_joint_action)
        append(
            "causal_safety_context",
            events["causal_safety_context"][source_rows[0]],
        )
        append(
            "current_safety_margin",
            events["current_safety_margin"][source_rows[0]],
        )
        append("per_agent_margins", margins)
        append("action_conditioned_ttc_s", np.asarray(ttc_values, np.float32))
        append("ttc_warning", np.asarray(warning_values, bool))
        append("ttc_emergency", np.asarray(emergency_values, bool))
        append(
            "minimum_predicted_clearance_m",
            np.asarray(clearance_values, np.float32),
        )
        append("per_agent_stage_delta_v_mps", delta_v)
        append("per_agent_stage_cost", per_agent_cost)
        append("team_stage_labels", team_labels)
        append("team_stage_severity", team_severity)
        append("team_stage_cost", team_severity)
        append("team_total_cost", total_cost)
        append("stage_eligible", stage_eligibility[0])
        append("recovery_active", events["recovery_active"][source_rows[0]])
        append("safe_brake_active", events["safe_brake_active"][source_rows[0]])
        append(
            "terminal_transition",
            events["terminal_transition"][source_rows[0]],
        )
    arrays = {key: np.asarray(values) for key, values in rows.items()}
    if len(np.unique(arrays["snapshot_index"])) != len(arrays["snapshot_index"]):
        raise RuntimeError("TeamDatasetV3 contains duplicate team-step rows")
    destination.mkdir(parents=True, exist_ok=False)
    events_path = destination / "team_events.npz"
    np.savez_compressed(events_path, **arrays)
    source_folds = json.loads(
        (source / "folds.json").read_text(encoding="utf-8")
    )
    folds_path = destination / "folds.json"
    save_json(source_folds, folds_path)
    supported: dict[str, bool] = {}
    fold_support: list[dict[str, Any]] = []
    layouts = arrays["layout_id"]
    labels = arrays["team_stage_labels"]
    eligible = arrays["stage_eligible"]
    folds = tuple(
        tuple(map(int, fold["held_out_layouts"]))
        for fold in source_folds["folds"]
    )
    for fold_index, heldout_layouts in enumerate(folds):
        heldout_mask = np.isin(layouts, heldout_layouts)
        train_mask = ~heldout_mask
        train_support = stage_support(
            labels[train_mask], eligible[train_mask]
        )
        heldout_support = stage_support(
            labels[heldout_mask], eligible[heldout_mask]
        )
        heads = {}
        for head, stage in enumerate(TRAINABLE_STAGES):
            passed = bool(
                train_support[head].sufficient(
                    minimum_positive=100, minimum_negative=100
                )
                and heldout_support[head].sufficient(
                    minimum_positive=20, minimum_negative=20
                )
            )
            heads[stage] = {
                "train": vars(train_support[head]),
                "held_out": vars(heldout_support[head]),
                "support_gate_pass": passed,
            }
        fold_support.append({"fold": fold_index, "heads": heads})
    for stage in TRAINABLE_STAGES:
        supported[stage] = all(
            fold["heads"][stage]["support_gate_pass"]
            for fold in fold_support
        )
    support_payload = {
        "schema_version": "IAMAPPOR2TeamSupportV3",
        "aggregation": TEAM_AGGREGATION_VERSION,
        "requirements": {
            "train_minimum_positive": 100,
            "train_minimum_negative": 100,
            "held_out_minimum_positive": 20,
            "held_out_minimum_negative": 20,
        },
        "folds": fold_support,
        "all_supported_heads_pass": supported,
    }
    support_path = destination / "support.json"
    save_json(support_payload, support_path)
    source_manifest_path = source / "manifest.json"
    artifacts = (events_path, folds_path, support_path)
    manifest = {
        "schema_version": TEAM_DATASET_SCHEMA,
        "source_dataset": str(source.resolve()),
        "source_manifest_sha256": _file_sha256(source_manifest_path),
        "source_schema_version": source_manifest["schema_version"],
        "task_config_fingerprint": source_manifest[
            "task_config_fingerprint"
        ],
        "policy_seed": int(source_manifest["policy_seed"]),
        "layout_count": int(len(np.unique(layouts))),
        "team_step_count": int(len(layouts)),
        "rows_per_snapshot": 1,
        "num_agents": num_agents,
        "stages": list(TRAINABLE_STAGES),
        "aggregation_version": TEAM_AGGREGATION_VERSION,
        "action_conditioned_ttc_version": ACTION_TTC_VERSION,
        "action_conditioned_ttc_contract": {
            **dynamics,
            "invalid_velocity_track_policy": "ignore_without_simulator_truth",
            "protected_radius": (
                "agent_radius + obstacle_radius + obstacle_safety_margin "
                "+ invariant_buffer"
            ),
        },
        "cost_contract": {
            "per_agent": (
                "MaterialIndicator * (1 + clip(delta_v_mps / 0.25, 0, 1))"
            ),
            "classification": "any UAV material intervention per stage",
            "severity": "mean UAV per-agent stage cost",
            "stage_cost": "mean UAV per-agent stage cost",
            "total_cost": "mean initial, TTC, and final stage costs",
        },
        "supported_heads": supported,
        "observable_only": True,
        "arrays": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in arrays.items()
        },
        "artifacts": {
            path.name: {
                "sha256": _file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in artifacts
        },
    }
    save_json(manifest, destination / "manifest.json")
    validate_team_dataset_v3(destination)
    return manifest


def validate_team_dataset_v3(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != TEAM_DATASET_SCHEMA:
        raise ValueError("unsupported team intervention dataset schema")
    for filename, metadata in manifest["artifacts"].items():
        artifact = root / filename
        if not artifact.is_file() or _file_sha256(artifact) != metadata["sha256"]:
            raise ValueError(f"team dataset artifact hash mismatch: {filename}")
    with np.load(root / "team_events.npz", allow_pickle=False) as data:
        required = {
            "snapshot_index", "local_observation_normalized",
            "raw_joint_mean_action", "action_conditioned_ttc_s",
            "per_agent_margins", "team_stage_labels",
            "team_stage_severity", "team_total_cost", "stage_eligible",
        }
        if not required.issubset(data.files):
            raise ValueError("team dataset is missing required arrays")
        rows = len(data["snapshot_index"])
        agents = int(manifest["num_agents"])
        if len(np.unique(data["snapshot_index"])) != rows:
            raise ValueError("team dataset repeats a snapshot")
        if data["local_observation_normalized"].shape[:2] != (rows, agents):
            raise ValueError("team local observation shape is invalid")
        if data["raw_joint_mean_action"].shape != (rows, agents, 2):
            raise ValueError("team raw action shape is invalid")
        if data["team_stage_labels"].shape != (rows, 3):
            raise ValueError("team stage label shape is invalid")
        excluded = (
            data["recovery_active"].astype(bool)
            | data["safe_brake_active"].astype(bool)
            | data["terminal_transition"].astype(bool)
        )
        if np.any(data["stage_eligible"][excluded]):
            raise ValueError("excluded team transitions contain eligible labels")
        if np.any(data["team_stage_severity"] < 0.0) or np.any(
            data["team_stage_severity"] > 2.0
        ):
            raise ValueError("team severity is outside the preregistered range")
        expected_total = np.mean(data["team_stage_severity"], axis=1)
        if not np.allclose(data["team_total_cost"], expected_total, atol=1e-6):
            raise ValueError("team total cost does not equal mean stage cost")
    return manifest
