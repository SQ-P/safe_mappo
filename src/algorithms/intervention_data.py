"""Layout-isolated R2 dataset contracts for intervention-risk learning."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.safety.contracts import (
    CausalSafetySnapshotV1,
    InterventionLevel,
    SAFETY_STAGES,
    StageInterventionTraceV1,
)
from src.utils.io_utils import read_jsonl, save_json, save_jsonl


DATASET_SCHEMA = "IAMAPPOR2DatasetV2"
SUPPORTED_DATASET_SCHEMAS = {"IAMAPPOR2DatasetV1", DATASET_SCHEMA}
TRAINABLE_STAGES = (
    "initial_joint_guard",
    "obstacle_ttc",
    "final_joint_guard",
)
FOLD_SEED = 20261109


@dataclass(frozen=True)
class HeadSupport:
    positive: int
    negative: int

    def sufficient(self, *, minimum_positive: int, minimum_negative: int) -> bool:
        return self.positive >= minimum_positive and self.negative >= minimum_negative


def grouped_layout_folds(
    layout_ids: Sequence[int], *, folds: int = 5, seed: int = FOLD_SEED
) -> tuple[tuple[int, ...], ...]:
    """Assign complete layouts to deterministic folds; steps never cross folds."""

    unique = np.asarray(sorted({int(value) for value in layout_ids}), dtype=np.int64)
    if folds <= 1 or len(unique) < folds:
        raise ValueError("grouped cross-validation requires at least one layout per fold")
    rng = np.random.default_rng(int(seed))
    shuffled = unique[rng.permutation(len(unique))]
    result = tuple(
        tuple(int(value) for value in chunk)
        for chunk in np.array_split(shuffled, folds)
    )
    if set().union(*(set(chunk) for chunk in result)) != set(unique.tolist()):
        raise RuntimeError("layout fold assignment lost or duplicated a layout")
    return result


def assert_layout_isolation(
    train_layouts: Iterable[int], validation_layouts: Iterable[int],
    test_layouts: Iterable[int],
) -> None:
    groups = [set(map(int, values)) for values in (
        train_layouts, validation_layouts, test_layouts
    )]
    if not all(groups):
        raise ValueError("train, validation, and test layout sets must be non-empty")
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("layout leakage detected across dataset partitions")


def stage_support(
    material_targets: np.ndarray,
    eligibility: np.ndarray | None = None,
) -> tuple[HeadSupport, HeadSupport, HeadSupport]:
    values = np.asarray(material_targets)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("material targets must have shape [events, 3]")
    if eligibility is None:
        mask = np.ones_like(values, dtype=bool)
    else:
        mask = np.asarray(eligibility, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("eligibility must match material target shape")
    return tuple(
        HeadSupport(
            positive=int(np.count_nonzero((values[:, head] > 0.5) & mask[:, head])),
            negative=int(np.count_nonzero((values[:, head] <= 0.5) & mask[:, head])),
        )
        for head in range(3)
    )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _minimum_observable_ttc(
    snapshot: CausalSafetySnapshotV1,
    *,
    agent_radius: float,
    obstacle_safety_margin: float,
    no_collision_value: float = 1_000.0,
) -> np.ndarray:
    """Constant-velocity TTC using only tracks in the causal snapshot."""

    result = np.full(len(snapshot.team_observation), no_collision_value, dtype=np.float32)
    for observer, tracks in enumerate(snapshot.obstacle_tracks):
        best = float(no_collision_value)
        for track in tracks:
            position = np.asarray(track.relative_position, dtype=np.float64)
            velocity = np.asarray(track.relative_velocity, dtype=np.float64)
            radius = float(agent_radius + obstacle_safety_margin + track.radius)
            c = float(np.dot(position, position) - radius * radius)
            if c <= 0.0:
                best = 0.0
                continue
            a = float(np.dot(velocity, velocity))
            if a <= 1e-12:
                continue
            b = 2.0 * float(np.dot(position, velocity))
            discriminant = b * b - 4.0 * a * c
            if discriminant < 0.0:
                continue
            root = (-b - np.sqrt(discriminant)) / (2.0 * a)
            if root >= 0.0:
                best = min(best, float(root))
        result[observer] = np.float32(best)
    return result


def _causal_context_from_snapshot(
    snapshot: CausalSafetySnapshotV1,
    *,
    agent_radius: float,
    map_size: Sequence[float],
) -> tuple[np.ndarray, float]:
    if len(snapshot.safety_state) != 5:
        raise ValueError("V2 causal context requires five snapshot safety values")
    positions = np.asarray(
        [row[:2] for row in snapshot.team_observation], dtype=np.float32
    )
    domain_margin = (
        float(agent_radius) + float(snapshot.safety_state[0])
        + float(snapshot.safety_state[1])
    )
    boundary_margin = float(np.min(np.concatenate((
        positions - domain_margin,
        np.asarray(map_size, dtype=np.float32) - domain_margin - positions,
    ))))
    pair_margin = 1_000.0
    for left in range(len(positions)):
        for right in range(left + 1, len(positions)):
            pair_margin = min(
                pair_margin,
                float(np.linalg.norm(positions[left] - positions[right]))
                - float(snapshot.safety_state[2]),
            )
    obstacle_margin = 1_000.0
    for tracks in snapshot.obstacle_tracks:
        for obstacle in tracks:
            obstacle_margin = min(
                obstacle_margin,
                float(np.linalg.norm(obstacle.relative_position))
                - float(agent_radius) - float(obstacle.radius)
                - float(snapshot.safety_state[3]),
            )
    current_margin = min(boundary_margin, pair_margin, obstacle_margin)
    context = np.asarray((
        *snapshot.safety_state[:4], boundary_margin, pair_margin,
        obstacle_margin, snapshot.safety_state[4],
    ), dtype=np.float32)
    return context, float(current_margin)


class InterventionDatasetAccumulatorV1:
    """Collect immutable per-agent R2 events without changing execution."""

    def __init__(
        self,
        *,
        policy_seed: int,
        checkpoint_update: int,
        task_config_fingerprint: str,
        checkpoint_sha256: str | None,
        agent_radius: float,
        map_size: Sequence[float],
        obstacle_safety_margin: float,
        any_action_l2: float = 1e-7,
        material_delta_v_mps: float = 0.05,
        large_delta_v_mps: float = 0.25,
    ) -> None:
        self.policy_seed = int(policy_seed)
        self.checkpoint_update = int(checkpoint_update)
        self.task_config_fingerprint = str(task_config_fingerprint)
        self.checkpoint_sha256 = checkpoint_sha256
        self.agent_radius = float(agent_radius)
        self.map_size = tuple(float(value) for value in map_size)
        if len(self.map_size) != 2:
            raise ValueError("map_size must contain two dimensions")
        self.obstacle_safety_margin = float(obstacle_safety_margin)
        self.thresholds = {
            "any_action_l2": float(any_action_l2),
            "material_delta_v_mps": float(material_delta_v_mps),
            "large_delta_v_mps": float(large_delta_v_mps),
            "direction_reversal_cosine_lt": 0.0,
        }
        self.rows: list[dict[str, Any]] = []
        self.snapshots: list[dict[str, Any]] = []
        self.episodes: list[dict[str, Any]] = []

    def append_step(
        self,
        *,
        layout_id: int,
        episode_index: int,
        step: int,
        local_observations_raw: np.ndarray,
        local_observations_normalized: np.ndarray,
        global_state_raw: np.ndarray,
        global_state_normalized: np.ndarray,
        bounded_joint_action: np.ndarray,
        causal_safety_context: np.ndarray,
        current_safety_margin: float,
        snapshot: CausalSafetySnapshotV1,
        trace: StageInterventionTraceV1,
        terminated: bool,
        truncated: bool,
    ) -> None:
        num_agents = len(snapshot.team_observation)
        local_raw = np.asarray(local_observations_raw, dtype=np.float32)
        local_normalized = np.asarray(local_observations_normalized, dtype=np.float32)
        global_raw = np.asarray(global_state_raw, dtype=np.float32).reshape(-1)
        global_normalized = np.asarray(global_state_normalized, dtype=np.float32).reshape(-1)
        joint_action = np.asarray(bounded_joint_action, dtype=np.float32)
        causal_context = np.asarray(causal_safety_context, dtype=np.float32).reshape(-1)
        expected_terminal = bool(terminated or truncated)
        if trace.snapshot_digest != snapshot.digest:
            raise ValueError("trace and causal snapshot digest disagree")
        if trace.step != snapshot.step:
            raise ValueError("trace and causal snapshot step disagree")
        if int(step) != int(snapshot.step) + 1:
            raise ValueError("dataset step must be one-based relative to snapshot step")
        if expected_terminal and not bool(trace.terminal_transition):
            raise ValueError("terminal transition is not represented in the safety trace")
        if local_raw.ndim != 2 or local_raw.shape[0] != num_agents:
            raise ValueError("raw local observations must be [agents, features]")
        if local_normalized.shape != local_raw.shape:
            raise ValueError("normalized local observations must match raw observations")
        if joint_action.shape != (num_agents, 2):
            raise ValueError("bounded joint action must have shape [agents, 2]")
        if not np.allclose(
            joint_action,
            np.asarray(snapshot.requested_actions, dtype=np.float32),
            atol=1e-6,
        ):
            raise ValueError("bounded policy mean differs from snapshot requested action")
        expected_context, expected_margin = _causal_context_from_snapshot(
            snapshot, agent_radius=self.agent_radius, map_size=self.map_size
        )
        if not np.allclose(causal_context, expected_context, atol=1e-5):
            raise ValueError("causal safety context is not derived from the pre-action snapshot")
        if not np.isclose(float(current_safety_margin), expected_margin, atol=1e-5):
            raise ValueError("current safety margin is not derived from the pre-action snapshot")
        for name, value in (
            ("local_raw", local_raw), ("local_normalized", local_normalized),
            ("global_raw", global_raw), ("global_normalized", global_normalized),
            ("joint_action", joint_action), ("causal_context", causal_context),
        ):
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
        expected_keys = {
            (agent, stage) for agent in range(num_agents) for stage in SAFETY_STAGES
        }
        records = {(record.uav_id, record.stage): record for record in trace.records}
        if set(records) != expected_keys or len(trace.records) != num_agents * len(SAFETY_STAGES):
            raise ValueError("trace must contain exactly one record for every UAV-stage pair")
        if trace.recovery_active or trace.safe_brake_active or trace.terminal_transition:
            if any(record.eligible_for_actor_risk for record in trace.records):
                raise ValueError("recovery/brake/terminal trace contains eligible labels")
        observable_ttc = _minimum_observable_ttc(
            snapshot,
            agent_radius=self.agent_radius,
            obstacle_safety_margin=self.obstacle_safety_margin,
        )
        snapshot_index = len(self.snapshots)
        self.snapshots.append({
            "schema_version": snapshot.schema_version,
            "layout_id": int(layout_id),
            "policy_seed": self.policy_seed,
            "episode_index": int(episode_index),
            "step": int(step),
            "snapshot_digest": snapshot.digest,
            "snapshot": snapshot.canonical_payload(),
        })
        for agent in range(num_agents):
            stage_records = [records[(agent, stage)] for stage in TRAINABLE_STAGES]
            trace_records = [records[(agent, stage)] for stage in SAFETY_STAGES]
            delta_v = np.asarray([item.delta_v_mps for item in stage_records], dtype=np.float32)
            cosine = np.asarray([item.direction_cosine for item in stage_records], dtype=np.float32)
            levels = np.asarray([int(item.modification_level) for item in stage_records], dtype=np.int8)
            eligible = np.asarray(
                [item.eligible_for_actor_risk for item in stage_records], dtype=bool
            )
            self.rows.append({
                "layout_id": int(layout_id),
                "policy_seed": self.policy_seed,
                "episode_index": int(episode_index),
                "step": int(step),
                "agent_id": int(agent),
                "snapshot_index": snapshot_index,
                "snapshot_digest": snapshot.digest,
                "local_observation_raw": local_raw[agent].copy(),
                "local_observation_normalized": local_normalized[agent].copy(),
                "global_state_raw": global_raw.copy(),
                "global_state_normalized": global_normalized.copy(),
                "single_bounded_mean_action": joint_action[agent].copy(),
                "joint_bounded_mean_action": joint_action.copy(),
                "causal_safety_context": causal_context.copy(),
                "current_safety_margin": float(current_safety_margin),
                "minimum_observable_ttc_s": float(observable_ttc[agent]),
                "stage_input_actions": np.asarray(
                    [item.input_action for item in stage_records], dtype=np.float32
                ),
                "stage_output_actions": np.asarray(
                    [item.output_action for item in stage_records], dtype=np.float32
                ),
                "stage_delta_v_mps": delta_v,
                "stage_direction_cosine": cosine,
                "stage_safety_margin": np.asarray([
                    np.nan if item.safety_margin is None else item.safety_margin
                    for item in stage_records
                ], dtype=np.float32),
                "stage_eligible": eligible,
                "material_targets": delta_v >= self.thresholds["material_delta_v_mps"],
                "large_targets": delta_v >= self.thresholds["large_delta_v_mps"],
                "reversal_targets": cosine < 0.0,
                "any_targets": levels != int(InterventionLevel.NONE),
                "modification_levels": levels,
                "stage_reasons": np.asarray([item.reason for item in stage_records]),
                "trace_input_actions": np.asarray(
                    [item.input_action for item in trace_records], dtype=np.float32
                ),
                "trace_output_actions": np.asarray(
                    [item.output_action for item in trace_records], dtype=np.float32
                ),
                "trace_delta_v_mps": np.asarray(
                    [item.delta_v_mps for item in trace_records], dtype=np.float32
                ),
                "trace_direction_cosine": np.asarray(
                    [item.direction_cosine for item in trace_records], dtype=np.float32
                ),
                "trace_safety_margin": np.asarray([
                    np.nan if item.safety_margin is None else item.safety_margin
                    for item in trace_records
                ], dtype=np.float32),
                "trace_eligible": np.asarray(
                    [item.eligible_for_actor_risk for item in trace_records],
                    dtype=bool,
                ),
                "trace_modification_levels": np.asarray(
                    [int(item.modification_level) for item in trace_records],
                    dtype=np.int8,
                ),
                "trace_reasons": np.asarray(
                    [item.reason for item in trace_records]
                ),
                "recovery_active": bool(trace.recovery_active),
                "safe_brake_active": bool(trace.safe_brake_active),
                "terminal_transition": bool(trace.terminal_transition),
            })

    def record_episode(self, *, layout_id: int, episode_index: int,
                       outcome: Mapping[str, Any]) -> None:
        self.episodes.append({
            "schema_version": DATASET_SCHEMA,
            "layout_id": int(layout_id),
            "policy_seed": self.policy_seed,
            "episode_index": int(episode_index),
            **dict(outcome),
        })

    def finalize(self, output_dir: str | Path) -> dict[str, Any]:
        output = Path(output_dir)
        if output.exists():
            raise FileExistsError(f"intervention dataset output already exists: {output}")
        if not self.rows or not self.snapshots or not self.episodes:
            raise RuntimeError("cannot finalize an empty or incomplete intervention dataset")
        output.mkdir(parents=True, exist_ok=False)
        text_keys = {"stage_reasons", "trace_reasons"}
        array_keys = tuple(key for key in self.rows[0] if key not in text_keys)
        arrays = {
            key: np.asarray([row[key] for row in self.rows])
            for key in array_keys
        }
        arrays["stage_reasons"] = np.asarray(
            [row["stage_reasons"] for row in self.rows], dtype=str
        )
        arrays["trace_reasons"] = np.asarray(
            [row["trace_reasons"] for row in self.rows], dtype=str
        )
        events_path = output / "events.npz"
        np.savez_compressed(events_path, **arrays)
        snapshots_path = output / "snapshots.jsonl"
        episodes_path = output / "episodes.jsonl"
        save_jsonl(self.snapshots, snapshots_path)
        save_jsonl(self.episodes, episodes_path)

        folds = grouped_layout_folds(arrays["layout_id"], folds=5, seed=FOLD_SEED)
        fold_payload = {
            "schema_version": "IAMAPPOR2LayoutFoldsV1",
            "seed": FOLD_SEED,
            "folds": [
                {"fold": index, "held_out_layouts": list(layouts)}
                for index, layouts in enumerate(folds)
            ],
        }
        save_json(fold_payload, output / "folds.json")
        support_folds: list[dict[str, Any]] = []
        layout_values = arrays["layout_id"].astype(np.int64)
        all_layouts = set(map(int, np.unique(layout_values)))
        for index, heldout in enumerate(folds):
            heldout_set = set(heldout)
            train_set = all_layouts - heldout_set
            train_mask = np.isin(layout_values, list(train_set))
            heldout_mask = np.isin(layout_values, list(heldout_set))
            train_support = stage_support(
                arrays["material_targets"][train_mask],
                arrays["stage_eligible"][train_mask],
            )
            heldout_support = stage_support(
                arrays["material_targets"][heldout_mask],
                arrays["stage_eligible"][heldout_mask],
            )
            support_folds.append({
                "fold": index,
                "train_layouts": sorted(train_set),
                "held_out_layouts": sorted(heldout_set),
                "heads": {
                    stage: {
                        "train": vars(train_support[head]),
                        "held_out": vars(heldout_support[head]),
                        "support_gate_pass": bool(
                            train_support[head].sufficient(
                                minimum_positive=100, minimum_negative=100
                            )
                            and heldout_support[head].sufficient(
                                minimum_positive=20, minimum_negative=20
                            )
                        ),
                    }
                    for head, stage in enumerate(TRAINABLE_STAGES)
                },
            })
        support_payload = {
            "schema_version": "IAMAPPOR2HeadSupportV1",
            "requirements": {
                "train_minimum_positive": 100,
                "train_minimum_negative": 100,
                "held_out_minimum_positive": 20,
                "held_out_minimum_negative": 20,
            },
            "folds": support_folds,
            "all_supported_heads_pass": {
                stage: all(
                    fold["heads"][stage]["support_gate_pass"]
                    for fold in support_folds
                )
                for stage in TRAINABLE_STAGES
            },
        }
        save_json(support_payload, output / "support.json")
        artifact_paths = (
            events_path, snapshots_path, episodes_path,
            output / "folds.json", output / "support.json",
        )
        manifest = {
            "schema_version": DATASET_SCHEMA,
            "policy_seed": self.policy_seed,
            "checkpoint_update": self.checkpoint_update,
            "checkpoint_sha256": self.checkpoint_sha256,
            "task_config_fingerprint": self.task_config_fingerprint,
            "layout_count": len(all_layouts),
            "episode_count": len(self.episodes),
            "team_step_count": len(self.snapshots),
            "agent_event_count": len(self.rows),
            "stages": list(TRAINABLE_STAGES),
            "trace_stages": list(SAFETY_STAGES),
            "thresholds": self.thresholds,
            "map_size": list(self.map_size),
            "label_policy": (
                "Recovery, SAFE_BRAKE, and terminal transitions are recorded but "
                "excluded from Actor-risk supervision by stage_eligible."
            ),
            "observable_only": True,
            "causal_context_semantics": "pre_action_snapshot_v2",
            "arrays": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in arrays.items()
            },
            "artifacts": {
                path.name: {
                    "sha256": _file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in artifact_paths
            },
            "support_summary": support_payload["all_supported_heads_pass"],
        }
        save_json(manifest, output / "manifest.json")
        validate_intervention_dataset(output)
        return manifest


def validate_intervention_dataset(output_dir: str | Path) -> dict[str, Any]:
    """Fail closed on schema, hashes, shapes, terminal eligibility and leakage."""

    output = Path(output_dir)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in SUPPORTED_DATASET_SCHEMAS:
        raise ValueError("unsupported intervention dataset schema")
    for filename, metadata in manifest["artifacts"].items():
        path = output / filename
        if not path.is_file() or _file_sha256(path) != metadata["sha256"]:
            raise ValueError(f"dataset artifact hash mismatch: {filename}")
    with np.load(output / "events.npz", allow_pickle=False) as data:
        required = {
            "layout_id", "agent_id", "snapshot_digest", "material_targets",
            "stage_eligible", "terminal_transition", "recovery_active",
            "safe_brake_active", "stage_input_actions", "stage_output_actions",
            "trace_input_actions", "trace_output_actions", "trace_eligible",
        }
        if not required.issubset(data.files):
            raise ValueError("intervention dataset is missing required arrays")
        row_count = len(data["layout_id"])
        if data["material_targets"].shape != (row_count, 3):
            raise ValueError("material target shape is invalid")
        if data["trace_input_actions"].shape != (row_count, 5, 2):
            raise ValueError("full stage trace shape is invalid")
        excluded = (
            data["terminal_transition"].astype(bool)
            | data["recovery_active"].astype(bool)
            | data["safe_brake_active"].astype(bool)
        )
        if np.any(data["stage_eligible"][excluded]):
            raise ValueError("excluded transitions contain eligible labels")
        if np.any(data["trace_eligible"][excluded]):
            raise ValueError("excluded full traces contain eligible labels")
        if not np.isfinite(data["stage_input_actions"]).all() or not np.isfinite(
            data["stage_output_actions"]
        ).all():
            raise ValueError("stage actions contain non-finite values")
        folds_payload = json.loads(
            (output / "folds.json").read_text(encoding="utf-8")
        )
        heldout_sets = [
            set(map(int, item["held_out_layouts"]))
            for item in folds_payload["folds"]
        ]
        if len(heldout_sets) != 5 or any(
            left & right
            for index, left in enumerate(heldout_sets)
            for right in heldout_sets[index + 1:]
        ):
            raise ValueError("layout folds overlap")
        if set().union(*heldout_sets) != set(map(int, np.unique(data["layout_id"]))):
            raise ValueError("layout folds do not cover the dataset")
    snapshots = read_jsonl(output / "snapshots.jsonl")
    if len(snapshots) != int(manifest["team_step_count"]):
        raise ValueError("snapshot count does not match manifest")
    return manifest
