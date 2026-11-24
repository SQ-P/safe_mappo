"""R4.4 dataset construction, training and one-shot certification."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.algorithms.intervention_advantage import (
    FEATURE_CONTRACT,
    REFERENCE_BANK_CONTRACT,
    SUPPORTED_STAGES,
    StructuredFeatureSpecV1,
    TwinInterventionAdvantageCriticV1,
)
from src.algorithms.intervention_r3 import validate_r3_shadow_dataset
from src.algorithms.intervention_r4_3_2 import (
    _correlation,
    classification_layout_bootstrap,
    classification_report,
    expected_cost_calibration_report,
    file_sha256,
    layout_bootstrap_report,
)
from src.utils.io_utils import save_json


DATASET_SCHEMA = "IAMAPPOR4_4AdvantageDatasetV1"
ARTIFACT_SCHEMA = "TwinInterventionAdvantageCriticArtifactV1"
PROTOCOL_SCHEMA = "IAMAPPOR4_4FrozenProtocolV1"
REFERENCE_FIELDS = {
    "reference_joint_actions", "reference_action_conditioned_ttc_s",
    "reference_ttc_warning", "reference_ttc_emergency",
    "reference_minimum_predicted_clearance_m", "reference_stage_costs",
    "reference_total_costs", "reference_valid",
}


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path / "shadow_events.npz", allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _source_row(name: str, path: Path, index: int, manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "index": int(index), "name": str(name), "dataset": str(path),
        "manifest_sha256": file_sha256(path / "manifest.json"),
        "checkpoint_sha256": str(manifest["checkpoint_sha256"]),
        "policy_seed": int(manifest["policy_seed"]),
        "checkpoint_update": int(manifest["checkpoint_update"]),
    }


def build_advantage_dataset(
    *, sources: Sequence[tuple[str, str | Path]], output_dir: str | Path,
    role: str, development_split_source: str | Path | None = None,
    forbidden_layout_panels: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Combine six same-panel collections without altering causal rows."""

    if role not in {"development", "fresh_certification"}:
        raise ValueError("R4.4 dataset role must be development or fresh_certification")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"R4.4 dataset output exists: {output}")
    if len(sources) != 6 or len({name for name, _ in sources}) != 6:
        raise ValueError("R4.4 requires exactly six unique policy anchors")

    loaded: list[tuple[dict[str, Any], dict[str, np.ndarray]]] = []
    common_layouts: set[int] | None = None
    common_keys: set[str] | None = None
    task_fingerprint: str | None = None
    gate_hash: str | None = None
    for index, (name, raw_path) in enumerate(sources):
        path = Path(raw_path).resolve()
        manifest = validate_r3_shadow_dataset(path)
        if manifest.get("action_query_kind") != "sampled_bounded_rollout_action":
            raise ValueError(f"R4.4 source is not sampled-action data: {name}")
        if manifest.get("reference_bank_contract") != REFERENCE_BANK_CONTRACT:
            raise ValueError(f"R4.4 source lacks the registered reference bank: {name}")
        arrays = _load_arrays(path)
        if not REFERENCE_FIELDS.issubset(arrays):
            raise ValueError(f"R4.4 source lacks reference arrays: {name}")
        layouts = set(map(int, np.unique(arrays["layout_id"])))
        if common_layouts is None:
            common_layouts = layouts
            common_keys = set(arrays)
            task_fingerprint = str(manifest["task_config_fingerprint"])
            gate_hash = str(manifest["r2r_gate_report_sha256"])
        elif layouts != common_layouts or set(arrays) != common_keys:
            raise ValueError("R4.4 sources must share panel and array schema")
        if str(manifest["task_config_fingerprint"]) != task_fingerprint:
            raise ValueError("R4.4 task fingerprint mismatch")
        if str(manifest["r2r_gate_report_sha256"]) != gate_hash:
            raise ValueError("R4.4 R2R gate provenance mismatch")
        loaded.append((_source_row(name, path, index, manifest), arrays))
    assert common_layouts is not None and common_keys is not None
    if len(common_layouts) != 200:
        raise ValueError("R4.4 requires exactly 200 shared layouts")

    forbidden: set[int] = set()
    for raw_path in forbidden_layout_panels:
        panel = json.loads(Path(raw_path).resolve().read_text(encoding="utf-8"))
        forbidden.update(map(int, panel.get("layout_seeds", [])))
    if role == "fresh_certification" and common_layouts & forbidden:
        raise ValueError("R4.4 fresh panel overlaps historical layouts")

    combined = {
        key: np.concatenate([arrays[key] for _, arrays in loaded], axis=0)
        for key in sorted(common_keys)
    }
    combined["source_policy_index"] = np.concatenate([
        np.full(len(arrays["layout_id"]), row["index"], dtype=np.int16)
        for row, arrays in loaded
    ])
    combined["actor_risk_eligible"] = (
        np.asarray(combined["stage_eligible"], dtype=bool)[:, :2].all(axis=1)
        & ~np.asarray(combined["recovery_active"], dtype=bool)
        & ~np.asarray(combined["safe_brake_active"], dtype=bool)
        & ~np.asarray(combined["terminal_transition"], dtype=bool)
    )
    combined["avoidable_event_labels"] = (
        np.asarray(combined["stage_avoidable_costs"], dtype=np.float32)[:, :2] > 1e-8
    )

    if role == "development":
        if development_split_source is None:
            raise ValueError("development dataset requires frozen layout splits")
        split_payload = json.loads(
            (Path(development_split_source).resolve() / "layout_splits.json").read_text(
                encoding="utf-8"
            )
        )["partitions"]
        splits = {
            "train": list(map(int, split_payload["train"])),
            "calibration": list(map(int, split_payload["calibration"])),
            "validation": list(map(int, split_payload.get("threshold_selection", split_payload["validation"]))),
            "test": list(map(int, split_payload["test"])),
        }
        if set().union(*map(set, splits.values())) != common_layouts:
            raise ValueError("R4.4 development splits do not cover the collection panel")
    else:
        splits = {"test": sorted(common_layouts)}

    output.mkdir(parents=True, exist_ok=False)
    events_path = output / "advantage_events.npz"
    np.savez_compressed(events_path, **combined)
    split_path = output / "layout_splits.json"
    save_json({
        "schema_version": "IAMAPPOR4_4LayoutSplitV1", "role": role,
        "partitions": splits,
        "partition_sizes": {key: len(value) for key, value in splits.items()},
    }, split_path)
    sources_path = output / "sources.json"
    save_json({"schema_version": "IAMAPPOR4_4SourcesV1", "sources": [r for r, _ in loaded]}, sources_path)
    artifacts = (events_path, split_path, sources_path)
    manifest = {
        "schema_version": DATASET_SCHEMA, "role": role,
        "development_only": role == "development",
        "fresh_certification_only": role == "fresh_certification",
        "feature_contract": FEATURE_CONTRACT,
        "reference_bank_contract": REFERENCE_BANK_CONTRACT,
        "supported_stages": list(SUPPORTED_STAGES),
        "severity_blocking": False, "final_guard_supported": False,
        "layout_count": 200, "source_policy_count": 6,
        "team_step_count": int(len(combined["layout_id"])),
        "task_config_fingerprint": task_fingerprint,
        "r2r_gate_report_sha256": gate_hash,
        "artifacts": {p.name: {"sha256": file_sha256(p), "size_bytes": p.stat().st_size} for p in artifacts},
    }
    save_json(manifest, output / "manifest.json")
    validate_advantage_dataset(output)
    return manifest


def validate_advantage_dataset(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != DATASET_SCHEMA:
        raise ValueError("unsupported R4.4 dataset schema")
    for filename, metadata in manifest["artifacts"].items():
        item = root / filename
        if not item.is_file() or file_sha256(item) != metadata["sha256"]:
            raise ValueError(f"R4.4 artifact hash mismatch: {filename}")
    arrays = _load_advantage_arrays(root)
    rows = len(arrays["layout_id"])
    required = REFERENCE_FIELDS | {
        "source_policy_index", "actor_risk_eligible", "avoidable_event_labels",
        "stage_avoidable_costs", "raw_stage_costs", "snapshot_digest_before",
        "snapshot_digest_after", "environment_fingerprint_before",
        "environment_fingerprint_after", "actor_hidden_digest_before",
        "actor_hidden_digest_after", "episode_index", "step",
    }
    if not required.issubset(arrays):
        raise ValueError("R4.4 dataset is missing required arrays")
    if arrays["avoidable_event_labels"].shape != (rows, 2):
        raise ValueError("R4.4 label shape is invalid")
    if not np.array_equal(arrays["snapshot_digest_before"], arrays["snapshot_digest_after"]):
        raise ValueError("R4.4 causal snapshot equality failed")
    for before, after in (
        ("environment_fingerprint_before", "environment_fingerprint_after"),
        ("actor_hidden_digest_before", "actor_hidden_digest_after"),
    ):
        if not np.array_equal(arrays[before], arrays[after]):
            raise ValueError(f"R4.4 invariance failed: {before}")
    identity = np.rec.fromarrays([
        np.asarray(arrays["source_policy_index"], dtype=np.int16),
        np.asarray(arrays["layout_id"], dtype=np.int64),
        np.asarray(arrays["episode_index"], dtype=np.int32),
        np.asarray(arrays["step"], dtype=np.int32),
    ], names="source,layout,episode,step")
    if len(np.unique(identity)) != rows:
        raise ValueError("R4.4 dataset duplicates a source team-step")
    if arrays["reference_joint_actions"].shape[1] != 6:
        raise ValueError("R4.4 reference bank cardinality is invalid")
    if not np.allclose(
        np.asarray(arrays["reference_stage_costs"]).mean(axis=2),
        np.asarray(arrays["reference_total_costs"]), atol=1e-6,
    ):
        raise ValueError("R4.4 reference stage costs do not conserve total cost")
    return manifest


def _load_advantage_arrays(root: Path) -> dict[str, np.ndarray]:
    with np.load(root / "advantage_events.npz", allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _feature_tensors(
    arrays: Mapping[str, np.ndarray], index: np.ndarray, device: torch.device,
    *, reference_index: np.ndarray | None = None,
) -> tuple[torch.Tensor, ...]:
    index = np.asarray(index, dtype=np.int64)
    static = (
        arrays["global_state_normalized"][index],
        arrays["causal_safety_context"][index],
        arrays["current_safety_margin"][index],
        arrays["per_agent_margins"][index],
    )
    if reference_index is None:
        dynamic = (
            arrays["raw_joint_mean_action"][index], arrays["action_conditioned_ttc_s"][index],
            arrays["ttc_warning"][index], arrays["ttc_emergency"][index],
            arrays["minimum_predicted_clearance_m"][index],
        )
    else:
        ref = np.asarray(reference_index, dtype=np.int64)
        dynamic = tuple(arrays[name][index, ref] for name in (
            "reference_joint_actions", "reference_action_conditioned_ttc_s",
            "reference_ttc_warning", "reference_ttc_emergency",
            "reference_minimum_predicted_clearance_m",
        ))
    state, context, margin, per_agent = static
    action, ttc, warning, emergency, clearance = dynamic
    values = (state, action, context, margin, per_agent, ttc, warning, emergency, clearance)
    return tuple(torch.as_tensor(v, dtype=torch.float32, device=device) for v in values)


def _best_reference(arrays: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(arrays["raw_stage_costs"], dtype=np.float32)[:, :2]
    costs = np.asarray(arrays["reference_stage_costs"], dtype=np.float32)[:, :, :2]
    valid = np.asarray(arrays["reference_valid"], dtype=bool)
    better = valid[:, :, None] & (costs < raw[:, None, :] - 1e-8)
    ranked = np.where(better, costs, np.inf)
    index = np.argmin(ranked, axis=1).astype(np.int64)
    mask = np.any(better, axis=1)
    return index, mask


def _fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    result = []
    for stage in range(logits.shape[1]):
        log_t = torch.zeros((), device=logits.device, requires_grad=True)
        optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=80)
        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                logits[:, stage] / log_t.exp().clamp(0.05, 20.0), labels[:, stage]
            )
            loss.backward()
            return loss
        optimizer.step(closure)
        result.append(log_t.detach().exp().clamp(0.05, 20.0))
    return torch.stack(result)


def _predict(
    model: TwinInterventionAdvantageCriticV1, arrays: Mapping[str, np.ndarray],
    indices: np.ndarray, *, batch_size: int, device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    members, references = [], []
    best_reference, pair_mask = _best_reference(arrays)
    for start in range(0, len(indices), batch_size):
        batch = indices[start:start + batch_size]
        with torch.no_grad():
            raw = model.member_probabilities(*_feature_tensors(arrays, batch, device))
            per_stage_reference = []
            for stage in range(2):
                ref_features = _feature_tensors(
                    arrays, batch, device, reference_index=best_reference[batch, stage]
                )
                per_stage_reference.append(
                    model.member_probabilities(*ref_features)[:, :, stage]
                )
            reference = torch.stack(per_stage_reference, dim=-1)
        members.append(raw.cpu().numpy())
        references.append(reference.cpu().numpy())
    return (
        np.concatenate(members, axis=1),
        np.concatenate(references, axis=1),
        pair_mask[indices],
    )


def train_probability_twin(
    *, dataset_dir: str | Path, output_dir: str | Path, alpha_pair: float,
    device: str = "cpu", epochs: int = 40, batch_size: int = 1024,
    learning_rate: float = 3e-4, seed: int = 20261116,
) -> Path:
    root, output = Path(dataset_dir).resolve(), Path(output_dir).resolve()
    manifest = validate_advantage_dataset(root)
    if manifest["role"] != "development":
        raise ValueError("R4.4 training requires development-only data")
    if output.exists():
        raise FileExistsError(f"R4.4 model output exists: {output}")
    arrays = _load_advantage_arrays(root)
    splits = json.loads((root / "layout_splits.json").read_text(encoding="utf-8"))["partitions"]
    layouts = np.asarray(arrays["layout_id"], dtype=np.int64)
    eligible = np.asarray(arrays["actor_risk_eligible"], dtype=bool)
    train_index = np.flatnonzero(np.isin(layouts, splits["train"]) & eligible)
    calibration_index = np.flatnonzero(np.isin(layouts, splits["calibration"]) & eligible)
    labels = np.asarray(arrays["avoidable_event_labels"], dtype=np.float32)
    best_reference, pair_mask = _best_reference(arrays)
    spec = StructuredFeatureSpecV1.from_arrays(arrays)
    torch_device = torch.device(device)
    torch.manual_seed(int(seed))
    model = TwinInterventionAdvantageCriticV1(spec).to(torch_device)
    action = np.asarray(arrays["raw_joint_mean_action"])[train_index]
    model.action_mean.copy_(torch.as_tensor(action.mean(axis=0), device=torch_device))
    model.action_std.copy_(torch.as_tensor(action.std(axis=0).clip(1e-6), device=torch_device))
    rng = np.random.default_rng(int(seed))
    source = np.asarray(arrays["source_policy_index"], dtype=np.int64)
    counts = np.bincount(source[train_index], minlength=6).clip(1)
    source_weight = len(train_index) / (6.0 * counts)
    positive = labels[train_index].sum(axis=0)
    negative = len(train_index) - positive
    pos_weight = torch.as_tensor(negative / np.maximum(positive, 1.0), dtype=torch.float32, device=torch_device)

    for critic_index, critic in enumerate(model.critics):
        torch.manual_seed(int(seed) + 1009 * critic_index)
        optimizer = torch.optim.Adam(critic.parameters(), lr=float(learning_rate))
        for _ in range(int(epochs)):
            order = rng.permutation(train_index)
            for start in range(0, len(order), int(batch_size)):
                batch = order[start:start + int(batch_size)]
                raw_logits = critic(*_feature_tensors(arrays, batch, torch_device))
                target = torch.as_tensor(labels[batch], device=torch_device)
                stage_eligible = torch.as_tensor(
                    np.asarray(arrays["stage_eligible"], dtype=np.float32)[batch, :2],
                    device=torch_device,
                )
                row_weight = torch.as_tensor(source_weight[source[batch]], dtype=torch.float32, device=torch_device).unsqueeze(-1)
                bce = F.binary_cross_entropy_with_logits(
                    raw_logits, target, reduction="none", pos_weight=pos_weight
                )
                classification = (bce * stage_eligible * row_weight).sum() / (stage_eligible * row_weight).sum().clamp_min(1.0)
                pair_terms = []
                for stage in range(2):
                    ref_logits = critic(*_feature_tensors(
                        arrays, batch, torch_device,
                        reference_index=best_reference[batch, stage],
                    ))[:, stage]
                    mask = torch.as_tensor(pair_mask[batch, stage], dtype=torch.bool, device=torch_device)
                    if bool(mask.any()):
                        pair_terms.append(F.softplus(-(raw_logits[mask, stage] - ref_logits[mask])).mean())
                pair_loss = torch.stack(pair_terms).mean() if pair_terms else raw_logits.new_zeros(())
                loss = classification + float(alpha_pair) * pair_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
                optimizer.step()

    model.eval()
    for member in range(2):
        logits = []
        for start in range(0, len(calibration_index), batch_size):
            batch = calibration_index[start:start + batch_size]
            with torch.no_grad():
                logits.append(model.critics[member](*_feature_tensors(arrays, batch, torch_device)))
        stacked = torch.cat(logits)
        model.temperatures[member].copy_(_fit_temperature(
            stacked, torch.as_tensor(labels[calibration_index], device=torch_device)
        ))

    output.mkdir(parents=True, exist_ok=False)
    artifact = output / "advantage_critic.pt"
    torch.save({
        "schema_version": ARTIFACT_SCHEMA,
        "model_state_dict": model.state_dict(), "feature_spec": spec.__dict__,
        "feature_contract": FEATURE_CONTRACT,
        "reference_bank_contract": REFERENCE_BANK_CONTRACT,
        "supported_stages": list(SUPPORTED_STAGES), "alpha_pair": float(alpha_pair),
        "dataset_manifest_sha256": file_sha256(root / "manifest.json"),
        "development_only": True, "severity_blocking": False,
        "final_guard_supported": False, "seed": int(seed),
    }, artifact)
    save_json({
        "schema_version": "IAMAPPOR4_4TrainingManifestV1",
        "artifact": str(artifact), "artifact_sha256": file_sha256(artifact),
        "dataset": str(root), "dataset_manifest_sha256": file_sha256(root / "manifest.json"),
        "alpha_pair": float(alpha_pair), "epochs": int(epochs),
        "batch_size": int(batch_size), "learning_rate": float(learning_rate),
    }, output / "training_manifest.json")
    return artifact


def load_advantage_artifact(path: str | Path, *, device: str) -> tuple[TwinInterventionAdvantageCriticV1, dict[str, Any]]:
    payload = torch.load(Path(path).resolve(), map_location=device, weights_only=False)
    if payload.get("schema_version") != ARTIFACT_SCHEMA:
        raise ValueError("unsupported R4.4 advantage artifact")
    spec = StructuredFeatureSpecV1(**payload["feature_spec"])
    model = TwinInterventionAdvantageCriticV1(spec).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def _pair_bootstrap(
    raw: np.ndarray, reference: np.ndarray, mask: np.ndarray, layouts: np.ndarray,
    *, replicates: int, seed: int,
) -> dict[str, float]:
    unique = np.unique(layouts[mask.any(axis=1)])
    if len(unique) < 10:
        return {"point": 0.0, "lower_95": 0.0, "upper_95": 0.0}
    def accuracy(index: np.ndarray) -> float:
        local = mask[index]
        return float(np.mean(raw[index][local] > reference[index][local])) if local.any() else 0.0
    point = accuracy(np.arange(len(layouts)))
    rng = np.random.default_rng(seed)
    values = []
    per_layout = {item: np.flatnonzero(layouts == item) for item in unique}
    for _ in range(replicates):
        chosen = rng.choice(unique, len(unique), replace=True)
        values.append(accuracy(np.concatenate([per_layout[x] for x in chosen])))
    return {"point": point, "lower_95": float(np.quantile(values, .025)), "upper_95": float(np.quantile(values, .975))}


def certify_advantage_critic(
    *, dataset_dir: str | Path, artifact_path: str | Path,
    output_path: str | Path, partition: str, device: str = "cpu",
    batch_size: int = 2048, bootstrap_replicates: int = 1000,
    seed: int = 20261116,
) -> dict[str, Any]:
    root = Path(dataset_dir).resolve()
    manifest = validate_advantage_dataset(root)
    arrays = _load_advantage_arrays(root)
    splits = json.loads((root / "layout_splits.json").read_text(encoding="utf-8"))["partitions"]
    if partition not in splits:
        raise ValueError(f"R4.4 partition is unavailable: {partition}")
    model, artifact = load_advantage_artifact(artifact_path, device=device)
    layouts = np.asarray(arrays["layout_id"], dtype=np.int64)
    source = np.asarray(arrays["source_policy_index"], dtype=np.int64)
    eligible = np.asarray(arrays["actor_risk_eligible"], dtype=bool)
    index = np.flatnonzero(np.isin(layouts, splits[partition]) & eligible)
    member, reference, pair_mask = _predict(
        model, arrays, index, batch_size=batch_size, device=torch.device(device)
    )
    labels = np.asarray(arrays["avoidable_event_labels"], dtype=bool)[index]
    stage_cost = np.asarray(arrays["stage_avoidable_costs"], dtype=np.float32)[index, :2]
    local_layouts, local_source = layouts[index], source[index]
    stage_reports: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for critic in range(2):
        for stage, name in enumerate(SUPPORTED_STAGES):
            report = classification_report(labels[:, stage], member[critic, :, stage])
            bootstrap = classification_layout_bootstrap(
                labels[:, stage], member[critic, :, stage], local_layouts,
                replicates=bootstrap_replicates, seed=seed + 100 * critic + 10 * stage,
            )
            key = f"critic_{critic}_{name}"
            local_checks = {
                "has_100_per_class": report["positive"] >= 100 and report["negative"] >= 100,
                "auroc_lower_95_above_0_65": bootstrap["auroc"]["lower_95"] > .65,
                "auprc_lift_point_at_least_1_5": report["auprc_lift"] >= 1.5,
                "auprc_lift_lower_95_above_1_0": bootstrap["auprc_lift"]["lower_95"] > 1.0,
                "ece_at_most_0_10": report["calibration"]["ece"] <= .10,
                "brier_beats_prevalence": report["brier_skill"] > 0.0,
                "nll_beats_prevalence": report["nll_improvement"] > 0.0,
            }
            checks.update({f"{key}_{k}": bool(v) for k, v in local_checks.items()})
            stage_reports[key] = {"classification": report, "layout_bootstrap": bootstrap, "checks": local_checks}

    conservative_stage = member.max(axis=0)
    conservative_reference = reference.max(axis=0)
    pair = _pair_bootstrap(
        conservative_stage, conservative_reference, pair_mask, local_layouts,
        replicates=bootstrap_replicates, seed=seed + 300,
    )
    cell_accuracy = {}
    for cell in range(6):
        mask = local_source == cell
        local_pair = pair_mask[mask]
        value = float(np.mean(conservative_stage[mask][local_pair] > conservative_reference[mask][local_pair])) if local_pair.any() else 0.0
        cell_accuracy[str(cell)] = value
    checks["pairwise_pooled_lower_95_above_0_65"] = pair["lower_95"] > .65
    checks["pairwise_all_cells_at_least_0_60"] = all(v >= .60 for v in cell_accuracy.values())
    checks["pairwise_no_cell_below_0_50"] = all(v >= .50 for v in cell_accuracy.values())

    total_prediction = conservative_stage.mean(axis=1)
    total_event_fraction = labels.astype(np.float32).mean(axis=1)
    total_cost = stage_cost.mean(axis=1)
    total_bootstrap = layout_bootstrap_report(
        total_cost, total_prediction, local_layouts,
        replicates=bootstrap_replicates, seed=seed + 400,
    )
    total_calibration = expected_cost_calibration_report(
        total_event_fraction, total_prediction, local_layouts
    )
    action = np.asarray(arrays["raw_joint_mean_action"], dtype=np.float32)[index]
    action_mean = model.action_mean.detach().cpu().numpy()
    action_std = model.action_std.detach().cpu().numpy().clip(1e-6)
    in_support = np.max(np.abs((action - action_mean) / action_std), axis=(1, 2)) <= 6.0
    ood_rate = float(1.0 - in_support.mean())
    checks.update({
        "total_finite_nonnegative": bool(np.isfinite(total_prediction).all() and np.all(total_prediction >= 0)),
        "total_auroc_lower_95_above_0_65": total_bootstrap["auroc"]["lower_95"] > .65,
        "total_spearman_lower_95_above_0_30": total_bootstrap["spearman"]["lower_95"] > .30,
        "risk_bin_calibration_at_most_0_10": total_calibration["risk_bin_calibration_error"] <= .10,
        "layout_calibration_at_most_0_10": total_calibration["layout_mean_calibration_error"] <= .10,
        "sampled_action_ood_at_most_10pct": ood_rate <= .10,
        "snapshot_equality": bool(np.array_equal(arrays["snapshot_digest_before"][index], arrays["snapshot_digest_after"][index])),
    })

    # Gradient usability is checked on deterministic rows with TTC held at the
    # causal pre-action feature value. R4.5, not R4.4, calibrates PPO ratios/lambda.
    gradient_index = index[: min(1024, len(index))]
    gradient_model = copy.deepcopy(model)
    gradient_model.freeze_for_actor()
    features = list(_feature_tensors(arrays, gradient_index, torch.device(device)))
    action_tensor = features[1].detach().clone().requires_grad_(True)
    features[1] = action_tensor
    risk = gradient_model.conservative_total_probability(*features)
    gradient = torch.autograd.grad(risk.sum(), action_tensor)[0]
    norm = gradient.flatten(1).norm(dim=1)
    stepped_action = (action_tensor - 1e-3 * gradient / norm.clamp_min(1e-8).view(-1, 1, 1)).detach()
    stepped = list(features)
    stepped[1] = stepped_action
    with torch.no_grad():
        stepped_risk = gradient_model.conservative_total_probability(*stepped)
    active = risk.detach() > 1e-6
    decrease = stepped_risk < risk.detach()
    nonzero_fraction = float((norm[active] > 1e-10).float().mean().cpu()) if bool(active.any()) else 0.0
    decrease_fraction = float(decrease[active].float().mean().cpu()) if bool(active.any()) else 0.0
    critic_has_grad = any(parameter.grad is not None for parameter in gradient_model.parameters())
    checks.update({
        "critic_parameters_frozen_without_grad": not critic_has_grad,
        "action_gradient_finite": bool(torch.isfinite(gradient).all()),
        "active_nonzero_gradient_at_least_90pct": nonzero_fraction >= .90,
        "negative_gradient_step_lowers_risk_at_least_95pct": decrease_fraction >= .95,
    })
    passed = all(checks.values())
    report = {
        "schema_version": "IAMAPPOR4_4CertificationV1",
        "status": "PASS" if passed else "FAIL",
        "certified": passed, "dataset_role": manifest["role"],
        "partition": partition, "artifact": str(Path(artifact_path).resolve()),
        "artifact_sha256": file_sha256(Path(artifact_path).resolve()),
        "dataset_manifest_sha256": file_sha256(root / "manifest.json"),
        "severity_blocking": False, "final_guard_supported": False,
        "single_critic_probability": stage_reports,
        "pairwise": {"pooled": pair, "cell_accuracy": cell_accuracy},
        "actor_conservative_total": {
            "layout_bootstrap": total_bootstrap, "calibration": total_calibration,
            "spearman_point": _correlation(total_cost, total_prediction, rank=True),
            "ood_rate": ood_rate,
        },
        "gradient_usability": {
            "rows": len(gradient_index), "nonzero_fraction": nonzero_fraction,
            "risk_decrease_fraction": decrease_fraction,
            "lambda_calibrated": False,
            "note": "R4.5 exclusively owns PPO-gradient-ratio and lambda calibration.",
        },
        "checks": checks,
    }
    save_json(report, Path(output_path).resolve())
    return report


def freeze_protocol(
    *, development_dataset: str | Path, proposed_artifact: str | Path,
    development_report: str | Path, output_path: str | Path,
    alpha_pair: float, eta_initial: float = 0.02, eta_ttc: float = 0.02,
    bootstrap_replicates: int = 1000, bootstrap_seed: int = 20261116,
) -> dict[str, Any]:
    report = json.loads(Path(development_report).resolve().read_text(encoding="utf-8"))
    if not report.get("certified"):
        raise ValueError("R4.4 protocol cannot freeze after failed development")
    payload = {
        "schema_version": PROTOCOL_SCHEMA, "status": "FROZEN",
        "development_only": False,
        "development_dataset_manifest_sha256": file_sha256(Path(development_dataset).resolve() / "manifest.json"),
        "proposed_artifact_sha256": file_sha256(Path(proposed_artifact).resolve()),
        "development_report_sha256": file_sha256(Path(development_report).resolve()),
        "feature_contract": FEATURE_CONTRACT,
        "reference_bank_contract": REFERENCE_BANK_CONTRACT,
        "supported_stages": list(SUPPORTED_STAGES),
        "alpha_pair": float(alpha_pair),
        "deadzone": {"initial_joint_guard": float(eta_initial), "obstacle_ttc": float(eta_ttc)},
        "bootstrap": {"replicates": int(bootstrap_replicates), "seed": int(bootstrap_seed), "unit": "layout"},
        "fresh_layout_count": 200, "source_policy_count": 6,
        "fresh_retraining_forbidden": True, "fresh_recalibration_forbidden": True,
        "lambda_calibration_stage": "R4.5", "actor_training_authorized": False,
        "severity_blocking": False, "final_guard_supported": False,
    }
    save_json(payload, Path(output_path).resolve())
    return payload
