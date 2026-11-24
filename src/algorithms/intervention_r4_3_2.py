"""R4.3.2 multi-policy sampled-action Initial-head remediation."""

from __future__ import annotations

import copy
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.algorithms.intervention_critic import (
    TwinInterventionCriticV1,
    full_encoder_stage_repair_parameter_groups,
    initial_head_repair_parameters,
    joint_stage_repair_parameter_groups,
    rebase_feature_normalizer_preserving_function,
)
from src.algorithms.intervention_r3 import (
    R3_FEATURE_CONTRACT,
    _critic_features,
    _split_layouts,
    validate_r3_shadow_dataset,
)
from src.utils.io_utils import save_json


DATASET_SCHEMA = "IAMAPPOR4_3_2MultiPolicySampledDatasetV1"
CERTIFICATION_SCHEMA = "IAMAPPOR4_3_2InitialRepairCertificationV1"
CANDIDATE_SCHEMA = "R4_3_2InitialRepairedCriticCandidateV1"
EXPECTED_RISK_CERTIFICATION_SCHEMA = (
    "IAMAPPOR4_3_3ExpectedRiskCertificationV1"
)
EXPECTED_RISK_CANDIDATE_SCHEMA = (
    "R4_3_3ExpectedRiskCriticCandidateV1"
)
JOINT_REPAIR_CERTIFICATION_SCHEMA = (
    "IAMAPPOR4_3_4JointStageSeverityCertificationV1"
)
JOINT_REPAIR_CANDIDATE_SCHEMA = "R4_3_4JointStageCriticCandidateV1"
ROBUST_SEVERITY_CERTIFICATION_SCHEMA = (
    "IAMAPPOR4_3_5RobustSeverityCertificationV1"
)
ROBUST_SEVERITY_CANDIDATE_SCHEMA = "R4_3_5RobustSeverityCriticCandidateV1"
LAYOUT_DRO_VALIDATION_SCHEMA = "IAMAPPOR4_3_6LayoutDROValidationV1"
LAYOUT_DRO_CANDIDATE_SCHEMA = "R4_3_6LayoutDRODiagnosticCandidateV1"
BASE_ARTIFACT_SCHEMA = "TwinInterventionCriticArtifactV5"
TRAINABLE_STAGE_INDICES = (0, 1)


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _actor_risk_eligibility(data: Mapping[str, np.ndarray]) -> np.ndarray:
    eligible = np.all(
        np.asarray(data["stage_eligible"], dtype=bool)[:, TRAINABLE_STAGE_INDICES],
        axis=1,
    )
    for name in ("recovery_active", "safe_brake_active", "terminal_transition"):
        if name in data:
            eligible &= ~np.asarray(data[name], dtype=bool)
    return eligible


def _support_payload(
    arrays: Mapping[str, np.ndarray], splits: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    labels = np.asarray(arrays["stage_avoidable_costs"]) > 1e-8
    layouts = np.asarray(arrays["layout_id"], dtype=np.int64)
    eligible = np.asarray(arrays["actor_risk_eligible"], dtype=bool)
    heads: dict[str, Any] = {}
    for index, stage in enumerate(
        ("initial_joint_guard", "obstacle_ttc", "final_joint_guard")
    ):
        partitions = {}
        for name, layout_ids in splits.items():
            mask = np.isin(layouts, layout_ids) & eligible
            partitions[name] = {
                "positive": int(np.count_nonzero(labels[mask, index])),
                "negative": int(np.count_nonzero(~labels[mask, index])),
            }
        enough = bool(
            min(partitions["train"].values()) >= 100
            and min(partitions["calibration"].values()) >= 20
            and min(partitions["threshold_selection"].values()) >= 20
            and min(partitions["test"].values()) >= 20
        )
        supported = enough and stage != "final_joint_guard"
        heads[stage] = {
            "partitions": partitions,
            "event_support_sufficient": enough,
            "supported": supported,
            "reason": (
                None if supported else
                "final_head_frozen_unsupported"
                if stage == "final_joint_guard" else
                "multi_policy_sampled_event_support_insufficient"
            ),
        }
    return {
        "schema_version": "IAMAPPOR4_3_2HeadSupportV1",
        "eligibility_contract": (
            "team_step_non_recovery_non_safe_brake_non_terminal_independent_of_label"
        ),
        "heads": heads,
    }


def combine_sampled_shadow_datasets(
    *, sources: Sequence[tuple[str, str | Path]], output_dir: str | Path,
    split_seed: int = 20261112,
) -> dict[str, Any]:
    """Merge six policy datasets while preserving layout-level isolation."""

    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"R4.3.2 merged dataset exists: {output}")
    if len(sources) < 4:
        raise ValueError("R4.3.2 requires multiple policies from both seeds")
    names = [str(name) for name, _ in sources]
    if len(names) != len(set(names)):
        raise ValueError("R4.3.2 source names must be unique")

    source_rows = []
    loaded_arrays: list[dict[str, np.ndarray]] = []
    reference_layouts: set[int] | None = None
    reference_keys: set[str] | None = None
    task_fingerprint: str | None = None
    gate_hash: str | None = None
    checkpoint_hashes: set[str] = set()
    for source_index, (name, raw_path) in enumerate(sources):
        path = Path(raw_path).resolve()
        manifest = validate_r3_shadow_dataset(path)
        if manifest.get("action_query_kind") != "sampled_bounded_rollout_action":
            raise ValueError(f"R4.3.2 source is not sampled-action data: {name}")
        with np.load(path / "shadow_events.npz", allow_pickle=False) as payload:
            arrays = {key: payload[key] for key in payload.files}
        layouts = set(map(int, np.unique(arrays["layout_id"])))
        if reference_layouts is None:
            reference_layouts = layouts
        elif layouts != reference_layouts:
            raise ValueError("R4.3.2 sources must use the identical fresh layout panel")
        keys = set(arrays)
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise ValueError("R4.3.2 source array schemas differ")
        current_fingerprint = str(manifest.get("task_config_fingerprint"))
        current_gate = str(manifest.get("r2r_gate_report_sha256"))
        if task_fingerprint is None:
            task_fingerprint = current_fingerprint
            gate_hash = current_gate
        elif current_fingerprint != task_fingerprint or current_gate != gate_hash:
            raise ValueError("R4.3.2 sources have incompatible task provenance")
        checkpoint_hash = str(manifest.get("checkpoint_sha256"))
        if checkpoint_hash in checkpoint_hashes:
            raise ValueError("R4.3.2 source checkpoint is duplicated")
        checkpoint_hashes.add(checkpoint_hash)
        loaded_arrays.append(arrays)
        source_rows.append({
            "index": source_index,
            "name": str(name),
            "dataset": str(path),
            "manifest_sha256": file_sha256(path / "manifest.json"),
            "checkpoint_sha256": checkpoint_hash,
            "policy_seed": int(manifest.get("policy_seed", -1)),
            "checkpoint_update": int(manifest.get("checkpoint_update", -1)),
            "team_step_count": int(manifest.get("team_step_count", -1)),
        })

    assert reference_layouts is not None and reference_keys is not None
    if len(reference_layouts) != 200:
        raise ValueError("R4.3.2 requires exactly 200 fresh layouts")
    combined = {
        key: np.concatenate([arrays[key] for arrays in loaded_arrays], axis=0)
        for key in sorted(reference_keys)
    }
    combined["source_policy_index"] = np.concatenate([
        np.full(len(arrays["layout_id"]), index, dtype=np.int16)
        for index, arrays in enumerate(loaded_arrays)
    ])
    combined["actor_risk_eligible"] = _actor_risk_eligibility(combined)
    splits = _split_layouts(sorted(reference_layouts), seed=int(split_seed))

    output.mkdir(parents=True, exist_ok=False)
    events_path = output / "shadow_events.npz"
    np.savez_compressed(events_path, **combined)
    splits_path = output / "layout_splits.json"
    save_json({
        "schema_version": "IAMAPPOR4_3_2LayoutSplitV1",
        "seed": int(split_seed),
        "partitions": splits,
        "partition_sizes": {key: len(value) for key, value in splits.items()},
    }, splits_path)
    support_path = output / "support.json"
    save_json(_support_payload(combined, splits), support_path)
    sources_path = output / "sources.json"
    save_json({
        "schema_version": "IAMAPPOR4_3_2SourcesV1",
        "sources": source_rows,
    }, sources_path)
    artifacts = (events_path, splits_path, support_path, sources_path)
    manifest = {
        "schema_version": DATASET_SCHEMA,
        "feature_contract": R3_FEATURE_CONTRACT,
        "action_query_kind": "sampled_bounded_rollout_action",
        "eligibility_contract": (
            "team_step_non_recovery_non_safe_brake_non_terminal_independent_of_label"
        ),
        "layout_count": len(reference_layouts),
        "source_policy_count": len(source_rows),
        "team_step_count": int(len(combined["layout_id"])),
        "task_config_fingerprint": task_fingerprint,
        "r2r_gate_report_sha256": gate_hash,
        "split_seed": int(split_seed),
        "sources": source_rows,
        "artifacts": {
            path.name: {
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in artifacts
        },
    }
    save_json(manifest, output / "manifest.json")
    validate_multi_policy_dataset(output)
    return manifest


def validate_multi_policy_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    dataset = Path(dataset_dir).resolve()
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != DATASET_SCHEMA:
        raise ValueError("unexpected R4.3.2 merged dataset schema")
    if (
        manifest.get("feature_contract") != R3_FEATURE_CONTRACT
        or manifest.get("action_query_kind") != "sampled_bounded_rollout_action"
    ):
        raise ValueError("R4.3.2 feature/action contract mismatch")
    for filename, metadata in manifest.get("artifacts", {}).items():
        path = dataset / filename
        if not path.is_file() or file_sha256(path) != metadata.get("sha256"):
            raise ValueError(f"R4.3.2 artifact hash mismatch: {filename}")
    with np.load(dataset / "shadow_events.npz", allow_pickle=False) as payload:
        required = {
            "layout_id", "global_state_normalized", "raw_joint_mean_action",
            "causal_safety_context", "current_safety_margin",
            "per_agent_margins", "action_conditioned_ttc_s", "ttc_warning",
            "ttc_emergency", "minimum_predicted_clearance_m",
            "stage_avoidable_costs", "stage_eligible", "source_policy_index",
            "actor_risk_eligible", "recovery_active", "safe_brake_active",
            "terminal_transition",
        }
        if not required.issubset(payload.files):
            raise ValueError("R4.3.2 merged dataset is missing required arrays")
        arrays = {key: payload[key] for key in payload.files}
    if not np.array_equal(
        np.asarray(arrays["actor_risk_eligible"], dtype=bool),
        _actor_risk_eligibility(arrays),
    ):
        raise ValueError("R4.3.2 actor-risk eligibility was not derived causally")
    layouts = np.unique(arrays["layout_id"])
    sources = np.unique(arrays["source_policy_index"])
    if len(layouts) != int(manifest["layout_count"]):
        raise ValueError("R4.3.2 layout count mismatch")
    if len(sources) != int(manifest["source_policy_count"]):
        raise ValueError("R4.3.2 source count mismatch")
    return manifest


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray, *, rank: bool) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if rank:
        left, right = _average_ranks(left), _average_ranks(right)
    if len(left) < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    positive = int(labels.sum())
    negative = int(len(labels) - positive)
    if positive == 0 or negative == 0:
        return 0.0
    ranks = _average_ranks(scores)
    statistic = float(ranks[labels].sum()) - positive * (positive + 1) / 2.0
    return statistic / float(positive * negative)


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    positive = int(labels.sum())
    if positive == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].sum() / positive)


def _binary_nll(labels: np.ndarray, probability: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1 - 1e-7)
    return float(np.mean(
        -labels * np.log(probability) - (1.0 - labels) * np.log(1.0 - probability)
    ))


def _ece(labels: np.ndarray, probability: np.ndarray, *, bins: int = 10) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.float64)
    probability = np.asarray(probability, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    rows = []
    error = 0.0
    for index in range(int(bins)):
        lower, upper = edges[index], edges[index + 1]
        mask = ((probability >= lower) & (probability < upper))
        if index == int(bins) - 1:
            mask = (probability >= lower) & (probability <= upper)
        count = int(mask.sum())
        if count == 0:
            continue
        predicted = float(probability[mask].mean())
        observed = float(labels[mask].mean())
        weight = count / len(labels)
        error += weight * abs(predicted - observed)
        rows.append({
            "lower": float(lower), "upper": float(upper), "count": count,
            "prediction_mean": predicted, "observed_rate": observed,
            "absolute_gap": abs(predicted - observed),
        })
    return {"bins": rows, "ece": float(error)}


def classification_report(labels: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=bool)
    probability = np.asarray(probability, dtype=np.float64)
    prevalence = float(labels.mean())
    brier = float(np.mean((probability - labels.astype(np.float64)) ** 2))
    baseline_brier = prevalence * (1.0 - prevalence)
    nll = _binary_nll(labels, probability)
    baseline_probability = np.full(len(labels), prevalence, dtype=np.float64)
    baseline_nll = _binary_nll(labels, baseline_probability)
    auprc = _average_precision(labels, probability)
    return {
        "rows": int(len(labels)),
        "positive": int(labels.sum()),
        "negative": int(len(labels) - labels.sum()),
        "prevalence": prevalence,
        "auroc": _auroc(labels, probability),
        "auprc": auprc,
        "auprc_lift": auprc / max(prevalence, 1e-12),
        "brier": brier,
        "prevalence_baseline_brier": baseline_brier,
        "brier_skill": 1.0 - brier / max(baseline_brier, 1e-12),
        "nll": nll,
        "prevalence_baseline_nll": baseline_nll,
        "nll_improvement": baseline_nll - nll,
        "calibration": _ece(labels, probability),
    }


def classification_layout_bootstrap(
    labels: np.ndarray, probability: np.ndarray, layouts: np.ndarray,
    *, replicates: int, seed: int,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=bool)
    probability = np.asarray(probability, dtype=np.float64)
    layouts = np.asarray(layouts, dtype=np.int64)
    unique = np.unique(layouts)
    if len(unique) < 10:
        raise ValueError("classification bootstrap requires at least ten layouts")
    rows = {layout: np.flatnonzero(layouts == layout) for layout in unique}
    rng = np.random.default_rng(int(seed))
    values = {key: [] for key in (
        "auroc", "auprc_lift", "brier_skill", "nll_improvement"
    )}
    for _ in range(int(replicates)):
        chosen = rng.choice(unique, len(unique), replace=True)
        index = np.concatenate([rows[int(layout)] for layout in chosen])
        report = classification_report(labels[index], probability[index])
        for key in values:
            values[key].append(float(report[key]))
    return {
        "replicates": int(replicates), "layout_count": int(len(unique)),
        **{
            key: {
                "lower_95": float(np.quantile(samples, 0.025)),
                "median": float(np.quantile(samples, 0.5)),
                "upper_95": float(np.quantile(samples, 0.975)),
            }
            for key, samples in values.items()
        },
    }


def _huber_array(error: np.ndarray, *, delta: float = 0.05) -> np.ndarray:
    absolute = np.abs(np.asarray(error, dtype=np.float64))
    return np.where(
        absolute <= float(delta),
        0.5 * absolute ** 2,
        float(delta) * (absolute - 0.5 * float(delta)),
    )


def positive_severity_report(
    *, train_positive_targets: np.ndarray, test_targets: np.ndarray,
    test_predictions: np.ndarray, test_layouts: np.ndarray,
    replicates: int, seed: int,
) -> dict[str, Any]:
    train_positive_targets = np.asarray(train_positive_targets, dtype=np.float64)
    test_targets = np.asarray(test_targets, dtype=np.float64)
    test_predictions = np.asarray(test_predictions, dtype=np.float64)
    test_layouts = np.asarray(test_layouts, dtype=np.int64)
    if not len(train_positive_targets) or not len(test_targets):
        raise ValueError("positive severity certification requires Material events")
    baseline = float(np.median(train_positive_targets))
    lower = float(np.min(train_positive_targets))
    upper = float(np.max(train_positive_targets))
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        huber_gradient = float(np.mean(np.clip(
            midpoint - train_positive_targets, -0.05, 0.05
        )))
        if huber_gradient > 0.0:
            upper = midpoint
        else:
            lower = midpoint
    huber_baseline = 0.5 * (lower + upper)
    model_error = test_predictions - test_targets
    baseline_error = baseline - test_targets
    huber_baseline_error = huber_baseline - test_targets
    point = {
        "positive_rows": int(len(test_targets)),
        "positive_layout_count": int(len(np.unique(test_layouts))),
        "train_positive_median_baseline": baseline,
        "train_positive_huber_optimal_baseline": huber_baseline,
        "mae": float(np.mean(np.abs(model_error))),
        "baseline_mae": float(np.mean(np.abs(baseline_error))),
        "mae_improvement": float(np.mean(np.abs(baseline_error)) - np.mean(np.abs(model_error))),
        "huber_delta": 0.05,
        "huber": float(np.mean(_huber_array(model_error))),
        "baseline_huber": float(np.mean(_huber_array(huber_baseline_error))),
        "huber_improvement": float(
            np.mean(_huber_array(huber_baseline_error))
            - np.mean(_huber_array(model_error))
        ),
    }
    unique = np.unique(test_layouts)
    rows = {layout: np.flatnonzero(test_layouts == layout) for layout in unique}
    rng = np.random.default_rng(int(seed))
    improvements = {"mae_improvement": [], "huber_improvement": []}
    for _ in range(int(replicates)):
        chosen = rng.choice(unique, len(unique), replace=True)
        index = np.concatenate([rows[int(layout)] for layout in chosen])
        model = test_predictions[index] - test_targets[index]
        reference = baseline - test_targets[index]
        huber_reference = huber_baseline - test_targets[index]
        improvements["mae_improvement"].append(float(
            np.mean(np.abs(reference)) - np.mean(np.abs(model))
        ))
        improvements["huber_improvement"].append(float(
            np.mean(_huber_array(huber_reference)) - np.mean(_huber_array(model))
        ))
    bootstrap = {
        key: {
            "lower_95": float(np.quantile(values, 0.025)),
            "median": float(np.quantile(values, 0.5)),
            "upper_95": float(np.quantile(values, 0.975)),
        }
        for key, values in improvements.items()
    }
    return {**point, "layout_bootstrap": bootstrap}


def expected_cost_calibration_report(
    target: np.ndarray, prediction: np.ndarray, layouts: np.ndarray,
    *, bins: int = 10,
) -> dict[str, Any]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    layouts = np.asarray(layouts, dtype=np.int64)
    order = np.argsort(prediction, kind="mergesort")
    bin_rows = []
    weighted_error = 0.0
    for index in np.array_split(order, min(int(bins), len(order))):
        if not len(index):
            continue
        predicted = float(prediction[index].mean())
        observed = float(target[index].mean())
        gap = abs(predicted - observed)
        weighted_error += len(index) / len(order) * gap
        bin_rows.append({
            "count": int(len(index)), "prediction_mean": predicted,
            "target_mean": observed, "absolute_gap": gap,
            "prediction_min": float(prediction[index].min()),
            "prediction_max": float(prediction[index].max()),
        })
    layout_rows = []
    for layout in np.unique(layouts):
        mask = layouts == layout
        layout_rows.append({
            "layout_id": int(layout),
            "prediction_mean": float(prediction[mask].mean()),
            "target_mean": float(target[mask].mean()),
        })
    layout_error = np.asarray([
        row["prediction_mean"] - row["target_mean"] for row in layout_rows
    ], dtype=np.float64)
    return {
        "rows": int(len(target)), "layout_count": int(len(layout_rows)),
        "prediction_mean": float(prediction.mean()),
        "target_mean": float(target.mean()),
        "mean_bias": float((prediction - target).mean()),
        "risk_bin_count": int(len(bin_rows)),
        "risk_bin_calibration_error": float(weighted_error),
        "risk_bins": bin_rows,
        "layout_mean_calibration_error": float(np.mean(np.abs(layout_error))),
        "layout_mean_bias": float(layout_error.mean()),
        "layout_rows": layout_rows,
    }


def regression_report(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    labels = target > 1e-8
    zero_mae = float(np.mean(np.abs(target)))
    mae = float(np.mean(np.abs(prediction - target)))
    return {
        "rows": int(len(target)),
        "positive": int(labels.sum()),
        "negative": int(len(labels) - labels.sum()),
        "prevalence": float(labels.mean()),
        "target_mean": float(target.mean()),
        "prediction_mean": float(prediction.mean()),
        "mean_bias": float((prediction - target).mean()),
        "mae": mae,
        "zero_baseline_mae": zero_mae,
        "mae_improvement_over_zero": zero_mae - mae,
        "spearman": _correlation(target, prediction, rank=True),
        "auroc": _auroc(labels, prediction),
    }


def layout_bootstrap_report(
    target: np.ndarray, prediction: np.ndarray, layouts: np.ndarray,
    *, replicates: int, seed: int,
) -> dict[str, Any]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    layouts = np.asarray(layouts, dtype=np.int64)
    unique = np.unique(layouts)
    if len(unique) < 10:
        raise ValueError("R4.3.2 bootstrap requires at least ten layouts")
    rows = {layout: np.flatnonzero(layouts == layout) for layout in unique}
    rng = np.random.default_rng(int(seed))
    values = {key: [] for key in ("auroc", "spearman", "mae_improvement_over_zero")}
    for _ in range(int(replicates)):
        chosen = rng.choice(unique, len(unique), replace=True)
        index = np.concatenate([rows[int(layout)] for layout in chosen])
        report = regression_report(target[index], prediction[index])
        for key in values:
            values[key].append(float(report[key]))
    return {
        "replicates": int(replicates),
        "layout_count": int(len(unique)),
        **{
            key: {
                "lower_95": float(np.quantile(sample, 0.025)),
                "median": float(np.quantile(sample, 0.5)),
                "upper_95": float(np.quantile(sample, 0.975)),
            }
            for key, sample in values.items()
        },
    }


def _fit_temperature(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_temperature = torch.zeros((), device=logits.device, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.1, max_iter=100, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.binary_cross_entropy_with_logits(logits / temperature, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return log_temperature.detach().exp().clamp(0.05, 20.0)


def _predict(
    model: TwinInterventionCriticV1, features: np.ndarray, *, batch_size: int,
) -> dict[str, np.ndarray]:
    device = model.feature_mean.device
    stage_risk, total_risk, probability, severity = [], [], [], []
    member_probability, member_severity, member_stage_risk = [], [], []
    with torch.no_grad():
        for start in range(0, len(features), int(batch_size)):
            raw = torch.as_tensor(
                features[start:start + int(batch_size)],
                dtype=torch.float32, device=device,
            )
            normalized = model.normalize_features(raw)
            twin_probability, twin_severity, twin_expected = [], [], []
            for index, member in enumerate(model.critics):
                logits, local_severity = member(normalized)
                local_probability = torch.sigmoid(
                    (logits - model.prior_logit_offsets[index])
                    / model.temperatures[index].clamp_min(1e-3)
                )
                local_expected = local_probability * torch.clamp(
                    local_severity, 0.0, 2.0
                )
                twin_probability.append(local_probability)
                twin_severity.append(torch.clamp(local_severity, 0.0, 2.0))
                twin_expected.append(local_expected)
            expected = torch.stack(twin_expected)
            winner = expected.argmax(dim=0)
            probability_tensor = torch.stack(twin_probability)
            severity_tensor = torch.stack(twin_severity)
            gather = winner.unsqueeze(0)
            probability.append(torch.gather(probability_tensor, 0, gather).squeeze(0).cpu())
            severity.append(torch.gather(severity_tensor, 0, gather).squeeze(0).cpu())
            member_probability.append(probability_tensor.movedim(0, 1).cpu())
            member_severity.append(severity_tensor.movedim(0, 1).cpu())
            member_stage_risk.append(expected.movedim(0, 1).cpu())
            stage_risk.append(model.conservative_stage_risk(raw).cpu())
            total_risk.append(model.conservative_risk(raw).cpu())
    return {
        "stage_risk": torch.cat(stage_risk).numpy(),
        "total_risk": torch.cat(total_risk).numpy(),
        "selected_probability": torch.cat(probability).numpy(),
        "selected_severity": torch.cat(severity).numpy(),
        "member_probability": torch.cat(member_probability).numpy(),
        "member_severity": torch.cat(member_severity).numpy(),
        "member_stage_risk": torch.cat(member_stage_risk).numpy(),
    }


def _source_stage_balanced_losses(
    member, normalized_features: torch.Tensor, stage_targets: torch.Tensor,
    source_ids: torch.Tensor, *, severity_weight: float,
    stages: Sequence[int] = (0, 1),
) -> dict[str, torch.Tensor]:
    """Average each source/stage task before combining the multi-task loss."""

    logits, severity = member(normalized_features)
    classification_losses: dict[int, torch.Tensor] = {}
    severity_losses: dict[int, torch.Tensor] = {}
    for stage_index in stages:
        local_classification = []
        local_severity = []
        labels = (stage_targets[:, stage_index] > 1e-8).to(logits.dtype)
        for source_id in torch.unique(source_ids):
            source_mask = source_ids == source_id
            if not bool(source_mask.any()):
                continue
            local_classification.append(F.binary_cross_entropy_with_logits(
                logits[source_mask, stage_index], labels[source_mask]
            ))
            positive = source_mask & (labels > 0.5)
            if bool(positive.any()):
                local_severity.append(F.huber_loss(
                    severity[positive, stage_index],
                    stage_targets[positive, stage_index],
                    delta=0.05,
                ))
        classification_losses[stage_index] = torch.stack(
            local_classification
        ).mean()
        severity_losses[stage_index] = (
            torch.stack(local_severity).mean()
            if local_severity else logits.new_zeros(())
        )
    classification = torch.stack(list(classification_losses.values())).mean()
    regression = torch.stack(list(severity_losses.values())).mean()
    return {
        "total": classification + float(severity_weight) * regression,
        "classification": classification,
        "severity": regression,
        **{
            f"classification_stage_{index}": value
            for index, value in classification_losses.items()
        },
        **{
            f"severity_stage_{index}": value
            for index, value in severity_losses.items()
        },
    }


def _source_stage_group_dro_losses(
    member, normalized_features: torch.Tensor, stage_targets: torch.Tensor,
    source_ids: torch.Tensor, *, severity_weight: float,
    stages: Sequence[int] = (0, 1),
) -> dict[str, torch.Tensor]:
    """Optimize mean classification while directly targeting worst severity group."""

    logits, severity = member(normalized_features)
    classification_losses: dict[int, torch.Tensor] = {}
    severity_losses: dict[int, torch.Tensor] = {}
    severity_groups = []
    for stage_index in stages:
        local_classification = []
        local_severity = []
        labels = (stage_targets[:, stage_index] > 1e-8).to(logits.dtype)
        for source_id in torch.unique(source_ids):
            source_mask = source_ids == source_id
            if not bool(source_mask.any()):
                continue
            local_classification.append(F.binary_cross_entropy_with_logits(
                logits[source_mask, stage_index], labels[source_mask]
            ))
            positive = source_mask & (labels > 0.5)
            if bool(positive.any()):
                loss = F.huber_loss(
                    severity[positive, stage_index],
                    stage_targets[positive, stage_index],
                    delta=0.05,
                )
                local_severity.append(loss)
                severity_groups.append(loss)
        classification_losses[stage_index] = torch.stack(
            local_classification
        ).mean()
        severity_losses[stage_index] = (
            torch.stack(local_severity).max()
            if local_severity else logits.new_zeros(())
        )
    classification = torch.stack(list(classification_losses.values())).mean()
    worst_severity = (
        torch.stack(severity_groups).max()
        if severity_groups else logits.new_zeros(())
    )
    return {
        "total": classification + float(severity_weight) * worst_severity,
        "classification": classification,
        "severity": worst_severity,
        "severity_group_dro": worst_severity,
        **{
            f"classification_stage_{index}": value
            for index, value in classification_losses.items()
        },
        **{
            f"severity_stage_{index}": value
            for index, value in severity_losses.items()
        },
    }


def _source_balanced_classification_loss(
    member, normalized_features: torch.Tensor, stage_targets: torch.Tensor,
    source_ids: torch.Tensor, *, stage_index: int,
) -> torch.Tensor:
    """Return a source-balanced BCE for one stage."""

    logits, _ = member(normalized_features)
    labels = (stage_targets[:, stage_index] > 1e-8).to(logits.dtype)
    losses = []
    for source_id in torch.unique(source_ids):
        mask = source_ids == source_id
        if bool(mask.any()):
            losses.append(F.binary_cross_entropy_with_logits(
                logits[mask, stage_index], labels[mask]
            ))
    if not losses:
        raise ValueError("classification refinement has no source rows")
    return torch.stack(losses).mean()


def _source_balanced_severity_loss(
    member, normalized_features: torch.Tensor, stage_targets: torch.Tensor,
    source_ids: torch.Tensor, *, stage_index: int,
) -> torch.Tensor:
    """Return source-balanced positive-only Huber loss for one stage."""

    _, severity = member(normalized_features)
    labels = stage_targets[:, stage_index] > 1e-8
    losses = []
    for source_id in torch.unique(source_ids):
        positive = (source_ids == source_id) & labels
        if bool(positive.any()):
            losses.append(F.huber_loss(
                severity[positive, stage_index],
                stage_targets[positive, stage_index],
                delta=0.05,
            ))
    if not losses:
        raise ValueError("severity refinement has no Material-positive rows")
    return torch.stack(losses).mean()


def _source_group_dro_severity_loss(
    member, normalized_features: torch.Tensor, stage_targets: torch.Tensor,
    source_ids: torch.Tensor, *, stage_index: int,
) -> torch.Tensor:
    """Return the worst source positive-only Huber loss for one stage."""

    _, severity = member(normalized_features)
    labels = stage_targets[:, stage_index] > 1e-8
    losses = []
    for source_id in torch.unique(source_ids):
        positive = (source_ids == source_id) & labels
        if bool(positive.any()):
            losses.append(F.huber_loss(
                severity[positive, stage_index],
                stage_targets[positive, stage_index],
                delta=0.05,
            ))
    if not losses:
        raise ValueError("Group-DRO severity refinement has no positive rows")
    return torch.stack(losses).max()


def _huber_optimal_constant(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        raise ValueError("Huber baseline requires positive targets")
    lower = float(np.min(values))
    upper = float(np.max(values))
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        gradient = float(np.mean(np.clip(midpoint - values, -0.05, 0.05)))
        if gradient > 0.0:
            upper = midpoint
        else:
            lower = midpoint
    return 0.5 * (lower + upper)


def _robust_severity_selection_report(
    *, train_targets: np.ndarray, validation_targets: np.ndarray,
    validation_predictions: np.ndarray, validation_sources: np.ndarray,
) -> dict[str, Any]:
    """Score severity by the worst source and the worst registered metric.

    The constants are fit only on the training Material positives.  Candidate
    heads are compared on validation positives separately for every source.
    The selected checkpoint minimizes the worst MAE/Huber ratio, so a large
    source cannot hide a regression in a smaller policy cell.
    """

    train = np.asarray(train_targets, dtype=np.float64)
    target = np.asarray(validation_targets, dtype=np.float64)
    prediction = np.asarray(validation_predictions, dtype=np.float64)
    source = np.asarray(validation_sources, dtype=np.int64)
    train_positive = train[train > 1e-8]
    if not len(train_positive):
        raise ValueError("robust severity selection requires train positives")
    median = float(np.median(train_positive))
    huber_constant = _huber_optimal_constant(train_positive)
    rows: dict[str, Any] = {}
    worst_ratio = 0.0
    missing_positive_sources = []
    for source_id in np.unique(source):
        mask = (source == source_id) & (target > 1e-8)
        if not np.any(mask):
            missing_positive_sources.append(int(source_id))
            continue
        local_target = target[mask]
        local_prediction = prediction[mask]
        model_mae = float(np.mean(np.abs(local_prediction - local_target)))
        baseline_mae = float(np.mean(np.abs(median - local_target)))
        model_huber = float(np.mean(_huber_array(
            local_prediction - local_target
        )))
        baseline_huber = float(np.mean(_huber_array(
            huber_constant - local_target
        )))
        mae_ratio = model_mae / max(baseline_mae, 1e-12)
        huber_ratio = model_huber / max(baseline_huber, 1e-12)
        local_worst = max(mae_ratio, huber_ratio)
        worst_ratio = max(worst_ratio, local_worst)
        rows[str(int(source_id))] = {
            "positive_rows": int(mask.sum()),
            "mae": model_mae,
            "baseline_mae": baseline_mae,
            "mae_ratio": mae_ratio,
            "huber": model_huber,
            "baseline_huber": baseline_huber,
            "huber_ratio": huber_ratio,
            "worst_metric_ratio": local_worst,
        }
    if missing_positive_sources:
        raise ValueError(
            "robust severity selection lacks validation positives for sources "
            f"{missing_positive_sources}"
        )
    if not rows:
        raise ValueError("robust severity selection has no validation positives")
    positive = target > 1e-8
    aggregate_mae = float(np.mean(np.abs(prediction[positive] - target[positive])))
    aggregate_baseline_mae = float(np.mean(np.abs(median - target[positive])))
    aggregate_huber = float(np.mean(_huber_array(
        prediction[positive] - target[positive]
    )))
    aggregate_baseline_huber = float(np.mean(_huber_array(
        huber_constant - target[positive]
    )))
    return {
        "selection_rule": "minimize_worst_source_max_mae_huber_ratio_v1",
        "train_positive_median_baseline": median,
        "train_positive_huber_optimal_baseline": huber_constant,
        "worst_source_metric_ratio": float(worst_ratio),
        "source_count": int(len(rows)),
        "all_sources_beat_both_baselines": bool(worst_ratio < 1.0),
        "aggregate_mae_ratio": (
            aggregate_mae / max(aggregate_baseline_mae, 1e-12)
        ),
        "aggregate_huber_ratio": (
            aggregate_huber / max(aggregate_baseline_huber, 1e-12)
        ),
        "sources": rows,
    }


def _layout_fold_severity_selection_report(
    *, train_targets: np.ndarray, validation_targets: np.ndarray,
    validation_predictions: np.ndarray, validation_sources: np.ndarray,
    validation_layouts: np.ndarray, fold_count: int = 5,
) -> dict[str, Any]:
    """Select by the worst source/layout-fold relative improvement."""

    train = np.asarray(train_targets, dtype=np.float64)
    target = np.asarray(validation_targets, dtype=np.float64)
    prediction = np.asarray(validation_predictions, dtype=np.float64)
    source = np.asarray(validation_sources, dtype=np.int64)
    layouts = np.asarray(validation_layouts, dtype=np.int64)
    train_positive = train[train > 1e-8]
    if not len(train_positive):
        raise ValueError("layout-fold selection requires train positives")
    median = float(np.median(train_positive))
    huber_constant = _huber_optimal_constant(train_positive)
    unique_layouts = np.sort(np.unique(layouts))
    if len(unique_layouts) < int(fold_count):
        raise ValueError("layout-fold selection has too few layouts")
    fold_by_layout = {
        int(layout): int(index % int(fold_count))
        for index, layout in enumerate(unique_layouts)
    }
    fold_id = np.asarray([fold_by_layout[int(value)] for value in layouts])
    worst_margin = float("inf")
    source_rows: dict[str, Any] = {}
    for source_id in np.unique(source):
        folds = []
        for fold in range(int(fold_count)):
            mask = (
                (source == source_id) & (fold_id == fold) & (target > 1e-8)
            )
            if not np.any(mask):
                raise ValueError(
                    f"source {int(source_id)} fold {fold} has no positives"
                )
            local_target = target[mask]
            local_prediction = prediction[mask]
            model_mae = float(np.mean(np.abs(local_prediction - local_target)))
            baseline_mae = float(np.mean(np.abs(median - local_target)))
            model_huber = float(np.mean(_huber_array(
                local_prediction - local_target
            )))
            baseline_huber = float(np.mean(_huber_array(
                huber_constant - local_target
            )))
            mae_relative_improvement = (
                (baseline_mae - model_mae) / max(baseline_mae, 1e-12)
            )
            huber_relative_improvement = (
                (baseline_huber - model_huber) / max(baseline_huber, 1e-12)
            )
            local_margin = min(
                mae_relative_improvement, huber_relative_improvement
            )
            worst_margin = min(worst_margin, local_margin)
            folds.append({
                "fold": int(fold),
                "layout_count": int(len(np.unique(layouts[mask]))),
                "positive_rows": int(mask.sum()),
                "mae_improvement": baseline_mae - model_mae,
                "huber_improvement": baseline_huber - model_huber,
                "mae_relative_improvement": mae_relative_improvement,
                "huber_relative_improvement": huber_relative_improvement,
                "worst_metric_relative_improvement": local_margin,
            })
        source_rows[str(int(source_id))] = {
            "folds": folds,
            "worst_fold_relative_improvement": min(
                row["worst_metric_relative_improvement"] for row in folds
            ),
        }
    return {
        "selection_rule": (
            "maximize_worst_source_layout_fold_mae_huber_relative_improvement_v1"
        ),
        "fold_count": int(fold_count),
        "layout_count": int(len(unique_layouts)),
        "train_positive_median_baseline": median,
        "train_positive_huber_optimal_baseline": huber_constant,
        "worst_source_fold_relative_improvement": float(worst_margin),
        "all_source_folds_beat_both_baselines": bool(worst_margin > 0.0),
        "sources": source_rows,
    }


def _positive_severity_per_layout(
    *, train_positive_targets: np.ndarray, test_targets: np.ndarray,
    test_predictions: np.ndarray, test_layouts: np.ndarray,
) -> list[dict[str, Any]]:
    """Return auditable layout-level severity improvements."""

    train = np.asarray(train_positive_targets, dtype=np.float64)
    target = np.asarray(test_targets, dtype=np.float64)
    prediction = np.asarray(test_predictions, dtype=np.float64)
    layouts = np.asarray(test_layouts, dtype=np.int64)
    median = float(np.median(train))
    huber_constant = _huber_optimal_constant(train)
    rows = []
    for layout in np.sort(np.unique(layouts)):
        mask = layouts == layout
        local_target = target[mask]
        local_prediction = prediction[mask]
        model_mae = float(np.mean(np.abs(local_prediction - local_target)))
        baseline_mae = float(np.mean(np.abs(median - local_target)))
        model_huber = float(np.mean(_huber_array(
            local_prediction - local_target
        )))
        baseline_huber = float(np.mean(_huber_array(
            huber_constant - local_target
        )))
        rows.append({
            "layout_id": int(layout),
            "positive_rows": int(mask.sum()),
            "target_mean": float(np.mean(local_target)),
            "target_std": float(np.std(local_target)),
            "mae": model_mae,
            "baseline_mae": baseline_mae,
            "mae_improvement": baseline_mae - model_mae,
            "huber": model_huber,
            "baseline_huber": baseline_huber,
            "huber_improvement": baseline_huber - model_huber,
        })
    return rows


def _decoupled_stage_head_refinement(
    *, member, critic_index: int, stage_index: int, stage_name: str,
    normalized: np.ndarray, targets: np.ndarray, source_index: np.ndarray,
    train_index: np.ndarray, validation_index: np.ndarray,
    learning_rate: float, head_refinement_epochs: int,
    early_stopping_patience: int, batch_size: int,
    generator: torch.Generator, torch_device: torch.device,
    history: list[dict[str, Any]],
    validation_layouts: np.ndarray | None = None,
    group_dro: bool = False,
) -> None:
    """Refine classification and severity independently on frozen features."""

    train_x = torch.as_tensor(
        normalized[train_index], dtype=torch.float32, device=torch_device
    )
    train_y = torch.as_tensor(
        targets[train_index, :2], dtype=torch.float32, device=torch_device
    )
    train_sources = torch.as_tensor(
        source_index[train_index], dtype=torch.int64, device=torch_device
    )
    validation_x = torch.as_tensor(
        normalized[validation_index], dtype=torch.float32, device=torch_device
    )
    validation_y = torch.as_tensor(
        targets[validation_index, :2], dtype=torch.float32, device=torch_device
    )
    validation_sources = torch.as_tensor(
        source_index[validation_index], dtype=torch.int64, device=torch_device
    )

    for parameter in member.parameters():
        parameter.requires_grad_(False)
    classification_parameters = list(
        member.classification_heads[stage_index].parameters()
    )
    for parameter in classification_parameters:
        parameter.requires_grad_(True)
    classification_optimizer = torch.optim.Adam(
        classification_parameters, lr=float(learning_rate)
    )
    member.eval()
    with torch.no_grad():
        best_classification_validation = float(
            _source_balanced_classification_loss(
                member, validation_x, validation_y, validation_sources,
                stage_index=stage_index,
            ).cpu()
        )
    best_classification = copy.deepcopy(
        member.classification_heads[stage_index].state_dict()
    )
    best_classification_epoch = 0
    stale = 0
    for epoch in range(int(head_refinement_epochs)):
        order = torch.randperm(len(train_index), generator=generator)
        train_losses = []
        for start in range(0, len(order), int(batch_size)):
            local = order[start:start + int(batch_size)].to(torch_device)
            loss = _source_balanced_classification_loss(
                member, train_x[local], train_y[local], train_sources[local],
                stage_index=stage_index,
            )
            classification_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classification_parameters, 5.0)
            classification_optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        member.eval()
        with torch.no_grad():
            validation_loss = float(_source_balanced_classification_loss(
                member, validation_x, validation_y, validation_sources,
                stage_index=stage_index,
            ).cpu())
        history.append({
            "phase": f"{stage_name}_classification_refinement",
            "critic": int(critic_index),
            "epoch": epoch + 1,
            "train_source_balanced_bce": float(np.mean(train_losses)),
            "validation_source_balanced_bce": validation_loss,
        })
        if validation_loss < best_classification_validation - 1e-4:
            best_classification_validation = validation_loss
            best_classification = copy.deepcopy(
                member.classification_heads[stage_index].state_dict()
            )
            best_classification_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
        if stale >= max(3, int(early_stopping_patience) // 2):
            break
    member.classification_heads[stage_index].load_state_dict(best_classification)
    history.append({
        "phase": f"{stage_name}_classification_selection",
        "critic": int(critic_index),
        "best_epoch": int(best_classification_epoch),
        "best_validation_source_balanced_bce": best_classification_validation,
        "selection_rule": "minimize_source_balanced_validation_bce_v1",
    })

    for parameter in member.parameters():
        parameter.requires_grad_(False)
    severity_parameters = list(member.severity_heads[stage_index].parameters())
    for parameter in severity_parameters:
        parameter.requires_grad_(True)
    severity_optimizer = torch.optim.Adam(
        severity_parameters, lr=float(learning_rate)
    )
    with torch.no_grad():
        _, initial_severity = member(validation_x)
    def severity_selection(prediction: np.ndarray) -> dict[str, Any]:
        if group_dro:
            if validation_layouts is None:
                raise ValueError("layout-aware refinement requires layouts")
            return _layout_fold_severity_selection_report(
                train_targets=targets[train_index, stage_index],
                validation_targets=targets[validation_index, stage_index],
                validation_predictions=prediction,
                validation_sources=source_index[validation_index],
                validation_layouts=validation_layouts,
            )
        return _robust_severity_selection_report(
            train_targets=targets[train_index, stage_index],
            validation_targets=targets[validation_index, stage_index],
            validation_predictions=prediction,
            validation_sources=source_index[validation_index],
        )

    def severity_score(report: Mapping[str, Any]) -> float:
        if group_dro:
            return -float(report["worst_source_fold_relative_improvement"])
        return float(report["worst_source_metric_ratio"])

    best_severity_report = severity_selection(
        initial_severity[:, stage_index].detach().cpu().numpy()
    )
    best_severity_validation = severity_score(best_severity_report)
    best_severity = copy.deepcopy(member.severity_heads[stage_index].state_dict())
    best_severity_epoch = 0
    stale = 0
    for epoch in range(int(head_refinement_epochs)):
        order = torch.randperm(len(train_index), generator=generator)
        train_losses = []
        for start in range(0, len(order), int(batch_size)):
            local = order[start:start + int(batch_size)].to(torch_device)
            severity_loss_fn = (
                _source_group_dro_severity_loss
                if group_dro else _source_balanced_severity_loss
            )
            loss = severity_loss_fn(
                member, train_x[local], train_y[local], train_sources[local],
                stage_index=stage_index,
            )
            severity_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(severity_parameters, 5.0)
            severity_optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        member.eval()
        with torch.no_grad():
            validation_loss_fn = (
                _source_group_dro_severity_loss
                if group_dro else _source_balanced_severity_loss
            )
            validation_huber = float(validation_loss_fn(
                member, validation_x, validation_y, validation_sources,
                stage_index=stage_index,
            ).cpu())
            _, validation_severity = member(validation_x)
        selection = severity_selection(
            validation_severity[:, stage_index].detach().cpu().numpy()
        )
        current = severity_score(selection)
        severity_history = {
            "phase": f"{stage_name}_severity_refinement",
            "critic": int(critic_index),
            "epoch": epoch + 1,
            "validation_severity_selection": selection,
        }
        if group_dro:
            severity_history.update({
                "train_worst_source_positive_huber": float(
                    np.mean(train_losses)
                ),
                "validation_worst_source_positive_huber": validation_huber,
            })
        else:
            severity_history.update({
                "train_source_balanced_positive_huber": float(
                    np.mean(train_losses)
                ),
                "validation_source_balanced_positive_huber": validation_huber,
            })
        history.append(severity_history)
        if current < best_severity_validation - 1e-4:
            best_severity_validation = current
            best_severity_report = selection
            best_severity = copy.deepcopy(
                member.severity_heads[stage_index].state_dict()
            )
            best_severity_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
        if stale >= max(3, int(early_stopping_patience) // 2):
            break
    member.severity_heads[stage_index].load_state_dict(best_severity)
    history.append({
        "phase": f"{stage_name}_severity_selection",
        "critic": int(critic_index),
        "best_epoch": int(best_severity_epoch),
        (
            "best_validation_negative_worst_fold_relative_improvement"
            if group_dro else "best_validation_worst_source_metric_ratio"
        ): best_severity_validation,
        "best_validation_selection": best_severity_report,
        "selection_rule": (
            "maximize_worst_source_layout_fold_mae_huber_relative_improvement_v1"
            if group_dro else "minimize_worst_source_max_mae_huber_ratio_v1"
        ),
    })


def train_and_certify_initial_repair(
    *, dataset_dir: str | Path, base_artifact: str | Path,
    output_dir: str | Path, device: str = "cuda", epochs: int = 40,
    batch_size: int = 1024, learning_rate: float = 3e-4,
    bootstrap_replicates: int = 1000, seed: int = 20261112,
    certification_protocol: str = "legacy_step_mae_v1",
    encoder_learning_rate_scale: float = 0.25,
    severity_loss_weight: float = 10.0,
    early_stopping_patience: int = 8,
    head_refinement_epochs: int = 15,
) -> dict[str, Any]:
    dataset = Path(dataset_dir).resolve()
    base_path = Path(base_artifact).resolve()
    output = Path(output_dir).resolve()
    if certification_protocol not in {
        "legacy_step_mae_v1", "expected_risk_calibration_v2",
        "joint_stage_finetune_v3", "decoupled_robust_severity_v4",
        "layout_group_dro_validation_v5",
    }:
        raise ValueError("unsupported Initial repair certification protocol")
    group_dro_protocol = certification_protocol == "layout_group_dro_validation_v5"
    robust_protocol = certification_protocol in {
        "decoupled_robust_severity_v4", "layout_group_dro_validation_v5"
    }
    joint_protocol = certification_protocol in {
        "joint_stage_finetune_v3", "decoupled_robust_severity_v4",
        "layout_group_dro_validation_v5",
    }
    expected_risk_protocol = certification_protocol in {
        "expected_risk_calibration_v2", "joint_stage_finetune_v3",
        "decoupled_robust_severity_v4", "layout_group_dro_validation_v5",
    }
    phase_name = (
        "R4.3.6V" if group_dro_protocol
        else "R4.3.5" if robust_protocol
        else "R4.3.4" if joint_protocol
        else "R4.3.3" if expected_risk_protocol
        else "R4.3.2"
    )
    if output.exists():
        raise FileExistsError(f"{phase_name} repair output exists: {output}")
    if robust_protocol:
        from src.algorithms.intervention_r4_3_5 import validate_fresh_test_dataset
        manifest = validate_fresh_test_dataset(dataset)
    elif joint_protocol:
        from src.algorithms.intervention_r4_3_4 import validate_fresh_test_dataset
        manifest = validate_fresh_test_dataset(dataset)
    else:
        manifest = validate_multi_policy_dataset(dataset)
    base_payload = torch.load(base_path, map_location="cpu", weights_only=False)
    if base_payload.get("schema_version") != BASE_ARTIFACT_SCHEMA:
        raise ValueError(f"{phase_name} requires the frozen V5 actor Critic")
    if base_payload.get("feature_contract") != R3_FEATURE_CONTRACT:
        raise ValueError(f"{phase_name} base Critic feature contract mismatch")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested for {phase_name} but unavailable")
    torch_device = torch.device(device)
    with np.load(dataset / "shadow_events.npz", allow_pickle=False) as payload:
        data = {key: payload[key] for key in payload.files}
    splits = json.loads(
        (dataset / "layout_splits.json").read_text(encoding="utf-8")
    )["partitions"]
    sources = json.loads(
        (dataset / "sources.json").read_text(encoding="utf-8")
    )["sources"]
    layouts = np.asarray(data["layout_id"], dtype=np.int64)
    source_index = np.asarray(data["source_policy_index"], dtype=np.int64)
    eligible = np.asarray(data["actor_risk_eligible"], dtype=bool)
    features = _critic_features(data)
    targets = np.asarray(data["stage_avoidable_costs"], dtype=np.float32)
    indices = {
        name: np.flatnonzero(np.isin(layouts, values) & eligible)
        for name, values in splits.items()
    }
    train_index = indices["train"]
    labels = (targets[:, 0] > 1e-8).astype(np.float32)
    for stage_index, stage_name in enumerate(("Initial", "TTC")):
        local_labels = targets[train_index, stage_index] > 1e-8
        if min(int(local_labels.sum()), int(len(local_labels) - local_labels.sum())) < 100:
            raise RuntimeError(f"{phase_name} {stage_name} train support is insufficient")

    new_mean = features[train_index].mean(axis=0, dtype=np.float64).astype(np.float32)
    new_std = features[train_index].std(axis=0, dtype=np.float64).astype(np.float32)
    new_std[new_std < 1e-6] = 1.0
    normalized = (features - new_mean) / new_std

    base_model = TwinInterventionCriticV1(int(base_payload["input_dim"])).to(torch_device)
    base_model.load_state_dict(base_payload["state_dict"])
    base_model.eval()
    repaired = TwinInterventionCriticV1(int(base_payload["input_dim"])).to(torch_device)
    repaired.load_state_dict(base_payload["state_dict"])
    repaired.eval()
    probe_index = train_index[:min(4096, len(train_index))]
    base_probe = _predict(base_model, features[probe_index], batch_size=batch_size)
    rebase_feature_normalizer_preserving_function(
        repaired,
        torch.as_tensor(new_mean, device=torch_device),
        torch.as_tensor(new_std, device=torch_device),
    )
    rebased_probe = _predict(repaired, features[probe_index], batch_size=batch_size)
    rebase_max_abs_error = float(np.max(np.abs(
        base_probe["stage_risk"] - rebased_probe["stage_risk"]
    )))
    if rebase_max_abs_error > 2e-5:
        raise RuntimeError("normalizer rebase changed the frozen Critic function")

    history: list[dict[str, Any]] = []
    if joint_protocol:
        parameter_groups = (
            full_encoder_stage_repair_parameter_groups(repaired)
            if robust_protocol
            else joint_stage_repair_parameter_groups(repaired)
        )
        validation_index = indices["validation"]
        for critic_index, member in enumerate(repaired.critics):
            local_seed = int(seed) + 10_000 * critic_index
            torch.manual_seed(local_seed)
            groups = parameter_groups[critic_index]
            optimizer = torch.optim.Adam([
                {
                    "params": groups["encoder"],
                    "lr": float(learning_rate) * float(encoder_learning_rate_scale),
                },
                {"params": groups["heads"], "lr": float(learning_rate)},
            ])
            generator = torch.Generator(device="cpu").manual_seed(local_seed)
            best_validation = float("inf")
            best_state = copy.deepcopy(member.state_dict())
            best_epoch = 0
            stale = 0
            for epoch in range(int(epochs)):
                member.train()
                order = torch.randperm(len(train_index), generator=generator)
                epoch_rows = []
                for start in range(0, len(order), int(batch_size)):
                    batch_rows = train_index[
                        order[start:start + int(batch_size)].numpy()
                    ]
                    joint_loss_fn = (
                        _source_stage_group_dro_losses
                        if group_dro_protocol else _source_stage_balanced_losses
                    )
                    losses = joint_loss_fn(
                        member,
                        torch.as_tensor(
                            normalized[batch_rows], dtype=torch.float32,
                            device=torch_device,
                        ),
                        torch.as_tensor(
                            targets[batch_rows, :2], dtype=torch.float32,
                            device=torch_device,
                        ),
                        torch.as_tensor(
                            source_index[batch_rows], dtype=torch.int64,
                            device=torch_device,
                        ),
                        severity_weight=severity_loss_weight,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    losses["total"].backward()
                    torch.nn.utils.clip_grad_norm_(
                        groups["encoder"] + groups["heads"], 5.0
                    )
                    optimizer.step()
                    epoch_rows.append({
                        key: float(value.detach().cpu())
                        for key, value in losses.items()
                    })
                member.eval()
                with torch.no_grad():
                    validation_loss_fn = (
                        _source_stage_group_dro_losses
                        if group_dro_protocol else _source_stage_balanced_losses
                    )
                    validation = validation_loss_fn(
                        member,
                        torch.as_tensor(
                            normalized[validation_index], dtype=torch.float32,
                            device=torch_device,
                        ),
                        torch.as_tensor(
                            targets[validation_index, :2], dtype=torch.float32,
                            device=torch_device,
                        ),
                        torch.as_tensor(
                            source_index[validation_index], dtype=torch.int64,
                            device=torch_device,
                        ),
                        severity_weight=severity_loss_weight,
                    )
                    robust_stage_reports = []
                    if robust_protocol:
                        validation_x = torch.as_tensor(
                            normalized[validation_index], dtype=torch.float32,
                            device=torch_device,
                        )
                        _, validation_severity = member(validation_x)
                        validation_severity_np = (
                            validation_severity.detach().cpu().numpy()
                        )
                        for stage_index in (0, 1):
                            if group_dro_protocol:
                                selection_report = (
                                    _layout_fold_severity_selection_report(
                                        train_targets=targets[
                                            train_index, stage_index
                                        ],
                                        validation_targets=targets[
                                            validation_index, stage_index
                                        ],
                                        validation_predictions=(
                                            validation_severity_np[:, stage_index]
                                        ),
                                        validation_sources=source_index[
                                            validation_index
                                        ],
                                        validation_layouts=layouts[
                                            validation_index
                                        ],
                                    )
                                )
                            else:
                                selection_report = _robust_severity_selection_report(
                                    train_targets=targets[train_index, stage_index],
                                    validation_targets=targets[
                                        validation_index, stage_index
                                    ],
                                    validation_predictions=validation_severity_np[
                                        :, stage_index
                                    ],
                                    validation_sources=source_index[
                                        validation_index
                                    ],
                                )
                            robust_stage_reports.append(selection_report)
                row = {
                    "phase": "joint_encoder_and_heads",
                    "critic": critic_index,
                    "epoch": epoch + 1,
                    **{
                        f"train_{key}": float(np.mean([
                            item[key] for item in epoch_rows
                        ]))
                        for key in epoch_rows[0]
                    },
                    **{
                        f"validation_{key}": float(value.detach().cpu())
                        for key, value in validation.items()
                    },
                }
                if robust_protocol:
                    row.update({
                        "validation_severity_selection_stage_0": (
                            robust_stage_reports[0]
                        ),
                        "validation_severity_selection_stage_1": (
                            robust_stage_reports[1]
                        ),
                    })
                    if group_dro_protocol:
                        row["validation_worst_stage_source_fold_relative_improvement"] = min(
                            report["worst_source_fold_relative_improvement"]
                            for report in robust_stage_reports
                        )
                    else:
                        row["validation_worst_stage_source_metric_ratio"] = max(
                            report["worst_source_metric_ratio"]
                            for report in robust_stage_reports
                        )
                history.append(row)
                current = (
                    -float(row[
                        "validation_worst_stage_source_fold_relative_improvement"
                    ])
                    if group_dro_protocol
                    else float(row["validation_worst_stage_source_metric_ratio"])
                    if robust_protocol
                    else float(validation["total"].detach().cpu())
                )
                if current < best_validation - 1e-4:
                    best_validation = current
                    best_state = copy.deepcopy(member.state_dict())
                    best_epoch = epoch + 1
                    stale = 0
                else:
                    stale += 1
                if stale >= int(early_stopping_patience):
                    break
            member.load_state_dict(best_state)
            history.append({
                "phase": "joint_selection",
                "critic": critic_index,
                "best_epoch": best_epoch,
                (
                    "best_validation_negative_worst_stage_source_fold_improvement"
                    if group_dro_protocol
                    else "best_validation_worst_stage_source_metric_ratio"
                    if robust_protocol else "best_validation_total"
                ): best_validation,
                "selection_rule": (
                    "maximize_worst_stage_source_layout_fold_improvement_v1"
                    if group_dro_protocol
                    else "minimize_worst_stage_source_max_mae_huber_ratio_v1"
                    if robust_protocol else "minimize_joint_validation_total_v1"
                ),
            })

            for stage_index, stage_name in enumerate(("initial", "ttc")):
                if robust_protocol:
                    _decoupled_stage_head_refinement(
                        member=member,
                        critic_index=critic_index,
                        stage_index=stage_index,
                        stage_name=stage_name,
                        normalized=normalized,
                        targets=targets,
                        source_index=source_index,
                        train_index=train_index,
                        validation_index=validation_index,
                        learning_rate=learning_rate,
                        head_refinement_epochs=head_refinement_epochs,
                        early_stopping_patience=early_stopping_patience,
                        batch_size=batch_size,
                        generator=generator,
                        torch_device=torch_device,
                        history=history,
                        validation_layouts=layouts[validation_index],
                        group_dro=group_dro_protocol,
                    )
                    continue
                for parameter in member.parameters():
                    parameter.requires_grad_(False)
                stage_parameters = list(
                    member.classification_heads[stage_index].parameters()
                ) + list(member.severity_heads[stage_index].parameters())
                for parameter in stage_parameters:
                    parameter.requires_grad_(True)
                stage_optimizer = torch.optim.Adam(
                    stage_parameters, lr=float(learning_rate)
                )
                best_stage_validation = float("inf")
                best_classification = copy.deepcopy(
                    member.classification_heads[stage_index].state_dict()
                )
                best_severity = copy.deepcopy(
                    member.severity_heads[stage_index].state_dict()
                )
                best_stage_epoch = 0
                stale = 0
                for epoch in range(int(head_refinement_epochs)):
                    order = torch.randperm(len(train_index), generator=generator)
                    epoch_rows = []
                    for start in range(0, len(order), int(batch_size)):
                        batch_rows = train_index[
                            order[start:start + int(batch_size)].numpy()
                        ]
                        losses = _source_stage_balanced_losses(
                            member,
                            torch.as_tensor(
                                normalized[batch_rows], dtype=torch.float32,
                                device=torch_device,
                            ),
                            torch.as_tensor(
                                targets[batch_rows, :2], dtype=torch.float32,
                                device=torch_device,
                            ),
                            torch.as_tensor(
                                source_index[batch_rows], dtype=torch.int64,
                                device=torch_device,
                            ),
                            severity_weight=severity_loss_weight,
                            stages=(stage_index,),
                        )
                        stage_optimizer.zero_grad(set_to_none=True)
                        losses["total"].backward()
                        torch.nn.utils.clip_grad_norm_(stage_parameters, 5.0)
                        stage_optimizer.step()
                        epoch_rows.append({
                            key: float(value.detach().cpu())
                            for key, value in losses.items()
                        })
                    member.eval()
                    with torch.no_grad():
                        validation = _source_stage_balanced_losses(
                            member,
                            torch.as_tensor(
                                normalized[validation_index], dtype=torch.float32,
                                device=torch_device,
                            ),
                            torch.as_tensor(
                                targets[validation_index, :2], dtype=torch.float32,
                                device=torch_device,
                            ),
                            torch.as_tensor(
                                source_index[validation_index], dtype=torch.int64,
                                device=torch_device,
                            ),
                            severity_weight=severity_loss_weight,
                            stages=(stage_index,),
                        )
                    current = float(validation["total"].detach().cpu())
                    history.append({
                        "phase": f"{stage_name}_head_refinement",
                        "critic": critic_index,
                        "epoch": epoch + 1,
                        **{
                            f"train_{key}": float(np.mean([
                                item[key] for item in epoch_rows
                            ]))
                            for key in epoch_rows[0]
                        },
                        **{
                            f"validation_{key}": float(value.detach().cpu())
                            for key, value in validation.items()
                        },
                    })
                    if current < best_stage_validation - 1e-4:
                        best_stage_validation = current
                        best_classification = copy.deepcopy(
                            member.classification_heads[stage_index].state_dict()
                        )
                        best_severity = copy.deepcopy(
                            member.severity_heads[stage_index].state_dict()
                        )
                        best_stage_epoch = epoch + 1
                        stale = 0
                    else:
                        stale += 1
                    if stale >= max(3, int(early_stopping_patience) // 2):
                        break
                member.classification_heads[stage_index].load_state_dict(
                    best_classification
                )
                member.severity_heads[stage_index].load_state_dict(best_severity)
                history.append({
                    "phase": f"{stage_name}_head_selection",
                    "critic": critic_index,
                    "best_epoch": best_stage_epoch,
                    "best_validation_total": best_stage_validation,
                })

            calibration_index = indices["calibration"]
            member.eval()
            with torch.no_grad():
                x = torch.as_tensor(
                    normalized[calibration_index], dtype=torch.float32,
                    device=torch_device,
                )
                calibration_logits, _ = member(x)
            for stage_index in (0, 1):
                temperature = _fit_temperature(
                    calibration_logits[:, stage_index],
                    torch.as_tensor(
                        targets[calibration_index, stage_index] > 1e-8,
                        dtype=torch.float32, device=torch_device,
                    ),
                )
                with torch.no_grad():
                    repaired.prior_logit_offsets[critic_index, stage_index] = 0.0
                    repaired.temperatures[critic_index, stage_index] = temperature
    else:
        initial_head_repair_parameters(repaired)
        train_positive = float(labels[train_index].sum())
        positive_weight = (len(train_index) - train_positive) / max(train_positive, 1.0)
        for critic_index, member in enumerate(repaired.critics):
            local_seed = int(seed) + 10_000 * critic_index
            torch.manual_seed(local_seed)
            parameters = list(member.classification_heads[0].parameters()) + list(
                member.severity_heads[0].parameters()
            )
            optimizer = torch.optim.Adam(parameters, lr=float(learning_rate))
            generator = torch.Generator(device="cpu").manual_seed(local_seed)
            member.eval()
            for epoch in range(int(epochs)):
                order = torch.randperm(len(train_index), generator=generator)
                losses = []
                for start in range(0, len(order), int(batch_size)):
                    batch_rows = train_index[order[start:start + int(batch_size)].numpy()]
                    x = torch.as_tensor(
                        normalized[batch_rows], dtype=torch.float32, device=torch_device
                    )
                    y = torch.as_tensor(
                        labels[batch_rows], dtype=torch.float32, device=torch_device
                    )
                    z = torch.as_tensor(
                        targets[batch_rows, 0], dtype=torch.float32, device=torch_device
                    )
                    optimizer.zero_grad(set_to_none=True)
                    encoded = member.encoder(x)
                    logits = member.classification_heads[0](encoded).squeeze(-1)
                    severity = F.softplus(
                        member.severity_heads[0](encoded).squeeze(-1)
                    )
                    classification = F.binary_cross_entropy_with_logits(
                        logits, y,
                        pos_weight=torch.as_tensor(positive_weight, device=torch_device),
                    )
                    positive = y > 0.5
                    regression = (
                        F.huber_loss(severity[positive], z[positive], delta=0.05)
                        if bool(positive.any()) else logits.new_zeros(())
                    )
                    loss = classification + regression
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, 5.0)
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                history.append({
                    "critic": critic_index,
                    "epoch": epoch + 1,
                    "loss": float(np.mean(losses)),
                })
            calibration_index = indices["calibration"]
            with torch.no_grad():
                x = torch.as_tensor(
                    normalized[calibration_index], dtype=torch.float32,
                    device=torch_device,
                )
                logits = member.classification_heads[0](member.encoder(x)).squeeze(-1)
            offset = torch.log(torch.as_tensor(positive_weight, device=torch_device))
            temperature = _fit_temperature(
                logits - offset,
                torch.as_tensor(labels[calibration_index], device=torch_device),
            )
            with torch.no_grad():
                repaired.prior_logit_offsets[critic_index, 0] = offset
                repaired.temperatures[critic_index, 0] = temperature
    repaired.eval()
    for parameter in repaired.parameters():
        parameter.requires_grad_(False)

    test_index = indices["test"]
    predictions = _predict(repaired, features, batch_size=batch_size)
    base_test = _predict(base_model, features[test_index], batch_size=batch_size)
    repaired_test = predictions["stage_risk"][test_index]
    ttc_preservation_max_abs_error = float(np.max(np.abs(
        base_test["stage_risk"][:, 1] - repaired_test[:, 1]
    )))
    if not joint_protocol and ttc_preservation_max_abs_error > 2e-5:
        raise RuntimeError("R4.3.2 Initial repair changed TTC predictions")

    action_start = int(data["global_state_normalized"].shape[1])
    action_end = action_start + int(np.prod(data["raw_joint_mean_action"].shape[1:]))
    old_mean = base_model.feature_mean[action_start:action_end].detach().cpu().numpy()
    old_std = base_model.feature_std[action_start:action_end].detach().cpu().numpy()
    new_action_mean = new_mean[action_start:action_end]
    new_action_std = new_std[action_start:action_end]
    action = features[:, action_start:action_end]
    old_in_support = np.max(np.abs((action - old_mean) / np.maximum(old_std, 1e-6)), axis=1) <= 6.0
    new_in_support = np.max(np.abs((action - new_action_mean) / np.maximum(new_action_std, 1e-6)), axis=1) <= 6.0

    cells: dict[str, Any] = {}
    all_cells_pass = True
    for source in sources:
        name = str(source["name"])
        source_mask = source_index == int(source["index"])
        raw_test_mask = np.isin(layouts, splits["test"]) & eligible & source_mask
        mask = raw_test_mask & new_in_support
        if not np.any(mask):
            raise RuntimeError(f"{phase_name} has no in-support test rows for {name}")
        initial_target = targets[mask, 0]
        initial_prediction = predictions["stage_risk"][mask, 0]
        total_target = targets[mask, :2].mean(axis=1)
        total_prediction = predictions["total_risk"][mask]
        old_ood = float(1.0 - old_in_support[raw_test_mask].mean())
        new_ood = float(1.0 - new_in_support[raw_test_mask].mean())
        if expected_risk_protocol:
            member_reports: dict[str, Any] = {}
            member_passes = []
            for critic_index in range(2):
                stage_reports: dict[str, Any] = {}
                stage_checks: dict[str, bool] = {}
                for stage_index, stage_name in enumerate(
                    ("initial_joint_guard", "obstacle_ttc")
                ):
                    stage_target = targets[mask, stage_index]
                    stage_labels = stage_target > 1e-8
                    stage_probability = predictions["member_probability"][
                        mask, critic_index, stage_index
                    ]
                    stage_severity = predictions["member_severity"][
                        mask, critic_index, stage_index
                    ]
                    stage_expected = predictions["member_stage_risk"][
                        mask, critic_index, stage_index
                    ]
                    classification = classification_report(
                        stage_labels, stage_probability
                    )
                    classification_bootstrap = classification_layout_bootstrap(
                        stage_labels, stage_probability, layouts[mask],
                        replicates=bootstrap_replicates,
                        seed=int(seed) + 1000 * critic_index + 100 * stage_index + 11,
                    )
                    train_stage_positive = targets[train_index, stage_index] > 1e-8
                    positive = stage_labels
                    severity_report = positive_severity_report(
                        train_positive_targets=targets[train_index, stage_index][
                            train_stage_positive
                        ],
                        test_targets=stage_target[positive],
                        test_predictions=stage_severity[positive],
                        test_layouts=layouts[mask][positive],
                        replicates=bootstrap_replicates,
                        seed=int(seed) + 1000 * critic_index + 100 * stage_index + 22,
                    )
                    if group_dro_protocol:
                        severity_report["per_layout"] = _positive_severity_per_layout(
                            train_positive_targets=targets[
                                train_index, stage_index
                            ][train_stage_positive],
                            test_targets=stage_target[positive],
                            test_predictions=stage_severity[positive],
                            test_layouts=layouts[mask][positive],
                        )
                    expected_calibration = expected_cost_calibration_report(
                        stage_target, stage_expected, layouts[mask]
                    )
                    prefix = f"{stage_name}_critic_{critic_index}"
                    local_checks = {
                        f"{prefix}_has_100_per_class": (
                            classification["positive"] >= 100
                            and classification["negative"] >= 100
                        ),
                        f"{prefix}_auroc_lower_95_above_0_65": (
                            classification_bootstrap["auroc"]["lower_95"] > 0.65
                        ),
                        f"{prefix}_auprc_lift_point_at_least_1_5": (
                            classification["auprc_lift"] >= 1.5
                        ),
                        f"{prefix}_auprc_lift_lower_95_above_1_0": (
                            classification_bootstrap["auprc_lift"]["lower_95"] > 1.0
                        ),
                        f"{prefix}_brier_beats_prevalence": (
                            classification["brier_skill"] > 0.0
                        ),
                        f"{prefix}_nll_beats_prevalence": (
                            classification["nll_improvement"] > 0.0
                        ),
                        f"{prefix}_ece_at_most_0_10": (
                            classification["calibration"]["ece"] <= 0.10
                        ),
                        f"{prefix}_positive_severity_mae_beats_median": (
                            severity_report["layout_bootstrap"]["mae_improvement"][
                                "lower_95"
                            ] > 0.0
                        ),
                        f"{prefix}_positive_severity_has_10_layouts": (
                            severity_report["positive_layout_count"] >= 10
                        ),
                        f"{prefix}_positive_severity_huber_beats_median": (
                            severity_report["layout_bootstrap"]["huber_improvement"][
                                "lower_95"
                            ] > 0.0
                        ),
                        f"{prefix}_risk_bin_calibration_error_at_most_0_10": (
                            expected_calibration["risk_bin_calibration_error"] <= 0.10
                        ),
                        f"{prefix}_layout_calibration_error_at_most_0_10": (
                            expected_calibration["layout_mean_calibration_error"] <= 0.10
                        ),
                        f"{prefix}_layout_mean_bias_at_most_0_10": (
                            abs(expected_calibration["layout_mean_bias"]) <= 0.10
                        ),
                    }
                    stage_checks.update(local_checks)
                    stage_reports[stage_name] = {
                        "classification": classification,
                        "classification_layout_bootstrap": classification_bootstrap,
                        "positive_severity": severity_report,
                        "expected_cost_calibration": expected_calibration,
                        "checks": local_checks,
                    }
                member_total_prediction = predictions["member_stage_risk"][
                    mask, critic_index, :2
                ].mean(axis=1)
                total_calibration = expected_cost_calibration_report(
                    total_target, member_total_prediction, layouts[mask]
                )
                total_checks = {
                    f"critic_{critic_index}_total_risk_bin_calibration_error_at_most_0_10": (
                        total_calibration["risk_bin_calibration_error"] <= 0.10
                    ),
                    f"critic_{critic_index}_total_layout_calibration_error_at_most_0_10": (
                        total_calibration["layout_mean_calibration_error"] <= 0.10
                    ),
                    f"critic_{critic_index}_total_layout_mean_bias_at_most_0_10": (
                        abs(total_calibration["layout_mean_bias"]) <= 0.10
                    ),
                }
                critic_checks = {**stage_checks, **total_checks}
                critic_passed = all(bool(value) for value in critic_checks.values())
                member_passes.append(critic_passed)
                member_reports[f"critic_{critic_index}"] = {
                    "status": "PASS" if critic_passed else "FAIL",
                    "stages": stage_reports,
                    "total_expected_cost_calibration": total_calibration,
                    "checks": critic_checks,
                }

            conservative = regression_report(total_target, total_prediction)
            conservative_bootstrap = layout_bootstrap_report(
                total_target, total_prediction, layouts[mask],
                replicates=bootstrap_replicates, seed=int(seed) + 303,
            )
            conservative_checks = {
                "finite_nonnegative": bool(
                    np.isfinite(total_prediction).all()
                    and np.all(total_prediction >= 0.0)
                ),
                "total_auroc_lower_95_above_0_65": (
                    conservative_bootstrap["auroc"]["lower_95"] > 0.65
                ),
                "total_spearman_lower_95_above_0_20": (
                    conservative_bootstrap["spearman"]["lower_95"] > 0.20
                ),
            }
            checks = {
                "sampled_action_ood_at_most_10pct": new_ood <= 0.10,
                "both_individual_critics_calibrated": all(member_passes),
                **(
                    {"ttc_retrained_prediction_finite": bool(
                        np.isfinite(predictions["member_stage_risk"][mask, :, 1]).all()
                    )}
                    if joint_protocol
                    else {"ttc_prediction_preserved": (
                        ttc_preservation_max_abs_error <= 2e-5
                    )}
                ),
                **{
                    f"actor_conservative_{key}": value
                    for key, value in conservative_checks.items()
                },
            }
            passed = all(bool(value) for value in checks.values())
            cells[name] = {
                "status": "PASS" if passed else "FAIL",
                "test_rows_before_support_filter": int(raw_test_mask.sum()),
                "test_rows": int(mask.sum()),
                "old_sampled_action_ood_rate": old_ood,
                "new_sampled_action_ood_rate": new_ood,
                "individual_critic_certification": member_reports,
                "actor_conservative_twin_max": {
                    "role": "actor_only_conservative_risk_not_probability_calibration",
                    "regression_diagnostic": conservative,
                    "layout_bootstrap": conservative_bootstrap,
                    "checks": conservative_checks,
                    "zero_baseline_mae_gate_used": False,
                },
                "checks": checks,
            }
        else:
            initial = regression_report(initial_target, initial_prediction)
            initial_bootstrap = layout_bootstrap_report(
                initial_target, initial_prediction, layouts[mask],
                replicates=bootstrap_replicates, seed=int(seed) + 101,
            )
            total = regression_report(total_target, total_prediction)
            total_bootstrap = layout_bootstrap_report(
                total_target, total_prediction, layouts[mask],
                replicates=bootstrap_replicates, seed=int(seed) + 202,
            )
            probability = predictions["selected_probability"][mask, 0]
            severity = predictions["selected_severity"][mask, 0]
            positive = initial_target > 1e-8
            probability_bias = float(probability.mean() - positive.mean())
            severity_bias = float(
                severity[positive].mean() - initial_target[positive].mean()
            )
            checks = {
                "sampled_action_ood_at_most_10pct": new_ood <= 0.10,
                "initial_has_100_per_class": (
                    initial["positive"] >= 100 and initial["negative"] >= 100
                ),
                "initial_auroc_lower_95_above_0_65": (
                    initial_bootstrap["auroc"]["lower_95"] > 0.65
                ),
                "initial_mae_equivalent_to_zero": (
                    initial_bootstrap["mae_improvement_over_zero"]["lower_95"] >= -0.01
                ),
                "initial_absolute_expected_cost_bias_at_most_0_10": (
                    abs(float(initial["mean_bias"])) <= 0.10
                ),
                "initial_absolute_probability_bias_at_most_0_10": (
                    abs(probability_bias) <= 0.10
                ),
                "initial_absolute_positive_severity_bias_at_most_0_10": (
                    abs(severity_bias) <= 0.10
                ),
                "total_has_100_per_class": (
                    total["positive"] >= 100 and total["negative"] >= 100
                ),
                "total_auroc_lower_95_above_0_65": (
                    total_bootstrap["auroc"]["lower_95"] > 0.65
                ),
                "total_spearman_lower_95_above_0_20": (
                    total_bootstrap["spearman"]["lower_95"] > 0.20
                ),
                "total_mae_equivalent_to_zero": (
                    total_bootstrap["mae_improvement_over_zero"]["lower_95"] >= -0.01
                ),
                "total_absolute_mean_bias_at_most_0_10": (
                    abs(float(total["mean_bias"])) <= 0.10
                ),
                "ttc_prediction_preserved": ttc_preservation_max_abs_error <= 2e-5,
            }
            passed = all(bool(value) for value in checks.values())
            cells[name] = {
                "status": "PASS" if passed else "FAIL",
                "test_rows_before_support_filter": int(raw_test_mask.sum()),
                "test_rows": int(mask.sum()),
                "old_sampled_action_ood_rate": old_ood,
                "new_sampled_action_ood_rate": new_ood,
                "initial": initial,
                "initial_layout_bootstrap": initial_bootstrap,
                "initial_probability_bias": probability_bias,
                "initial_positive_severity_bias": severity_bias,
                "total": total,
                "total_layout_bootstrap": total_bootstrap,
                "checks": checks,
            }
        all_cells_pass = all_cells_pass and passed

    output.mkdir(parents=True, exist_ok=False)
    history_path = output / "training_history.json"
    save_json({
        "schema_version": (
            "IAMAPPOR4_3_6LayoutDROTrainingHistoryV1"
            if group_dro_protocol
            else "IAMAPPOR4_3_5RobustSeverityTrainingHistoryV1"
            if robust_protocol
            else "IAMAPPOR4_3_4JointStageTrainingHistoryV1"
            if joint_protocol
            else "IAMAPPOR4_3_3ExpectedRiskTrainingHistoryV1"
            if expected_risk_protocol
            else "IAMAPPOR4_3_2InitialRepairHistoryV1"
        ),
        "rows": history,
    }, history_path)
    report = {
        "schema_version": (
            LAYOUT_DRO_VALIDATION_SCHEMA
            if group_dro_protocol
            else ROBUST_SEVERITY_CERTIFICATION_SCHEMA
            if robust_protocol
            else JOINT_REPAIR_CERTIFICATION_SCHEMA
            if joint_protocol
            else EXPECTED_RISK_CERTIFICATION_SCHEMA
            if expected_risk_protocol
            else CERTIFICATION_SCHEMA
        ),
        "dataset": str(dataset),
        "dataset_manifest_sha256": file_sha256(dataset / "manifest.json"),
        "base_artifact": str(base_path),
        "base_artifact_sha256": file_sha256(base_path),
        "feature_contract": R3_FEATURE_CONTRACT,
        "feature_normalization": "multi_policy_sampled_train_layout_mean_std_v1",
        "normalizer_rebase": "first_linear_exact_affine_reparameterization_v1",
        "rebase_max_abs_error": rebase_max_abs_error,
        "trained_parameters": (
            "full_encoder_plus_group_dro_decoupled_initial_and_ttc_heads"
            if group_dro_protocol
            else "full_encoder_plus_decoupled_initial_and_ttc_heads"
            if robust_protocol
            else "last_encoder_block_plus_initial_and_ttc_classification_and_severity_heads"
            if joint_protocol
            else "initial_classification_and_severity_heads_only"
        ),
        "ttc_frozen": not joint_protocol,
        "ttc_preservation_max_abs_error": (
            None if joint_protocol else ttc_preservation_max_abs_error
        ),
        "ttc_retraining_change_max_abs_error": (
            ttc_preservation_max_abs_error if joint_protocol else None
        ),
        "support_z_limit": 6.0,
        "cells": cells,
        "all_cells_passed": bool(all_cells_pass),
        "ready_for_fresh_certification": bool(
            group_dro_protocol and all_cells_pass
        ),
        "certified_for_online_reaudit": bool(
            all_cells_pass and not group_dro_protocol
        ),
        "certified_for_lambda_calibration": bool(
            all_cells_pass and expected_risk_protocol and not group_dro_protocol
        ),
        "certified_for_actor_training": False,
        "actor_training_block_reason": (
            "requires_new_unopened_r4_3_6_fresh_certification"
            if group_dro_protocol and all_cells_pass
            else "requires_fresh_frozen_critic_gradient_and_lambda_calibration"
            if all_cells_pass else (
                "r4_3_6_layout_group_dro_validation_failed"
                if group_dro_protocol
                else "r4_3_5_robust_severity_certification_failed"
                if robust_protocol
                else "r4_3_4_joint_stage_certification_failed"
                if joint_protocol
                else "r4_3_3_expected_risk_certification_failed"
                if expected_risk_protocol
                else "r4_3_2_initial_repair_certification_failed"
            )
        ),
        "protocol": {
            "certification_protocol": certification_protocol,
            "per_step_zero_baseline_mae_is_blocking": False if expected_risk_protocol else True,
            "individual_critic_probability_calibration_is_blocking": bool(expected_risk_protocol),
            "twin_max_role": (
                "actor_conservative_risk_only"
                if expected_risk_protocol else "legacy_selected_member"
            ),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "encoder_learning_rate_scale": (
                float(encoder_learning_rate_scale) if joint_protocol else None
            ),
            "severity_loss_weight": (
                float(severity_loss_weight) if joint_protocol else 1.0
            ),
            "source_stage_balanced_loss": bool(
                joint_protocol and not group_dro_protocol
            ),
            "severity_group_dro_worst_source_stage_loss": bool(
                group_dro_protocol
            ),
            "encoder_selection_rule": (
                "maximize_worst_stage_source_layout_fold_improvement_v1"
                if group_dro_protocol
                else "minimize_worst_stage_source_max_mae_huber_ratio_v1"
                if robust_protocol
                else "minimize_joint_validation_total_v1"
                if joint_protocol else None
            ),
            "classification_head_selection_rule": (
                "minimize_source_balanced_validation_bce_v1"
                if robust_protocol else None
            ),
            "severity_head_selection_rule": (
                "maximize_worst_source_layout_fold_mae_huber_relative_improvement_v1"
                if group_dro_protocol
                else "minimize_worst_source_max_mae_huber_ratio_v1"
                if robust_protocol else None
            ),
            "selection_layout_fold_count": 5 if group_dro_protocol else None,
            "opened_r4_3_5_test_used_for_validation_only": bool(
                group_dro_protocol
            ),
            "classification_severity_refinement_decoupled": bool(robust_protocol),
            "early_stopping_patience": (
                int(early_stopping_patience) if joint_protocol else None
            ),
            "head_refinement_epochs": (
                int(head_refinement_epochs) if joint_protocol else 0
            ),
            "bootstrap_replicates": int(bootstrap_replicates),
            "seed": int(seed),
            "layout_split": (
                "60_train_10_calibration_10_validation_20_fresh_test"
                if joint_protocol
                else "60_train_10_calibration_10_threshold_unused_20_test"
            ),
        },
    }
    report_path = output / "certification.json"
    save_json(report, report_path)
    if all_cells_pass or group_dro_protocol:
        candidate_path = output / (
            "r4_3_6_validation_candidate.pt"
            if group_dro_protocol
            else "r4_3_5_robust_severity_candidate.pt"
            if robust_protocol
            else "r4_3_4_joint_stage_candidate.pt"
            if joint_protocol
            else "r4_3_3_expected_risk_candidate.pt"
            if expected_risk_protocol
            else "r4_3_2_initial_repaired_candidate.pt"
        )
        certification_hash = file_sha256(report_path)
        torch.save({
            "schema_version": (
                LAYOUT_DRO_CANDIDATE_SCHEMA
                if group_dro_protocol
                else ROBUST_SEVERITY_CANDIDATE_SCHEMA
                if robust_protocol
                else JOINT_REPAIR_CANDIDATE_SCHEMA
                if joint_protocol
                else EXPECTED_RISK_CANDIDATE_SCHEMA
                if expected_risk_protocol
                else CANDIDATE_SCHEMA
            ),
            "input_dim": int(base_payload["input_dim"]),
            "state_dict": repaired.state_dict(),
            "feature_contract": R3_FEATURE_CONTRACT,
            "dataset_manifest_sha256": file_sha256(dataset / "manifest.json"),
            "source_base_artifact_sha256": file_sha256(base_path),
            "certification_report_sha256": certification_hash,
            "diagnostic_only": bool(group_dro_protocol),
            "ready_for_fresh_certification": bool(
                group_dro_protocol and all_cells_pass
            ),
            "certified_for_online_reaudit": bool(
                all_cells_pass and not group_dro_protocol
            ),
            "certified_for_lambda_calibration": bool(
                all_cells_pass and expected_risk_protocol
                and not group_dro_protocol
            ),
            "certified_for_actor_training": False,
            "requires_lambda_gradient_calibration": bool(
                not group_dro_protocol
            ),
            "required_actor_mode": "stage_counterfactual_v5",
        }, candidate_path)
        report["candidate"] = str(candidate_path)
        report["candidate_sha256"] = file_sha256(candidate_path)
        save_json({
            "schema_version": (
                "IAMAPPOR4_3_6ValidationCandidateManifestV1"
                if group_dro_protocol
                else "IAMAPPOR4_3_5CandidateManifestV1"
                if robust_protocol
                else "IAMAPPOR4_3_4CandidateManifestV1"
                if joint_protocol
                else "IAMAPPOR4_3_3CandidateManifestV1"
                if expected_risk_protocol
                else "IAMAPPOR4_3_2CandidateManifestV1"
            ),
            "certification": str(report_path),
            "certification_sha256": certification_hash,
            "candidate": str(candidate_path),
            "candidate_sha256": report["candidate_sha256"],
            "diagnostic_only": bool(group_dro_protocol),
            "ready_for_fresh_certification": bool(
                group_dro_protocol and all_cells_pass
            ),
            "certified_for_actor_training": False,
        }, output / "candidate_manifest.json")
    return report
