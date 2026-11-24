"""Layout-grouped learnability metrics and preregistered R2/R3 gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


def _binary(values: Sequence[int | bool]) -> np.ndarray:
    result = np.asarray(values, dtype=np.int8).reshape(-1)
    if not np.isin(result, (0, 1)).all():
        raise ValueError("labels must be binary")
    return result


def binary_auroc(labels, scores) -> float:
    labels = _binary(labels)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("AUROC requires matching finite scores")
    positive_count = int(np.count_nonzero(labels == 1))
    negative_count = int(np.count_nonzero(labels == 0))
    if not positive_count or not negative_count:
        raise ValueError("AUROC requires both classes")
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    positive_rank_sum = float(np.sum(ranks[labels == 1]))
    statistic = positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    return statistic / float(positive_count * negative_count)


def binary_average_precision(labels, scores) -> float:
    labels = _binary(labels)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.shape != scores.shape or not np.count_nonzero(labels):
        raise ValueError("average precision requires matching scores and positives")
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision * ranked) / np.count_nonzero(ranked))


def expected_calibration_error(labels, probabilities, bins: int = 10) -> float:
    labels = _binary(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if labels.shape != probabilities.shape or not np.isfinite(probabilities).all():
        raise ValueError("calibration inputs must be matching and finite")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("probabilities must be in [0,1]")
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    total = max(len(labels), 1)
    error = 0.0
    for index in range(int(bins)):
        lower, upper = edges[index], edges[index + 1]
        mask = (probabilities >= lower) & (
            probabilities <= upper if index == bins - 1 else probabilities < upper
        )
        if np.any(mask):
            error += float(np.count_nonzero(mask) / total) * abs(
                float(np.mean(probabilities[mask])) - float(np.mean(labels[mask]))
            )
    return float(error)


@dataclass(frozen=True)
class LearnabilityGateResultV1:
    passed: bool
    train_positive: int
    train_negative: int
    heldout_positive: int
    heldout_negative: int
    prevalence: float
    auroc: float
    auroc_lower_95: float
    average_precision: float
    auprc_lift: float
    auprc_lift_lower_95: float
    ece: float
    material_recall: float
    material_precision: float
    failures: tuple[str, ...]
    schema_version: str = "LearnabilityGateResultV1"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


TEAM_AUPRC_NORMALIZED_GAIN_MIN = 0.30
TEAM_AUPRC_NORMALIZED_GAIN_LOWER_MIN = 0.0


def normalized_auprc_gain(average_precision: float, prevalence: float) -> float:
    """Return the fraction of attainable AP headroom above prevalence.

    Unlike AP/prevalence lift, this quantity remains attainable when a
    team-level any-agent label has prevalence above two thirds.  Zero is the
    prevalence baseline and one is perfect ranking.
    """

    average_precision = float(average_precision)
    prevalence = float(prevalence)
    if not 0.0 < prevalence < 1.0:
        raise ValueError("normalized AUPRC gain requires prevalence in (0,1)")
    return (average_precision - prevalence) / (1.0 - prevalence)


@dataclass(frozen=True)
class LearnabilityGateResultV2:
    passed: bool
    train_positive: int
    train_negative: int
    heldout_positive: int
    heldout_negative: int
    prevalence: float
    auroc: float
    auroc_lower_95: float
    average_precision: float
    auprc_lift: float
    auprc_lift_lower_95: float
    auprc_normalized_gain: float
    auprc_normalized_gain_lower_95: float
    auprc_normalized_gain_min: float
    auprc_normalized_gain_lower_min: float
    ece: float
    material_recall: float
    material_precision: float
    failures: tuple[str, ...]
    gate_contract: str = "team_any_agent_prevalence_normalized_v2"
    schema_version: str = "LearnabilityGateResultV2"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_learnability_gate(
    *,
    train_labels,
    heldout_labels,
    heldout_probabilities,
    heldout_layout_ids,
    threshold: float = 0.5,
    bootstrap_replicates: int = 2000,
    seed: int = 0,
) -> LearnabilityGateResultV1:
    """Evaluate one head with bootstrap resampling at the layout level."""

    train = _binary(train_labels)
    heldout = _binary(heldout_labels)
    probabilities = np.asarray(heldout_probabilities, dtype=np.float64).reshape(-1)
    layouts = np.asarray(heldout_layout_ids).reshape(-1)
    if not (heldout.shape == probabilities.shape == layouts.shape):
        raise ValueError("held-out labels, scores and layout ids must align")
    unique_layouts = np.unique(layouts)
    if len(unique_layouts) < 2 or bootstrap_replicates <= 0:
        raise ValueError("layout bootstrap requires at least two layouts and replicates")

    auroc = binary_auroc(heldout, probabilities)
    average_precision = binary_average_precision(heldout, probabilities)
    prevalence = float(np.mean(heldout))
    lift = average_precision / prevalence
    ece = expected_calibration_error(heldout, probabilities)
    predicted = probabilities >= float(threshold)
    true_positive = int(np.count_nonzero(predicted & (heldout == 1)))
    recall = true_positive / max(int(np.count_nonzero(heldout == 1)), 1)
    precision = true_positive / max(int(np.count_nonzero(predicted)), 1)

    rng = np.random.default_rng(int(seed))
    auc_samples: list[float] = []
    lift_samples: list[float] = []
    rows_by_layout = {
        layout: np.flatnonzero(layouts == layout) for layout in unique_layouts
    }
    for _ in range(int(bootstrap_replicates)):
        sampled_layouts = rng.choice(unique_layouts, len(unique_layouts), replace=True)
        rows = np.concatenate([rows_by_layout[item] for item in sampled_layouts])
        sample_labels = heldout[rows]
        if not np.any(sample_labels == 0) or not np.any(sample_labels == 1):
            continue
        sample_scores = probabilities[rows]
        sample_prevalence = float(np.mean(sample_labels))
        auc_samples.append(binary_auroc(sample_labels, sample_scores))
        lift_samples.append(
            binary_average_precision(sample_labels, sample_scores)
            / sample_prevalence
        )
    if not auc_samples:
        raise ValueError("layout bootstrap produced no two-class replicates")
    auc_lower = float(np.quantile(auc_samples, 0.025))
    lift_lower = float(np.quantile(lift_samples, 0.025))

    counts = {
        "train_positive": int(np.count_nonzero(train == 1)),
        "train_negative": int(np.count_nonzero(train == 0)),
        "heldout_positive": int(np.count_nonzero(heldout == 1)),
        "heldout_negative": int(np.count_nonzero(heldout == 0)),
    }
    failures = []
    if min(counts["train_positive"], counts["train_negative"]) < 100:
        failures.append("train_support_below_100_per_class")
    if min(counts["heldout_positive"], counts["heldout_negative"]) < 20:
        failures.append("heldout_support_below_20_per_class")
    if auc_lower <= 0.65:
        failures.append("auroc_lower_not_above_0.65")
    if lift_lower <= 1.0 or lift < 1.5:
        failures.append("auprc_lift_gate_failed")
    if ece > 0.10:
        failures.append("ece_above_0.10")
    if recall < 0.80 or precision < 0.50:
        failures.append("material_recall_precision_gate_failed")
    return LearnabilityGateResultV1(
        passed=not failures, prevalence=prevalence, auroc=auroc,
        auroc_lower_95=auc_lower, average_precision=average_precision,
        auprc_lift=lift, auprc_lift_lower_95=lift_lower, ece=ece,
        material_recall=float(recall), material_precision=float(precision),
        failures=tuple(failures), **counts,
    )


def evaluate_team_learnability_gate_v2(
    *,
    train_labels,
    heldout_labels,
    heldout_probabilities,
    heldout_layout_ids,
    threshold: float = 0.5,
    bootstrap_replicates: int = 2000,
    seed: int = 0,
    normalized_gain_min: float = TEAM_AUPRC_NORMALIZED_GAIN_MIN,
    normalized_gain_lower_min: float = TEAM_AUPRC_NORMALIZED_GAIN_LOWER_MIN,
) -> LearnabilityGateResultV2:
    """Evaluate a team-level head with a prevalence-attainable AUPRC gate.

    The legacy AP/prevalence lift is retained for diagnostics only.  The V2
    decision uses normalized AP headroom, (AP - prevalence)/(1 - prevalence),
    so a perfect classifier always scores one regardless of event prevalence.
    """

    train = _binary(train_labels)
    heldout = _binary(heldout_labels)
    probabilities = np.asarray(heldout_probabilities, dtype=np.float64).reshape(-1)
    layouts = np.asarray(heldout_layout_ids).reshape(-1)
    if not (heldout.shape == probabilities.shape == layouts.shape):
        raise ValueError("held-out labels, scores and layout ids must align")
    if not 0.0 <= float(normalized_gain_min) <= 1.0:
        raise ValueError("normalized gain point threshold must be in [0,1]")
    if not 0.0 <= float(normalized_gain_lower_min) <= 1.0:
        raise ValueError("normalized gain lower threshold must be in [0,1]")
    unique_layouts = np.unique(layouts)
    if len(unique_layouts) < 2 or bootstrap_replicates <= 0:
        raise ValueError("layout bootstrap requires at least two layouts and replicates")

    auroc = binary_auroc(heldout, probabilities)
    average_precision = binary_average_precision(heldout, probabilities)
    prevalence = float(np.mean(heldout))
    lift = average_precision / prevalence
    normalized_gain = normalized_auprc_gain(average_precision, prevalence)
    ece = expected_calibration_error(heldout, probabilities)
    predicted = probabilities >= float(threshold)
    true_positive = int(np.count_nonzero(predicted & (heldout == 1)))
    recall = true_positive / max(int(np.count_nonzero(heldout == 1)), 1)
    precision = true_positive / max(int(np.count_nonzero(predicted)), 1)

    rng = np.random.default_rng(int(seed))
    auc_samples: list[float] = []
    lift_samples: list[float] = []
    normalized_gain_samples: list[float] = []
    rows_by_layout = {
        layout: np.flatnonzero(layouts == layout) for layout in unique_layouts
    }
    for _ in range(int(bootstrap_replicates)):
        sampled_layouts = rng.choice(unique_layouts, len(unique_layouts), replace=True)
        rows = np.concatenate([rows_by_layout[item] for item in sampled_layouts])
        sample_labels = heldout[rows]
        if not np.any(sample_labels == 0) or not np.any(sample_labels == 1):
            continue
        sample_scores = probabilities[rows]
        sample_prevalence = float(np.mean(sample_labels))
        sample_ap = binary_average_precision(sample_labels, sample_scores)
        auc_samples.append(binary_auroc(sample_labels, sample_scores))
        lift_samples.append(sample_ap / sample_prevalence)
        normalized_gain_samples.append(
            normalized_auprc_gain(sample_ap, sample_prevalence)
        )
    if not auc_samples:
        raise ValueError("layout bootstrap produced no two-class replicates")
    auc_lower = float(np.quantile(auc_samples, 0.025))
    lift_lower = float(np.quantile(lift_samples, 0.025))
    normalized_gain_lower = float(
        np.quantile(normalized_gain_samples, 0.025)
    )

    counts = {
        "train_positive": int(np.count_nonzero(train == 1)),
        "train_negative": int(np.count_nonzero(train == 0)),
        "heldout_positive": int(np.count_nonzero(heldout == 1)),
        "heldout_negative": int(np.count_nonzero(heldout == 0)),
    }
    failures = []
    if min(counts["train_positive"], counts["train_negative"]) < 100:
        failures.append("train_support_below_100_per_class")
    if min(counts["heldout_positive"], counts["heldout_negative"]) < 20:
        failures.append("heldout_support_below_20_per_class")
    if auc_lower <= 0.65:
        failures.append("auroc_lower_not_above_0.65")
    if (
        normalized_gain_lower <= float(normalized_gain_lower_min)
        or normalized_gain < float(normalized_gain_min)
    ):
        failures.append("auprc_normalized_gain_gate_failed")
    if ece > 0.10:
        failures.append("ece_above_0.10")
    if recall < 0.80 or precision < 0.50:
        failures.append("material_recall_precision_gate_failed")
    return LearnabilityGateResultV2(
        passed=not failures,
        prevalence=prevalence,
        auroc=auroc,
        auroc_lower_95=auc_lower,
        average_precision=average_precision,
        auprc_lift=lift,
        auprc_lift_lower_95=lift_lower,
        auprc_normalized_gain=normalized_gain,
        auprc_normalized_gain_lower_95=normalized_gain_lower,
        auprc_normalized_gain_min=float(normalized_gain_min),
        auprc_normalized_gain_lower_min=float(normalized_gain_lower_min),
        ece=ece,
        material_recall=float(recall),
        material_precision=float(precision),
        failures=tuple(failures),
        **counts,
    )
