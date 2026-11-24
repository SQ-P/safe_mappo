"""Read-only quality classification of existing signed parameter measurements.

Observed flatness is not proof of a zero derivative. No gradient export occurs.
"""
from collections import Counter
from copy import deepcopy

from src.algorithms.ppcsa_parameter_sensitivity import (
    SCALES, REPEATS, ABS_RESPONSE_TOL, parameterization, protocol, digest,
    metrics, validate_point, stage_signature, close_response, analyze_axis, max_calls)
from src.algorithms.ppcsa_update_acceptance import finite

NUMERIC_FLAGS = ("ONE_SIDED_INCONSISTENCY", "SCALE_INCONSISTENCY", "HELDOUT_SCALE_PREDICTION_MISMATCH")


def reproduce_axis(baseline, samples):
    reproduced = analyze_axis(baseline, samples)
    fields = ("status", "flags", "errors", "usable", "gradient", "central_estimates_by_scale",
        "forward_estimates_by_scale", "backward_estimates_by_scale", "heldout_prediction_residuals",
        "unresolved_metric_indices")
    return {field: reproduced[field] for field in fields if field in reproduced}


def validate_structure(report, registry, historical_baselines, *, validate_samples=True):
    """Coverage/identity preflight only; no classification or new evaluation."""
    if (report.get("schema_version") != "PPCSAParameterSensitivityDiagnosticV1"
            or report.get("protocol") != protocol()
            or report.get("measurement_protocol_sha256") != digest(protocol())
            or report.get("parameterization_sha256") != digest(parameterization())
            or report.get("fixed_state_registry") != registry):
        raise ValueError("Measurement schema/protocol/registry mismatch")
    keys = [f["key"] for f in registry]
    if not keys or len(set(keys)) != len(keys) or set(historical_baselines) != set(keys):
        raise ValueError("Incomplete fixed historical baseline registry")
    states = report.get("states")
    if not isinstance(states, list) or [s.get("key") for s in states] != keys:
        raise ValueError("Missing, reordered or duplicate measured states")
    progress_ids = ["progress:"+key for key in keys]
    risk_ids = [f"risk:{key}:{t['observer_id']}:{t['track_id']}" for key in keys for t in historical_baselines[key]["tracks"]]
    if report.get("progress_ids") != progress_ids or report.get("risk_ids") != risk_ids:
        raise ValueError("Progress/risk identity coverage differs")
    if report.get("shadow_calls") != max_calls(len(keys)) or report.get("maximum_shadow_calls") != max_calls(len(keys)):
        raise ValueError("Fixed signed measurement call coverage differs")
    for fixed, state in zip(registry, states):
        if state.get("state_index") != fixed["state_index"] or len(state.get("axes", [])) != 6:
            raise ValueError("Coordinate or state-index coverage differs")
        bases = state.get("baseline_repeats")
        if not isinstance(bases, list) or len(bases) != REPEATS:
            raise ValueError("Missing baseline repeats")
        for base in bases if validate_samples else []:
            try:
                validate_point(base)
            except (ValueError, TypeError, KeyError):
                if not state.get("errors"):
                    raise ValueError("Unrecorded invalid baseline evidence")
            if (isinstance(base, dict) and base.get("tracks") != historical_baselines[fixed["key"]]["tracks"]
                    and "HISTORICAL_BASELINE_MISMATCH:tracks" not in state.get("errors", [])):
                raise ValueError("Unrecorded historical track identity/context mismatch")
        for axis_index, axis in enumerate(state["axes"]):
            samples = axis.get("samples")
            if not isinstance(samples, list) or len(samples) != len(SCALES):
                raise ValueError("Missing scale coverage")
            recorded_invalid = axis.get("status") == "INVALID_MEASUREMENT"
            if validate_samples and recorded_invalid:
                reproduced = reproduce_axis(bases[0], samples)
                if (reproduced["status"] != "INVALID_MEASUREMENT" or not reproduced["errors"]
                        or any(axis.get(k) != v for k, v in reproduced.items())):
                    raise ValueError("Recorded invalid measurement does not reproduce")
            for h, sample in zip(SCALES, samples):
                if sample.get("h") != h:
                    raise ValueError("Tuned/reordered difference scale")
                for sign in ("minus", "plus"):
                    points = sample.get(sign)
                    if not isinstance(points, list) or len(points) != REPEATS:
                        raise ValueError("Missing signed repeats")
                    for point in points:
                        if validate_samples and not recorded_invalid:
                            validate_point(point, bases[0])
                        if isinstance(point, dict) and point.get("evaluation_error"):
                            continue  # Explicit failed call remains an invalid metric during classification.
                        probe = point.get("parameter_probe", {})
                        expected_delta = h if sign == "plus" else -h
                        if (probe.get("axis") != axis_index or probe.get("requested_delta") != expected_delta
                                or not finite(probe.get("actual_float32_delta"))):
                            raise ValueError("Signed coordinate metadata differs from fixed axis order")
    return {"state_count": len(keys), "coordinate_count": 6, "axis_count": len(keys)*6,
        "progress_constraint_count": len(progress_ids), "risk_constraint_count": len(risk_ids),
        "progress_metric_count": len(progress_ids)*6, "risk_metric_count": len(risk_ids)*6,
        "progress_ids": progress_ids, "risk_ids": risk_ids}


def classify_axis(baseline, samples, metric_ids, *, baseline_issues=()):
    """Two separate axes: amplitude class and nonexclusive instability flags."""
    reproduced = reproduce_axis(baseline, samples)
    original_check = reproduced
    invalid_context = "HISTORICAL_BASELINE_MISMATCH:tracks" in baseline_issues
    if reproduced["status"] == "INVALID_MEASUREMENT" or invalid_context:
        return {"reproduced": original_check, "stage_switches": [], "metrics": [{"metric_id": name,
            "response_class": "INVALID_MEASUREMENT", "amplitude_class": "UNKNOWN",
            "instability_flags": ["INVALID_MEASUREMENT", *baseline_issues],
            "errors": reproduced["errors"] + (["Historical track identity/context mismatch"] if invalid_context else []),
            "gradient_exported": False} for name in metric_ids]}
    if len(metric_ids) != len(metrics(baseline)):
        raise ValueError("Observable identity/measurement dimension mismatch")
    base_signature = stage_signature(baseline)
    switches = []
    for scale, sample in enumerate(samples):
        for sign in ("minus", "plus"):
            for repeat, point in enumerate(sample[sign]):
                sig = stage_signature(point)
                changed = [k for k in base_signature if base_signature[k] != sig[k]]
                if changed:
                    switches.append({"scale_index": scale, "h": SCALES[scale], "sign": sign,
                        "repeat_index": repeat, "changed_signature_fields": changed})
    details = []
    for j, (name, base) in enumerate(zip(metric_ids, metrics(baseline))):
        values = {sign: [[metrics(sample[sign][repeat])[j] for repeat in range(REPEATS)] for sample in samples]
                  for sign in ("minus", "plus")}
        amplitude = max(abs(v-base) for signs in values.values() for repeats in signs for v in repeats)
        amplitude_class = ("EXACT_OBSERVED_FLAT" if amplitude == 0 else
                           "SUB_RESOLUTION_NONZERO" if amplitude <= ABS_RESPONSE_TOL else "RESOLVED_NONZERO")
        flags = list(baseline_issues)
        for flag in ("REPEATABILITY_MISMATCH", "STAGE_SWITCH"):
            if flag in reproduced["flags"]:
                flags.append(flag)
        central = [row[j] for row in reproduced["central_estimates_by_scale"]]
        forward = [row[j] for row in reproduced["forward_estimates_by_scale"]]
        backward = [row[j] for row in reproduced["backward_estimates_by_scale"]]
        one_sided = [not close_response(h*a, h*b) for h, a, b in zip(SCALES, forward, backward)]
        scale_failure = not close_response(SCALES[1]*central[0], SCALES[1]*central[1])
        residuals = reproduced["heldout_prediction_residuals"][j]
        heldout_failure = [not close_response(values[sign][2][0]-base, multiplier*SCALES[2]*central[1])
            for sign, multiplier in (("minus", -1), ("plus", 1))]
        if any(one_sided): flags.append(NUMERIC_FLAGS[0])
        if scale_failure: flags.append(NUMERIC_FLAGS[1])
        if any(heldout_failure): flags.append(NUMERIC_FLAGS[2])
        response = ("INSTABILITY_OBSERVED" if flags else "OBSERVED_STABLE_FLAT" if amplitude == 0
                    else "RESOLUTION_LIMITED_NONZERO" if amplitude <= ABS_RESPONSE_TOL else "LOCAL_RESPONSE_CONSISTENT")
        details.append({"metric_id": name, "baseline": base, "signed_values_by_scale_and_repeat": values,
            "risk_baseline_zero": name.startswith("risk:") and base == 0.,
            "response_class": response, "amplitude_class": amplitude_class,
            "amplitude": amplitude, "instability_flags": list(dict.fromkeys(flags)),
            "one_sided_failure_by_scale": one_sided, "scale_inconsistency": scale_failure,
            "heldout_failure_by_sign": heldout_failure, "heldout_prediction_residuals": residuals,
            "central_estimates_by_scale": central, "forward_estimates_by_scale": forward,
            "backward_estimates_by_scale": backward,
            "original_resolution_veto": j in reproduced["unresolved_metric_indices"],
            "zero_derivative_proven": False, "gradient_exported": False})
    return {"reproduced": original_check, "stage_switches": switches, "metrics": details}


def summarize(items):
    return {"count": len(items),
        "response_class_counts": dict(Counter(i["response_class"] for i in items)),
        "amplitude_class_counts": dict(Counter(i["amplitude_class"] for i in items)),
        "instability_flag_counts_nonexclusive": dict(Counter(f for i in items for f in i["instability_flags"])),
        "original_resolution_veto_count": sum(i.get("original_resolution_veto", False) for i in items),
        "observed_stable_flat_zero_risk_baseline_count": sum(i["response_class"] == "OBSERVED_STABLE_FLAT" and i.get("risk_baseline_zero", False) for i in items),
        "observed_stable_flat_positive_risk_baseline_count": sum(i["response_class"] == "OBSERVED_STABLE_FLAT" and i["metric_id"].startswith("risk:") and i.get("baseline", 0) > 0 for i in items)}


def audit_quality(report, registry, historical_baselines, common_constraints):
    coverage = validate_structure(report, registry, historical_baselines, validate_samples=False)
    common = set(common_constraints)
    if not common.issubset(set(coverage["progress_ids"]+coverage["risk_ids"])):
        raise ValueError("Common-veto constraint absent from measurement cohort")
    entries, axis_details, state_summaries, evidence_errors = [], [], [], []
    original_flags = Counter()
    explained_flags = Counter()
    for state_index, state in enumerate(report["states"]):
        key = state["key"]
        base = state["baseline_repeats"][0]
        baseline_issues = []
        if state["baseline_repeats"][0] != state["baseline_repeats"][1]:
            baseline_issues.append("BASELINE_REPEATABILITY_MISMATCH")
        historical = historical_baselines[key]
        for field in ("tracks", "requested_actions", "executed_actions", "executed_progress", "risk_proxy"):
            if base.get(field) != historical[field]:
                baseline_issues.append("HISTORICAL_BASELINE_MISMATCH:"+field)
        if state.get("errors"):
            baseline_issues.append("RECORDED_STATE_ERRORS")
        ids = ["progress:"+key]+[f"risk:{key}:{t['observer_id']}:{t['track_id']}" for t in historical["tracks"]]
        start = len(entries)
        for axis, coordinate in enumerate(parameterization()["coordinates"]):
            recorded = state["axes"][axis]
            classified = classify_axis(base, recorded["samples"], ids, baseline_issues=baseline_issues)
            check = classified["reproduced"]
            mismatch = [field for field, value in check.items() if recorded.get(field) != value]
            if mismatch:
                evidence_errors.append(f"{key}:axis{axis}:recorded_diagnostic_reproduction_mismatch:"+",".join(mismatch))
            original_flags.update(recorded.get("flags", []))
            explained_flags.update(check["flags"])
            axis_details.append({"key": key, "coordinate": coordinate["name"], "axis_index": axis,
                "source_pointer": f"/states/{state_index}/axes/{axis}", "original_status": recorded["status"],
                "original_flags": recorded.get("flags", []), "reproduced_flags": check["flags"],
                "original_veto_reproduced": not mismatch, "stage_switches": classified["stage_switches"],
                "baseline_issues": baseline_issues,
                "observables": summarize(classified["metrics"])})
            for item in classified["metrics"]:
                item.update(key=key, coordinate=coordinate["name"], axis_index=axis,
                    common_veto_constraint=item["metric_id"] in common,
                    source_pointer=f"/states/{state_index}/axes/{axis}/samples")
                entries.append(item)
        state_summaries.append({"key": key, "state_index": state["state_index"],
            "baseline_issues": baseline_issues, "observables": summarize(entries[start:])})
    common_coverage = []
    for constraint in common_constraints:
        rows = [i for i in entries if i["metric_id"] == constraint]
        common_coverage.append({"constraint_id": constraint, "expected_coordinates": 6,
            "observed_coordinates": len(rows), "original_constraint_retained": True, **summarize(rows)})
    invalid = sum(i["response_class"] == "INVALID_MEASUREMENT" for i in entries)
    return {"schema_version": "PPCSASensitivityQualityReviewV1",
        "status": "INVALID_EVIDENCE" if evidence_errors else "QUALITY_AUDIT_HAS_INVALID_MEASUREMENTS" if invalid else "QUALITY_AUDIT_COMPLETE",
        "coverage": coverage, "fixed_state_registry": deepcopy(registry), "measurement_status_unchanged": report["status"],
        "evidence_errors": evidence_errors, "all_original_axis_vetoes_reproduced": not evidence_errors,
        "original_axis_flag_counts_nonexclusive": dict(original_flags), "reproduced_axis_flag_counts_nonexclusive": dict(explained_flags),
        "original_axis_flag_patterns": dict(Counter("|".join(sorted(a.get("flags", []))) or "NONE" for s in report["states"] for a in s["axes"])),
        "progress_summary": summarize([i for i in entries if i["metric_id"].startswith("progress:")]),
        "risk_summary": summarize([i for i in entries if i["metric_id"].startswith("risk:")]),
        "per_state": state_summaries,
        "per_coordinate": [{"coordinate": c["name"],
            "progress": summarize([i for i in entries if i["coordinate"] == c["name"] and i["metric_id"].startswith("progress:")]),
            "risk": summarize([i for i in entries if i["coordinate"] == c["name"] and i["metric_id"].startswith("risk:")])}
            for c in parameterization()["coordinates"]],
        "per_risk_constraint": [{"constraint_id": name, **summarize([i for i in entries if i["metric_id"] == name])} for name in coverage["risk_ids"]],
        "common_veto_coverage": common_coverage, "axis_details": axis_details, "observable_details": entries,
        "limitations": ["Observed stable flatness at finite probes does not prove zero derivative or identify a risk plateau cause.",
            "Instability flags overlap and are descriptive; branch changes do not establish causal responsibility.",
            "Original protocol scales, numeric floor, admission and blocked sensitivity export remain unchanged.",
            "All original risk/progress constraints are retained; no unknown or flat entry is dropped or zero-imputed.",
            "Repeated states/coordinates/tracks are dependent exploratory evidence, not independent efficacy validation."],
        "model_execution": False, "new_shadow_calls": 0, "gradient_exported": False,
        "direction_generated": False, "actual_update_accepted": False, "independent_validation": False,
        "actor_training_authorized": False}
