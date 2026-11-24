"""Frozen six-coordinate finite-difference protocol; pure measurement analysis.

No optimizer or actual update admission. Callback execution is user-run only.
"""
from copy import deepcopy
import hashlib
import json
import math

from src.algorithms.ppcsa_common_veto_diagnostics import STAGES, action_matrix, validate_parameter_sensitivities
from src.algorithms.ppcsa_direction_feasibility import complete_track
from src.algorithms.ppcsa_update_acceptance import finite

COORDINATES = (
    ("up.weight", (0, 0), 0), ("up.weight", (0, 1), 1),
    ("up.weight", (1, 0), 16), ("up.weight", (1, 1), 17),
    ("up.bias", (0,), 32), ("up.bias", (1,), 33))
SCALES = (1e-3, 5e-4, 2.5e-4)
REPEATS = 2
ABS_RESPONSE_TOL = 1e-7
REL_RESPONSE_TOL = .1


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def parameterization():
    return {"effective_up_dimension": 34, "measured_dimension": 6,
        "scope": "six_fixed_coordinate_subspace_not_full_adapter_gradient",
        "full_up_shapes": {"up.weight": [2, 16], "up.bias": [2]},
        "coordinates": [{"parameter": name, "index": list(index), "up_flat_index": flat,
            "name": name+"["+",".join(map(str, index))+"]"} for name, index, flat in COORDINATES],
        "unmeasured_up_flat_indices": [i for i in range(34) if i not in [c[2] for c in COORDINATES]],
        "unmeasured_values": "NOT_MEASURED_NEVER_ZERO_IMPUTED"}


def protocol():
    return {"schema_version": "PPCSAParameterMeasurementProtocolV1", "device": "cpu",
        "adapter_initialization_seed": 20261121, "hidden_dim": 128, "adapter_rank": 16, "adapter_scale": .15,
        "parameterization": parameterization(), "scales": list(SCALES), "repeats": REPEATS,
        "difference": "central; estimate h/2, h and one-sided checks; held-out h/4 prediction",
        "absolute_response_diagnostic_tolerance": ABS_RESPONSE_TOL,
        "relative_response_diagnostic_tolerance": REL_RESPONSE_TOL,
        "repeatability": "exact payload equality", "risk_admission_tolerance": 0., "progress_admission_tolerance": 0.,
        "selection": "fixed source coordinate order; no outcome-dependent selection",
        "unresolved_response": "any response <= absolute diagnostic floor blocks whole export",
        "stage_switch": "any discrete guard signature or TTC intervention-mask change blocks whole export",
        "independent_validation": False, "optimizer_authorized": False}


def max_calls(states):
    return states * (REPEATS + len(COORDINATES)*len(SCALES)*2*REPEATS)


def stage_signature(point):
    checks = point.get("safety_evidence", {}).get("stage_checks", {})
    required = ("initial_guard", "final_guard", "ttc_fallback", "safe_brake_applied", "recovery_overlay")
    if any(k not in checks for k in required):
        raise ValueError("Missing stage branch evidence")
    signature = {k: checks[k] for k in required[2:]}
    for guard in required[:2]:
        obj = checks[guard]
        fields = ("shielded_axes", "pair_infeasible", "recovery_required", "safe_brake", "pair_iterations")
        if not isinstance(obj, dict) or any(k not in obj for k in fields):
            raise ValueError("Missing guard branch signature")
        signature[guard] = {k: deepcopy(obj[k]) for k in fields}
    stages = point["stage_actions"]
    signature["ttc_intervention_mask"] = [[x != y for x, y in zip(a, b)]
        for a, b in zip(stages["initial_joint_guard"], stages["obstacle_ttc"])]
    return signature


def validate_point(point, baseline=None):
    if not isinstance(point, dict) or point.get("supported") is not True or point.get("hard_safe") is not True:
        raise ValueError("Unsupported or uncertified perturbation")
    if (point.get("source_unchanged") is not True or point.get("track_capture_unchanged") is not True
            or not point.get("snapshot_fingerprint") or not finite(point.get("executed_progress"))):
        raise ValueError("Invalid source or progress evidence")
    tracks, risk = point.get("tracks"), point.get("risk_proxy")
    if (not isinstance(tracks, list) or not isinstance(risk, list) or len(tracks) != len(risk)
            or not all(complete_track(t) for t in tracks) or not all(finite(v) and v >= 0 for v in risk)):
        raise ValueError("Unknown/nonfinite risk or incomplete track context")
    identities = [(t["observer_id"], t["track_id"]) for t in tracks]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate track identity")
    actions = point.get("requested_actions")
    if not action_matrix(actions) or not action_matrix(point.get("executed_actions"), len(actions)):
        raise ValueError("Invalid action matrix")
    if any(not action_matrix(point.get("stage_actions", {}).get(s), len(actions)) for s in STAGES):
        raise ValueError("Missing/nonfinite action stage")
    stage_signature(point)
    if baseline is not None and (tracks != baseline["tracks"] or point["snapshot_fingerprint"] != baseline["snapshot_fingerprint"]):
        raise ValueError("Perturbation track/state identity differs")


def metrics(point):
    return [point["executed_progress"], *point["risk_proxy"]]


def close_response(a, b):
    return abs(a-b) <= ABS_RESPONSE_TOL + REL_RESPONSE_TOL*max(abs(a), abs(b))


def analyze_axis(baseline, samples):
    """Retain every signed endpoint/repeat; reject unstable evidence as a whole."""
    flags, errors = [], []
    try:
        validate_point(baseline)
        if len(samples) != len(SCALES):
            raise ValueError("Missing fixed difference scale")
        for h, sample in zip(SCALES, samples):
            if sample.get("h") != h:
                raise ValueError("Reordered or tuned difference scales")
            for sign in ("minus", "plus"):
                points = sample.get(sign)
                if not isinstance(points, list) or len(points) != REPEATS:
                    raise ValueError("Missing sign or repeat")
                for point in points:
                    validate_point(point, baseline)
                if points[0] != points[1]:
                    flags.append("REPEATABILITY_MISMATCH")
                if any(stage_signature(p) != stage_signature(baseline) for p in points):
                    flags.append("STAGE_SWITCH")
    except (ValueError, TypeError, KeyError) as error:
        errors.append(str(error))
    if errors:
        return {"status": "INVALID_MEASUREMENT", "errors": errors, "flags": list(dict.fromkeys(flags)),
            "usable": False, "gradient": None, "samples": deepcopy(samples)}
    base = metrics(baseline)
    central, forward, backward = [], [], []
    for h, sample in zip(SCALES, samples):
        minus, plus = metrics(sample["minus"][0]), metrics(sample["plus"][0])
        central.append([(p-m)/(2*h) for m, p in zip(minus, plus)])
        forward.append([(p-b)/h for p, b in zip(plus, base)])
        backward.append([(b-m)/h for m, b in zip(minus, base)])
    heldout_residuals = []
    unresolved = []
    for j, b in enumerate(base):
        for i, h in enumerate(SCALES):
            if not close_response(h*forward[i][j], h*backward[i][j]):
                flags.append("ONE_SIDED_INCONSISTENCY")
        if not close_response(SCALES[1]*central[0][j], SCALES[1]*central[1][j]):
            flags.append("SCALE_INCONSISTENCY")
        smallest = samples[2]
        residuals = []
        for sign, multiplier in (("minus", -1), ("plus", 1)):
            observed = metrics(smallest[sign][0])[j]-b
            predicted = multiplier*SCALES[2]*central[1][j]
            residuals.append(observed-predicted)
            if not close_response(observed, predicted):
                flags.append("HELDOUT_SCALE_PREDICTION_MISMATCH")
        heldout_residuals.append(residuals)
        amplitude = max(abs(metrics(sample[sign][0])[j]-b) for sample in samples for sign in ("minus", "plus"))
        if amplitude <= ABS_RESPONSE_TOL:
            unresolved.append(j)
    if unresolved:
        flags.append("ZERO_OR_RESOLUTION_LIMITED_RESPONSE")
    flags = list(dict.fromkeys(flags))
    return {"status": "NUMERICAL_FIDELITY_UNRESOLVED" if flags else "LOCAL_FD_ESTIMATE_CONSISTENT",
        "usable": not flags, "errors": [], "flags": flags,
        "central_estimates_by_scale": central, "forward_estimates_by_scale": forward,
        "backward_estimates_by_scale": backward, "heldout_prediction_residuals": heldout_residuals,
        "unresolved_metric_indices": unresolved, "gradient": central[1] if not flags else None,
        "samples": deepcopy(samples), "empirical_derivatives_verified": False}


def measure_cohort(registry, historical_baselines, evaluate, bindings):
    """User runner supplies a frozen evaluation callback; tests use analytic data."""
    keys = [fixed["key"] for fixed in registry]
    if not keys or len(set(keys)) != len(keys) or set(historical_baselines) != set(keys):
        raise ValueError("Empty, duplicate or incomplete fixed measurement cohort")
    states, errors = [], []
    calls = 0
    def call(key, axis, delta):
        nonlocal calls
        calls += 1
        try:
            return evaluate(key, axis, delta)
        except Exception as error:
            return {"evaluation_error": type(error).__name__+": "+str(error)}
    for fixed in registry:
        key = fixed["key"]
        bases = [call(key, None, 0.) for _ in range(REPEATS)]
        base = bases[0]
        state_errors = []
        try:
            validate_point(base)
            if bases[0] != bases[1]:
                state_errors.append("BASELINE_REPEATABILITY_MISMATCH")
            historical = historical_baselines[key]
            # Historical composite RNG fingerprints may differ across runs.
            for field in ("tracks", "requested_actions", "executed_actions", "executed_progress", "risk_proxy"):
                if base[field] != historical[field]:
                    state_errors.append("HISTORICAL_BASELINE_MISMATCH:"+field)
        except (ValueError, KeyError, TypeError) as error:
            state_errors.append(str(error))
        axes = []
        for axis in range(len(COORDINATES)):
            samples = [{"h": h, **{sign: [call(key, axis, multiplier*h) for _ in range(REPEATS)]
                for sign, multiplier in (("minus", -1), ("plus", 1))}} for h in SCALES]
            axes.append(analyze_axis(base, samples))
        states.append({"key": key, "state_index": fixed["state_index"], "baseline_repeats": bases,
            "errors": state_errors, "axes": axes})
        errors.extend(key+":"+e for e in state_errors)
    complete = not errors and all(axis["usable"] for state in states for axis in state["axes"])
    payload = None
    progress_ids = ["progress:"+fixed["key"] for fixed in registry]
    risk_ids = [f"risk:{fixed['key']}:{t['observer_id']}:{t['track_id']}" for fixed in registry for t in historical_baselines[fixed["key"]]["tracks"]]
    if complete:
        progress, risk = [], []
        for state in states:
            progress.append({"id": "progress:"+state["key"], "gradient": [a["gradient"][0] for a in state["axes"]]})
            for i, t in enumerate(historical_baselines[state["key"]]["tracks"]):
                risk.append({"id": f"risk:{state['key']}:{t['observer_id']}:{t['track_id']}",
                    "gradient": [a["gradient"][i+1] for a in state["axes"]]})
        payload = {"schema_version": "PPCSAParameterSensitivitiesV1", "quantity": "executed_parameter_derivatives",
            "method": "parameter_finite_difference", "endpoint_secants_used": False, "bindings": bindings,
            "measurement_protocol_sha256": digest(protocol()), "parameterization_sha256": digest(parameterization()),
            "parameter_order": [{"name": c["name"], "shape": [1]} for c in parameterization()["coordinates"]],
            "parameter_count": 6, "parameterization": parameterization(), "progress": progress, "risk": risk,
            "fidelity_scope": "fixed signed scales and repeated numerical consistency, not true derivative certification"}
    checked = validate_parameter_sensitivities(payload, progress_ids=progress_ids, risk_ids=risk_ids, bindings=bindings)
    if payload is not None and checked["status"] != "DECLARED_PARAMETER_SENSITIVITIES_VALIDATED":
        errors.append("SENSITIVITY_DATA_CONTRACT_INVALID")
        payload = None
    conflicts = []
    for axis in range(len(COORDINATES)):
        counts = {}
        for metric_kind in ("progress", "risk"):
            values = []
            missing = 0
            for state in states:
                estimates = state["axes"][axis].get("central_estimates_by_scale")
                if estimates is None:
                    missing += 1; continue
                values.extend(estimates[1][:1] if metric_kind == "progress" else estimates[1][1:])
            counts[metric_kind] = {"positive": sum(v > 0 for v in values), "negative": sum(v < 0 for v in values),
                "zero": sum(v == 0 for v in values), "invalid_states": missing,
                "opposed_signs_observed": any(v > 0 for v in values) and any(v < 0 for v in values)}
        conflicts.append({"coordinate": parameterization()["coordinates"][axis]["name"], **counts})
    return {"status": "MEASUREMENT_INVALID" if errors or any(a["status"] == "INVALID_MEASUREMENT" for s in states for a in s["axes"])
                else "MEASUREMENT_FIDELITY_UNRESOLVED" if not complete else "SUBSPACE_ESTIMATES_REQUIRE_REVIEW",
        "states": states, "errors": errors, "fixed_state_registry": deepcopy(registry),
        "progress_ids": progress_ids, "risk_ids": risk_ids, "shadow_calls": calls,
        "maximum_shadow_calls": max_calls(len(registry)), "protocol": protocol(),
        "measurement_protocol_sha256": digest(protocol()), "parameterization_sha256": digest(parameterization()),
        "parameter_sensitivities": payload, "sensitivity_contract": checked,
        "response_sign_conflicts": conflicts,
        "response_sign_scope": "Descriptive dependent state/track counts, including unresolved estimates; not shared-update feasibility.",
        "empirical_derivatives_verified": False, "independent_validation": False,
        "direction_generated": False, "actual_update_accepted": False, "actor_training_authorized": False}
