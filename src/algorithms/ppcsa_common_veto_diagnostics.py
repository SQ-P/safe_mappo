"""Read-only common-veto evidence review and future sensitivity data contract.

No optimizer, parameter reconstruction, model execution or Shadow evaluation.
"""
from copy import deepcopy
from fractions import Fraction
import math
import re

from src.algorithms.ppcsa_direction_feasibility import analyze_admission, rational
from src.algorithms.ppcsa_update_acceptance import CANDIDATES, PARAMETER_STEP, finite

STAGES = ("actor_raw", "recovery_overlay", "goal_corridor", "initial_joint_guard",
          "obstacle_ttc", "final_joint_guard", "total", "executed")


def action_matrix(value, agents=None):
    return (isinstance(value, list) and bool(value)
            and (agents is None or len(value) == agents)
            and all(isinstance(row, list) and len(row) == 2
                    and all(finite(x) for x in row) for row in value))


def difference(a, b):
    delta = [[y-x for x, y in zip(aa, bb)] for aa, bb in zip(a, b)]
    return {"delta": delta, "l2": math.sqrt(math.fsum(x*x for row in delta for x in row))}


def validate_evidence(admission, feasibility):
    """Validate the complete old cohort; never solve a new LP or select a subset."""
    checked = analyze_admission(admission, solve=False)
    if checked["evidence_errors"]:
        raise ValueError("Invalid admission evidence: " + ";".join(checked["evidence_errors"]))
    if (feasibility.get("schema_version") != "PPCSADirectionSecantFeasibilityV1"
            or feasibility.get("status") != "PROXY_INFEASIBLE"
            or feasibility.get("proxy", {}).get("status") != "PROXY_INFEASIBLE"
            or feasibility.get("evidence_errors") != []):
        raise ValueError("Expected complete PROXY_INFEASIBLE evidence")
    if (feasibility.get("fixed_state_registry") != admission["fixed_state_registry"]
            or feasibility.get("fixed_state_registry_sha256") != admission["fixed_state_registry_sha256"]):
        raise ValueError("Fixed registry differs between reports")
    for field in ("fixed_state_count", "complete_progress_constraint_count", "complete_risk_constraint_count"):
        if feasibility.get(field) != checked[field]:
            raise ValueError("Constraint coverage mismatch: " + field)
    rows = {name: admission["candidates"][name]["raw_state_evidence"] for name in CANDIDATES}
    progress, risk = [], []
    for index, fixed in enumerate(admission["fixed_state_registry"]):
        aligned = [rows[name][index] for name in CANDIDATES]
        for row in aligned:
            for endpoint in (row["baseline"], row["candidate"]):
                actions = endpoint.get("requested_actions")
                if not action_matrix(actions):
                    raise ValueError("Missing/nonfinite requested actions: " + fixed["key"])
                agents = len(actions)
                if (not action_matrix(endpoint.get("executed_actions"), agents)
                        or any(not action_matrix(endpoint.get("stage_actions", {}).get(s), agents) for s in STAGES)):
                    raise ValueError("Missing/nonfinite action-stage evidence: " + fixed["key"])
                # Registry state_fingerprint hashes the environment state;
                # Shadow snapshot_fingerprint hashes state + cache + RNG/GRU.
                # Compare each within its domain: registry across bound reports,
                # Shadow across endpoints/directions via analyze_admission above.
        progress.append({"id": "progress:" + fixed["key"], "normal_ge_zero_exact": [
            str(rational(row["candidate"]["executed_progress"] - row["baseline"]["executed_progress"])) for row in aligned]})
        for track_index, track in enumerate(aligned[0]["baseline"]["tracks"]):
            risk.append({"id": f"risk:{fixed['key']}:{track['observer_id']}:{track['track_id']}",
                "normal_ge_zero_exact": [str(-rational(row["candidate"]["risk_proxy"][track_index]
                    - row["baseline"]["risk_proxy"][track_index])) for row in aligned]})
    constraints = progress + risk
    if feasibility["proxy"].get("constraints") != constraints:
        raise ValueError("Exact constraints/order differ from raw admission evidence")
    common = [c["id"] for c in constraints if max(map(Fraction, c["normal_ge_zero_exact"])) < 0]
    if not common or feasibility["proxy"].get("universal_veto_constraints") != common:
        raise ValueError("Common veto list does not reproduce exactly")
    return {"fixed_state_count": checked["fixed_state_count"],
        "progress_constraint_count": len(progress), "risk_constraint_count": len(risk),
        "constraints": constraints, "common_veto_constraints": common}


def endpoint_actions(base, candidate):
    endpoints = {}
    for name, point in (("baseline", base), ("candidate", candidate)):
        stages = point["stage_actions"]
        endpoints[name] = {"requested_actions": deepcopy(point["requested_actions"]),
            "executed_actions": deepcopy(point["executed_actions"]),
            "stage_actions": deepcopy(stages),
            "within_pipeline_changes": {after: difference(stages[before], stages[after])
                for before, after in zip(STAGES, STAGES[1:])},
            "requested_to_executed": difference(point["requested_actions"], point["executed_actions"])}
    requested = difference(base["requested_actions"], candidate["requested_actions"])
    executed = difference(base["executed_actions"], candidate["executed_actions"])
    return {**endpoints, "between_endpoints": {"requested": requested, "executed": executed,
        "stage_changes": {s: difference(base["stage_actions"][s], candidate["stage_actions"][s]) for s in STAGES},
        "execution_to_request_change_ratio": executed["l2"]/requested["l2"] if requested["l2"] else None},
        "interpretation": "Recorded association only; stage intervention is not a causal explanation of risk/progress loss."}


def validate_parameter_sensitivities(payload, *, progress_ids, risk_ids, bindings):
    """Validate declarations/shape/provenance only, never certify measured derivatives.

    Full cohort rows share one explicitly ordered parameterization. Endpoint
    direction responses are not parameter sensitivities, even if dimensions fit.
    """
    base = {"empirical_derivatives_verified": False, "direction_generated": False,
            "actual_safety_certified": False}
    if payload is None:
        return {**base, "status": "PARAMETER_SENSITIVITIES_UNAVAILABLE",
                "errors": [], "reason": "Historical endpoints do not retain measured parameter derivatives."}
    errors = []
    if not isinstance(payload, dict):
        return {**base, "status": "INVALID_PARAMETER_SENSITIVITIES", "errors": ["payload_not_object"]}
    if (payload.get("schema_version") != "PPCSAParameterSensitivitiesV1"
            or payload.get("quantity") != "executed_parameter_derivatives"
            or payload.get("method") not in ("autograd_of_executed_map", "parameter_finite_difference")
            or payload.get("endpoint_secants_used") is not False):
        errors.append("not_declared_executed_parameter_derivatives")
    binding_names = {"admission_sha256", "registry_sha256", "checkpoint_sha256"}
    if (set(bindings) != binding_names
            or any(not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v) for v in bindings.values())
            or payload.get("bindings") != bindings):
        errors.append("source_bindings_mismatch")
    for field in ("measurement_protocol_sha256", "parameterization_sha256"):
        if not isinstance(payload.get(field), str) or not re.fullmatch(r"[0-9a-f]{64}", payload[field]):
            errors.append("invalid_" + field)
    parameters = payload.get("parameter_order")
    count = 0
    names = []
    if not isinstance(parameters, list) or not parameters:
        errors.append("missing_parameter_order")
    else:
        for param in parameters:
            if not isinstance(param, dict):
                errors.append("invalid_parameter_entry"); continue
            shape = param.get("shape")
            if (not isinstance(param.get("name"), str) or not param["name"]
                    or not isinstance(shape, list) or not shape
                    or any(not isinstance(s, int) or isinstance(s, bool) or s <= 0 for s in shape)):
                errors.append("invalid_parameter_shape_or_name"); continue
            names.append(param["name"]); count += math.prod(shape)
        if len(set(names)) != len(names):
            errors.append("duplicate_parameter_name")
    if (not isinstance(payload.get("parameter_count"), int) or isinstance(payload.get("parameter_count"), bool)
            or payload.get("parameter_count") != count or count <= 0):
        errors.append("parameter_count_mismatch")
    for kind, ids in (("progress", progress_ids), ("risk", risk_ids)):
        rows = payload.get(kind)
        if not isinstance(rows, list) or [r.get("id") if isinstance(r, dict) else None for r in rows] != ids:
            errors.append(kind + "_coverage_or_order_mismatch"); continue
        for row in rows:
            vector = row.get("gradient")
            if (not isinstance(vector, list) or len(vector) != count
                    or not all(finite(x) for x in vector)):
                errors.append(kind + "_nonfinite_or_wrong_parameter_dimension")
    return {**base, "status": "INVALID_PARAMETER_SENSITIVITIES" if errors else "DECLARED_PARAMETER_SENSITIVITIES_VALIDATED",
        "errors": errors, "parameter_count": count,
        "interpretation": "Data contract validation only. Provenance declarations do not prove derivative fidelity or authorize optimization."}


def check_declared_linear_direction(payload, update, *, progress_ids, risk_ids, bindings):
    """Pure synthetic witness check, not a generator or real update admission.

    No actual parameter installation or categorical hard-safety checks can occur.
    """
    checked = validate_parameter_sensitivities(payload, progress_ids=progress_ids, risk_ids=risk_ids, bindings=bindings)
    if checked["status"] != "DECLARED_PARAMETER_SENSITIVITIES_VALIDATED":
        raise ValueError("Invalid sensitivity contract")
    if (not isinstance(update, list) or len(update) != payload["parameter_count"]
            or not all(finite(x) for x in update)):
        raise ValueError("Invalid parameter-space witness")
    u = list(map(rational, update))
    dot = lambda v: sum((rational(a)*b for a, b in zip(v, u)), Fraction(0))
    g = [dot(row["gradient"]) for row in payload["progress"]]
    r = [dot(row["gradient"]) for row in payload["risk"]]
    norm_squared = sum((v*v for v in u), Fraction(0))
    violated = (any(v < 0 for v in g) or any(v > 0 for v in r)
                or norm_squared > rational(PARAMETER_STEP)**2)
    status = ("ZERO_DIRECTION" if not any(u) else "LINEAR_CONSTRAINT_VIOLATION" if violated
              else "LINEAR_POSITIVE_UNVERIFIED" if sum(g) > 0 else "LINEAR_NONPOSITIVE")
    return {"status": status, "progress_exact": list(map(str, g)), "risk_exact": list(map(str, r)),
        "norm_squared_exact": str(norm_squared), "actual_safety_certified": False,
        "actual_update_accepted": False, "direction_generated": False}


def review_common_veto(admission, feasibility):
    validation = validate_evidence(admission, feasibility)
    registry = admission["fixed_state_registry"]
    indices = {f["key"]: i for i, f in enumerate(registry)}
    constraints = {c["id"]: c for c in validation["constraints"]}
    details = []
    for constraint_id in validation["common_veto_constraints"]:
        parts = constraint_id.split(":")
        kind, key = parts[0], ":".join(parts[1:3])
        index = indices[key]
        identity = tuple(map(int, parts[3:])) if kind == "risk" else None
        entry = {"constraint_id": constraint_id, "kind": kind, "key": key,
            "layout": int(parts[1]), "step": int(parts[2]), "state_index": registry[index]["state_index"],
            "registry_state_fingerprint": registry[index].get("state_fingerprint"),
            "observer_id": identity[0] if identity else None, "track_id": identity[1] if identity else None,
            "normal_ge_zero_exact": constraints[constraint_id]["normal_ge_zero_exact"],
            "all_three_strictly_violate": True, "directions": {},
            "parameter_response_status": "PARAMETER_SENSITIVITIES_UNAVAILABLE",
            "causal_mechanism_status": "UNIDENTIFIED"}
        for name in CANDIDATES:
            row = admission["candidates"][name]["raw_state_evidence"][index]
            b, c = row["baseline"], row["candidate"]
            track_index = (next(i for i, t in enumerate(b["tracks"]) if (t["observer_id"], t["track_id"]) == identity)
                           if identity else None)
            value = {"baseline_progress": b["executed_progress"], "candidate_progress": c["executed_progress"],
                "progress_delta": c["executed_progress"]-b["executed_progress"],
                "baseline_shadow_fingerprint": b["snapshot_fingerprint"],
                "candidate_shadow_fingerprint": c["snapshot_fingerprint"],
                "actions": endpoint_actions(b, c),
                "baseline_hard_safe": b["hard_safe"], "candidate_hard_safe": c["hard_safe"],
                "baseline_safety_evidence": deepcopy(b.get("safety_evidence")),
                "candidate_safety_evidence": deepcopy(c.get("safety_evidence"))}
            if identity:
                value.update(track_index=track_index, track_context=deepcopy(b["tracks"][track_index]),
                    baseline_risk=b["risk_proxy"][track_index], candidate_risk=c["risk_proxy"][track_index],
                    risk_delta=c["risk_proxy"][track_index]-b["risk_proxy"][track_index])
            entry["directions"][name] = value
        details.append(entry)
    return {**validation, "status": "COMMON_VETO_REVIEW_COMPLETE", "direction_order": list(CANDIDATES),
        "fixed_state_registry": deepcopy(registry), "common_veto_count": len(details), "common_veto_details": details,
        "parameter_sensitivities": validate_parameter_sensitivities(None, progress_ids=[], risk_ids=[], bindings={}),
        "future_direction_contract": {"implemented": "input_validation_only_no_optimizer",
            "objective": "maximize sum_i w_i grad_parameter(G_i) dot update with prespecified weights",
            "constraints": ["every state progress derivative dot update >= 0", "every track risk derivative dot update <= 0",
                "parameter norm <= 0.01", "strictly positive total predicted progress", "nonzero actual parameter vector"],
            "requires": ["full fixed state/track coverage", "one ordered shared parameterization",
                "frozen source bindings and measurement protocol", "separate review before derivative measurement or optimization"],
            "real_derivative_fidelity": "UNVERIFIED", "generation_authorized": False},
        "limitations": ["Stage action differences are descriptive associations, not causal attribution.",
            "Recorded float deltas and storage precision are not physical error bounds; original zero tolerances remain unchanged.",
            "These common vetoes concern the fixed endpoint convex family, not the full Adapter parameter space.",
            "No true parameter response is measured by endpoint secants; derivatives remain unavailable.",
            "All exposed Fit/Calibration rows are dependent development evidence, not independent validation."],
        "model_execution": False, "new_shadow_calls": 0, "parameter_reconstruction": False,
        "optimizer_executed": False, "actor_training_authorized": False, "actual_update_accepted": False}
