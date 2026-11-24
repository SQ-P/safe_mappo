"""Pure whole-update admission. Decisions describe virtual updates only."""
import math
from collections import Counter

CANDIDATES = ("ordinary","paired","kinematic")
PARAMETER_STEP = .01


def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def state_admission(evidence):
    reasons = []
    if evidence.get("replay_match") is not True:
        reasons.append("replay_or_fingerprint_mismatch")
    if evidence.get("evaluation_error"):
        reasons.append("evaluation_error")
    base,candidate = evidence.get("baseline"),evidence.get("candidate")
    if not isinstance(base,dict) or not isinstance(candidate,dict):
        return {"key":evidence.get("key"),"passed":False,"gain":None,
            "reasons":reasons+["missing_shadow_evidence"]}
    for name,value in (("baseline",base),("candidate",candidate)):
        if value.get("supported") is not True:
            reasons.append(name+"_not_NORMAL_supported")
        if value.get("hard_safe") is not True:
            reasons.append(name+"_not_hard_safe")
        if value.get("source_unchanged") is not True or value.get("track_capture_unchanged") is not True:
            reasons.append(name+"_snapshot_or_track_mutation")
    if (not base.get("snapshot_fingerprint") or base.get("snapshot_fingerprint") != candidate.get("snapshot_fingerprint")):
        reasons.append("shadow_snapshot_mismatch")
    tracks_a,tracks_b = base.get("tracks"),candidate.get("tracks")
    risk_a,risk_b = base.get("risk_proxy"),candidate.get("risk_proxy")
    mapping = all(isinstance(x,list) for x in (tracks_a,tracks_b,risk_a,risk_b))
    if mapping:
        mapping = len(tracks_a) == len(tracks_b) == len(risk_a) == len(risk_b)
    if mapping:
        try:
            keys_a = [(t["observer_id"],t["track_id"]) for t in tracks_a]
            keys_b = [(t["observer_id"],t["track_id"]) for t in tracks_b]
            mapping = (len(set(keys_a)) == len(keys_a) and keys_a == keys_b and tracks_a == tracks_b)
        except (KeyError,TypeError):
            mapping = False
    if not mapping:
        reasons.append("risk_track_identity_order_or_context_mismatch")
    else:
        known = all(finite(v) and v>=0 for v in risk_a+risk_b)
        known = known and all(t.get("velocity_valid") is True for t in tracks_a+tracks_b)
        if not known:
            reasons.append("unknown_or_invalid_track_risk")
        elif any(c>b for b,c in zip(risk_a,risk_b)):
            reasons.append("per_track_risk_regression")
    a,b = base.get("executed_progress"),candidate.get("executed_progress")
    gain = float(b-a) if finite(a) and finite(b) else None
    if gain is None or not math.isfinite(gain):
        gain = None
        reasons.append("unknown_or_nonfinite_executed_progress")
    elif gain<0:
        reasons.append("executed_progress_regression")
    return {"key":evidence.get("key"),"passed":not reasons,"gain":gain,"reasons":reasons}


def candidate_admission(expected_keys,evidence,parameter_update):
    """Any missing, unsafe, risk-worse or progress-worse state vetoes all rows.

    Predicted values never enter this predicate. No averaging can undo a veto.
    Tolerances for risk/progress are exactly zero in this V1 protocol.
    """
    reasons = []
    expected = list(expected_keys)
    keys = [e.get("key") for e in evidence]
    if not expected or len(set(expected)) != len(expected):
        reasons.append("invalid_fixed_state_registry")
    if len(keys) != len(set(keys)) or set(keys) != set(expected):
        reasons.append("missing_duplicate_or_unregistered_state")
    states = [state_admission(e) for e in evidence]
    if any(not s["passed"] for s in states):
        reasons.append("state_veto")
    gains = [s["gain"] for s in states if s["gain"] is not None]
    net = math.fsum(gains) if len(gains) == len(states) else None
    if not states or net is None or net<=0 or not any(g>0 for g in gains):
        reasons.append("no_strictly_positive_executed_improvement")
    update_valid = bool(parameter_update) and all(finite(v) for v in parameter_update)
    if not update_valid or not any(v != 0 for v in parameter_update):
        reasons.append("zero_or_invalid_parameter_update")
    accepted = not reasons
    effective = list(parameter_update) if accepted else [0.]*len(parameter_update)
    return {"would_accept":accepted,"decision":"WOULD_ACCEPT_VIRTUAL" if accepted else "REJECT_WHOLE_UPDATE",
        "expected_states":len(expected),"evaluated_states":len(states),"states":states,
        "state_rejection_counts_nonexclusive":dict(Counter(r for s in states for r in s["reasons"])),
        "candidate_rejection_reasons":reasons,"executed_progress_net_all_states":net,
        "strictly_improving_states":sum(g>0 for g in gains),"effective_virtual_update":effective,
        "actor_update_applied":False}


def candidate_set_admission(decisions,parameter_count):
    if set(decisions) != set(CANDIDATES):
        raise ValueError("All three fixed candidates must be retained")
    accepted = [name for name in CANDIDATES if decisions[name]["would_accept"]]
    return {"would_accept_candidates":accepted,"selected_candidate":None,
        "zero_update_returned":not accepted,"fallback_update":[0.]*parameter_count if not accepted else None,
        "status":"NO_ACCEPTABLE_SHARED_UPDATE" if not accepted else "VIRTUAL_CANDIDATES_REQUIRE_REVIEW",
        "automatic_model_selection":False,"all_virtual_copies_discarded":True,"actor_update_applied":False}
