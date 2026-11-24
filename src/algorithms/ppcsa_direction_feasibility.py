"""Evidence-only endpoint-secant attribution and exact three-weight linear program.

No model, parameter reconstruction, admission modification or Shadow execution.
"""
from collections import Counter,defaultdict
from fractions import Fraction
import math
import struct

from src.algorithms.ppcsa_update_acceptance import CANDIDATES,PARAMETER_STEP,candidate_admission,state_admission,finite


def rational(value):
    if isinstance(value,Fraction):
        return value
    if not isinstance(value,(float,int)) or isinstance(value,bool) or not math.isfinite(value):
        raise ValueError("Nonfinite or nonnumeric linear coefficient")
    return Fraction(str(value))


def dot(a,b):
    return sum((x*y for x,y in zip(a,b)),Fraction(0))


def clip_polygon(polygon,normal):
    """Intersect the simplex polygon with normal @ weights >= 0 exactly."""
    if not polygon:
        return []
    result = []
    previous = polygon[-1]
    before = dot(normal,previous)
    for current in polygon:
        after = dot(normal,current)
        if (before>=0) != (after>=0):
            t = before/(before-after)
            result.append(tuple(x+t*(y-x) for x,y in zip(previous,current)))
        if after>=0:
            result.append(current)
        previous,before = current,after
    # Degenerate feasible points/segments must survive without duplicate edges.
    return list(dict.fromkeys(result))


def solve_simplex(progress,risk,*,progress_ids=None,risk_ids=None):
    """Maximize full-cohort secant net progress over the nonnegative simplex.

    G @ alpha >= 0, R @ alpha <= 0, alpha >= 0, sum(alpha)=1.
    Fractions represent the decimal round-trip of recorded float deltas exactly;
    no numerical feasibility tolerance can relax a zero admission boundary.
    """
    contract = {"solver":"exact_fraction_simplex_polygon_linear_program",
        "numeric_status":"EXACT_RECORDED_DECIMAL_COEFFICIENTS","constraint_tolerance":0.,
        "parameter_norm_analytic_upper_bound":PARAMETER_STEP,"parameter_norm_lower_bound":0.,
        "actual_nonzero_parameter_status":"UNKNOWN","actual_direction_vectors_available":False,
        "real_shadow_validation":False,"actual_safety_certified":False}
    try:
        g = [tuple(map(rational,row)) for row in progress]
        r = [tuple(map(rational,row)) for row in risk]
        if not g or any(len(row)!=3 for row in g+r):
            raise ValueError("Linear rows must contain all three fixed directions")
    except (TypeError,ValueError) as error:
        return {**contract,"status":"INVALID_EVIDENCE","error":str(error)}
    progress_ids = progress_ids if progress_ids is not None else [f"progress:{i}" for i in range(len(g))]
    risk_ids = risk_ids if risk_ids is not None else [f"risk:{i}" for i in range(len(r))]
    if len(progress_ids)!=len(g) or len(risk_ids)!=len(r):
        return {**contract,"status":"INVALID_EVIDENCE","error":"Constraint identity coverage differs"}
    constraints = [(name,row) for name,row in zip(progress_ids,g)]+[(name,tuple(-v for v in row)) for name,row in zip(risk_ids,r)]
    serialized = [{"id":name,"normal_ge_zero_exact":[str(x) for x in row]} for name,row in constraints]
    universal = [name for name,row in constraints if max(row)<0]
    polygon = [(Fraction(1),Fraction(0),Fraction(0)),(Fraction(0),Fraction(1),Fraction(0)),(Fraction(0),Fraction(0),Fraction(1))]
    for index,(name,normal) in enumerate(constraints):
        before = polygon
        polygon = clip_polygon(polygon,normal)
        if not polygon:
            return {**contract,"status":"PROXY_INFEASIBLE","weights":None,"objective_net_progress":None,
                "max_constraint_violation":None,"constraints":serialized,"universal_veto_constraints":universal,
                "processed_constraints":index+1,"first_empty_intersection":name,
                "last_feasible_vertices_exact":[[str(x) for x in p] for p in before],
                "interpretation":"No simplex combination satisfies this complete endpoint-secant proxy; not a proof about the full Adapter space."}
    objective = tuple(sum((row[i] for row in g),Fraction(0)) for i in range(3))
    best = max(polygon,key=lambda p:(dot(objective,p),p))
    optimum = dot(objective,best)
    violation = max([Fraction(0)]+[-dot(row,best) for _,row in constraints])
    if violation>0 or sum(best)!=1 or min(best)<0:
        raise RuntimeError("Exact solver produced an invalid simplex witness")
    return {**contract,"status":"PROXY_POSITIVE_DIRECTION_UNVERIFIED" if optimum>0 else "PROXY_NONPOSITIVE",
        "weights":{name:float(v) for name,v in zip(CANDIDATES,best)},
        "weights_exact":{name:str(v) for name,v in zip(CANDIDATES,best)},
        "objective_net_progress":float(optimum),"objective_net_progress_exact":str(optimum),
        "max_constraint_violation":float(violation),"max_constraint_violation_exact":str(violation),
        "constraints":serialized,"universal_veto_constraints":universal,
        "feasible_vertices_exact":[[str(x) for x in p] for p in polygon],
        "proxy_state_progress":[float(dot(row,best)) for row in g],
        "proxy_track_risk_changes":[float(dot(row,best)) for row in r],
        "tie_break":"Lexicographically greatest exact weights among optimum vertices; no model installation",
        "interpretation":"Positive endpoint interpolation is only a proposal. Parameter cancellation and nonlinear execution remain unverified."}


def stats(values):
    values = sorted(values)
    if not values:
        return {"count":0}
    n = len(values)
    median = values[n//2] if n%2 else (values[n//2-1]+values[n//2])/2
    return {"count":n,"min":values[0],"median":median,"max":values[-1],"sum":math.fsum(values)}


def float32_ulp(value):
    """Value-storage resolution reference only, never a physical error bound."""
    try:
        x = abs(struct.unpack("!f",struct.pack("!f",value))[0])
    except OverflowError:
        return None
    return 2.**max(-149,math.frexp(x)[1]-24) if x else 2.**-149


def action_change(base,candidate,field):
    a,b = base.get(field),candidate.get(field)
    if not isinstance(a,list) or not isinstance(b,list) or len(a)!=len(b):
        return None
    try:
        if any(len(x)!=len(y) for x,y in zip(a,b)):
            return None
        return math.sqrt(math.fsum((float(x)-float(y))**2 for aa,bb in zip(a,b) for x,y in zip(aa,bb)))
    except (TypeError,ValueError):
        return None


def complete_track(track):
    if not isinstance(track,dict):
        return False
    identities = (track.get("observer_id"),track.get("track_id"))
    vectors = (track.get("relative_position"),track.get("relative_velocity"))
    return (all(isinstance(v,int) and not isinstance(v,bool) and v>=0 for v in identities)
        and finite(track.get("radius")) and track["radius"]>0 and track.get("velocity_valid") is True
        and all(isinstance(v,list) and len(v)==2 and all(finite(x) for x in v) for v in vectors))


def analyze_admission(report,*,solve=True):
    """Keep every fixed state and align every original per-track response."""
    registry = report.get("fixed_state_registry",[])
    expected = [r["key"] for r in registry]
    errors = []
    if report.get("parameter_step") != PARAMETER_STEP or report.get("candidate_registry") != list(CANDIDATES):
        errors.append("fixed_step_or_candidate_order_mismatch")
    if not expected or len(set(expected)) != len(expected):
        errors.append("invalid_fixed_registry")
    if set(report.get("candidates",{})) != set(CANDIDATES):
        errors.append("fixed_candidate_registry_mismatch")
    if report.get("execution_errors") or report.get("mechanism_conclusion_available") is not True:
        errors.append("source_diagnostic_errors_or_missing_conclusion")
    rows_by_candidate = {}
    attribution = {}
    for name in CANDIDATES:
        original = report.get("candidates",{}).get(name,{})
        rows = original.get("raw_state_evidence",[])
        keys = [row.get("key") for row in rows]
        if keys != expected:
            errors.append(name+":missing_duplicate_reordered_or_extra_states")
        mapping = {row.get("key"):row for row in rows}
        rows_by_candidate[name] = mapping
        metadata = original.get("direction_metadata",{})
        if metadata.get("parameter_step") != PARAMETER_STEP or metadata.get("parameter_step_search") is not False:
            errors.append(name+":fixed_direction_budget_metadata_mismatch")
        norm = metadata.get("aggregate_gradient_norm")
        declared_nonzero = isinstance(norm,(float,int)) and math.isfinite(norm) and norm>0
        # Reproduce admission predicates using the declared nonzero/zero norm,
        # never pretend these placeholder scalars are original direction vectors.
        reproduced = candidate_admission(expected,rows,[PARAMETER_STEP if declared_nonzero else 0.])
        for field in ("would_accept","states","state_rejection_counts_nonexclusive","candidate_rejection_reasons",
                      "executed_progress_net_all_states","strictly_improving_states"):
            if original.get(field) != reproduced[field]:
                errors.append(name+":admission_reproduction_mismatch:"+field)
        attribution[name] = {"original_rejection_reproduced":not any(e.startswith(name+":") for e in errors),
            "declared_nonzero_direction":declared_nonzero,"direction_sha256":metadata.get("direction_sha256"),
            "state_verdicts":reproduced["states"],"state_rejection_counts_nonexclusive":reproduced["state_rejection_counts_nonexclusive"],
            "reproduction_basis":"Original state/coverage predicates and reported direction norm; no original parameter-vector reconstruction"}
    progress,risk,progress_ids,risk_ids = [],[],[],[]
    state_details,track_groups = [],defaultdict(list)
    fatal = {"replay_or_fingerprint_mismatch","evaluation_error","missing_shadow_evidence",
        "baseline_not_NORMAL_supported","baseline_not_hard_safe","candidate_not_NORMAL_supported",
        "baseline_snapshot_or_track_mutation","candidate_snapshot_or_track_mutation","shadow_snapshot_mismatch",
        "risk_track_identity_order_or_context_mismatch","unknown_or_invalid_track_risk","unknown_or_nonfinite_executed_progress"}
    for fixed in registry:
        key = fixed["key"]
        rows = [rows_by_candidate[name].get(key) for name in CANDIDATES]
        detail = {"key":key,"directions":{}}
        if any(row is None for row in rows):
            errors.append(key+":missing_direction_evidence")
            state_details.append(detail)
            continue
        verdicts = [state_admission(row) for row in rows]
        valid = True
        for name,row,v in zip(CANDIDATES,rows,verdicts):
            if fatal.intersection(v["reasons"]):
                errors.append(key+":"+name+":incomplete_or_unaligned_evidence")
                valid = False
            base,candidate = row.get("baseline") or {},row.get("candidate") or {}
            if any(not isinstance(value.get("tracks"),list) or not all(complete_track(t) for t in value["tracks"]) for value in (base,candidate)):
                errors.append(key+":"+name+":missing_or_invalid_track_context")
                valid = False
            if row.get("state_index") != fixed.get("state_index") or row.get("key") != f"{row.get('layout')}:{row.get('step')}":
                errors.append(key+":"+name+":state_identity_mismatch")
                valid = False
            requested = action_change(base,candidate,"requested_actions")
            executed = action_change(base,candidate,"executed_actions")
            ulps = [float32_ulp(value["executed_progress"]) for value in (base,candidate)] if v["gain"] is not None else [None]
            resolution = sum(ulps) if all(v is not None for v in ulps) else None
            stage_changes = {stage:action_change({"a":base.get("stage_actions",{}).get(stage)},
                {"a":candidate.get("stage_actions",{}).get(stage)},"a") for stage in ("initial_joint_guard","obstacle_ttc","final_joint_guard")}
            detail["directions"][name] = {"progress_delta":v["gain"],"rejection_reasons":v["reasons"],
                "float32_value_ulp_reference":resolution,
                "abs_gain_over_value_ulp_reference":abs(v["gain"])/resolution if resolution and v["gain"] is not None else None,
                "requested_action_change_l2":requested,"executed_action_change_l2":executed,
                "execution_to_request_change_ratio":executed/requested if requested and executed is not None else None,
                "stage_action_change_l2":stage_changes,
                "endpoint_hard_safe":candidate.get("hard_safe"),"risk_changes":[],
                "baseline_risk_vector":base.get("risk_proxy"),"candidate_risk_vector":candidate.get("risk_proxy"),
                "baseline_tracks":base.get("tracks"),"candidate_tracks":candidate.get("tracks")}
        # Identical baselines across all directions are necessary for a common
        # secant origin. No subset, tolerance or imputation replaces this check.
        if any(row.get("baseline") != rows[0].get("baseline") for row in rows[1:]):
            errors.append(key+":cross_direction_baseline_mismatch")
            valid = False
        state_details.append(detail)
        if not valid:
            continue
        progress.append([v["gain"] for v in verdicts]); progress_ids.append("progress:"+key)
        tracks = rows[0]["baseline"]["tracks"]
        for index,track in enumerate(tracks):
            changes = []
            track_key = f"{rows[0]['layout']}:{track['observer_id']}:{track['track_id']}"
            for name,row in zip(CANDIDATES,rows):
                delta = row["candidate"]["risk_proxy"][index]-row["baseline"]["risk_proxy"][index]
                changes.append(delta)
                item = {"key":key,"direction":name,"observer_id":track["observer_id"],"track_id":track["track_id"],
                    "baseline_risk":row["baseline"]["risk_proxy"][index],"candidate_risk":row["candidate"]["risk_proxy"][index],"delta":delta}
                detail["directions"][name]["risk_changes"].append(item)
                if delta>0:
                    track_groups[track_key].append(item)
            risk.append(changes); risk_ids.append(f"risk:{key}:{track['observer_id']}:{track['track_id']}")
    for name in CANDIDATES:
        values = [s["directions"][name] for s in state_details if name in s["directions"]]
        attribution[name].update(negative_progress_magnitudes=stats([-s["progress_delta"] for s in values if s["progress_delta"] is not None and s["progress_delta"]<0]),
            positive_risk_deltas=stats([item["delta"] for s in values for item in s["risk_changes"] if item["delta"]>0]),
            negative_progress_within_value_ulp_reference=sum(s["float32_value_ulp_reference"] is not None and s["progress_delta"] is not None and s["progress_delta"]<0 and abs(s["progress_delta"])<=s["float32_value_ulp_reference"] for s in values))
    proxy = (solve_simplex(progress,risk,progress_ids=progress_ids,risk_ids=risk_ids) if solve else {"status":"NOT_SOLVED_PREFLIGHT"}) if not errors else {
        "status":"INVALID_EVIDENCE","errors":errors,"actual_nonzero_parameter_status":"UNKNOWN","real_shadow_validation":False,"actual_safety_certified":False}
    patterns = Counter()
    for state in expected:
        patterns[''.join('?' if state not in rows_by_candidate[name] else
            ('1' if state_admission(rows_by_candidate[name][state])["passed"] else '0') for name in CANDIDATES)] += 1
    return {"attribution":attribution,"state_details":state_details,
        "state_pass_patterns":{"direction_order":list(CANDIDATES),"counts":dict(patterns)},
        "recurring_risk_veto_tracks":dict(track_groups),"risk_track_identity_scope":"layout-local observer/track identity; repeated steps/directions are dependent",
        "fixed_state_count":len(expected),"complete_progress_constraint_count":len(progress),"complete_risk_constraint_count":len(risk),
        "evidence_errors":errors,"proxy":proxy,
        "limitations":["Recorded finite-step endpoint secants are not true derivatives or measured mixed-direction outcomes.",
            "Strict zero risk/progress thresholds are unchanged; ULP value-storage references are neither full error bounds nor reasons to relax admission.",
            "Stage action-change attenuation is descriptive, not causal attribution of execution failures.",
            "Endpoint hard-safety predicates are categorical and are not certified for mixtures by this linear model.",
            "Original nonzero direction vectors are unavailable. Convex weights give an analytic norm bound <=0.01 but a possible zero lower bound.",
            "Any weights are selected using exposed Fit/Calibration evidence, not independent validation. No Audit-based selection.",
            "No Actor/Critic/Shadow execution or parameter reconstruction; phase3 requires a separate reviewed protocol."],
        "actor_training_authorized":False,"refit_authorized":False,"shadow_validation_authorized":False}
