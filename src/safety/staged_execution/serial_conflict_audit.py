"""P0 diagnostics for the frozen serial Guard/TTC path; never an executor."""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import random

import numpy as np

from src.safety.staged_execution.execution_consistency import registered_ttc_headroom_actions
from src.safety.staged_execution.obstacle_guard import assess_obstacle_actions, minimum_predicted_obstacle_clearance_m

STAGES = ("actor_raw", "recovery_overlay", "initial_joint_guard", "obstacle_ttc", "final_joint_guard", "executed")
MATERIAL_MPS = .05
PROGRESS_MPS = .10
LOCAL_LIMIT = 8
JOINT_LIMIT = 64


def protocol():
    return {"schema_version": "SerialGuardTTCProtocolV1", "e1_nominal_delta_v_mps": MATERIAL_MPS,
        "e3_final_goal_velocity_loss_mps": PROGRESS_MPS, "goal_direction_epsilon_m": 1e-8,
        "local_candidate_limit": LOCAL_LIMIT, "joint_candidate_limit": JOINT_LIMIT, "dedup_decimals": 10,
        "shadow_steps": 3, "minimum_opportunity_states": 20, "minimum_opportunity_layouts": 10,
        "scope": "exposed development only; P0 observable screen, not unified safety certification",
        "p1_authorized": False, "training_authorized": False}


def matrix(value, shape=None):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 2 or (shape is not None and result.shape != shape) or not np.isfinite(result).all():
        raise ValueError("Expected finite matching UAV-by-2 matrices")
    return result


def next_velocity(velocity, action, context):
    return np.clip(velocity + np.clip(action, -1., 1.)*context["max_acceleration"]*context["dt"],
                   -context["max_speed"], context["max_speed"])


def goal_velocity(velocity, positions, goals):
    delta = goals-positions
    norms = np.linalg.norm(delta, axis=1)
    unit = np.divide(delta, norms[:, None], out=np.zeros_like(delta), where=norms[:, None] > 1e-8)
    values = np.sum(velocity*unit, axis=1)
    return [float(v) if n > 1e-8 else None for v, n in zip(values, norms)]


def event_metrics(context, stages, ttc_risks, final_risks):
    velocity = matrix(context["velocities"])
    actions = {s: matrix(stages[s], velocity.shape) for s in STAGES}
    if min(context["max_acceleration"], context["dt"], context["max_speed"]) <= 0:
        raise ValueError("Positive dynamics constants required")
    ttc, final = actions["obstacle_ttc"], actions["final_joint_guard"]
    nominal = np.linalg.norm(final-ttc, axis=1)*context["max_acceleration"]*context["dt"]
    vt, vf = next_velocity(velocity, ttc, context), next_velocity(velocity, final, context)
    positions, goals = matrix(context["positions"], velocity.shape), matrix(context["goals"], velocity.shape)
    pt, pf = goal_velocity(vt, positions, goals), goal_velocity(vf, positions, goals)
    losses = [None if a is None or b is None else a-b for a, b in zip(pt, pf)]
    local = None
    if context.get("subgoals") is not None:
        local_goals = matrix(context["subgoals"], velocity.shape)
        local = {"ttc": goal_velocity(vt, positions, local_goals), "final": goal_velocity(vf, positions, local_goals)}
    previous = {(r["agent_id"], r["track_id"]) for r in ttc_risks}
    added = [r for r in final_risks if (r["agent_id"], r["track_id"]) not in previous]
    return {"e1_material_agents": np.flatnonzero(nominal >= MATERIAL_MPS).tolist(),
        "nominal_delta_v_mps": nominal.tolist(), "action_delta_l2": np.linalg.norm(final-ttc, axis=1).tolist(),
        "actual_clipped_velocity_delta_mps": np.linalg.norm(vf-vt, axis=1).tolist(),
        "e2_rule_retrigger": not ttc_risks and bool(final_risks), "e2_new_ttc_risks": added,
        "ttc_risks": ttc_risks, "final_risks": final_risks,
        "e3_progress_loss_mps": losses, "e3_loss_agents": [i for i, v in enumerate(losses) if v is not None and v >= PROGRESS_MPS],
        "goal_direction_undefined_agents": [i for i, v in enumerate(losses) if v is None],
        "final_goal_velocity_mps": {"ttc": pt, "final": pf}, "subgoal_velocity_mps": local,
        "interpretation": "E2 is legacy rule triggering; E3 is a same-state velocity contrast, not proof of avoidable loss."}


@contextmanager
def preserve_rng():
    import torch
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state); np.random.set_state(numpy_state); torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _token(env, hidden):
    from src.safety.ppcsa_one_step import integrity_token
    return integrity_token(env, hidden)


class _Captured(Exception):
    pass


def capture_prefix(env, action, hidden=None):
    """Clone actual serial prefix, including Recovery; do not restrict to NORMAL."""
    from src.safety.ppcsa_one_step import jsonable
    from scripts.diagnose_ppcsa_direction_feasibility import json_safe
    before = _token(env, hidden)
    result = {}
    with preserve_rng():
        branch = deepcopy(env)
        branch.set_runtime_telemetry_level("full")
        def capture(executed, active, checks):
            result.update(stages=jsonable(deepcopy(active._current_action_stage_actions)),
                checks=json_safe(jsonable(checks)), executed_actions=np.asarray(executed).tolist(),
                recovery_mode_after=str(active._safety_mode))
            raise _Captured()
        branch._ppcsa_capture_control = capture
        try:
            branch.step(np.asarray(action, dtype=np.float32).copy())
            raise RuntimeError("Legacy prefix capture hook was not reached")
        except _Captured:
            pass
    if _token(env, hidden) != before:
        raise RuntimeError("Audit prefix mutated source env/cache/Recovery/GRU/RNG")
    if any(s not in result["stages"] for s in STAGES):
        raise ValueError("Legacy prefix missing required stage actions")
    result["source_unchanged"] = True
    return result


def obstacle_arguments(env, actions, visible):
    return dict(actions=actions, velocities=env.velocities,
        visible_obstacles=visible, agent_radius=env.agent_radius,
        obstacle_safety_margin=float(env.oracle_config["obstacle_safety_margin"]),
        invariant_buffer=env.boundary_guard.invariant_buffer_m, max_acceleration=env.agent_max_acceleration,
        dt=env.dt, horizon_seconds=env.obstacle_guard_horizon_seconds,
        warning_ttc_seconds=env.obstacle_guard_warning_ttc_seconds,
        emergency_ttc_seconds=env.obstacle_guard_emergency_ttc_seconds)


def risk_rows(env, actions, visible):
    return [asdict(r) for r in assess_obstacle_actions(**obstacle_arguments(env, actions, visible))]


def obstacle_clearance(env, actions, visible):
    args = obstacle_arguments(env, actions, visible)
    args.pop("warning_ttc_seconds"); args.pop("emergency_ttc_seconds")
    value = minimum_predicted_obstacle_clearance_m(**args)
    return float(value) if np.isfinite(value) else None


def current_state_evidence(env):
    """Action-independent current geometry/braking diagnostics, not a verifier."""
    positions, velocities = matrix(env.positions), matrix(env.velocities)
    boundary = []
    for i, (position, velocity) in enumerate(zip(positions, velocities)):
        distances = env.boundary_guard.surface_clearances(position, env.map_size, env.agent_radius)
        margins = []
        for axis in range(2):
            for side, speed in ((0, max(0., -velocity[axis])), (1, max(0., velocity[axis]))):
                available = distances[axis*2+side]-env.boundary_guard.surface_clearance_m-env.boundary_guard.invariant_buffer_m
                margins.append(float(available-env.boundary_guard.stopping_distance(speed, dt=env.dt,
                    max_acceleration=env.agent_max_acceleration)))
        boundary.append({"agent_id": i, "surface_clearances_m": distances.tolist(),
            "current_braking_margins_m": margins, "violated": any(v < -1e-9 for v in margins)})
    required = float(env.oracle_config["teammate_safety_distance"])+float(getattr(env.boundary_guard, "pair_invariant_buffer_m", 0.))
    pairs = []
    for left in range(len(positions)):
        for right in range(left+1, len(positions)):
            relative = positions[left]-positions[right]
            distance = float(np.linalg.norm(relative))
            closing = max(0., -float(np.dot(velocities[left]-velocities[right], relative/distance))) if distance > 1e-9 else 0.
            braking = env.boundary_guard.stopping_distance(closing, dt=env.dt, max_acceleration=2*env.agent_max_acceleration)
            pairs.append({"agents": [left, right], "distance_m": distance, "required_distance_m": required,
                "current_geometry_margin_m": distance-required, "current_radial_closing_mps": closing,
                "current_braking_margin_m": distance-required-braking,
                "violated": distance-required-braking < -1e-9})
    return {"boundary": boundary, "pairs": pairs,
        "preexisting_violation": any(r["violated"] for r in boundary+pairs),
        "basis": "Current geometry and discrete braking only; no requested or projected action is used."}


def guard_evidence(env, initial):
    current = current_state_evidence(env)
    required = float(env.oracle_config["teammate_safety_distance"])+float(getattr(env.boundary_guard, "pair_invariant_buffer_m", 0.))
    requested_distance = initial.get("requested_pair_min_distance")
    action_violation = (float(requested_distance) < required-1e-6
        if isinstance(requested_distance, (float, int)) and np.isfinite(requested_distance) else None)
    return {"current_state": current,
        "requested_action_diagnostics": {"requested_swept_pair_min_distance_m": requested_distance,
            "requested_pair_clearance_violation": action_violation},
        "projection_residual_diagnostics": {name: deepcopy(initial.get(name)) for name in (
            "invariant_violations", "pair_invariant_violations", "pair_infeasible", "recovery_required", "minimum_pair_margin")},
        "interpretation": "Projection failure is action-dependent and does not establish preexisting state violation or physical infeasibility."}


def recovery_state(snapshot):
    """Extract explicitly from EnvironmentShadowSnapshotV1 nested sections."""
    from src.safety.ppcsa_one_step import jsonable
    state = snapshot.environment_state
    scalars, arrays = state.get("scalars", {}), state.get("arrays", {})
    counters = {k: deepcopy(v) for k, v in scalars.items() if
        ("recover" in k or "safe_brake" in k or "closure" in k)
        and isinstance(v, (str, int, float, bool, type(None)))}
    masks = {k: deepcopy(v) for k, v in arrays.items() if "recover" in k or "safe_brake" in k or "closure" in k}
    targets = {k: deepcopy(v) for k, v in state.items() if "recover" in k or "closure" in k or "safe_brake" in k}
    return jsonable({"mode": scalars.get("_safety_mode"), "brake_active": scalars.get("_safe_brake_active"),
        "phase": scalars.get("recovery_phase"), "affected_agent_mask": arrays.get("_star_recovery_affected_mask"),
        "scalars_and_counters": counters, "arrays_and_masks": masks, "targets_and_top_level_context": targets})


def candidate_actions(context, stages, visible, implicated):
    """Finite diagnostic coverage, not a deployed selector or exhaustive search."""
    raw, final = matrix(stages["actor_raw"]), matrix(stages["final_joint_guard"])
    velocity = matrix(context["velocities"], raw.shape)
    goal_delta = matrix(context["goals"], raw.shape)-matrix(context["positions"], raw.shape)
    rows = [(s, matrix(stages[s])) for s in STAGES]
    brake = np.clip(-velocity/(context["max_acceleration"]*context["dt"]), -1., 1.)
    rows.append(("brake_joint", brake))
    local = {}
    for i in sorted(set(implicated)):
        if not 0 <= i < len(raw):
            raise ValueError("Invalid implicated UAV")
        registered = registered_ttc_headroom_actions(raw_action=raw[i], velocity=velocity[i],
            goal_delta=goal_delta[i], risk_rows=[(np.asarray(t.relative_position), np.asarray(t.relative_velocity)) for t in visible[i] if t.velocity_valid],
            max_acceleration=context["max_acceleration"], dt=context["dt"])
        # Source priority, then action lexicographic tie-break; never outcome selection.
        priority = {"raw": 0, "hold_brake": 1, "goal_aligned": 2}
        registered = sorted(registered, key=lambda p: (priority.get(p[0], 3), tuple(np.round(p[1], 10)), p[0]))[:LOCAL_LIMIT]
        local[i] = registered
        for label, value in registered:
            trial = final.copy(); trial[i] = value
            rows.append((f"uav{i}:{label}", trial))
    for rank in range(max((len(v) for v in local.values()), default=0)):
        trial = final.copy()
        for i, choices in local.items():
            if rank < len(choices): trial[i] = choices[rank][1]
        rows.append((f"synchronized_rank{rank}", trial))
    unique = {}
    for label, value in rows:
        key = tuple(np.round(value, 10).reshape(-1))
        unique.setdefault(key, (label, value.copy()))
    complete = list(unique.values())
    return complete[:JOINT_LIMIT], {"local_limit": LOCAL_LIMIT, "joint_limit": JOINT_LIMIT,
        "registered_unique_joint_count": len(complete), "evaluated_joint_count": min(len(complete), JOINT_LIMIT),
        "budget_exhausted": len(complete) > JOINT_LIMIT, "exhaustive_action_space": False,
        "local_action_counts": {str(i): len(v) for i, v in local.items()}}


def screen_candidates(env, context, stages, visible, implicated, evidence_reasons, preexisting):
    candidates, coverage = candidate_actions(context, stages, visible, implicated)
    rows = []
    base_progress = goal_velocity(next_velocity(matrix(context["velocities"]), matrix(stages["final_joint_guard"]), context),
        matrix(context["positions"]), matrix(context["goals"]))
    for label, action in candidates:
        isolated = deepcopy(env)
        outcome = isolated._ttc_candidate_shadow(action.copy(), actor_reference_actions=matrix(stages["actor_raw"]),
            agent_id=0, horizon_steps=3, visible_obstacles=visible)
        executed = matrix(outcome["first_executed_actions"], action.shape)
        reprojected = bool(np.any(np.abs(executed-action) > 1e-7))
        nominal_delta = action*context["max_acceleration"]*context["dt"]
        actual_delta = next_velocity(matrix(context["velocities"]), action, context)-matrix(context["velocities"])
        dynamics_mismatch = not np.allclose(nominal_delta, actual_delta, atol=1e-9, rtol=0)
        status = ("PREEXISTING_VIOLATION" if preexisting else "EVIDENCE_INSUFFICIENT" if evidence_reasons
            or dynamics_mismatch
            else "OBSERVABLE_SCREEN_PASS" if outcome["safe"] and outcome["minimum_safety_margin_m"] >= 0 and not reprojected
            else "OBSERVABLE_SCREEN_REJECT")
        progress = goal_velocity(next_velocity(matrix(context["velocities"]), executed, context), matrix(context["positions"]), matrix(context["goals"]))
        gains = [None if a is None or b is None else a-b for a, b in zip(progress, base_progress)]
        # No per-agent progress loss and at least one material progress opportunity.
        opportunity = (status == "OBSERVABLE_SCREEN_PASS" and all(v is not None and v >= 0 for v in gains)
                       and any(v >= PROGRESS_MPS for v in gains))
        rows.append({"label": label, "actions": action.tolist(), "screen_status": status,
            "legacy_three_step_result": {k: v for k, v in outcome.items() if k != "first_executed_actions"},
            "first_executed_actions": executed.tolist(), "internally_reprojected": reprojected,
            "candidate_clipping_model_mismatch": dynamics_mismatch,
            "goal_velocity_gain_mps": gains, "potential_opportunity": opportunity,
            "actual_safety_certified": False})
    counts = {"PASS": sum(r["screen_status"] == "OBSERVABLE_SCREEN_PASS" for r in rows),
        "REJECT": sum(r["screen_status"] == "OBSERVABLE_SCREEN_REJECT" for r in rows),
        "UNKNOWN": sum(r["screen_status"] == "EVIDENCE_INSUFFICIENT" for r in rows),
        "PREEXISTING": sum(r["screen_status"] == "PREEXISTING_VIOLATION" for r in rows)}
    outcome = ("CANDIDATE_SCREEN_PASS_FOUND" if counts["PASS"] else "CANDIDATES_UNRESOLVED" if counts["UNKNOWN"]
        else "PREEXISTING_VIOLATION" if counts["PREEXISTING"] else "NO_CANDIDATE_FOUND" if counts["REJECT"]
        else "NO_CANDIDATES_REGISTERED")
    coverage.update(screen_status_counts=counts, screen_outcome=outcome,
        no_candidate_found=True if outcome == "NO_CANDIDATE_FOUND" else False if counts["PASS"] else None,
        budget_status="CANDIDATE_BUDGET_EXHAUSTED" if coverage["budget_exhausted"] else "WITHIN_REGISTERED_BUDGET",
        potential_opportunity_found=any(r["potential_opportunity"] for r in rows),
        interpretation="Finite observable screening only; no candidate found is not physical infeasibility.")
    return rows, coverage


def _audit_state(env, raw_action, layout_seed, hidden=None):
    """Read-only audit before the factual old-policy step; parity checked later."""
    from src.safety.ppcsa_one_step import jsonable, digest
    before = _token(env, hidden)
    with preserve_rng():
        reference = deepcopy(env)
        reference.set_runtime_telemetry_level("full")
        visible = deepcopy(reference._safety_planning_snapshot().visible_obstacles)
        context = {"positions": env.positions.tolist(), "velocities": env.velocities.tolist(), "goals": env.goals.tolist(),
            "subgoals": env.subgoals.tolist(), "max_acceleration": float(env.agent_max_acceleration),
            "dt": float(env.dt), "max_speed": float(env.agent_max_speed)}
        first, second = capture_prefix(env, raw_action, hidden), capture_prefix(env, raw_action, hidden)
        deterministic = digest(first) == digest(second)
        stages = first["stages"]
        rt, rf = risk_rows(reference, stages["obstacle_ttc"], visible), risk_rows(reference, stages["final_joint_guard"], visible)
        events = event_metrics(context, stages, rt, rf)
        clearances = {s: obstacle_clearance(reference, stages[s], visible) for s in ("obstacle_ttc", "final_joint_guard")}
        reasons = []
        invalid = [(i, t.track_id) for i, row in enumerate(visible) for t in row if not t.velocity_valid]
        if invalid: reasons.append("INVALID_TRACK_VELOCITY")
        saturated = [i for i, row in enumerate(visible) if len(row) >= env.obstacle_slots]
        if saturated: reasons.append("OBSERVATION_SLOT_COMPLETENESS_UNPROVEN")
        if not any(visible): reasons.append("NO_OBSERVABLE_TRACK_EVIDENCE")
        if env._safety_mode != "NORMAL": reasons.append("RECOVERY_ACTIVE")
        recovery_required = (first["recovery_mode_after"] != "NORMAL"
            or any(first["checks"].get(k, False) for k in ("ttc_fallback", "safe_brake_applied", "recovery_overlay")))
        if recovery_required: reasons.append("RECOVERY_REQUIRED_IN_PREFIX")
        if not deterministic: reasons.append("PREFIX_REPEAT_MISMATCH")
        mismatch_stages = []
        for name in ("actor_raw", "obstacle_ttc", "final_joint_guard"):
            value = matrix(stages[name])
            dv = value*env.agent_max_acceleration*env.dt
            if not np.allclose(next_velocity(matrix(context["velocities"]), value, context)-matrix(context["velocities"]), dv, atol=1e-9, rtol=0):
                mismatch_stages.append(name)
        if mismatch_stages:
            reasons.append("TTC_UNCLIPPED_VS_ACTUAL_DYNAMICS_MISMATCH")
        initial = first["checks"]["initial_guard"]
        guard_diagnostics = guard_evidence(reference, initial)
        preexisting = guard_diagnostics["current_state"]["preexisting_violation"]
        implicated = sorted(set(events["e1_material_agents"]+events["e3_loss_agents"]+[r["agent_id"] for r in events["e2_new_ttc_risks"]]))
        if implicated and env._safety_mode == "NORMAL" and not recovery_required:
            candidates, coverage = screen_candidates(reference, context, stages, visible, implicated, reasons, preexisting)
        else:
            candidates, coverage = [], {"evaluated_joint_count": 0, "potential_opportunity_found": False,
                "no_candidate_found": None, "budget_exhausted": False, "exhaustive_action_space": False,
                "screen_status_counts": {"PASS": 0, "REJECT": 0, "UNKNOWN": 0, "PREEXISTING": 0},
                "screen_outcome": "NOT_EVALUATED",
                "reason": "RECOVERY_STATEFUL_CONTROL_NOT_A_CANDIDATE_SOURCE" if env._safety_mode != "NORMAL" or recovery_required else "NO_REGISTERED_EVENT"}
    if _token(env, hidden) != before:
        raise RuntimeError("P0 audit changed source env/cache/Recovery/GRU/RNG")
    return {"schema_version": "SerialGuardTTCStateAuditV1", "layout_seed": int(layout_seed), "episode_step": int(env.step_count),
        "snapshot_digest": before, "context": context, "observable_tracks": [[asdict(t) for t in row] for row in visible],
        "invalid_velocity_tracks": invalid, "slot_saturation_agents": saturated, "recovery_mode": str(env._safety_mode),
        "clipping_model_mismatch_stages": mismatch_stages,
        "recovery_state": recovery_state(env.export_counterfactual_snapshot(hidden)),
        "guard_state_and_projection_evidence": guard_diagnostics,
        "stages": stages, "prefix_checks": first["checks"], **events, "prefix_deterministic": deterministic,
        "minimum_legacy_predicted_obstacle_clearance_m": clearances,
        "preexisting_violation": preexisting, "evidence_reasons": reasons,
        "evidence_status": "PREEXISTING_VIOLATION" if preexisting else "EVIDENCE_INSUFFICIENT" if reasons else "OBSERVABLE_SCREEN_ONLY",
        "candidates": candidates, "candidate_coverage": coverage, "source_unchanged": True,
        "factual_parity": "PENDING", "mechanism_eligible": False, "actual_safety_certified": False}


def audit_state(env, raw_action, layout_seed, hidden=None):
    before = _token(env, hidden)
    try:
        return _audit_state(env, raw_action, layout_seed, hidden)
    except (ValueError, RuntimeError, KeyError) as error:
        if _token(env, hidden) != before:
            raise RuntimeError("Unsupported audit changed source state") from error
        return {"schema_version": "SerialGuardTTCStateAuditV1", "layout_seed": int(layout_seed),
            "episode_step": int(env.step_count), "snapshot_digest": before, "actor_action": matrix(raw_action).tolist(),
            "context": {"positions": env.positions.tolist(), "velocities": env.velocities.tolist(),
                "goals": env.goals.tolist(), "subgoals": env.subgoals.tolist()},
            "recovery_mode": str(env._safety_mode), "stages": {},
            "recovery_state": recovery_state(env.export_counterfactual_snapshot(hidden)),
            "e1_material_agents": [], "e2_rule_retrigger": None, "e3_loss_agents": [],
            "events_known": False, "evidence_status": "AUDIT_UNSUPPORTED", "evidence_reasons": [str(error)],
            "source_unchanged": True, "prefix_deterministic": False,
            "candidate_coverage": {"evaluated_joint_count": 0, "potential_opportunity_found": False},
            "factual_parity": "NOT_EVALUABLE", "mechanism_eligible": False, "actual_safety_certified": False}


def attach_factual_parity(row, info):
    actual = info.get("action_stage_actions", {})
    if row["evidence_status"] == "AUDIT_UNSUPPORTED":
        row["factual_stages"] = {k: np.asarray(v).tolist() for k, v in actual.items()}
        row["potential_progress_opportunity"] = False
        return row
    equal = {s: s in actual and np.array_equal(np.asarray(row["stages"][s]), np.asarray(actual[s])) for s in STAGES}
    row["factual_stage_parity"] = equal
    row["factual_parity"] = "PASS" if all(equal.values()) else "FACTUAL_PARITY_FAILED"
    row["mechanism_eligible"] = bool(all(equal.values()) and row["prefix_deterministic"] and row["source_unchanged"]
        and not row["preexisting_violation"] and not row["evidence_reasons"])
    row["potential_progress_opportunity"] = bool(row["mechanism_eligible"] and row["candidate_coverage"]["potential_opportunity_found"])
    return row


def summarize(rows, completed_layouts):
    opportunities = [r for r in rows if r.get("potential_progress_opportunity")]
    layouts = sorted({r["layout_seed"] for r in opportunities})
    complete = (list(completed_layouts) == list(range(121000, 121200))
                and {r["layout_seed"] for r in rows} == set(completed_layouts))
    parity_ok = not any(r["factual_parity"] == "FACTUAL_PARITY_FAILED" for r in rows)
    ready = complete and parity_ok and len(opportunities) >= 20 and len(layouts) >= 10
    return {"schema_version": "SerialGuardTTCReportV1", "protocol": protocol(), "audited_states": len(rows),
        "completed_layouts": list(completed_layouts), "event_state_counts": {
            "E1": sum(bool(r["e1_material_agents"]) for r in rows), "E2": sum(bool(r["e2_rule_retrigger"]) for r in rows),
            "E3": sum(bool(r["e3_loss_agents"]) for r in rows)},
        "evidence_status_counts": dict(Counter(r["evidence_status"] for r in rows)),
        "factual_parity_counts": dict(Counter(r["factual_parity"] for r in rows)),
        "potential_opportunity_states": len(opportunities), "potential_opportunity_layouts": layouts,
        "panel_complete": complete, "factual_parity_failures_absent": parity_ok,
        "next_stage_decision": "P1_CONTRACT_REVIEW_ONLY" if ready else "INSUFFICIENT_MECHANISM_EVIDENCE",
        "p1_implementation_authorized": False, "p2_authorized": False, "training_authorized": False,
        "actual_safety_certified": False, "independent_validation": False}
