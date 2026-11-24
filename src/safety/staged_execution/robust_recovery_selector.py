"""Pure candidate geometry and deterministic progress-tolerance selection."""
from __future__ import annotations

import numpy as np
from src.safety.staged_execution.contracts import RobustRecoveryResultV1


def relative_prediction(relative, obstacle_velocity, own_displacement, seconds):
    return np.asarray(relative) + np.asarray(obstacle_velocity) * seconds - np.asarray(own_displacement)


def swept_margin(start, end, distance):
    start, end = np.asarray(start), np.asarray(end)
    delta = end - start
    denominator = float(delta @ delta)
    t = float(np.clip(-start @ delta / denominator, 0., 1.)) if denominator > 1e-15 else 0.
    return float(np.linalg.norm(start + t * delta) - distance)


def evaluate_candidate(snapshot, position_path, stage_feasible, radius, affected, actor_deviation,
                       *, obstacle_speed_bound=None, lost_tracks=()):
    """Check all observed tracks, all pairs and walls over executed segments."""
    original = np.asarray(snapshot.positions)
    previous = original.copy()
    velocities = np.asarray(snapshot.velocities)
    minimum = float("inf")
    reachable_rejected = False
    if obstacle_speed_bound is not None and (not np.isfinite(obstacle_speed_bound) or obstacle_speed_bound < 0):
        raise ValueError("physical obstacle speed bound must be finite and nonnegative")
    for horizon, positions in enumerate(position_path, 1):
        positions = np.asarray(positions)
        wall = snapshot.agent_radius + snapshot.boundary_surface_clearance + snapshot.boundary_invariant_buffer
        minimum = min(minimum, float(np.min(positions - wall)),
                      float(np.min(np.asarray(snapshot.map_size) - wall - positions)))
        for left in range(snapshot.num_agents):
            for right in range(left + 1, snapshot.num_agents):
                minimum = min(minimum, swept_margin(previous[left] - previous[right], positions[left] - positions[right], snapshot.teammate_safety_distance + snapshot.boundary_invariant_buffer))
            for track in snapshot.visible_obstacles[left]:
                key = (left, track.track_id)
                rho = radius(key, horizon)
                if (not track.velocity_valid or not np.isfinite(rho)) and obstacle_speed_bound is None:
                    return RobustRecoveryResultV1(False, None, 0., 0., actor_deviation, "missing_velocity_or_history")
                unknown = not track.velocity_valid or not np.isfinite(rho)
                obstacle_velocity = np.zeros(2) if unknown else velocities[left] + track.relative_velocity
                start = relative_prediction(track.relative_position, obstacle_velocity, previous[left] - original[left], (horizon - 1) * snapshot.dt)
                end = relative_prediction(track.relative_position, obstacle_velocity, positions[left] - original[left], horizon * snapshot.dt)
                uncertainty = obstacle_speed_bound * horizon * snapshot.dt if unknown else rho
                required = snapshot.agent_radius + track.radius + snapshot.obstacle_safety_margin + snapshot.boundary_invariant_buffer + uncertainty
                minimum = min(minimum, swept_margin(start, end, required))
                reachable_rejected |= bool(unknown and swept_margin(start, end, required) < 0.)
        for agent, relative, obstacle_radius, age_seconds in lost_tracks:
            # A ball around the last observation includes all motion allowed by
            # the configured physical speed bound, including the unseen interval.
            start = np.asarray(relative) - (previous[agent] - original[agent])
            end = np.asarray(relative) - (positions[agent] - original[agent])
            required = (snapshot.agent_radius + obstacle_radius + snapshot.obstacle_safety_margin
                        + snapshot.boundary_invariant_buffer
                        + obstacle_speed_bound * (age_seconds + horizon*snapshot.dt))
            minimum = min(minimum, swept_margin(start, end, required))
            reachable_rejected |= swept_margin(start, end, required) < 0.
        previous = positions
    progress = np.linalg.norm(original - snapshot.final_goals, axis=1) - np.linalg.norm(previous - snapshot.final_goals, axis=1)
    affected_progress = progress[np.asarray(affected, dtype=bool)]
    safe = bool(stage_feasible and minimum >= 0.)
    reason = "accepted" if safe else ("reachable_set_intersection" if reachable_rejected else "robust_or_stage_infeasible")
    return RobustRecoveryResultV1(safe, float(minimum), float(progress.sum()), float(affected_progress.min()) if len(affected_progress) else 0., actor_deviation, reason)


def select_progress_candidate(rows, tolerance):
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("progress tie tolerance must be nonnegative")
    feasible = [row for row in rows if row[0].safe]
    if not feasible:
        return None
    best = max(row[0].team_progress_m for row in feasible)
    near = [row for row in feasible if row[0].team_progress_m >= best - tolerance]
    return max(near, key=lambda row: (row[0].robust_margin_m,
               row[0].affected_min_progress_m, -row[0].actor_deviation, row[1]))


def selected_actions_differ(left, right):
    """Compare executed joint commands, not merely candidate direction names."""
    if left is None or right is None:
        return (left is None) != (right is None)
    return not np.allclose(left[3]["first_executed_actions"], right[3]["first_executed_actions"],
                           atol=1e-7, rtol=0.)


def diagnose_candidate(snapshot, position_path, radius, *, obstacle_speed_bound, lost_tracks=()):
    """Pure observed-geometry evidence, never used to admit or rank a candidate.

    Nominal obstacle motion is an observable constant-velocity prediction, not
    ground-truth collision evidence. Lost/unknown tracks retain physical bounds.
    Initial and terminal margins use time-matched uncertainty; swept margins use
    the segment-end ball, matching evaluate_candidate's conservative screen.
    """
    original = np.asarray(snapshot.positions, dtype=float)
    velocities = np.asarray(snapshot.velocities)
    details = []
    path = [original, *[np.asarray(p, dtype=float) for p in position_path]]
    for horizon, positions in enumerate(path):
        previous = path[max(0, horizon-1)]
        wall = snapshot.agent_radius + snapshot.boundary_surface_clearance + snapshot.boundary_invariant_buffer
        for agent in range(snapshot.num_agents):
            for axis in range(2):
                for side in (0, 1):
                    start = previous[agent, axis]-wall if side == 0 else snapshot.map_size[axis]-wall-previous[agent, axis]
                    end = positions[agent, axis]-wall if side == 0 else snapshot.map_size[axis]-wall-positions[agent, axis]
                    details.append(dict(kind="wall", key=[agent, axis, side], horizon=horizon,
                        nominal_margin_m=float(min(start,end)), calibrated_margin_m=float(min(start,end)),
                        endpoint_margin_m=float(end), uncertainty_m=0.))
            for other in range(agent+1, snapshot.num_agents):
                required = snapshot.teammate_safety_distance + snapshot.boundary_invariant_buffer
                margin = swept_margin(previous[agent]-previous[other], positions[agent]-positions[other], required)
                details.append(dict(kind="pair", key=[agent, other], horizon=horizon,
                    nominal_margin_m=margin, calibrated_margin_m=margin,
                    endpoint_margin_m=float(np.linalg.norm(positions[agent]-positions[other])-required), uncertainty_m=0.))
            for track in snapshot.visible_obstacles[agent]:
                # h=0 is an initial-state diagnostic, not a calibration horizon.
                rho = float(radius((agent, track.track_id), max(1,horizon)))
                unknown = not track.velocity_valid or not np.isfinite(rho)
                velocity = np.zeros(2) if unknown else velocities[agent]+track.relative_velocity
                begin_seconds = max(0,horizon-1)*snapshot.dt
                end_seconds = horizon*snapshot.dt
                start = relative_prediction(track.relative_position, velocity, previous[agent]-original[agent], begin_seconds)
                end = relative_prediction(track.relative_position, velocity, positions[agent]-original[agent], end_seconds)
                uncertainty = obstacle_speed_bound*end_seconds if unknown else (0. if horizon == 0 else rho)
                # The currently observed position has no future prediction error.
                # Also report overlap of the first future calibrated ball at t=0.
                initial_inflation = obstacle_speed_bound*snapshot.dt if unknown else rho
                required = snapshot.agent_radius+track.radius+snapshot.obstacle_safety_margin+snapshot.boundary_invariant_buffer
                nominal = swept_margin(start,end,required)
                details.append(dict(kind="observed_obstacle", key=[agent,track.track_id], horizon=horizon,
                    mode="physical_reachable" if unknown else "calibrated",
                    nominal_margin_m=nominal, calibrated_margin_m=nominal-uncertainty,
                    segment_start_margin_m=float(np.linalg.norm(start)-required-uncertainty),
                    endpoint_margin_m=float(np.linalg.norm(end)-required-uncertainty),
                    uncertainty_m=float(uncertainty),
                    first_future_ball_initial_margin_m=float(np.linalg.norm(track.relative_position)-required-initial_inflation)))
        for agent, relative, obstacle_radius, age_seconds in lost_tracks:
            start = np.asarray(relative)-(previous[agent]-original[agent])
            end = np.asarray(relative)-(positions[agent]-original[agent])
            required = snapshot.agent_radius+obstacle_radius+snapshot.obstacle_safety_margin+snapshot.boundary_invariant_buffer
            uncertainty = obstacle_speed_bound*(age_seconds+horizon*snapshot.dt)
            nominal = swept_margin(start,end,required)
            details.append(dict(kind="lost_track", key=[agent], horizon=horizon,
                mode="physical_reachable", age_seconds=float(age_seconds),
                nominal_margin_m=nominal, calibrated_margin_m=nominal-uncertainty,
                segment_start_margin_m=float(np.linalg.norm(start)-required-uncertainty),
                endpoint_margin_m=float(np.linalg.norm(end)-required-uncertainty), uncertainty_m=float(uncertainty)))
    swept = [d for d in details if d["horizon"] > 0]
    initial = [d for d in details if d["horizon"] == 0]
    terminal = [d for d in details if d["horizon"] == len(position_path)]
    minimum = lambda rows, field: min((d[field] for d in rows), default=None)
    initial_margin = minimum(initial,"calibrated_margin_m")
    terminal_margin = minimum(terminal,"endpoint_margin_m")
    kinds = ("wall", "pair", "observed_obstacle", "lost_track")
    nominal = minimum(swept,"nominal_margin_m")
    calibrated = minimum(swept,"calibrated_margin_m")
    first_ball = [d["first_future_ball_initial_margin_m"] for d in initial if "first_future_ball_initial_margin_m" in d]
    initial_expanded = min([initial_margin, *first_ball])
    return {"robust_nominal_margin_m": nominal, "robust_calibrated_margin_m": calibrated,
            **{f"{kind}_margin_m": minimum([d for d in swept if d["kind"] == kind],"calibrated_margin_m") for kind in kinds},
            "initial_time_matched_margin_m": initial_margin,
            "initial_first_future_ball_margin_m": initial_expanded,
            "initial_robust_violation": initial_expanded < 0.,
            "terminal_robust_margin_m": terminal_margin,
            "risk_margin_improving": terminal_margin > initial_expanded,
            "time_matched_risk_margin_improving": terminal_margin > initial_margin,
            "uncertainty_only_violation": nominal >= 0. and calibrated < 0.,
            "calibration_only_violation": nominal >= 0. and calibrated < 0. and
                all(d["kind"] == "observed_obstacle" and d.get("mode") == "calibrated" for d in swept if d["calibrated_margin_m"] < 0.),
            "violated_constraint_kinds": [kind for kind in kinds if any(d["kind"] == kind and d["calibrated_margin_m"] < 0. for d in swept)],
            "constraints": details,
            "interpretation": "observable model margins; improvement is evidence, never safety permission"}


def decision_diagnostics(snapshot, rows, history, artifact, speed_bound, selected, *, arm, sources, reference_actions, tolerance):
    """All four selectors share the already computed candidate shadows."""
    from dataclasses import replace
    candidates, fixed_rows, adaptive_rows = [], [], []
    priority = {"normal": 2, "left_tangent": 1, "right_tangent": 0}
    lost = history.lost_reachable(snapshot)
    for row in rows:
        candidate, envelope = row["candidate"], row["envelope"]
        modes = {}
        for name, adaptive in (("fixed",False),("adaptive",True)):
            # Cache diagnostic radii once per key/horizon; do not write online stats.
            radii = {(a,t.track_id,h): history.radius_v2((a,t.track_id),h,artifact,adaptive,record=False)
                     for a,tracks in enumerate(snapshot.visible_obstacles) for t in tracks for h in (1,2,3)}
            radius = lambda key,h: radii[(*key,h)]
            geometry = diagnose_candidate(snapshot,envelope["position_path"],radius,
                obstacle_speed_bound=speed_bound,lost_tracks=lost)
            result = evaluate_candidate(snapshot,envelope["position_path"],envelope["feasible"],radius,
                row["affected"],row["actor_deviation"],obstacle_speed_bound=speed_bound,lost_tracks=lost)
            if not row["legacy_eligible"]:
                result = replace(result,safe=False,reason="legacy_candidate_ineligible")
            modes[name] = {**geometry,"screen_result":vars(result),
                "radii": [{"agent":a,"track":t,"horizon":h,"radius_m":float(v) if np.isfinite(v) else None,
                           "mode":"calibrated" if np.isfinite(v) else "reachable"} for (a,t,h),v in radii.items()]}
            (adaptive_rows if adaptive else fixed_rows).append((result,priority.get(candidate.label,-1),candidate,envelope))
        candidates.append({"label":candidate.label,"legacy_eligible":row["legacy_eligible"],
            "legacy_margin_gain_m":row["margin_gain"],"legacy_rejection_reason":row["legacy_reason"],
            "legacy_minimum_margin_m":float(envelope["minimum_margin"]),
            "team_progress_m":row["team_progress"],"affected_progress_m":row["affected_progress"],
            "actor_deviation":row["actor_deviation"],"requested_actions":candidate.actions.tolist(),
            "first_executed_actions":np.asarray(envelope["first_executed_actions"]).tolist(),
            "position_path":np.asarray(envelope["position_path"]).tolist(),
            "stage_trace":envelope["stage_trace"],"stage_feasible":bool(envelope["feasible"]),
            "fixed":modes["fixed"],"adaptive":modes["adaptive"]})
    eligible = [r for r in rows if r["legacy_eligible"]]
    r0 = max(eligible,key=lambda r:(r["margin_gain"],r["envelope"]["minimum_margin"],r["affected_progress"],
                                   -r["actor_deviation"],priority.get(r["candidate"].label,-1)),default=None)
    r1_rows = [(RobustRecoveryResultV1(r["legacy_eligible"],
                    float(r["envelope"]["minimum_margin"]) if np.isfinite(r["envelope"]["minimum_margin"]) else 1e9,
                    r["team_progress"],r["affected_min_progress"],r["actor_deviation"],"audit"),
                priority.get(r["candidate"].label,-1),r["candidate"],r["envelope"]) for r in rows]
    choices = {"R0": r0["candidate"].label if r0 else None}
    selected_rows = {name:select_progress_candidate(values,tolerance) for name,values in
                     (("R1",r1_rows),("R2",fixed_rows),("R3",adaptive_rows))}
    choices.update({name:choice[2].label if choice else None for name,choice in selected_rows.items()})
    def executed(value):
        return value["envelope"]["first_executed_actions"] if value is not None else None
    r0_action = executed(r0)
    r1_choice = selected_rows["R1"]
    r1_action = r1_choice[3]["first_executed_actions"] if r1_choice else None
    r0_r1_different = ((r0_action is None)!=(r1_action is None)) or (r0_action is not None and r1_action is not None and not np.allclose(r0_action,r1_action,atol=1e-7,rtol=0.))
    from dataclasses import asdict
    return {"schema_version":"CUPJRGateATraceV1","arm":arm,"step":snapshot.step,
            "snapshot":asdict(snapshot),"entry_sources":list(sources),
            "observed_prediction_features":[{"agent":a,"track":t.track_id,
                "recent_residual_scale":history.recent((a,t.track_id)),
                "valid_residual_count":len(history.residuals.get((a,t.track_id),())),
                "track_age":history.age.get((a,t.track_id),0)}
                for a,tracks in enumerate(snapshot.visible_obstacles) for t in tracks],
            "reference_actions":np.asarray(reference_actions).tolist(),"candidates":candidates,
            "same_state_choices":choices,"R0_R1_action_different":bool(r0_r1_different),
            "R2_R3_action_different":selected_actions_differ(selected_rows["R2"],selected_rows["R3"]),
            "selected_label":selected[0] if selected else None,
            "selected_action":np.asarray(selected[1]).tolist() if selected else None,
            "no_feasible_candidate":selected is None,
            "interpretation":"same-state model diagnostics; no alternative environment rollout or causal outcome claim"}
