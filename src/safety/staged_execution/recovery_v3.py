"""Opt-in CUP-JR V3 selection; no simulator access or candidate generation."""
from dataclasses import dataclass, field

import numpy as np

from .robust_recovery_selector import evaluate_candidate, relative_prediction, swept_margin


@dataclass(frozen=True)
class RecoveryResultV3:
    mode: str
    reason: str
    legacy_margin_gain_m: float
    robust_margin_m: float
    team_progress_m: float
    affected_min_progress_m: float
    actor_deviation: float
    physical_margin_m: float | None = None
    risk_improvement_m: float | None = None
    certificate: dict = field(default_factory=dict)

    @property
    def eligible(self):
        return self.mode != "unavailable"


def select_safety_candidate_v3(rows):
    """Mode first, then lexicographic safety; only exact ties use progress."""
    eligible = [row for row in rows if row[0].eligible]
    def key(row):
        result = row[0]
        if result.mode == "constrained_recovery":
            safety = (0, result.risk_improvement_m, result.physical_margin_m,
                      result.legacy_margin_gain_m, result.robust_margin_m)
        else:
            safety = (1, result.legacy_margin_gain_m, result.robust_margin_m, 0., 0.)
        return (*safety, result.team_progress_m, result.affected_min_progress_m,
                -result.actor_deviation, row[1])
    return max(eligible, key=key, default=None)


def shadow_consistent_v3(snapshot, envelope):
    path = np.asarray(envelope["position_path"],dtype=float)
    final = np.asarray(envelope["positions"],dtype=float)
    first = np.asarray(envelope["first_executed_actions"],dtype=float)
    return bool(path.shape == (3,snapshot.num_agents,2) and final.shape == first.shape == (snapshot.num_agents,2)
                and np.isfinite(path).all() and np.isfinite(final).all() and np.isfinite(first).all()
                and np.allclose(path[-1],final,atol=1e-7,rtol=0.))


def evaluate_candidate_v3(snapshot, position_path, stage_feasible, radius, affected,
                          actor_deviation, *, legacy_eligible, legacy_margin_gain,
                          obstacle_speed_bound, lost_tracks=()):
    """A full robust screen or a distinct, conservative constrained certificate.

    For constrained recovery, every observed/lost obstacle is additionally
    enclosed by a ball at its last observed center, radius v_max*(age+k*dt).
    The END-time ball protects the WHOLE segment. Nominal swept geometry must
    also pass. At each k, candidate and stationary-position geometric reference
    use the SAME obstacle prediction and rho(k). No new calibrated violation;
    every already-negative reference constraint must not worsen. At least one
    currently expanded-overlapping track must strictly improve at the terminal
    time. This is not the full calibrated coverage claim and does not replace
    the execution chain, time budget, or any future real observation.
    """
    path = np.asarray(position_path, dtype=float)
    original = np.asarray(snapshot.positions, dtype=float)
    velocities = np.asarray(snapshot.velocities, dtype=float)
    certificate = {"schema_version":"CUPJRConstrainedCertificateV3", "coverage_claim":False,
                   "reference":"stationary position geometry at matched future time; not an executable brake",
                   "risk_checks":[]}
    def result(mode, reason, robust=-1e9, physical=None, improvement=None):
        progress = np.linalg.norm(original-snapshot.final_goals,axis=1)-np.linalg.norm(path[-1]-snapshot.final_goals,axis=1) if path.shape == (3,snapshot.num_agents,2) and np.isfinite(path).all() else np.zeros(snapshot.num_agents)
        active = progress[np.asarray(affected,dtype=bool)]
        return RecoveryResultV3(mode,reason,float(legacy_margin_gain),float(robust),float(progress.sum()),
            float(active.min()) if len(active) else 0.,float(actor_deviation),physical,improvement,dict(certificate))
    if (path.shape != (3,snapshot.num_agents,2) or not np.isfinite(path).all()
            or not np.isfinite(legacy_margin_gain) or not np.isfinite(actor_deviation)
            or not np.isfinite(obstacle_speed_bound) or obstacle_speed_bound < 0):
        return result("unavailable","invalid_evidence")
    if not stage_feasible:
        return result("unavailable","stage_infeasible")
    if not legacy_eligible:
        return result("unavailable","legacy_candidate_ineligible")
    if (not np.isfinite(original).all() or not np.isfinite(velocities).all()
            or not np.isfinite(snapshot.dt) or snapshot.dt <= 0
            or any(not np.isfinite([*t.relative_position,*t.relative_velocity,t.radius]).all() or t.radius < 0
                   for tracks in snapshot.visible_obstacles for t in tracks)
            or any(not np.isfinite([*relative,obstacle_radius,age]).all() or obstacle_radius < 0 or age < 0
                   for agent,relative,obstacle_radius,age in lost_tracks)):
        return result("unavailable","invalid_evidence")
    # Unknown velocity/OOD are explicit physical-bound cases. NaN or negative
    # calibration radii are malformed evidence rather than an OOD signal.
    radii = {(a,t.track_id,h):float(radius((a,t.track_id),h))
             for a,tracks in enumerate(snapshot.visible_obstacles) for t in tracks for h in (1,2,3)}
    if any(np.isnan(v) or v < 0 for v in radii.values()):
        return result("unavailable","invalid_radius")
    cached_radius = lambda key,h:radii[(*key,h)]
    robust = evaluate_candidate(snapshot,path,stage_feasible,cached_radius,affected,actor_deviation,
        obstacle_speed_bound=obstacle_speed_bound,lost_tracks=lost_tracks)
    if robust.safe:
        return result("robust_feasible","full_robust_screen",robust.robust_margin_m)
    # Conservative dynamic-risk certificate: all visible tracks, including
    # apparently static ones, use the task's public physical speed bound.
    physical = evaluate_candidate(snapshot,path,stage_feasible,lambda key,h:float("inf"),affected,actor_deviation,
        obstacle_speed_bound=obstacle_speed_bound,lost_tracks=lost_tracks)
    if not physical.safe:
        return result("unavailable","physical_reachable_intersection",robust.robust_margin_m,physical.robust_margin_m)
    nominal = evaluate_candidate(snapshot,path,stage_feasible,lambda key,h:0.,affected,actor_deviation,
        obstacle_speed_bound=obstacle_speed_bound,lost_tracks=lost_tracks)
    if not nominal.safe:
        return result("unavailable","nominal_swept_intersection",robust.robust_margin_m,physical.robust_margin_m)
    # Initial hard wall/pair violations cannot be excused as uncertainty.
    wall = snapshot.agent_radius+snapshot.boundary_surface_clearance+snapshot.boundary_invariant_buffer
    if np.any(original < wall) or np.any(original > np.asarray(snapshot.map_size)-wall):
        return result("unavailable","initial_hard_wall_violation",robust.robust_margin_m,physical.robust_margin_m)
    for a in range(snapshot.num_agents):
        for b in range(a+1,snapshot.num_agents):
            if np.linalg.norm(original[a]-original[b]) < snapshot.teammate_safety_distance+snapshot.boundary_invariant_buffer:
                return result("unavailable","initial_hard_pair_violation",robust.robust_margin_m,physical.robust_margin_m)
    terminal_gains = []
    initial_overlap = False
    for agent,tracks in enumerate(snapshot.visible_obstacles):
        for track in tracks:
            # Unknown/OOD have already passed their physical balls and cannot
            # justify a calibration-only escape certificate themselves.
            if not track.velocity_valid or any(not np.isfinite(radii[agent,track.track_id,h]) for h in (1,2,3)):
                continue
            required = snapshot.agent_radius+track.radius+snapshot.obstacle_safety_margin+snapshot.boundary_invariant_buffer
            initial = np.linalg.norm(track.relative_position)-required-radii[agent,track.track_id,1]
            initial_overlap |= initial < 0.
            for h,positions in enumerate(path,1):
                prediction = np.asarray(track.relative_position)+(velocities[agent]+track.relative_velocity)*h*snapshot.dt
                rho = radii[agent,track.track_id,h]
                reference = float(np.linalg.norm(prediction)-required-rho)
                candidate = float(np.linalg.norm(prediction-(positions[agent]-original[agent]))-required-rho)
                gain = candidate-reference
                certificate["risk_checks"].append({"agent":agent,"track":track.track_id,"horizon":h,
                    "reference_margin_m":reference,"candidate_margin_m":candidate,"gain_m":gain,
                    "initial_expanded_margin_m":float(initial)})
                if reference >= 0. and candidate < 0.:
                    return result("unavailable","new_calibrated_violation",robust.robust_margin_m,physical.robust_margin_m)
                if reference < 0. and candidate < reference:
                    return result("unavailable","existing_risk_worsened",robust.robust_margin_m,physical.robust_margin_m)
                if h == 3 and initial < 0.:
                    terminal_gains.append(gain)
    certificate["initial_expanded_overlap"] = bool(initial_overlap)
    if not initial_overlap:
        return result("unavailable","no_initial_expanded_violation",robust.robust_margin_m,physical.robust_margin_m)
    if not terminal_gains or max(terminal_gains) <= 0.:
        return result("unavailable","no_strict_terminal_risk_improvement",robust.robust_margin_m,physical.robust_margin_m)
    certificate["all_hard_and_time_matched_checks_passed"] = True
    return result("constrained_recovery","physical_and_matched_risk_certificate",robust.robust_margin_m,
                  physical.robust_margin_m,float(min(terminal_gains)))
