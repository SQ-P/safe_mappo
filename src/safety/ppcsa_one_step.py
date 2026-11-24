"""PPCSA's isolated, observable, NORMAL-only frozen V4 control shadow."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, asdict
import hashlib
import json
import random

import numpy as np
import torch

from src.safety.staged_execution.robust_recovery_selector import evaluate_candidate, swept_margin

PROTOCOL = "ppcsa_normal_v4_one_step_v1"
STAGES = ("initial_joint_guard", "obstacle_ttc", "final_joint_guard")


def jsonable(value):
    if isinstance(value,torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value,np.ndarray):
        return value.tolist()
    if isinstance(value,np.generic):
        return value.item()
    if isinstance(value,dict):
        return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value,set):
        return sorted((jsonable(v) for v in value),key=str)
    return value


def digest(value):
    return hashlib.sha256(json.dumps(jsonable(value),sort_keys=True,allow_nan=False).encode()).hexdigest()


def cache_payload(env):
    return [[{k:deepcopy(v) for k,v in row.items() if k != "obstacle"} for row in tracks]
            for tracks in env._obstacle_track_cache]


def integrity_token(env,gru=None):
    snapshot = env.export_counterfactual_snapshot(gru)
    return digest({"state":snapshot.state_fingerprint,"cache":cache_payload(env),
        "cache_step":env._obstacle_track_cache_step,"gru":gru,
        "python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state()})


def capture_packet(env,layout,gru=None):
    return {"layout_seed":int(layout),"snapshot":env.export_counterfactual_snapshot(gru),
            "cache":cache_payload(env),"cache_step":env._obstacle_track_cache_step}


def restore_packet(env,packet):
    env.reset(seed=int(packet["layout_seed"]))
    gru = env.restore_counterfactual_snapshot(packet["snapshot"])
    env._obstacle_track_cache = [[{**deepcopy(row),"obstacle":env.obstacles[int(row["obstacle_id"])]}
                                 for row in tracks] for tracks in packet["cache"]]
    env._obstacle_track_cache_step = packet["cache_step"]
    return gru


def normal_support(env):
    reasons = []
    if env.terminated or env.truncated:
        reasons.append("terminal_state")
    if env._safety_mode != "NORMAL" or env._safe_brake_active:
        reasons.append("recovery_or_safe_brake")
    if env._safe_recovery_target is not None or env._recovery_incident_id is not None:
        reasons.append("active_recovery_context")
    return reasons


@dataclass(frozen=True)
class NormalOneStepResult:
    supported: bool
    hard_safe: bool
    requested_actions: list
    executed_actions: list
    stage_actions: dict
    predicted_next_positions: list
    predicted_next_velocities: list
    executed_progress: float | None
    per_agent_distance_gain_m: list
    safety_evidence: dict
    risk_proxy: list
    rejection_reasons: list
    snapshot_fingerprint: str
    protocol: str = PROTOCOL

    def payload(self):
        return asdict(self)


class _ControlCaptured(Exception):
    def __init__(self,actions,branch,checks):
        self.actions = np.asarray(actions,dtype=np.float32).copy()
        self.stages = {k:np.asarray(v).copy() for k,v in branch._current_action_stage_actions.items()}
        self.reasons = normal_support(branch)
        self.checks = deepcopy(checks)


def _capture(actions,branch,checks):
    raise _ControlCaptured(actions,branch,checks)


def evaluate_ppcsa_normal_shadow(env,requested_joint_action,gru=None):
    """Run the actual V4 control prefix on a clone, before physical advancement.

    Obstacle step methods on the clone are disabled: no true future position
    or velocity is advanced into the control prefix. The prefix includes the
    original progress-aware TTC candidate selection. Source state/cache/GRU
    and all global RNG states must remain unchanged. Parity with the factual
    step is checked separately, not assumed from sharing this code path.
    """
    if (env.star_safety_state_machine_version != "star_scoped_recovery_v4"
            or env.obstacle_execution_guard_version != "causal_ttc_progress_projection_v2"
            or env.goal_boundary_corridor_enabled or not env.safety_execution_enabled
            or env.num_agents != 3 or env.obstacle_slots != 6):
        raise ValueError("PPCSA NORMAL shadow requires the frozen 3U/6-slot V4/TTCv2 contract")
    action = np.asarray(requested_joint_action,dtype=np.float32)
    if action.shape != (env.num_agents,2) or not np.isfinite(action).all() or np.any(abs(action)>1.):
        raise ValueError("PPCSA request must be a finite bounded [UAV,2] action")
    before = integrity_token(env,gru)
    reasons = normal_support(env)
    if reasons:
        return NormalOneStepResult(False,False,action.tolist(),[],{},[],[],None,[],{},[],reasons,before)
    branch = deepcopy(env)
    branch.set_runtime_telemetry_level("full")
    # Instance overrides affect this diagnostic clone only, not the source or
    # CircularObstacle class. The physical state is never advanced for labels.
    for obstacle in branch.obstacles:
        obstacle.step = lambda *args,**kwargs:None
    snapshot = branch._safety_planning_snapshot()
    branch._ppcsa_capture_control = _capture
    try:
        branch.step(action.copy())
        raise RuntimeError("PPCSA control capture hook was not reached")
    except _ControlCaptured as captured:
        executed,stages = captured.actions,captured.stages
        control_reasons = list(captured.reasons)
        checks = captured.checks
        if checks["ttc_fallback"] or checks["safe_brake_applied"] or checks["recovery_overlay"]:
            control_reasons.append("unsupported_control_overlay")
        if any(checks[name].get("invariant_violations",0) for name in ("initial_guard","final_guard")):
            control_reasons.append("guard_invariant_violation")
        # Capture precedes V4's Recovery-state update. A Guard request is
        # already outside NORMAL support even if its projected action is safe.
        for name in ("initial_guard","final_guard"):
            for field in ("pair_infeasible","pair_invariant_violations","recovery_required"):
                if checks[name].get(field,0):
                    control_reasons.append(f"{name}_{field}")
        reasons.extend(control_reasons)
    if integrity_token(env,gru) != before:
        raise RuntimeError("PPCSA shadow changed source state/cache/GRU or RNG")
    if any(stage not in stages for stage in STAGES):
        raise RuntimeError("PPCSA shadow missing factual safety stages")
    positions,velocities = np.asarray(snapshot.positions,dtype=np.float32),np.asarray(snapshot.velocities,dtype=np.float32)
    next_velocities = np.clip(velocities+executed*env.agent_max_acceleration*env.dt,-env.agent_max_speed,env.agent_max_speed)
    next_positions = positions+next_velocities*env.dt
    old_distances = np.linalg.norm(positions-env.goals,axis=1)
    new_distances = np.linalg.norm(next_positions-env.goals,axis=1)
    progress = env._team_goal_potential(new_distances)-env._team_goal_potential(old_distances)
    bound = np.sqrt(2.)*max(abs(float(v)) for v in env.config["dynamic_obstacle_velocity_range"])
    geometry = evaluate_candidate(snapshot,[next_positions],not reasons,lambda key,h:float("inf"),
        [True]*env.num_agents,0.,obstacle_speed_bound=bound)
    risk = []
    unknown = []
    for agent,tracks in enumerate(snapshot.visible_obstacles):
        for track in tracks:
            if not track.velocity_valid:
                unknown.append([agent,track.track_id])
                risk.append(None)
                continue
            relative_velocity = velocities[agent]+track.relative_velocity-next_velocities[agent]
            required = env.agent_radius+track.radius+snapshot.obstacle_safety_margin+snapshot.boundary_invariant_buffer
            margin = swept_margin(track.relative_position,np.asarray(track.relative_position)+relative_velocity*env.obstacle_guard_horizon_seconds,required)
            risk.append(max(0.,-margin))
    # Missing dynamic motion is reported, never silently a positive teacher.
    if unknown:
        reasons.append("unknown_obstacle_motion")
    # Detection/slot truncation is a support boundary for positive certification.
    # This check uses the counts from CURRENT observable detections, not hidden
    # positions or future outcomes. A full slot list cannot prove completeness.
    if any(len(tracks) >= env.obstacle_slots for tracks in snapshot.visible_obstacles):
        reasons.append("obstacle_slot_completeness_unproven")
    if not geometry.safe:
        reasons.append("observable_physical_reachable_intersection")
    stages = {k:np.asarray(v).tolist() for k,v in stages.items()}
    return NormalOneStepResult(not control_reasons,not reasons,action.tolist(),executed.tolist(),stages,
        next_positions.tolist(),next_velocities.tolist(),float(progress),(old_distances-new_distances).tolist(),
        {"physical_margin_m":geometry.robust_margin_m,"physical_bound_mps":float(bound),"unknown_tracks":unknown,
         "stage_checks":jsonable(checks),"stage_checks_passed":not control_reasons,
         "scope":"observable one-step; not global/closed-loop safety"},risk,reasons,before)


def safe_improvement(candidate,baseline,deadzone=0.):
    if not candidate.supported or not candidate.hard_safe or not baseline.supported:
        return False
    if len(candidate.risk_proxy) != len(baseline.risk_proxy) or any(v is None for v in (*candidate.risk_proxy,*baseline.risk_proxy)):
        return False
    return bool(all(a<=b for a,b in zip(candidate.risk_proxy,baseline.risk_proxy))
                and candidate.executed_progress > baseline.executed_progress+deadzone)


def factual_parity(shadow,env_after,info,atol=1e-6):
    """Numerical tolerance accommodates float32 physical integration only."""
    if not shadow.executed_actions:
        return False,{"unsupported_before_action":True}
    factual = info["action_stage_actions"]
    stage_equal = {stage:bool(stage in factual and np.allclose(shadow.stage_actions[stage],factual[stage],atol=atol,rtol=0.)) for stage in STAGES}
    actions = bool(np.allclose(shadow.executed_actions,info["executed_actions"],atol=atol,rtol=0.))
    collision = bool(info.get("collision_count",0) or info.get("rolled_back_agent_ids"))
    comparable_positions = not collision
    positions = bool(np.allclose(shadow.predicted_next_positions,env_after.positions,atol=atol,rtol=0.)) if comparable_positions else None
    return bool(actions and all(stage_equal.values()) and (positions is not False)),{
        "stage_equal":stage_equal,"actions_equal":actions,"positions_equal":positions,"position_comparable":comparable_positions,"atol":atol}
