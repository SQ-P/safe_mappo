"""Frozen-Actor CUP-JR R0-R3 evaluation. Runs only when explicitly invoked."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.algorithms.factory import create_trainer
from src.safety.staged_execution.uncertainty import ObservablePredictionHistory
from src.utils.io_utils import load_yaml, save_json, append_jsonl
from scripts.evaluate_safety_ablation import CHECKPOINT, _paired_bootstrap


def v3_registry():
    registry = load_yaml(ROOT / "configs/experiments/recovery_seed23/panels_v3_dev.json")
    old = load_yaml(ROOT / "configs/experiments/recovery_seed23/panels.json")
    used = set().union(*(set(old[name]) for name in ("fit","calibration","validation","final")))
    for name in ("v3_dev_fit","v3_dev_calibration","v3_dev_validation"):
        values = registry[name]
        if not values or len(values) != len(set(values)) or used.intersection(values):
            raise ValueError("V3 panels must be unique and disjoint from all historical panels")
        used.update(values)
    return registry


def validate_v3_request(arm, seeds, mechanism_only):
    old = load_yaml(ROOT / "configs/experiments/recovery_seed23/panels.json")
    v3_registry()
    if arm not in {"R0","R1","R2"}:
        raise ValueError("V3 R3 is deferred pending Fit-only predictive-feature evidence")
    if not mechanism_only or list(seeds) != old["validation"]:
        raise ValueError("Formal V3 validation is paused by the user; use --mechanism-only on the exposed validation panel")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def resolve(path):
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def run_arm(arm, checkpoint, seeds, output, device, artifact_path=None, collect=False, protocol="v1", gate_a_diagnostics=False, mechanism_only=False):
    if protocol == "v3":
        validate_v3_request(arm,seeds,mechanism_only)
        if collect:
            raise ValueError("V3 data collection is deferred; existing fixed calibration is reused")
    if gate_a_diagnostics and (protocol != "v2" or not set(seeds).issubset(set(load_yaml(ROOT / "configs/experiments/recovery_seed23/panels.json")["validation"]))):
        raise ValueError("Gate A diagnostics require v2 and the already exposed validation panel")
    output = resolve(output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    checkpoint = resolve(checkpoint)
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = deepcopy(saved["config"])
    if (int(config.get("seed", -1)) != 23 or config.get("algorithm") != "mappo"
            or int(saved.get("update", -1)) != 800
            or int(saved.get("obs_dim", -1)) != 52):
        raise ValueError("first study requires the registered seed23 MAPPO checkpoint")
    study = load_yaml(ROOT / f"configs/experiments/recovery_seed23/{arm}.yaml")
    if protocol not in {"v1", "v2", "v3"}:
        raise ValueError("unknown study protocol")
    study["protocol"] = protocol
    if protocol in {"v2","v3"}:
        # Component-wise initialization bound is preserved by mirror reflection.
        # This is public task configuration, never simulator obstacle velocity.
        bound = config["env"]["dynamic_obstacle_velocity_range"]
        study["obstacle_speed_bound_mps"] = float(np.sqrt(2.) * max(abs(float(v)) for v in bound))
    if protocol == "v3":
        study.update(safety_tie_tolerance_m=0.,mechanism_only=True,
                     certificate_version="physical_reachable_matched_risk_v3",adaptive_model_changed=False)
    if (study.get("version") != "calibrated_joint_recovery_v1"
            or study.get("arm") != arm or study.get("horizon_steps") != 3
            or study.get("parent_recovery") != "star_persistent_recovery_v5_4"
            or not np.isfinite(study.get("progress_tie_tolerance_m", np.nan))
            or study["progress_tie_tolerance_m"] < 0):
        raise ValueError("unsupported recovery study contract")
    artifact = None
    if arm in {"R2", "R3"} or gate_a_diagnostics:
        if not artifact_path:
            raise ValueError("R2/R3 require an independently calibrated artifact")
        artifact = json.loads(resolve(artifact_path).read_text(encoding="utf-8"))
        if artifact["schema_version"] != "CUPJRCalibrationV1":
            raise ValueError("unsupported calibration artifact")
        expected_model = "normalized_ridge_v2" if protocol in {"v2","v3"} else "legacy"
        if artifact.get("scale_model_version", "legacy") != expected_model:
            raise ValueError("study protocol and calibration scale model version must match")
        model = artifact["adaptive_model"]
        if protocol in {"v2","v3"} and (not np.isfinite(model.get("feature_scale", np.nan))
                                 or model["feature_scale"] <= 0
                                 or not np.isfinite(model.get("feature_max", np.nan))
                                 or model["feature_max"] < 0):
            raise ValueError("invalid normalized calibration feature contract")
        if artifact.get("calibration_scope", "strict") not in {"strict", "observed_valid"}:
            raise ValueError("unknown calibration scope")
        for key in ("fixed_q", "adaptive_q"):
            if not np.isfinite(artifact[key]) or artifact[key] < 0:
                raise ValueError("calibration quantiles must be finite and nonnegative")
        for key in ("fixed_model", "adaptive_model"):
            for coefficient in ("b", "c"):
                values = np.asarray(artifact[key][coefficient], dtype=float)
                if values.shape != (3,) or not np.isfinite(values).all() or np.any(values < 0):
                    raise ValueError("invalid calibration scale coefficients")
        if artifact["checkpoint_sha256"] != checkpoint_sha:
            raise ValueError("calibration checkpoint mismatch")
        panels = load_yaml(ROOT / "configs/experiments/recovery_seed23/panels.json")
        if (artifact["fit_layouts"] != panels["fit"]
                or artifact["calibration_layouts"] != panels["calibration"]):
            raise ValueError("calibration artifact does not match registered panels")
        if set(seeds) & set(artifact["fit_layouts"] + artifact["calibration_layouts"]):
            raise ValueError("evaluation layouts overlap fit/calibration data")
    safety = config["safety_execution"]
    if not (safety["enabled"] and safety["recovery_enabled"]
            and safety["initial_joint_guard"]["enabled"]
            and safety["obstacle_ttc"]["enabled"] and safety["final_joint_guard"]["enabled"]):
        raise ValueError("registered complete safety executor is required")
    safety["safety_state_machine_version"] = "star_persistent_recovery_v5_4"
    safety["goal_boundary_corridor"]["enabled"] = True
    config["device"] = device
    config["output"] = {"output_dir": str(output / "runtime"),
                        "checkpoint_dir": str(output / "runtime/checkpoints"), "reject_nonempty": True}
    trainer = create_trainer(config, project_root=ROOT)
    trainer.load_checkpoint(checkpoint, allow_recovery_study_evaluation=True)
    env = trainer._base_env()
    study.update(artifact=artifact)
    env.recovery_study = study
    history = ObservablePredictionHistory(env.dt,
        sink=(lambda row: append_jsonl(row, output / "predictions.jsonl")) if collect else None)
    env.recovery_prediction_history = history
    if gate_a_diagnostics:
        def gate_a_sink(row):
            row["layout_seed"] = int(env._episode_seed)
            append_jsonl(row, output / "gate_a_decisions.jsonl")
        env.recovery_gate_a_sink = gate_a_sink
    reset, step = env.reset, env.step
    times = []
    radius_values = []
    radius_modes = Counter()
    candidate_times = []
    candidate_times_without_probe = []
    step_times_without_probe = []
    step_times_without_diagnostics = []
    candidate_times_without_diagnostics = []
    original_select = env._v5_4_select_recovery_command

    def timed_select(actions):
        probe_before = getattr(env, "_cupjr_probe_seconds", 0.)
        audit_before = getattr(env, "_cupjr_audit_seconds", 0.)
        started = perf_counter()
        result = original_select(actions)
        candidate_times.append(perf_counter() - started)
        candidate_times_without_probe.append(max(0., candidate_times[-1] - (getattr(env, "_cupjr_probe_seconds", 0.) - probe_before)))
        candidate_times_without_diagnostics.append(max(0., candidate_times_without_probe[-1] - (getattr(env, "_cupjr_audit_seconds", 0.) - audit_before)))
        return result

    def observed_reset(*args, **kwargs):
        history.reset()
        result = reset(*args, **kwargs)
        history.observe(env._safety_planning_snapshot(), env._episode_seed)
        return result

    def observed_step(actions):
        history.observe(env._safety_planning_snapshot(), env._episode_seed)
        started = perf_counter()
        probe_before = getattr(env, "_cupjr_probe_seconds", 0.)
        audit_before = getattr(env, "_cupjr_audit_seconds", 0.)
        result = step(actions)
        times.append(perf_counter() - started)
        step_times_without_probe.append(max(0., times[-1] - (getattr(env, "_cupjr_probe_seconds", 0.) - probe_before)))
        step_times_without_diagnostics.append(max(0., step_times_without_probe[-1] - (getattr(env, "_cupjr_audit_seconds", 0.) - audit_before)))
        terminal = bool(result[2] or result[3])
        history.observe(env._safety_planning_snapshot(), env._episode_seed, terminal=terminal)
        if protocol in {"v2","v3"}:
            for event in history.radius_events:
                radius_modes[event["mode"]] += 1
                if event.get("radius_m") is not None:
                    radius_values.append(event["radius_m"])
            history.radius_events.clear()
            if collect:
                for event in history.latest_risk_events:
                    append_jsonl(event, output / "track_loss_events.jsonl")
        return result

    env._v5_4_select_recovery_command = timed_select
    env.reset, env.step = observed_reset, observed_step
    started = perf_counter()
    result = trainer.evaluate(layout_seeds=seeds,
        episode_records_path=output / "episodes.jsonl", summary_output_path=output / "summary.json")
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checkpoint_sha:
        raise RuntimeError("checkpoint changed during evaluation")
    def timing(values):
        return {"count": len(values), "p95_seconds": float(np.quantile(values, .95)) if values else None,
                "p99_seconds": float(np.quantile(values, .99)) if values else None}
    save_json({"wall_seconds": perf_counter() - started, "environment_step": timing(times),
               "candidate_selection_including_shadow": timing(candidate_times),
               "environment_step_excluding_R2_probe": timing(step_times_without_probe),
               "candidate_selection_excluding_R2_probe": timing(candidate_times_without_probe),
               "environment_step_excluding_all_diagnostics": timing(step_times_without_diagnostics),
               "candidate_selection_excluding_all_diagnostics": timing(candidate_times_without_diagnostics),
               "includes_gate_a_diagnostics": gate_a_diagnostics,
               "includes_R2_counterfactual_probe": protocol == "v2" and arm == "R3"}, output / "timing.json")
    if protocol == "v2":
        episodes = read_rows(output / "episodes.jsonl")
        events = [e for row in episodes for e in row.get("safe_recovery_events", [])]
        choices = [e for e in events if e.get("event") == "R2_R3_same_state_choice"]
        radii = radius_values
        save_json({"protocol": protocol, "same_state_decisions": len(choices),
                   "R2_R3_different_count": sum(e["different"] for e in choices),
                   "R2_R3_difference_rate": sum(e["different"] for e in choices)/len(choices) if choices else None,
                   "note": "same-state choices on R3-induced states; includes None; no second environment step",
                   "calibrated_radius_calls": len(radii), "radius_modes": dict(radius_modes),
                   "calibrated_radius_quantiles": {str(q): float(np.quantile(radii, q)) if radii else None for q in (.5,.95,.99)}}, output / "selection_diagnostics.json")
    if protocol == "v3":
        episodes = read_rows(output / "episodes.jsonl")
        events = [e for row in episodes for e in row.get("safe_recovery_events",[])]
        scores = [e for e in events if e.get("event") == "v3_candidate_score"]
        selections = [e for e in events if e.get("event") == "v3_candidate_selected"]
        active_checks = [e for e in events if e.get("event") == "v3_active_certificate"]
        save_json({"protocol":"v3","mechanism_only":True,"formal_efficacy_decision":None,
            "candidate_modes":dict(Counter(e["mode"] for e in scores)),
            "candidate_reasons":dict(Counter(e["reason"] for e in scores)),
            "selection_modes":dict(Counter(e["mode"] for e in selections)),
            "active_command_modes":dict(Counter(e["mode"] for e in active_checks)),
            "active_command_reasons":dict(Counter(e["reason"] for e in active_checks)),
            "no_feasible_decisions":sum(e.get("event") == "v3_no_feasible_candidate" for e in events),
            "decisions":len(selections)+sum(e.get("event") == "v3_no_feasible_candidate" for e in events),
            "selection_count":len(selections),"R0_legacy_selection_count":sum(r.get("recovery_candidate_selection_count",0) for r in episodes) if arm == "R0" else None,
            "safety_tie_tolerance_m":0.,"adaptive_contribution_tested":False,
            "note":"Exposed-layout mechanism diagnostics; constrained recovery is not full calibrated coverage; no complete policy safety claim"},output/"selection_diagnostics.json")
    contract = {"study": {k: v for k, v in study.items() if k != "artifact"},
                "safety_execution": safety, "artifact_sha256": digest(artifact) if artifact else None}
    save_json({"schema_version": "CUPJRProvenanceV1", "arm": arm,
               "checkpoint_sha256": checkpoint_sha,
               "parent_task_fingerprint": saved["task_config_fingerprint"],
               "runtime_task_fingerprint": trainer.task_config_fingerprint(),
               "study_contract": contract, "study_contract_sha256": digest(contract),
               "layout_seeds": seeds, "frozen_actor": True,
               "gate_a_diagnostics": gate_a_diagnostics,
               "executor_transition": "V4-to-V5.4 plus required goal corridor; identical across R0-R3",
               "coverage_claim": "mechanism-only; constrained mode has no full calibrated guarantee" if protocol == "v3" else "empirical; conditional on observed valid tracks"}, output / "provenance.json")
    return result


def compare(root, seeds, delta):
    provenance = {arm: json.loads((root / arm / "provenance.json").read_text())
                  for arm in ("R0", "R1", "R2", "R3")}
    for field in ("checkpoint_sha256", "parent_task_fingerprint", "runtime_task_fingerprint"):
        if len({p[field] for p in provenance.values()}) != 1:
            raise ValueError(f"paired study provenance differs: {field}")
    if provenance["R2"]["study_contract"]["artifact_sha256"] != provenance["R3"]["study_contract"]["artifact_sha256"]:
        raise ValueError("R2/R3 calibration artifacts differ")
    rows = {arm: read_rows(root / arm / "episodes.jsonl") for arm in ("R0", "R1", "R2", "R3")}
    if any([r["seed"] for r in values] != seeds for values in rows.values()):
        raise ValueError("paired panels differ")
    rng = np.random.default_rng(20261120)
    stats = {}
    for left, right in (("R1", "R0"), ("R2", "R1"), ("R3", "R2")):
        metrics = {}
        for name, getter in {
            "cfms": lambda r: float(r["safe_success"]),
            "collision": lambda r: float(r["collision_count"] > 0),
            "dynamic_collision": lambda r: float(r["dynamic_obstacle_collision_count"] > 0),
            "timeout": lambda r: float(r["reason"] == "timeout"),
            "progress": lambda r: r["goal_progress_ratio"],
        }.items():
            metrics[name] = _paired_bootstrap(np.asarray([getter(r) for r in rows[left]]),
                np.asarray([getter(r) for r in rows[right]]), samples=10000, rng=rng)
        metrics["new_success_layouts"] = [a["seed"] for a,b in zip(rows[left], rows[right]) if a["safe_success"] and not b["safe_success"]]
        metrics["lost_success_layouts"] = [a["seed"] for a,b in zip(rows[left], rows[right]) if b["safe_success"] and not a["safe_success"]]
        stats[f"{left}-{right}"] = metrics
    save_json({"comparisons": stats, "collision_noninferiority_delta": delta,
               "R3_R2_noninferior": stats["R3-R2"]["collision"]["ci95_high"] < delta if delta is not None else None}, root / "paired_analysis.json")


def compare_v3(root,seeds):
    """Descriptive paired comparisons on exposed layouts; never approve efficacy."""
    arms = ("R0","R1","R2")
    provenance = {a:json.loads((root/a/"provenance.json").read_text()) for a in arms}
    for field in ("checkpoint_sha256","runtime_task_fingerprint","parent_task_fingerprint"):
        if len({p[field] for p in provenance.values()}) != 1:
            raise ValueError(f"V3 study identity differs: {field}")
    if any(p["study_contract"]["study"].get("protocol") != "v3" for p in provenance.values()):
        raise ValueError("V3 comparisons cannot mix protocol versions")
    rows = {a:read_rows(root/a/"episodes.jsonl") for a in arms}
    if any([r["seed"] for r in values] != seeds for values in rows.values()):
        raise ValueError("V3 paired panels differ")
    rng = np.random.default_rng(20261121)
    comparisons = {}
    for left,right in (("R1","R0"),("R2","R1"),("R2","R0")):
        comparisons[f"{left}-{right}"] = {name:_paired_bootstrap(
            np.asarray([getter(r) for r in rows[left]]),np.asarray([getter(r) for r in rows[right]]),samples=10000,rng=rng)
            for name,getter in {"cfms":lambda r:float(r["safe_success"]),
                "collision":lambda r:float(r["collision_count"]>0),"progress":lambda r:r["goal_progress_ratio"],
                "timeout":lambda r:float(r["reason"] == "timeout")}.items()}
    save_json({"protocol":"v3","mechanism_only":True,"comparisons":comparisons,
        "efficacy_decision":None,"noninferiority_decision":None,"fresh_development_validation_run":False,
        "note":"Exposed design layouts; CIs are descriptive, not independent efficacy evidence; R3 not tested"},root/"paired_analysis.json")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", nargs="+", choices=["R0", "R1", "R2", "R3"])
    p.add_argument("--checkpoint", default=CHECKPOINT)
    p.add_argument("--panel", choices=["fit", "calibration", "validation", "final","v3_dev_fit","v3_dev_calibration","v3_dev_validation"], default="validation")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--calibration-artifact")
    p.add_argument("--device", default="cpu")
    p.add_argument("--protocol", choices=["v1", "v2","v3"], default="v1")
    p.add_argument("--mechanism-only",action="store_true",help="V3 only: replay exposed validation for mechanism checks; formal validation remains paused")
    p.add_argument("--noninferiority-delta", type=float)
    p.add_argument("--gate-a-diagnostics", action="store_true", help="v2 exposed-validation diagnostic replay only; writes candidate traces without changing selection")
    args = p.parse_args(argv)
    panels = load_yaml(ROOT / "configs/experiments/recovery_seed23/panels.json")
    args.arm = args.arm or (["R0","R1","R2"] if args.protocol == "v3" else ["R0","R1","R2","R3"])
    if args.panel.startswith("v3_dev_"):
        panels.update(v3_registry())
        if args.protocol != "v3":
            raise ValueError("Fresh V3 panels cannot be used by V1/V2")
    seeds = panels[args.panel]
    if args.protocol == "v3":
        for arm in args.arm:
            validate_v3_request(arm,seeds,args.mechanism_only)
    elif args.mechanism_only:
        raise ValueError("--mechanism-only is a V3 diagnostic flag")
    if args.gate_a_diagnostics and (args.protocol != "v2" or args.panel != "validation"):
        raise ValueError("Gate A diagnostics require --protocol v2 --panel validation; Final remains sealed")
    if args.panel == "final" and args.noninferiority_delta is None:
        raise ValueError("freeze --noninferiority-delta before final testing")
    if args.panel == "final" and args.protocol == "v2":
        raise ValueError("v2 is a development protocol; freeze a reviewed final protocol after validation")
    if args.panel == "final" and args.calibration_artifact:
        artifact = json.loads(resolve(args.calibration_artifact).read_text(encoding="utf-8"))
        if (artifact.get("calibration_scope", "strict") != "strict"
                or artifact.get("strict_gate_passed") is False):
            raise ValueError("conditional development calibration does not pass the original final-test gate")
    for arm in args.arm:
        result = run_arm(arm, args.checkpoint, seeds, resolve(args.output_dir) / arm,
                         args.device, args.calibration_artifact, protocol=args.protocol, gate_a_diagnostics=args.gate_a_diagnostics, mechanism_only=args.mechanism_only)
        print(f"{arm}: CFMS={result['cfms']:.3f}; output={args.output_dir}/{arm}", flush=True)
    root = resolve(args.output_dir)
    if args.protocol == "v3":
        if all((root/arm/"episodes.jsonl").exists() for arm in ("R0","R1","R2")):
            compare_v3(root,seeds)
        return
    if all((root / arm / "episodes.jsonl").exists() for arm in ("R0", "R1", "R2", "R3")):
        compare(root, seeds, args.noninferiority_delta)


if __name__ == "__main__":
    main()
