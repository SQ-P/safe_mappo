"""Read historical CUP-JR outputs only. Never creates an environment or fits a model."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from src.utils.io_utils import save_json

ARMS = ("R0","R1","R2","R3")


def resolve(path):
    value = Path(path)
    return value if value.is_absolute() else ROOT/value


def audit(root, artifact_path=None):
    root = Path(root)
    hashes = {}
    def read(path, lines=False):
        data = path.read_bytes()
        hashes[str(path.resolve())] = hashlib.sha256(data).hexdigest()
        text = data.decode("utf-8-sig")
        return [json.loads(s) for s in text.splitlines() if s.strip()] if lines else json.loads(text)
    episodes = {arm:read(root/arm/"episodes.jsonl",True) for arm in ARMS}
    provenance = {arm:read(root/arm/"provenance.json") for arm in ARMS}
    seeds = [r["seed"] for r in episodes["R0"]]
    if not seeds or len(seeds) != len(set(seeds)) or any([r["seed"] for r in episodes[a]] != seeds for a in ARMS):
        raise ValueError("complete, uniquely paired episode layouts are required")
    for field in ("checkpoint_sha256","runtime_task_fingerprint","parent_task_fingerprint"):
        if len({p[field] for p in provenance.values()}) != 1:
            raise ValueError(f"study identity mismatch: {field}")
    panels = json.loads((ROOT/"configs/experiments/recovery_seed23/panels.json").read_text())
    if not set(seeds).issubset(set(panels["validation"])):
        raise ValueError("Gate A reads the exposed validation layouts only; Final remains sealed")
    summaries = {}
    for arm,rows in episodes.items():
        events = [e for r in rows for e in r.get("safe_recovery_events",[])]
        choices = [e for e in events if e["event"] == "R2_R3_same_state_choice"]
        summary = {"episodes":len(rows),"cfms":sum(bool(r["safe_success"]) for r in rows)/len(rows),
            "collision_rate":sum(r["collision_count"]>0 for r in rows)/len(rows),
            "mean_progress":sum(r["goal_progress_ratio"] for r in rows)/len(rows),
            "timeout_rate":sum(r["reason"] == "timeout" for r in rows)/len(rows),
            "selection_count":sum(r.get("recovery_candidate_selection_count",0) for r in rows),
            "event_counts":dict(Counter(e["event"] for e in events)),
            "rejection_reasons":dict(Counter(e["reason"] for e in events if e["event"] == "robust_candidate_score" and not e["safe"])),
            "same_state_decisions":len(choices),"R2_R3_action_differences":sum(e["different"] for e in choices)}
        trace_path = root/arm/"gate_a_decisions.jsonl"
        summary["diagnostic_trace_present"] = trace_path.exists()
        if trace_path.exists():
            traces = read(trace_path,True)
            if any(t.get("schema_version") != "CUPJRGateATraceV1" or t.get("arm") != arm or t.get("layout_seed") not in seeds for t in traces):
                raise ValueError("diagnostic trace identity mismatch")
            counts = Counter()
            for t in traces:
                counts["decision_calls"] += 1
                counts["no_feasible_candidate"] += t["no_feasible_candidate"]
                counts["R0_R1_action_different"] += t["R0_R1_action_different"]
                r0,r1 = (next((c for c in t["candidates"] if c["label"] == t["same_state_choices"][a]),None) for a in ("R0","R1"))
                if r0 is not None and r1 is not None:
                    counts["R1_lower_nominal_safety_margin_than_R0"] += r1["legacy_minimum_margin_m"] < r0["legacy_minimum_margin_m"]
                for c in t["candidates"]:
                    if not c["legacy_eligible"]:
                        continue
                    for mode in ("fixed","adaptive"):
                        d = c[mode]
                        if not d["screen_result"]["safe"]:
                            counts[f"{mode}:legacy_eligible_rejected"] += 1
                            for kind in d["violated_constraint_kinds"]:
                                counts[f"{mode}:violated:{kind}"] += 1
                            for field in ("calibration_only_violation","uncertainty_only_violation","initial_robust_violation","risk_margin_improving"):
                                counts[f"{mode}:rejected:{field}"] += d[field]
                            counts[f"{mode}:initial_violation_and_improving"] += d["initial_robust_violation"] and d["risk_margin_improving"]
            summary["diagnostic_counts_nonexclusive"] = dict(counts)
        summaries[arm] = summary
    artifact_path = Path(artifact_path) if artifact_path else root.parent/"calibration_observed_dev_v2.json"
    model_evidence = {"status":"unknown_artifact_missing"}
    if artifact_path.exists():
        artifact = read(artifact_path)
        if artifact["checkpoint_sha256"] != provenance["R0"]["checkpoint_sha256"]:
            raise ValueError("calibration checkpoint differs from study")
        for arm in ("R2","R3"):
            declared = provenance[arm].get("study_contract",{}).get("artifact_sha256")
            canonical = hashlib.sha256(json.dumps(artifact,sort_keys=True,allow_nan=False).encode()).hexdigest()
            if declared != canonical:
                raise ValueError("calibration artifact differs from evaluated study")
        adaptive,fixed = artifact["adaptive_model"],artifact["fixed_model"]
        model_evidence = {"status":"artifact_verified","adaptive_slopes":adaptive["c"],
            "all_adaptive_slopes_zero":all(c == 0 for c in adaptive["c"]),
            "same_base_scales_and_quantiles":adaptive["b"] == fixed["b"] and artifact["adaptive_q"] == artifact["fixed_q"],
            "feature_max":adaptive.get("feature_max"),"feature_scale":adaptive.get("feature_scale"),
            "feature_predictiveness":"unestablished; zero coefficients do not prove all causal features uninformative",
            "strict_gate_passed":artifact.get("strict_gate_passed")}
    paired_path = root/"paired_analysis.json"
    paired = read(paired_path) if paired_path.exists() else None
    return {"schema_version":"CUPJRGateAAuditV1","source_root":str(root.resolve()),"source_sha256":hashes,
            "layout_seeds":seeds,"arms":summaries,"model_evidence":model_evidence,"paired_analysis":paired,
            "gate_a_status":"PARTIALLY_ATTRIBUTED_REQUIRES_REVIEW",
            "v3_authorized":False,"final_authorized":False,
            "unknowns":["No full causal attribution from counters or same-state shadows alone",
                        "Observed nominal prediction is not simulator-truth physical safety",
                        "No-candidate to brake/collision causality requires matched stepwise review",
                        "Predictive information of replacement causal features not established"],
            "next_step":"review detailed traces before any bounded V3; no automatic gate pass",
            "unresolved_scientific_thresholds":{"minimum_meaningful_cfms_gain":None,"collision_noninferiority_delta":None},
            "interpretation":"Historical diagnostic evidence only; rejection categories overlap; no new rollout, fitting, or efficacy claim"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--calibration-artifact")
    args = parser.parse_args(argv)
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError(output)
    report = audit(resolve(args.root),resolve(args.calibration_artifact) if args.calibration_artifact else None)
    save_json(report,output)
    print(f"GATE_A=REQUIRES_REVIEW; V3=NOT_AUTHORIZED; DETAIL_LOG={output}")


if __name__ == "__main__":
    main()
