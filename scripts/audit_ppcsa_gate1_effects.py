"""Read existing Gate 1 evidence only; never collect, fit or choose hyperparameters."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from src.algorithms.ppcsa_protocol import resolve,read_manifest,file_sha,new_output


def distribution(values):
    a = np.asarray(values,dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite effect evidence")
    if not len(a):
        return {"count":0}
    return {"count":len(a),"min":float(a.min()),"p10":float(np.quantile(a,.1)),
        "p25":float(np.quantile(a,.25)),"median":float(np.median(a)),
        "p75":float(np.quantile(a,.75)),"p90":float(np.quantile(a,.9)),
        "max":float(a.max()),"mean":float(a.mean()),"sum":float(a.sum())}


def input_ulp_scale(base,candidate,goals,start_distances):
    """Descriptive paired distance-statistic quantization scale, not error bound."""
    a = np.linalg.norm(np.asarray(base,dtype=np.float32)-goals,axis=1)
    b = np.linalg.norm(np.asarray(candidate,dtype=np.float32)-goals,axis=1)
    maximum = float(np.spacing(a.max())+np.spacing(b.max()))/max(float(start_distances.max()),1e-6)
    mean = float(np.spacing(a.mean())+np.spacing(b.mean()))/max(float(start_distances.mean()),1e-6)
    # Max over the two components avoids assuming a reward mixing weight.
    return max(maximum,mean)


def summarize(data,partitions):
    states,records,examples = data["states"],data["records"],data["examples"]
    roles = {g:role for role,layouts in partitions.items() for g in layouts}
    if len(roles) != sum(map(len,partitions.values())):
        raise ValueError("Layout partition overlap")
    starts = {}
    for packet in states:
        if packet["layout_seed"] not in roles:
            raise ValueError("State outside registered development partitions")
        if packet["step"] == 0:
            snapshot = packet["snapshot"].environment_state
            positions = np.asarray([a["position"] for a in snapshot["agents"]],dtype=np.float32)
            goals = np.asarray(snapshot["arrays"]["goals"],dtype=np.float32)
            starts[packet["layout_seed"]] = np.linalg.norm(positions-goals,axis=1)
    rows,best_rows = [],[]
    for record in records:
        index,layout = record["state_index"],record["layout"]
        packet = states[index]
        if packet["layout_seed"] != layout or layout not in starts:
            raise ValueError("Missing original step-zero normalization evidence or state/layout mismatch")
        baseline = next(c["result"] for c in record["candidates"] if c["name"] == "base")
        goals = np.asarray(packet["snapshot"].environment_state["arrays"]["goals"],dtype=np.float32)
        local = []
        for candidate in record["candidates"]:
            result = candidate["result"]
            gain = (result["executed_progress"]-baseline["executed_progress"]
                if result["executed_progress"] is not None and baseline["executed_progress"] is not None else None)
            row = {"state_index":index,"layout":layout,"step":record["step"],"role":roles[layout],
                "name":candidate["name"],"kind":candidate["kind"],
                "supported":bool(result["supported"]),"hard_safe":bool(result["hard_safe"]),
                "positive":bool(candidate["improved"]),"gain":gain,
                "baseline_hard_safe":bool(baseline["hard_safe"]),
                "baseline_supported":bool(baseline["supported"]),
                "baseline_progress":baseline["executed_progress"],
                "rejection_reasons":result["rejection_reasons"]}
            if row["positive"]:
                if not (row["supported"] and row["hard_safe"] and gain>0):
                    raise ValueError("Invalid stored positive teacher")
                scale = input_ulp_scale(baseline["predicted_next_positions"],result["predicted_next_positions"],goals,starts[layout])
                row.update(input_statistic_ulp_scale=scale,gain_over_input_ulp_scale=gain/scale if scale>0 else None,
                    relative_gain_over_abs_baseline=gain/abs(baseline["executed_progress"]) if baseline["executed_progress"] != 0 else None,
                    gain_after_float32_target_cast=float(np.float32(result["executed_progress"])-np.float32(baseline["executed_progress"])))
                if row["kind"] == "local":
                    local.append(row)
            rows.append(row)
        if local:
            best_rows.append(max(local,key=lambda r:r["gain"]))

    def block(chosen_records,chosen_rows,chosen_examples,chosen_best,registered_layouts):
        positives = [r for r in chosen_rows if r["positive"]]
        local = [r for r in positives if r["kind"] == "local"]
        refs = [r for r in positives if r["kind"] == "reference"]
        baselines = [r for r in chosen_rows if r["kind"] == "base"]
        order = sorted((r["gain"] for r in chosen_best),reverse=True)
        def share(n):
            return float(sum(order[:n])/sum(order)) if order else None
        layouts = sorted({r["layout"] for r in chosen_records})
        return {"registered_layouts":registered_layouts,"analyzed_states":len(chosen_records),"analyzed_layouts":layouts,
            "baseline_hard_safe_states":sum(r["hard_safe"] for r in baselines),
            "baseline_uncertifiable_states":sum(not r["hard_safe"] for r in baselines),
            "candidate_rows":len(chosen_rows),"supported_candidates":sum(r["supported"] for r in chosen_rows),
            "hard_safe_candidates":sum(r["hard_safe"] for r in chosen_rows),
            "rejection_counts_nonexclusive":dict(Counter(reason for r in chosen_rows for reason in r["rejection_reasons"])),
            "positive_local_candidates":len(local),"positive_reference_candidates":len(refs),
            "positive_local_candidate_gains":distribution([r["gain"] for r in local]),
            "positive_reference_candidate_gains":distribution([r["gain"] for r in refs]),
            "best_positive_local_per_state_gains":distribution([r["gain"] for r in chosen_best]),
            "positive_local_layouts":sorted({r["layout"] for r in chosen_best}),
            "best_local_gain_by_baseline":{"hard_safe":distribution([r["gain"] for r in chosen_best if r["baseline_hard_safe"]]),
                "uncertifiable":distribution([r["gain"] for r in chosen_best if not r["baseline_hard_safe"]])},
            "best_gain_top1_state_share":share(1),"best_gain_top5_state_share":share(5),
            "best_gain_relative_to_abs_baseline":distribution([r["relative_gain_over_abs_baseline"] for r in chosen_best if r["relative_gain_over_abs_baseline"] is not None]),
            "relative_gain_undefined_zero_baseline":sum(r["relative_gain_over_abs_baseline"] is None for r in chosen_best),
            "best_gain_over_input_ulp_scale":distribution([r["gain_over_input_ulp_scale"] for r in chosen_best if r["gain_over_input_ulp_scale"] is not None]),
            "best_gain_at_or_below_input_ulp_scale":sum(r["gain"]<=r["input_statistic_ulp_scale"] for r in chosen_best),
            "positive_local_float32_target_ties":sum(r["gain_after_float32_target_cast"] == 0 for r in local),
            "eligible_value_examples":len(chosen_examples),
            "eligible_value_states":len({e["state_index"] for e in chosen_examples}),
            "eligible_value_layouts":sorted({e["layout"] for e in chosen_examples}),
            "eligible_positive_teacher_examples":sum(bool(e["teacher_eligible"]) for e in chosen_examples),
            "eligible_value_target_distribution":distribution([e["target"] for e in chosen_examples]),
            "eligible_value_unique_targets":len({e["target"] for e in chosen_examples})}

    def select(layouts):
        return block([r for r in records if r["layout"] in layouts],[r for r in rows if r["layout"] in layouts],
            [e for e in examples if e["layout"] in layouts],[r for r in best_rows if r["layout"] in layouts],sorted(layouts))
    return {"overall":select(set(roles)),"partitions":{role:select(set(layouts)) for role,layouts in partitions.items()},
        "per_layout":{str(g):select({g}) for g in sorted(roles)},"best_positive_local_states":best_rows,
        "candidate_evidence":rows,
        "precision_note":"ULP scale propagates one float32 ULP of paired max/mean goal-distance statistics through original step-zero normalizers, maximized over mixing weights. It is a descriptive reference, NOT a bound on all integration/norm errors, statistical significance, or meaningful efficacy. No rows filtered or thresholds selected.",
        "sampling_note":"Candidates share states and layouts; counts are not independent sample sizes. No confidence claim. Best-per-state is an oracle opportunity envelope, not a learned policy."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate1-dir",required=True)
    parser.add_argument("--output-dir",required=True)
    args = parser.parse_args(argv)
    gate1 = resolve(args.gate1_dir)
    source = read_manifest(gate1)
    gate0 = resolve(source["gate0_dir"])
    original = {str(p):file_sha(p) for d in (gate0,gate1) for p in d.rglob("*") if p.is_file()}
    if file_sha(gate0/"manifest.json") != source["gate0_manifest_sha256"]:
        raise ValueError("Gate 0 manifest drift")
    read_manifest(gate0)
    data = torch.load(gate1/"opportunities.pt",map_location="cpu",weights_only=False)
    report = {"schema_version":"PPCSAGate1EffectAuditV1","status":"REVIEW_REQUIRED",
        "source_gate1_manifest_sha256":file_sha(gate1/"manifest.json"),
        "source_report_sha256":file_sha(gate1/"ppcsa_opportunity_report.json"),
        "source_opportunities_sha256":file_sha(gate1/"opportunities.pt"),
        **summarize(data,source["partitions"]),
        "shared_jacobian":source["shared_parameter_representability"],
        "limitations":["Shared Jacobian is a first-order unconstrained fit, not finite update realizability.",
            "Calibration state/layout support is limited; no data sufficiency claim.",
            "No failure-proximate Gate1 states; early capped NORMAL evidence only.",
            "Audit partition is descriptive only, never used to select thresholds, models or hyperparameters."],
        "critic_fit_authorized":False,"actor_training_authorized":False,"formal_validation_authorized":False,
        "new_collection":False,"model_fitting":False,"source_file_sha256":original}
    current = {str(p):file_sha(p) for d in (gate0,gate1) for p in d.rglob("*") if p.is_file()}
    if current != original:
        raise RuntimeError("Original Gate0/1 evidence changed during read-only audit")
    output_path = resolve(args.output_dir)
    for protected in (gate0,gate1):
        if output_path == protected or protected in output_path.parents:
            raise ValueError("Effect audit must be separate from original Gate0/1 artifacts")
    output = new_output(args.output_dir)
    report["source_artifacts_unchanged"] = True
    (output/"effect_audit.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    lines = ["# PPCSA Gate 1 offline effect audit","","Read-only existing-data audit. REVIEW_REQUIRED; no fitting or Actor update.","",
        "| Partition | States | Value examples | Value layouts | Best positive local states | Median gain | Median gain / input ULP scale |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for role,b in report["partitions"].items():
        gain,ulp = b["best_positive_local_per_state_gains"],b["best_gain_over_input_ulp_scale"]
        lines.append(f"| {role} | {b['analyzed_states']} | {b['eligible_value_examples']} | {len(b['eligible_value_layouts'])} | {gain['count']} | {gain.get('median')} | {ulp.get('median')} |")
    lines += ["",report["precision_note"],"",report["sampling_note"],"","Source Gate0/1 files verified unchanged.","","## Limitations",""]
    lines += ["- "+s for s in report["limitations"]]
    lines += ["","Full per-layout/candidate evidence and source SHA256 values: `effect_audit.json`."]
    (output/"effect_audit.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"EFFECT_AUDIT=REVIEW_REQUIRED; SOURCE_UNCHANGED=True; DETAIL_LOG={output/'effect_audit.json'}")
    return report


if __name__ == "__main__":
    main()
