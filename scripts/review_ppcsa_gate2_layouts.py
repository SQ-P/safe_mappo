"""Offline paired/layout review of an existing Gate 2 report; no model execution."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from src.algorithms.ppcsa_protocol import resolve,read_manifest,file_sha,new_output


def stats(values):
    values = np.asarray(values,dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite recorded gain")
    if not len(values):
        return {"count":0,"net":None,"mean":None}
    return {"count":len(values),"net":float(values.sum()),"mean":float(values.mean()),
        "median":float(np.median(values)),"min":float(values.min()),"max":float(values.max()),
        "positive":int((values>0).sum()),"negative":int((values<0).sum()),"zero":int((values==0).sum())}


def eligible(row):
    return bool(row["certifiable"] and row["risk_proxy_regression"] is False and row["gain"] is not None)


def coverage(rows):
    return {"rows":len(rows),"states":len({(r["layout"],r["step"]) for r in rows}),
        "layouts":sorted({r["layout"] for r in rows}),
        "outcome_counts":dict(Counter(r["outcome"] for r in rows)),
        "certifiable":sum(bool(r["certifiable"]) for r in rows),
        "uncertifiable":sum(not r["certifiable"] for r in rows),
        "risk_proxy_regressions":sum(r["risk_proxy_regression"] is True for r in rows),
        "risk_proxy_unknown":sum(r["risk_proxy_regression"] is None for r in rows),
        "hard_certification_losses":sum(bool(r["hard_safety_regression"]) for r in rows),
        "eligible_progress":stats([r["gain"] for r in rows if eligible(r)]),
        "certifiable_progress_including_risk_worsening":stats([r["gain"] for r in rows if r["certifiable"] and r["gain"] is not None]),
        "raw_progress_including_uncertifiable":stats([r["gain"] for r in rows if r["gain"] is not None]),
        "rejection_counts_nonexclusive":dict(Counter(reason for r in rows for reason in r["rejection_reasons"]))}


def index_rows(rows,keys):
    result = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        if key in result:
            raise ValueError("Duplicate paired key")
        if row["gain"] is not None and not np.isfinite(row["gain"]):
            raise ValueError("Nonfinite recorded gain")
        result[key] = row
    return result


def paired(left,right,layouts,*,action=False):
    keys = ("layout","step","epsilon") if action else ("layout","step")
    a,b = index_rows(left,keys),index_rows(right,keys)
    if a.keys() != b.keys():
        raise ValueError("Paired state/epsilon keys differ; do not silently drop rows")
    pairs = []
    for key in sorted(a):
        l,r = a[key],b[key]
        norm_matched = (not action or (np.isclose(l["delta_l2"],r["delta_l2"],atol=1e-7,rtol=1e-6)
            and np.isclose(l["delta_l2"],l["epsilon"],atol=1e-7,rtol=1e-6)
            and np.isclose(r["delta_l2"],r["epsilon"],atol=1e-7,rtol=1e-6)))
        common = bool(eligible(l) and eligible(r) and norm_matched)
        pairs.append({"layout":l["layout"],"step":l["step"],
            **({"epsilon":l["epsilon"]} if action else {}),
            "left_outcome":l["outcome"],"right_outcome":r["outcome"],
            "left_eligible":eligible(l),"right_eligible":eligible(r),
            "both_certifiable":bool(l["certifiable"] and r["certifiable"]),
            "matched_norm":bool(norm_matched),"common_eligible":common,
            "left_gain":l["gain"],"right_gain":r["gain"],
            "delta_left_minus_right":float(l["gain"]-r["gain"]) if common else None,
            "left_positive_only":bool(l["positive_safe"] and not r["positive_safe"]),
            "right_positive_only":bool(r["positive_safe"] and not l["positive_safe"]),
            "left_risk_regression":l["risk_proxy_regression"],"right_risk_regression":r["risk_proxy_regression"]})
    by_layout = {}
    for layout in layouts:
        group = [p for p in pairs if p["layout"] == layout]
        common = [p for p in group if p["common_eligible"]]
        by_layout[str(layout)] = {"pairs":len(group),"common_eligible":len(common),
            "left_progress":stats([p["left_gain"] for p in common]),
            "right_progress":stats([p["right_gain"] for p in common]),
            "delta_left_minus_right":stats([p["delta_left_minus_right"] for p in common])}
    layout_means = [r["delta_left_minus_right"]["mean"] for r in by_layout.values() if r["common_eligible"]]
    return {"paired_rows":len(pairs),"unique_states":len({(p["layout"],p["step"]) for p in pairs}),
        "both_certifiable":sum(p["both_certifiable"] for p in pairs),
        "common_eligible":sum(p["common_eligible"] for p in pairs),
        "left_only_eligible":sum(p["left_eligible"] and not p["right_eligible"] for p in pairs),
        "right_only_eligible":sum(p["right_eligible"] and not p["left_eligible"] for p in pairs),
        "norm_mismatch_pairs":sum(not p["matched_norm"] for p in pairs),
        "left_positive_only":sum(p["left_positive_only"] for p in pairs),
        "right_positive_only":sum(p["right_positive_only"] for p in pairs),
        "common_gain_difference":stats([p["delta_left_minus_right"] for p in pairs if p["common_eligible"]]),
        "layout_balanced_mean_gain_difference":float(np.mean(layout_means)) if layout_means else None,
        "layouts_with_common_eligible":len(layout_means),"per_layout":by_layout,"paired_evidence":pairs}


def review_report(source):
    layouts = source["audit_layouts"]
    if layouts != source["partitions"]["audit"] or source["probe_epsilons"] != [.01,.05,.10] or source["parameter_step"] != .01:
        raise ValueError("Fixed Gate2 protocol differs")
    models = source["models"]
    if set(models) != {"ordinary","paired"}:
        raise ValueError("Both registered models must be retained")
    reports = {}
    for name,model in models.items():
        action = model["action_probes"]
        index_rows(action,("layout","step","epsilon","direction"))
        if any(r["layout"] not in layouts or r["epsilon"] not in source["probe_epsilons"] or r["direction"] not in ("critic","random","goal") for r in action):
            raise ValueError("Unregistered action probe")
        comparisons,action_summary = {},{}
        for epsilon in source["probe_epsilons"]:
            group = [r for r in action if r["epsilon"] == epsilon]
            directions = {d:[r for r in group if r["direction"] == d] for d in ("critic","random","goal")}
            action_summary[str(epsilon)] = {d:{"overall":coverage(rows),
                "per_layout":{str(g):coverage([r for r in rows if r["layout"] == g]) for g in layouts}} for d,rows in directions.items()}
            comparisons[str(epsilon)] = {"critic_minus_"+d:paired(directions["critic"],directions[d],layouts,action=True) for d in ("random","goal")}
        parameters = model["parameter_probe"]["heldout_outcomes"]
        index_rows(parameters,("layout","step"))
        if any(r["layout"] not in layouts for r in parameters):
            raise ValueError("Unregistered parameter probe")
        if {(r["layout"],r["step"]) for r in parameters} != {(r["layout"],r["step"]) for r in action}:
            raise ValueError("Action/parameter states differ")
        risk_cases = [r for r in parameters if r["risk_proxy_regression"] is True]
        metadata = {k:model["parameter_probe"][k] for k in ("gradient_norm","calibration_states","calibration_layouts","per_agent_gradient_cosine","gradient_cancellation_ratio","parameter_step")}
        reports[name] = {"action_by_epsilon":action_summary,"action_matched_comparisons":comparisons,
            "parameter_probe":{"overall":coverage(parameters),"per_layout":{str(g):coverage([r for r in parameters if r["layout"] == g]) for g in layouts},
                "risk_regression_cases":risk_cases,"calibration_direction_metadata":metadata},
            "predicted_positive_but_absorbed":model["predicted_positive_but_absorbed"]}
    model_comparisons = {str(e):paired([r for r in models["paired"]["action_probes"] if r["direction"] == "critic" and r["epsilon"] == e],
        [r for r in models["ordinary"]["action_probes"] if r["direction"] == "critic" and r["epsilon"] == e],layouts,action=True) for e in source["probe_epsilons"]}
    return {"models":reports,"paired_minus_ordinary_action":model_comparisons,
        "paired_minus_ordinary_parameter":paired(models["paired"]["parameter_probe"]["heldout_outcomes"],models["ordinary"]["parameter_probe"]["heldout_outcomes"],layouts),
        "analysis_contract":["All three predeclared epsilon values retained separately; no best-scale selection or refitting.",
            "Pairs joined by layout/step/epsilon, not by row order; duplicate or missing keys fail.",
            "Net comparisons use the intersection of certifiable, risk-nonworse outcomes for BOTH directions; coverage discordance remains visible.",
            "Equal layout weighting averages within-layout mean differences over observed eligible layouts; missing layouts are not imputed as zero.",
            "Certification is the stored observable one-step predicate; losses are not observed collisions. Uncertifiable is not safe.",
            "Repeated scales/states within layouts are dependent. Descriptive comparisons have no significance or formal efficacy claim.",
            "Common eligibility depends on both candidate outcomes, so conditional gain contrasts do not establish unconditional benefit.",
            "No matched kinematic-gradient control exists in this report; no claim of superiority over that heuristic.",
            "Gradient cosine/cancellation are observed metadata, not causal proof of the shared-parameter failure mechanism."],
        "status":"RESEARCH_REVIEW_REQUIRED","actor_training_authorized":False,"pilot_authorized":False,
        "refit_authorized":False,"formal_validation_authorized":False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate2-dir",required=True)
    parser.add_argument("--gate1-dir",required=True)
    parser.add_argument("--critic-dir",required=True)
    parser.add_argument("--output-dir",required=True)
    args = parser.parse_args(argv)
    gate2,gate1,critic = map(resolve,(args.gate2_dir,args.gate1_dir,args.critic_dir))
    protected = [gate2,gate1,critic,resolve("results/ppcsa_seed23/gate0")]
    output = resolve(args.output_dir).resolve()
    if any(output == p.resolve() or p.resolve() in output.parents for p in protected):
        raise ValueError("Review output must be separate from original artifacts")
    before = {str(p):file_sha(p) for d in protected for p in d.rglob("*") if p.is_file()}
    manifest = read_manifest(gate2)
    read_manifest(gate1)
    read_manifest(critic)
    if manifest["gate1_manifest_sha256"] != file_sha(gate1/"manifest.json") or manifest["critic_manifest_sha256"] != file_sha(critic/"manifest.json"):
        raise ValueError("Gate2 source manifests differ")
    source = json.loads((gate2/"ppcsa_gradient_report.json").read_text())
    report = {"schema_version":"PPCSAGate2PairedLayoutReviewV1",**review_report(source),
        "source_file_sha256":before,"source_gate2_manifest_sha256":file_sha(gate2/"manifest.json"),
        "source_gradient_report_sha256":file_sha(gate2/"ppcsa_gradient_report.json"),
        "script_sha256":file_sha(__file__),"new_rollouts":False,"model_execution":False}
    after = {str(p):file_sha(p) for d in protected for p in d.rglob("*") if p.is_file()}
    if after != before:
        raise RuntimeError("Original experiment files changed during review")
    report["source_artifacts_unchanged"] = True
    output = new_output(output)
    (output/"paired_layout_review.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    lines = ["# Gate 2 paired layout review","","Offline existing-result review; no model execution, fitting or Actor update.","",
        "## Shared parameter update","","| Layout | Ordinary eligible states | Ordinary net progress | Paired eligible states | Paired net progress |",
        "|---|---:|---:|---:|---:|"]
    for layout in source["audit_layouts"]:
        a,b = (report["models"][m]["parameter_probe"]["per_layout"][str(layout)]["eligible_progress"] for m in ("ordinary","paired"))
        lines.append(f"| {layout} | {a['count']} | {a['net']} | {b['count']} | {b['net']} |")
    lines += ["","## Matched action comparisons","","| Model | Epsilon | Comparison | Common eligible states | Mean gain difference | Equal-layout mean difference |",
        "|---|---:|---|---:|---:|---:|"]
    for model,m in report["models"].items():
        for epsilon,comparisons in m["action_matched_comparisons"].items():
            for comparison,r in comparisons.items():
                lines.append(f"| {model} | {epsilon} | {comparison} | {r['common_eligible']} | {r['common_gain_difference']['mean']} | {r['layout_balanced_mean_gain_difference']} |")
    lines += ["","## Interpretation boundaries",""]+["- "+s for s in report["analysis_contract"]]
    lines += ["","Gate3/Pilot and Actor training remain blocked pending research review. Full paired rows, coverage, risks and provenance: `paired_layout_review.json`."]
    (output/"paired_layout_review.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"GATE2_REVIEW=RESEARCH_REVIEW_REQUIRED; SOURCE_UNCHANGED=True; DETAIL_LOG={output/'paired_layout_review.json'}")
    return report


if __name__ == "__main__":
    main()
