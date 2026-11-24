"""Read-only rejection attribution and endpoint-secant LP; no model or Shadow calls."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_FILES = (Path(__file__).resolve(),ROOT/"src/algorithms/ppcsa_direction_feasibility.py")
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from src.algorithms.ppcsa_direction_feasibility import analyze_admission


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(path):
    path = Path(path)
    return (path if path.is_absolute() else ROOT/path).resolve()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def json_safe(value):
    # Preserve invalid evidence explicitly in an INVALID_EVIDENCE report rather
    # than emitting non-standard JSON or converting it into a usable coefficient.
    if isinstance(value,float) and not math.isfinite(value):
        return {"invalid_numeric_value":repr(value)}
    if isinstance(value,dict):
        return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,list):
        return [json_safe(v) for v in value]
    return value


def bind_report(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != "PPCSASharedUpdateAdmissionV1":
        raise ValueError("Shared admission report schema mismatch")
    if report.get("audit_used") is not False or report.get("used_partitions") != ["fit","calibration"]:
        raise ValueError("Only exposed Fit/Calibration evidence is authorized")
    for name in ("source_actor_unchanged","source_critics_unchanged","source_adapter_unchanged","source_artifacts_unchanged"):
        if report.get(name) is not True:
            raise ValueError("Source integrity was not established")
    if digest(report["fixed_state_registry"]) != report["fixed_state_registry_sha256"]:
        raise ValueError("Fixed state registry hash mismatch")
    protected = {str(path):file_sha(path)}
    hashes = {**report["input_file_sha256"],**report["implementation_sha256"],**report["source_sha256"],
              report["checkpoint"]:report["checkpoint_sha256"]}
    for name,expected in hashes.items():
        source = resolve(name)
        try:
            source.relative_to(ROOT.resolve())
        except ValueError:
            raise ValueError("Evidence path is outside this workspace")
        actual = file_sha(source)
        if actual != expected:
            raise ValueError("Source evidence/implementation hash mismatch: "+name)
        protected[str(source)] = actual
    # The referenced historical Phase1 report must be one of the bound inputs.
    if report["source_mechanism_report_sha256"] not in report["input_file_sha256"].values():
        raise ValueError("Historical mechanism report binding missing")
    allowed = set(report["partitions"]["fit"]+report["partitions"]["calibration"])
    if allowed.intersection(report["partitions"]["audit"]):
        raise ValueError("Partition overlap")
    for candidate in report["candidates"].values():
        if any(row["layout"] not in allowed for row in candidate["raw_state_evidence"]):
            raise ValueError("Audit/unregistered state cannot participate in proxy selection")
        metadata = candidate["direction_metadata"]
        if metadata.get("source") != "Calibration only" or any(int(k.split(':')[0]) not in report["partitions"]["calibration"] for k in metadata["teacher_keys"]):
            raise ValueError("Fixed directions were not constructed only from Calibration")
    return report,protected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission-report",required=True)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--preflight-only",action="store_true")
    args = parser.parse_args(argv)
    path,output = resolve(args.admission_report),resolve(args.output_dir)
    report,before = bind_report(path)
    if output.exists() or any(output == Path(name).parent or Path(name).parent in output.parents for name in before):
        raise ValueError("Output must be new and outside all historical artifact/source directories")
    started = perf_counter()
    result = analyze_admission(report,solve=not args.preflight_only)
    if args.preflight_only:
        if result["evidence_errors"]:
            raise ValueError("Invalid or incomplete proxy evidence: "+';'.join(result["evidence_errors"]))
        print(f"PREFLIGHT=PASS; FIXED_STATES={result['fixed_state_count']}; PROGRESS_ROWS={result['complete_progress_constraint_count']}; RISK_ROWS={result['complete_risk_constraint_count']}; MODELS=NOT_EXECUTED; SHADOW_CALLS=0; SOLVER=NOT_RUN; OUTPUT_CREATED=NO")
        return {"status":"READY","diagnostic_run":False}
    result.update(schema_version="PPCSADirectionSecantFeasibilityV1",status=result["proxy"]["status"],
        source_admission_report_sha256=before[str(path)],input_file_sha256=before,
        implementation_sha256={str(p):file_sha(p) for p in IMPLEMENTATION_FILES},
        fixed_state_registry=report["fixed_state_registry"],fixed_state_registry_sha256=report["fixed_state_registry_sha256"],
        partitions=report["partitions"],used_partitions=["fit","calibration"],parameter_step=.01,
        model_execution=False,parameter_reconstruction=False,new_shadow_calls=0,new_collection=False,
        independent_validation=False,wall_seconds=perf_counter()-started)
    if any(file_sha(name)!=sha for name,sha in before.items()):
        raise RuntimeError("Historical evidence changed during read-only analysis")
    result["source_artifacts_unchanged"] = True
    output.mkdir(parents=True)
    (output/"direction_feasibility_report.json").write_text(json.dumps(json_safe(result),indent=2,allow_nan=False),encoding="utf-8")
    lines = ["# PPCSA rejection attribution and endpoint-secant feasibility","",
        "Status: **"+result["status"]+"**. No actual direction reconstruction, model execution or new Shadow evaluation.","",
        "| Direction | Progress-regression states | Risk-regression states | Negative progress magnitude median | Positive risk delta median |",
        "|---|---:|---:|---:|---:|"]
    for name,a in result["attribution"].items():
        counts = a["state_rejection_counts_nonexclusive"]
        lines.append(f"| {name} | {counts.get('executed_progress_regression',0)} | {counts.get('per_track_risk_regression',0)} | {a['negative_progress_magnitudes'].get('median')} | {a['positive_risk_deltas'].get('median')} |")
    lines += ["","## Proxy linear program","",json.dumps({k:v for k,v in result["proxy"].items() if k not in ('constraints','feasible_vertices_exact','proxy_state_progress','proxy_track_risk_changes')},indent=2),
        "","## Evidence boundaries",""]+["- "+s for s in result["limitations"]]
    lines += ["","All state/track attributions, exact constraint coefficients and solver witnesses: `direction_feasibility_report.json`."]
    (output/"direction_feasibility_report.md").write_text('\n'.join(lines)+'\n',encoding="utf-8")
    print(f"DIRECTION_FEASIBILITY={result['status']}; REAL_DIRECTION=UNVERIFIED; SHADOW_CALLS=0; DETAIL_LOG={output/'direction_feasibility_report.json'}")
    return result


if __name__ == "__main__":
    main()
