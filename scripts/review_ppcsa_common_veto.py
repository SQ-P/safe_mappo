"""User-run read-only common-veto review; no parameter search or model calls."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.diagnose_ppcsa_direction_feasibility import bind_report, file_sha
from src.algorithms.ppcsa_common_veto_diagnostics import review_common_veto, validate_evidence

IMPLEMENTATION_FILES = (Path(__file__).resolve(), ROOT/"src/algorithms/ppcsa_common_veto_diagnostics.py")


def resolve(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT/path).resolve()


def bind_inputs(admission_path, feasibility_path):
    admission, protected = bind_report(admission_path)
    feasibility = json.loads(feasibility_path.read_text(encoding="utf-8"))
    if feasibility.get("source_admission_report_sha256") != protected[str(admission_path)]:
        raise ValueError("Feasibility/admission SHA binding mismatch")
    if feasibility.get("input_file_sha256") != protected:
        raise ValueError("Feasibility historical input bindings differ")
    if (feasibility.get("source_artifacts_unchanged") is not True
            or feasibility.get("model_execution") is not False
            or feasibility.get("parameter_reconstruction") is not False
            or feasibility.get("new_shadow_calls") != 0
            or feasibility.get("used_partitions") != ["fit", "calibration"]
            or feasibility.get("partitions") != admission["partitions"]):
        raise ValueError("Feasibility source integrity/scope mismatch")
    implementation = feasibility.get("implementation_sha256")
    if not isinstance(implementation, dict) or not implementation:
        raise ValueError("Missing historical feasibility implementation binding")
    for name, expected in implementation.items():
        path = resolve(name)
        path.relative_to(ROOT.resolve())
        actual = file_sha(path)
        if actual != expected:
            raise ValueError("Historical feasibility implementation SHA mismatch")
        protected[str(path)] = actual
    protected[str(feasibility_path)] = file_sha(feasibility_path)
    return admission, feasibility, protected


def unchanged(protected):
    if any(file_sha(path) != expected for path, expected in protected.items()):
        raise RuntimeError("Historical source changed during read-only validation/review")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission-report", required=True)
    parser.add_argument("--feasibility-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    admission_path, feasibility_path, output = map(resolve, (args.admission_report, args.feasibility_report, args.output_dir))
    for path in (admission_path, feasibility_path, output):
        path.relative_to(ROOT.resolve())
    admission, feasibility, protected = bind_inputs(admission_path, feasibility_path)
    if output.exists() or any(Path(p).parent == output or Path(p).parent in output.parents for p in protected):
        raise ValueError("Output must be new and outside historical artifact/source directories")
    if args.preflight_only:
        validation = validate_evidence(admission, feasibility)
        unchanged(protected)
        print(f"PREFLIGHT=PASS; FIXED_STATES={validation['fixed_state_count']}; PROGRESS_ROWS={validation['progress_constraint_count']}; RISK_ROWS={validation['risk_constraint_count']}; COMMON_VETOES={len(validation['common_veto_constraints'])}; MODELS=NOT_EXECUTED; ATTRIBUTION=NOT_RUN; OUTPUT_CREATED=NO")
        return {"status": "READY", "diagnostic_run": False}
    result = review_common_veto(admission, feasibility)
    result.update(schema_version="PPCSACommonVetoReviewV1", input_file_sha256=protected,
        source_admission_report_sha256=protected[str(admission_path)],
        source_feasibility_report_sha256=protected[str(feasibility_path)],
        implementation_sha256={str(p): file_sha(p) for p in IMPLEMENTATION_FILES},
        used_partitions=["fit", "calibration"], independent_validation=False)
    unchanged(protected)
    result["source_artifacts_unchanged"] = True
    body = json.dumps(result, indent=2, allow_nan=False)
    lines = ["# PPCSA common-veto review", "", "Status: **"+result["status"]+"**.",
        "Parameter sensitivities: **PARAMETER_SENSITIVITIES_UNAVAILABLE**. No new direction generated.", "",
        "| Constraint | Ordinary endpoint delta | Paired endpoint delta | Kinematic endpoint delta | Causal mechanism |",
        "|---|---:|---:|---:|---|"]
    for entry in result["common_veto_details"]:
        field = "risk_delta" if entry["kind"] == "risk" else "progress_delta"
        values = [str(entry["directions"][name][field]) for name in result["direction_order"]]
        lines.append("| "+" | ".join([entry["constraint_id"]]+values+["UNIDENTIFIED"])+" |")
    lines += ["", "Full action-stage arrays, within-pipeline interventions and endpoint changes are in `common_veto_report.json`.",
        "", "## Evidence boundaries", ""]+["- "+text for text in result["limitations"]]
    output.mkdir(parents=True)
    (output/"common_veto_report.json").write_text(body, encoding="utf-8")
    (output/"common_veto_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"COMMON_VETO={result['status']}; PARAMETER_SENSITIVITIES=UNAVAILABLE; DIRECTION=NOT_GENERATED; SHADOW_CALLS=0; DETAIL_LOG={output/'common_veto_report.json'}")
    return result


if __name__ == "__main__":
    main()
