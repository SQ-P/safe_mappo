"""User-run offline quality audit; never run models, Shadow or gradient export."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.diagnose_ppcsa_parameter_sensitivity import bind_sources
from scripts.review_ppcsa_common_veto import unchanged
from scripts.diagnose_ppcsa_direction_feasibility import file_sha, json_safe
from src.algorithms.ppcsa_parameter_sensitivity import protocol, parameterization, digest
from src.algorithms.ppcsa_sensitivity_quality import audit_quality, validate_structure

IMPLEMENTATION_FILES = (Path(__file__).resolve(), ROOT/"src/algorithms/ppcsa_sensitivity_quality.py")


def resolve(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT/path).resolve()


def bind_quality_sources(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    recorded = report.get("input_file_sha256", {})
    def unique(name):
        matches = [resolve(p) for p in recorded if Path(p).name == name]
        if len(matches) != 1:
            raise ValueError("Missing or ambiguous historical source: "+name)
        return matches[0]
    a, f, c = map(unique, ("shared_admission_report.json", "direction_feasibility_report.json", "common_veto_report.json"))
    admission, protected, _, bindings = bind_sources(a, f, c)
    if recorded != protected or report.get("bindings") != bindings:
        raise ValueError("Measurement/historical source SHA binding differs")
    if (report.get("source_actor_unchanged") is not True or report.get("source_adapter_unchanged") is not True
            or report.get("source_artifacts_unchanged") is not True
            or report.get("used_partitions") != ["fit", "calibration"] or report.get("audit_used") is not False
            or report.get("direction_generated") is not False or report.get("actual_update_accepted") is not False
            or report.get("empirical_derivatives_verified") is not False):
        raise ValueError("Measurement isolation/scope evidence missing")
    implementations = report.get("implementation_sha256")
    if not isinstance(implementations, dict) or not implementations:
        raise ValueError("Measurement implementation binding missing")
    for name, sha in implementations.items():
        source = resolve(name); source.relative_to(ROOT.resolve())
        if file_sha(source) != sha:
            raise ValueError("Historical measurement implementation drift")
        protected[str(source)] = sha
    frozen_path = path.parent/"measurement_protocol.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if (frozen.get("protocol") != protocol() or frozen.get("protocol_sha256") != digest(protocol())
            or frozen.get("parameterization_sha256") != digest(parameterization())
            or frozen.get("input_file_sha256") != recorded or frozen.get("implementation_sha256") != implementations):
        raise ValueError("Frozen measurement protocol/source binding differs")
    protected[str(frozen_path)] = file_sha(frozen_path)
    protected[str(path)] = file_sha(path)
    baselines = {row["key"]: row["baseline"] for row in admission["candidates"]["ordinary"]["raw_state_evidence"]}
    common = json.loads(c.read_text(encoding="utf-8"))["common_veto_constraints"]
    return report, admission["fixed_state_registry"], baselines, common, protected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensitivity-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    path, output = resolve(args.sensitivity_report), resolve(args.output_dir)
    path.relative_to(ROOT.resolve()); output.relative_to(ROOT.resolve())
    report, registry, baselines, common, protected = bind_quality_sources(path)
    if output.exists() or any(output == Path(p).parent or Path(p).parent in output.parents for p in protected):
        raise ValueError("Output must be new and outside historical source/artifact directories")
    if args.preflight_only:
        coverage = validate_structure(report, registry, baselines)
        unchanged(protected)
        print(f"PREFLIGHT=PASS; STATES={coverage['state_count']}; AXES={coverage['axis_count']}; RISK_ROWS={coverage['risk_constraint_count']}; QUALITY_AGGREGATION=NOT_RUN; MODELS=NOT_EXECUTED; SHADOW_CALLS=0; OUTPUT_CREATED=NO")
        return {"status": "READY", "diagnostic_run": False}
    result = audit_quality(report, registry, baselines, common)
    unchanged(protected)
    result.update(source_sensitivity_report_sha256=protected[str(path)], input_file_sha256=protected,
        measurement_protocol_sha256=report["measurement_protocol_sha256"],
        parameterization_sha256=report["parameterization_sha256"], source_artifacts_unchanged=True,
        implementation_sha256={str(p): file_sha(p) for p in IMPLEMENTATION_FILES}, used_partitions=["fit", "calibration"], audit_used=False)
    body = json.dumps(json_safe(result), indent=2, allow_nan=False)
    lines = ["# PPCSA parameter-sensitivity quality audit", "", "Status: **"+result["status"]+"**.",
        "Original measurement status: **"+result["measurement_status_unchanged"]+"** (unchanged).", "",
        "| Observable | Count | Stable flat observed | Small nonzero unresolved | Instability observed | Consistent nonzero | Invalid |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for kind in ("progress", "risk"):
        summary = result[kind+"_summary"]; counts = summary["response_class_counts"]
        lines.append("| "+" | ".join([kind, str(summary["count"])]+[str(counts.get(name, 0)) for name in
            ("OBSERVED_STABLE_FLAT", "RESOLUTION_LIMITED_NONZERO", "INSTABILITY_OBSERVED", "LOCAL_RESPONSE_CONSISTENT", "INVALID_MEASUREMENT")])+" |")
    lines += ["", "## Original axis-level vetoes (nonexclusive)", "", "| Flag | Count |", "|---|---:|"]
    lines += [f"| {flag} | {count} |" for flag, count in result["original_axis_flag_counts_nonexclusive"].items()]
    lines += ["", "## Common-veto coverage", "", "| Constraint | Coordinates retained |", "|---|---:|"]
    lines += [f"| {row['constraint_id']} | {row['observed_coordinates']}/6 |" for row in result["common_veto_coverage"]]
    lines += ["", "## Evidence boundaries", ""]+["- "+text for text in result["limitations"]]
    output.mkdir(parents=True)
    (output/"sensitivity_quality_report.json").write_text(body, encoding="utf-8")
    (output/"sensitivity_quality_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"SENSITIVITY_QUALITY={result['status']}; ORIGINAL_EXPORT=UNCHANGED; SHADOW_CALLS=0; DETAIL_LOG={output/'sensitivity_quality_report.json'}")
    return result


if __name__ == "__main__":
    main()
