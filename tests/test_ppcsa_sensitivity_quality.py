from copy import deepcopy
import json

import pytest

from src.algorithms.ppcsa_parameter_sensitivity import analyze_axis, measure_cohort, protocol, parameterization, digest
from src.algorithms.ppcsa_sensitivity_quality import classify_axis, audit_quality, validate_structure
from tests.test_ppcsa_parameter_sensitivity import point, samples, bindings


def test_flat_small_nonzero_and_resolved_responses_remain_different_without_zero_proof():
    base = point()
    flat = classify_axis(base, samples(lambda x: point()), ["progress:1:0", "risk:1:0:0:3"])
    assert all(m["response_class"] == "OBSERVED_STABLE_FLAT" for m in flat["metrics"])
    assert all(m["original_resolution_veto"] and not m["zero_derivative_proven"] for m in flat["metrics"])
    tiny = classify_axis(base, samples(lambda x: point(x, progress=1e-5*x, risk=10+1e-5*x)), ["progress:1:0", "risk:1:0:0:3"])
    assert all(m["response_class"] == "RESOLUTION_LIMITED_NONZERO" for m in tiny["metrics"])
    assert all(m["amplitude_class"] == "SUB_RESOLUTION_NONZERO" for m in tiny["metrics"])
    active = classify_axis(base, samples(), ["progress:1:0", "risk:1:0:0:3"])
    assert all(m["response_class"] == "LOCAL_RESPONSE_CONSISTENT" for m in active["metrics"])
    assert not any(m["gradient_exported"] for group in (flat, tiny, active) for m in group["metrics"])


def test_zero_risk_platform_is_observed_not_causally_inferred():
    base = point(risk=0.)
    result = classify_axis(base, samples(lambda x: point(x, risk=0.)), ["progress:1:0", "risk:1:0:0:3"])
    risk = result["metrics"][1]
    assert risk["risk_baseline_zero"] and risk["response_class"] == "OBSERVED_STABLE_FLAT"
    assert risk["baseline"] == 0 and not risk["zero_derivative_proven"]


@pytest.mark.parametrize("kind", ["kink", "jump", "scale", "heldout", "repeat", "branch"])
def test_nonexclusive_instability_evidence_is_preserved(kind):
    data = samples()
    if kind == "kink": data = samples(lambda x: point(x, progress=abs(x)))
    if kind == "jump": data = samples(lambda x: point(x, progress=float(x > 0)))
    if kind == "scale":
        for sign in ("minus", "plus"):
            for p in data[0][sign]: p["executed_progress"] *= 3
    if kind == "heldout":
        for p in data[2]["plus"]: p["executed_progress"] += .01
    if kind == "repeat": data[0]["plus"][1]["executed_progress"] += 1e-15
    if kind == "branch": data[0]["plus"][0]["safety_evidence"]["stage_checks"]["initial_guard"]["pair_iterations"] = 2
    result = classify_axis(point(), data, ["progress:1:0", "risk:1:0:0:3"])
    progress = result["metrics"][0]
    assert progress["response_class"] == "INSTABILITY_OBSERVED"
    expected = {"kink": "ONE_SIDED_INCONSISTENCY", "jump": "ONE_SIDED_INCONSISTENCY",
        "scale": "SCALE_INCONSISTENCY", "heldout": "HELDOUT_SCALE_PREDICTION_MISMATCH",
        "repeat": "REPEATABILITY_MISMATCH", "branch": "STAGE_SWITCH"}[kind]
    assert expected in progress["instability_flags"]
    if kind == "branch":
        assert result["stage_switches"][0]["changed_signature_fields"] == ["initial_guard"]


@pytest.mark.parametrize("bad", ["track", "unknown", "nonfinite", "missing_sign", "stage"])
def test_invalid_measurements_are_never_called_stable_flat(bad):
    data = samples()
    p = data[0]["plus"][0]
    if bad == "track": p["tracks"][0]["track_id"] = 8
    if bad == "unknown": p["risk_proxy"] = [None]
    if bad == "nonfinite": p["executed_progress"] = float('nan')
    if bad == "missing_sign": data[0]["plus"] = []
    if bad == "stage": p["stage_actions"].pop("obstacle_ttc")
    result = classify_axis(point(), data, ["progress:1:0", "risk:1:0:0:3"])
    assert len(result["metrics"]) == 2
    assert all(m["response_class"] == "INVALID_MEASUREMENT" for m in result["metrics"])


def measurement_fixture():
    registry = [{"key": "1:0", "state_index": 0}, {"key": "2:0", "state_index": 1}]
    baselines = {row["key"]: point() for row in registry}
    def evaluate(key, axis, delta):
        p = (point() if axis == 0 else point(delta, progress=1e-5*delta, risk=10+1e-5*delta) if axis == 1
             else point(delta, progress=abs(delta)) if axis == 2 else point(delta))
        p["parameter_probe"] = {"axis": axis, "requested_delta": delta, "actual_float32_delta": delta}
        return p
    report = measure_cohort(registry, baselines, evaluate, bindings())
    report["schema_version"] = "PPCSAParameterSensitivityDiagnosticV1"
    return report, registry, baselines


def test_full_aggregation_reproduces_original_vetoes_and_retains_common_constraints():
    report, registry, baselines = measurement_fixture(); original = deepcopy(report)
    common = ["progress:1:0", "risk:2:0:0:3"]
    result = audit_quality(report, registry, baselines, common)
    assert report == original and result["status"] == "QUALITY_AUDIT_COMPLETE"
    assert result["all_original_axis_vetoes_reproduced"]
    assert result["original_axis_flag_counts_nonexclusive"] == result["reproduced_axis_flag_counts_nonexclusive"]
    assert result["coverage"]["state_count"] == 2 and result["coverage"]["axis_count"] == 12
    assert result["progress_summary"]["count"] == result["risk_summary"]["count"] == 12
    assert len(result["per_risk_constraint"]) == 2 and len(result["observable_details"]) == 24
    assert all(r["observed_coordinates"] == 6 and r["original_constraint_retained"] for r in result["common_veto_coverage"])
    assert result["measurement_status_unchanged"] == "MEASUREMENT_FIDELITY_UNRESOLVED"
    assert not result["gradient_exported"] and not result["direction_generated"] and result["new_shadow_calls"] == 0
    for kind in ("progress", "risk"):
        assert sum(result[kind+"_summary"]["response_class_counts"].values()) == 12


def test_rewritten_original_diagnostic_is_rejected_and_missing_state_cannot_be_excluded():
    report, registry, baselines = measurement_fixture()
    report["states"][0]["axes"][0]["flags"] = []
    result = audit_quality(report, registry, baselines, [])
    assert result["status"] == "INVALID_EVIDENCE" and result["evidence_errors"]
    report["states"].pop()
    with pytest.raises(ValueError, match="measured states"):
        audit_quality(report, registry, baselines, [])


def test_recorded_invalid_track_is_retained_and_baseline_mismatch_is_not_stable():
    report, registry, baselines = measurement_fixture()
    state = report["states"][0]
    data = state["axes"][0]["samples"]
    data[0]["plus"][0]["risk_proxy"] = [None]
    state["axes"][0] = analyze_axis(state["baseline_repeats"][0], data)
    state["baseline_repeats"][1]["executed_progress"] += .1
    result = audit_quality(report, registry, baselines, [])
    assert result["status"] == "QUALITY_AUDIT_HAS_INVALID_MEASUREMENTS"
    assert len(result["observable_details"]) == 24
    assert result["progress_summary"]["response_class_counts"]["INVALID_MEASUREMENT"] == 1
    assert result["per_state"][0]["baseline_issues"] == ["BASELINE_REPEATABILITY_MISMATCH"]
    assert validate_structure(report, registry, baselines)["state_count"] == 2


def cli_fixture(tmp_path, monkeypatch):
    from tests.test_ppcsa_common_veto import evidence_fixture
    import scripts.review_ppcsa_common_veto as common_cli
    import scripts.diagnose_ppcsa_direction_feasibility as feasibility_cli
    import scripts.diagnose_ppcsa_parameter_sensitivity as sensitivity_cli
    import scripts.review_ppcsa_sensitivity_quality as quality_cli
    from src.algorithms.ppcsa_direction_feasibility import analyze_admission
    root = tmp_path/"project"; root.mkdir()
    for module in (common_cli, feasibility_cli, sensitivity_cli, quality_cli): monkeypatch.setattr(module, "ROOT", root)
    source = root/"src/frozen.txt"; source.parent.mkdir(); source.write_text("frozen test implementation")
    monkeypatch.setattr(common_cli, "IMPLEMENTATION_FILES", (source,))
    phase1 = root/"results/phase1/source.json"; phase1.parent.mkdir(parents=True); phase1.write_text("{}")
    parent = root/"results/parent/source.stub"; parent.parent.mkdir(); parent.write_bytes(b"not loaded")
    a, _ = evidence_fixture()
    for candidate in a["candidates"].values():
        for row in candidate["raw_state_evidence"]:
            for endpoint in ("baseline", "candidate"):
                row[endpoint]["safety_evidence"] = point()["safety_evidence"]
    a.update(schema_version="PPCSASharedUpdateAdmissionV1", audit_used=False, used_partitions=["fit", "calibration"],
        source_actor_unchanged=True, source_critics_unchanged=True, source_adapter_unchanged=True, source_artifacts_unchanged=True,
        input_file_sha256={str(phase1): quality_cli.file_sha(phase1)}, implementation_sha256={str(source): quality_cli.file_sha(source)},
        source_sha256={}, checkpoint=str(parent), checkpoint_sha256=quality_cli.file_sha(parent),
        source_mechanism_report_sha256=quality_cli.file_sha(phase1), partitions={"fit": [121000], "calibration": [121024], "audit": [121032]},
        hidden_dim=128, adapter_rank=16, adapter_scale=.15)
    admission = root/"results/admission/shared_admission_report.json"; admission.parent.mkdir(); admission.write_text(json.dumps(a))
    _, before = feasibility_cli.bind_report(admission)
    f = analyze_admission(a)
    f.update(schema_version="PPCSADirectionSecantFeasibilityV1", status=f["proxy"]["status"],
        fixed_state_registry=a["fixed_state_registry"], fixed_state_registry_sha256=a["fixed_state_registry_sha256"],
        source_admission_report_sha256=quality_cli.file_sha(admission), input_file_sha256=before,
        implementation_sha256={str(source): quality_cli.file_sha(source)}, source_artifacts_unchanged=True,
        model_execution=False, parameter_reconstruction=False, new_shadow_calls=0, used_partitions=["fit", "calibration"], partitions=a["partitions"])
    feasibility = root/"results/feasibility/direction_feasibility_report.json"; feasibility.parent.mkdir(); feasibility.write_text(json.dumps(f))
    common_output = root/"results/common"
    common_cli.main(["--admission-report", str(admission), "--feasibility-report", str(feasibility), "--output-dir", str(common_output)])
    common = common_output/"common_veto_report.json"
    _, protected, _, source_bindings = sensitivity_cli.bind_sources(admission, feasibility, common)
    baselines = {r["key"]: r["baseline"] for r in a["candidates"]["ordinary"]["raw_state_evidence"]}
    def evaluate(key, axis, delta):
        p = deepcopy(baselines[key]); p["executed_progress"] += 2*delta; p["risk_proxy"][0] += delta
        p["parameter_probe"] = {"axis": axis, "requested_delta": delta, "actual_float32_delta": delta}
        return p
    result = measure_cohort(a["fixed_state_registry"], baselines, evaluate, source_bindings)
    implementation = {str(source): quality_cli.file_sha(source)}
    result.update(schema_version="PPCSAParameterSensitivityDiagnosticV1", input_file_sha256=protected,
        implementation_sha256=implementation, bindings=source_bindings, used_partitions=["fit", "calibration"], audit_used=False,
        source_actor_unchanged=True, source_adapter_unchanged=True, source_artifacts_unchanged=True)
    measurement = root/"results/measurement/parameter_sensitivity_report.json"; measurement.parent.mkdir(); measurement.write_text(json.dumps(result))
    (measurement.parent/"measurement_protocol.json").write_text(json.dumps({"protocol": protocol(),
        "protocol_sha256": digest(protocol()), "parameterization_sha256": digest(parameterization()),
        "input_file_sha256": protected, "implementation_sha256": implementation}))
    return quality_cli, measurement, root/"results/quality"


def test_synthetic_cli_preflight_does_not_aggregate_then_audit_does_not_change_sources(tmp_path, monkeypatch):
    cli, measurement, output = cli_fixture(tmp_path, monkeypatch)
    args = ["--sensitivity-report", str(measurement), "--output-dir", str(output)]
    audit = cli.audit_quality
    monkeypatch.setattr(cli, "audit_quality", lambda *a: (_ for _ in ()).throw(AssertionError("preflight aggregated")))
    before = cli.file_sha(measurement)
    assert not cli.main(args+["--preflight-only"])["diagnostic_run"] and not output.exists()
    monkeypatch.setattr(cli, "audit_quality", audit)
    result = cli.main(args)
    assert result["status"] == "QUALITY_AUDIT_COMPLETE" and result["source_artifacts_unchanged"]
    assert cli.file_sha(measurement) == before
    assert set(p.name for p in output.iterdir()) == {"sensitivity_quality_report.json", "sensitivity_quality_report.md"}
    assert not result["gradient_exported"] and result["new_shadow_calls"] == 0
    with pytest.raises(ValueError, match="new and outside"): cli.main(args)


def recorded_invalid_fixture(tmp_path, monkeypatch):
    cli, measurement, output = cli_fixture(tmp_path, monkeypatch)
    report = json.loads(measurement.read_text())
    state = report["states"][0]
    data = state["axes"][0]["samples"]
    data[0]["plus"][0]["risk_proxy"] = [None]
    state["axes"][0] = analyze_axis(state["baseline_repeats"][0], data)
    report.update(status="MEASUREMENT_INVALID", parameter_sensitivities=None)
    from src.algorithms.ppcsa_common_veto_diagnostics import validate_parameter_sensitivities
    report["sensitivity_contract"] = validate_parameter_sensitivities(None,
        progress_ids=report["progress_ids"], risk_ids=report["risk_ids"], bindings=report["bindings"])
    measurement.write_text(json.dumps(report))
    return cli, measurement, output


def test_cli_archives_reproducible_historical_invalid_measurement_without_flat_or_gradient(tmp_path, monkeypatch):
    cli, measurement, output = recorded_invalid_fixture(tmp_path, monkeypatch)
    args = ["--sensitivity-report", str(measurement), "--output-dir", str(output)]
    before = cli.file_sha(measurement)
    audit = cli.audit_quality
    monkeypatch.setattr(cli, "audit_quality", lambda *a: (_ for _ in ()).throw(AssertionError("preflight aggregated")))
    assert cli.main(args+["--preflight-only"])["status"] == "READY" and not output.exists()
    monkeypatch.setattr(cli, "audit_quality", audit)
    result = cli.main(args)
    assert result["status"] == "QUALITY_AUDIT_HAS_INVALID_MEASUREMENTS"
    assert result["all_original_axis_vetoes_reproduced"] and len(result["observable_details"]) == 24
    invalid = [m for m in result["observable_details"] if m["response_class"] == "INVALID_MEASUREMENT"]
    assert len(invalid) == 2 and all(m["errors"] and not m["gradient_exported"] for m in invalid)
    assert result["measurement_status_unchanged"] == "MEASUREMENT_INVALID" and not result["gradient_exported"]
    saved = json.loads((output/"sensitivity_quality_report.json").read_text())
    assert any(m["response_class"] == "INVALID_MEASUREMENT" for m in saved["observable_details"])
    assert cli.file_sha(measurement) == before and result["new_shadow_calls"] == 0


@pytest.mark.parametrize("bad", ["original_error", "false_invalid", "missing_repeat"])
def test_cli_rejects_nonreproducible_invalid_claim_or_structural_loss(tmp_path, monkeypatch, bad):
    cli, measurement, output = recorded_invalid_fixture(tmp_path, monkeypatch)
    report = json.loads(measurement.read_text())
    axis = report["states"][0]["axes"][0]
    if bad == "original_error": axis["errors"] = ["invented error"]
    if bad == "false_invalid": axis["samples"][0]["plus"][0]["risk_proxy"] = [1.001]
    if bad == "missing_repeat": axis["samples"][0]["plus"].pop()
    measurement.write_text(json.dumps(report))
    with pytest.raises(ValueError): cli.main(["--sensitivity-report", str(measurement), "--output-dir", str(output), "--preflight-only"])
    assert not output.exists()


@pytest.mark.parametrize("bad", ["source_sha", "implementation", "protocol", "state", "track", "axis_order", "sign"])
def test_cli_provenance_or_coverage_failure_stops_before_output(tmp_path, monkeypatch, bad):
    cli, measurement, output = cli_fixture(tmp_path, monkeypatch)
    r = json.loads(measurement.read_text())
    if bad == "source_sha": r["input_file_sha256"] = {}
    if bad == "implementation": r["implementation_sha256"] = {str(measurement): "0"*64}
    if bad == "protocol": r["protocol"]["scales"][0] = .1
    if bad == "state": r["states"].pop()
    if bad == "track": r["states"][0]["axes"][0]["samples"][0]["plus"][0]["tracks"][0]["track_id"] = 99
    if bad == "axis_order": r["states"][0]["axes"].reverse()
    if bad == "sign": r["states"][0]["axes"][0]["samples"][0]["plus"][0]["parameter_probe"]["requested_delta"] *= -1
    measurement.write_text(json.dumps(r))
    with pytest.raises(ValueError): cli.main(["--sensitivity-report", str(measurement), "--output-dir", str(output), "--preflight-only"])
    assert not output.exists()
