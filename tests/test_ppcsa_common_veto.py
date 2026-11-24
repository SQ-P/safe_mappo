from copy import deepcopy
import json

import pytest

from src.algorithms.ppcsa_common_veto_diagnostics import (
    STAGES, review_common_veto, validate_evidence, validate_parameter_sensitivities,
    check_declared_linear_direction)
from src.algorithms.ppcsa_direction_feasibility import analyze_admission
from src.algorithms.ppcsa_update_acceptance import CANDIDATES, candidate_admission
from scripts.diagnose_ppcsa_direction_feasibility import digest


def evidence_fixture():
    registry = [{"key": "121000:0", "state_index": 0, "state_fingerprint": "environment0"},
                {"key": "121024:0", "state_index": 1, "state_fingerprint": "environment1"}]
    candidates = {}
    for name in CANDIDATES:
        rows = []
        for i, fixed in enumerate(registry):
            base = {"supported": True, "hard_safe": True, "source_unchanged": True,
                "track_capture_unchanged": True, "snapshot_fingerprint": "shadow-composite-"+str(i),
                "executed_progress": 1., "risk_proxy": [1.],
                "tracks": [{"observer_id": 0, "track_id": 2, "velocity_valid": True,
                    "relative_position": [3., 0.], "relative_velocity": [-1., 0.], "radius": 1.}],
                "requested_actions": [[.5, 0.]], "executed_actions": [[.1, 0.]],
                "stage_actions": {s: [[.5 if n < 3 else .1, 0.]] for n, s in enumerate(STAGES)}}
            candidate = deepcopy(base)
            candidate["executed_progress"] += -.01 if i == 0 else .02
            candidate["risk_proxy"] = [1. + (.1 if i == 1 else 0.)]
            candidate["requested_actions"] = [[.6, 0.]]
            candidate["stage_actions"]["actor_raw"] = [[.6, 0.]]
            candidate["executed_actions"] = [[.15, 0.]]
            candidate["stage_actions"]["executed"] = [[.15, 0.]]
            rows.append({**fixed, "layout": int(fixed["key"].split(':')[0]), "step": 0,
                         "replay_match": True, "baseline": base, "candidate": candidate})
        decision = candidate_admission([r["key"] for r in registry], rows, [.01])
        candidates[name] = {**decision, "raw_state_evidence": rows, "direction_metadata": {
            "aggregate_gradient_norm": 1., "parameter_step": .01, "parameter_step_search": False,
            "direction_sha256": name, "source": "Calibration only", "teacher_keys": ["121024:0"]}}
    admission = {"fixed_state_registry": registry, "fixed_state_registry_sha256": digest(registry),
        "candidates": candidates, "candidate_registry": list(CANDIDATES), "parameter_step": .01,
        "execution_errors": [], "mechanism_conclusion_available": True}
    feasibility = analyze_admission(admission)
    feasibility.update(schema_version="PPCSADirectionSecantFeasibilityV1", status=feasibility["proxy"]["status"],
        fixed_state_registry=deepcopy(registry), fixed_state_registry_sha256=digest(registry))
    return admission, feasibility


def test_exact_common_veto_attribution_retains_full_coverage_without_causal_claims():
    admission, feasibility = evidence_fixture()
    original = deepcopy((admission, feasibility))
    result = review_common_veto(admission, feasibility)
    assert (admission, feasibility) == original
    assert result["fixed_state_count"] == result["progress_constraint_count"] == result["risk_constraint_count"] == 2
    assert len(result["constraints"]) == 4 and len(result["fixed_state_registry"]) == 2
    assert result["common_veto_constraints"] == ["progress:121000:0", "risk:121024:0:0:2"]
    assert result["parameter_sensitivities"]["status"] == "PARAMETER_SENSITIVITIES_UNAVAILABLE"
    assert result["common_veto_count"] == 2
    assert result["common_veto_details"][0]["registry_state_fingerprint"] == "environment0"
    assert result["common_veto_details"][0]["directions"]["ordinary"]["baseline_shadow_fingerprint"] == "shadow-composite-0"
    for entry in result["common_veto_details"]:
        assert entry["causal_mechanism_status"] == "UNIDENTIFIED"
        assert entry["all_three_strictly_violate"]
        actions = entry["directions"]["ordinary"]["actions"]
        assert actions["baseline"]["within_pipeline_changes"]["initial_joint_guard"]["delta"] == [[-.4, 0.]]
        assert actions["between_endpoints"]["requested"]["l2"] == pytest.approx(.1)
        assert actions["between_endpoints"]["executed"]["l2"] == pytest.approx(.05)
    track = result["common_veto_details"][1]["directions"]["kinematic"]
    assert track["track_index"] == 0 and track["track_context"]["track_id"] == 2
    assert track["baseline_risk"] == 1. and track["risk_delta"] == pytest.approx(.1)
    assert not result["actual_update_accepted"] and not result["optimizer_executed"]


@pytest.mark.parametrize("bad", ["coefficient", "common_list", "coverage", "registry", "missing_track",
    "track_context", "stage", "nonfinite_stage", "nonfinite_risk", "baseline", "fingerprint", "direction_order"])
def test_incomplete_or_rewritten_evidence_is_rejected_without_subset(bad):
    a, f = evidence_fixture()
    row = a["candidates"]["ordinary"]["raw_state_evidence"][0]
    if bad == "coefficient": f["proxy"]["constraints"][0]["normal_ge_zero_exact"][0] = "0"
    if bad == "common_list": f["proxy"]["universal_veto_constraints"].pop()
    if bad == "coverage": f["complete_risk_constraint_count"] -= 1
    if bad == "registry": f["fixed_state_registry"][0]["state_index"] = 8
    if bad == "missing_track": row["candidate"]["tracks"] = []
    if bad == "track_context": row["candidate"]["tracks"][0]["radius"] = 9.
    if bad == "stage": row["candidate"]["stage_actions"].pop("obstacle_ttc")
    if bad == "nonfinite_stage": row["candidate"]["stage_actions"]["obstacle_ttc"] = [[float('inf'), 0.]]
    if bad == "nonfinite_risk": row["candidate"]["risk_proxy"] = [float('nan')]
    if bad == "baseline": row["baseline"]["executed_progress"] = 2.
    if bad == "fingerprint": row["candidate"]["snapshot_fingerprint"] = "other"
    if bad == "direction_order": a["candidate_registry"].reverse()
    with pytest.raises(ValueError): validate_evidence(a, f)


def sensitivity_fixture():
    ids = {"progress_ids": ["progress:a", "progress:b"], "risk_ids": ["risk:a:0:2"],
           "bindings": {"admission_sha256": "1"*64, "registry_sha256": "2"*64, "checkpoint_sha256": "3"*64}}
    payload = {"schema_version": "PPCSAParameterSensitivitiesV1", "quantity": "executed_parameter_derivatives",
        "method": "parameter_finite_difference", "endpoint_secants_used": False, "bindings": ids["bindings"],
        "measurement_protocol_sha256": "4"*64, "parameterization_sha256": "5"*64,
        "parameter_order": [{"name": "adapter.up.bias", "shape": [2]}], "parameter_count": 2,
        "progress": [{"id": x, "gradient": [1., 1.]} for x in ids["progress_ids"]],
        "risk": [{"id": ids["risk_ids"][0], "gradient": [1., -1.]}]}
    return payload, ids


def test_parameter_contract_does_not_confuse_common_endpoint_veto_with_full_space():
    payload, ids = sensitivity_fixture()
    checked = validate_parameter_sensitivities(payload, **ids)
    assert checked["status"] == "DECLARED_PARAMETER_SENSITIVITIES_VALIDATED"
    assert not checked["empirical_derivatives_verified"] and not checked["direction_generated"]
    # Three old directions could all have positive first-coordinate risk response,
    # but an orthogonal second-coordinate update can improve G and reduce R.
    result = check_declared_linear_direction(payload, [0., .005], **ids)
    assert result["status"] == "LINEAR_POSITIVE_UNVERIFIED"
    assert not result["actual_update_accepted"] and not result["actual_safety_certified"]
    assert check_declared_linear_direction(payload, [0., 0.], **ids)["status"] == "ZERO_DIRECTION"
    assert check_declared_linear_direction(payload, [.005, 0.], **ids)["status"] == "LINEAR_CONSTRAINT_VIOLATION"
    assert check_declared_linear_direction(payload, [0., .02], **ids)["status"] == "LINEAR_CONSTRAINT_VIOLATION"
    assert check_declared_linear_direction(payload, [0., -.005], **ids)["status"] == "LINEAR_CONSTRAINT_VIOLATION"
    zero_gain = deepcopy(payload)
    for row in zero_gain["progress"]: row["gradient"] = [0., 0.]
    assert check_declared_linear_direction(zero_gain, [0., .005], **ids)["status"] == "LINEAR_NONPOSITIVE"
    assert validate_parameter_sensitivities(None, **ids)["status"] == "PARAMETER_SENSITIVITIES_UNAVAILABLE"


@pytest.mark.parametrize("bad", ["secants", "shape", "nonfinite", "missing_row", "reorder", "bindings", "names", "protocol", "count", "float_count", "binding_format"])
def test_parameter_sensitivity_contract_rejects_missing_false_or_unaligned_data(bad):
    p, ids = sensitivity_fixture()
    if bad == "secants": p["quantity"] = "endpoint_secants"
    if bad == "shape": p["risk"][0]["gradient"].append(0.)
    if bad == "nonfinite": p["progress"][0]["gradient"][0] = float('nan')
    if bad == "missing_row": p["progress"].pop()
    if bad == "reorder": p["progress"].reverse()
    if bad == "bindings": p["bindings"] = {}
    if bad == "names": p["parameter_order"] *= 2
    if bad == "protocol": p.pop("measurement_protocol_sha256")
    if bad == "count": p["parameter_count"] = True
    if bad == "float_count": p["parameter_count"] = 2.
    if bad == "binding_format": p["bindings"] = ids["bindings"] = {"admission_sha256": "not-a-sha"}
    assert validate_parameter_sensitivities(p, **ids)["status"] == "INVALID_PARAMETER_SENSITIVITIES"
    with pytest.raises(ValueError, match="sensitivity contract"):
        check_declared_linear_direction(p, [0., .005], **ids)


def cli_fixture(tmp_path, monkeypatch):
    import scripts.review_ppcsa_common_veto as cli
    import scripts.diagnose_ppcsa_direction_feasibility as prior
    root = tmp_path/"project"; root.mkdir()
    monkeypatch.setattr(cli, "ROOT", root); monkeypatch.setattr(prior, "ROOT", root)
    source = root/"src/source.txt"; source.parent.mkdir(); source.write_text("source remains frozen")
    phase1 = root/"results/phase1/report.json"; phase1.parent.mkdir(parents=True); phase1.write_text("{}")
    parent = root/"results/parent/frozen.stub"; parent.parent.mkdir(); parent.write_bytes(b"not a model")
    a, f = evidence_fixture()
    a.update(schema_version="PPCSASharedUpdateAdmissionV1", audit_used=False, used_partitions=["fit", "calibration"],
        source_actor_unchanged=True, source_critics_unchanged=True, source_adapter_unchanged=True, source_artifacts_unchanged=True,
        input_file_sha256={str(phase1): cli.file_sha(phase1)}, implementation_sha256={str(source): cli.file_sha(source)},
        source_sha256={}, checkpoint=str(parent), checkpoint_sha256=cli.file_sha(parent),
        source_mechanism_report_sha256=cli.file_sha(phase1), partitions={"fit": [121000], "calibration": [121024], "audit": [121032]})
    admission = root/"results/admission/report.json"; admission.parent.mkdir(); admission.write_text(json.dumps(a))
    _, protected = prior.bind_report(admission)
    f.update(source_admission_report_sha256=cli.file_sha(admission), input_file_sha256=protected,
        implementation_sha256={str(source): cli.file_sha(source)}, source_artifacts_unchanged=True,
        model_execution=False, parameter_reconstruction=False, new_shadow_calls=0,
        used_partitions=["fit", "calibration"], partitions=a["partitions"])
    feasibility = root/"results/feasibility/report.json"; feasibility.parent.mkdir(); feasibility.write_text(json.dumps(f))
    output = root/"results/common"
    return cli, source, admission, feasibility, output


def test_cli_preflight_does_not_review_then_synthetic_review_preserves_sources(tmp_path, monkeypatch):
    cli, source, a, f, output = cli_fixture(tmp_path, monkeypatch)
    args = ["--admission-report", str(a), "--feasibility-report", str(f), "--output-dir", str(output)]
    original = cli.review_common_veto
    monkeypatch.setattr(cli, "review_common_veto", lambda *args: (_ for _ in ()).throw(AssertionError("preflight ran review")))
    before = {str(p): cli.file_sha(p) for p in (source, a, f)}
    assert not cli.main(args+["--preflight-only"])["diagnostic_run"] and not output.exists()
    monkeypatch.setattr(cli, "review_common_veto", original)
    result = cli.main(args)
    assert result["source_artifacts_unchanged"] and result["new_shadow_calls"] == 0 and not result["model_execution"]
    assert set(p.name for p in output.iterdir()) == {"common_veto_report.json", "common_veto_report.md"}
    assert json.loads((output/"common_veto_report.json").read_text())["parameter_sensitivities"]["status"] == "PARAMETER_SENSITIVITIES_UNAVAILABLE"
    assert before == {p: cli.file_sha(p) for p in before}
    with pytest.raises(ValueError, match="new and outside"): cli.main(args)
    source.write_text("tampered")
    with pytest.raises(ValueError, match="hash mismatch"): cli.main(args+["--preflight-only"])


@pytest.mark.parametrize("bad", ["admission_sha", "input_sha", "implementation_sha", "scope"])
def test_cli_rejects_cross_report_binding_or_scope_tampering(tmp_path, monkeypatch, bad):
    cli, _, a, f, output = cli_fixture(tmp_path, monkeypatch)
    payload = json.loads(f.read_text())
    if bad == "admission_sha": payload["source_admission_report_sha256"] = "0"*64
    if bad == "input_sha": payload["input_file_sha256"] = {}
    if bad == "implementation_sha": payload["implementation_sha256"] = {str(a): "0"*64}
    if bad == "scope": payload["used_partitions"] = ["audit"]
    f.write_text(json.dumps(payload))
    with pytest.raises(ValueError): cli.main(["--admission-report", str(a), "--feasibility-report", str(f), "--output-dir", str(output), "--preflight-only"])
    assert not output.exists()
