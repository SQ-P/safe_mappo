from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from src.algorithms.ppcsa_parameter_sensitivity import (
    COORDINATES, SCALES, REPEATS, protocol, parameterization, digest,
    max_calls, analyze_axis, measure_cohort)
from src.algorithms.ppcsa_common_veto_diagnostics import STAGES


def point(x=0., progress=None, risk=None):
    guard = {"shielded_axes": [[False, False]], "pair_infeasible": False,
             "recovery_required": False, "safe_brake": False, "pair_iterations": 1}
    return {"supported": True, "hard_safe": True, "source_unchanged": True, "track_capture_unchanged": True,
        "snapshot_fingerprint": "composite", "executed_progress": 2*x if progress is None else progress,
        "risk_proxy": [10.+x if risk is None else risk],
        "tracks": [{"observer_id": 0, "track_id": 3, "relative_position": [3., 0.],
            "relative_velocity": [-1., 0.], "radius": 1., "velocity_valid": True}],
        "requested_actions": [[x, 0.]], "executed_actions": [[x, 0.]],
        "stage_actions": {s: [[x, 0.]] for s in STAGES},
        "safety_evidence": {"stage_checks": {"initial_guard": deepcopy(guard), "final_guard": deepcopy(guard),
            "ttc_fallback": False, "safe_brake_applied": False, "recovery_overlay": False}}}


def samples(fn=point):
    return [{"h": h, "minus": [fn(-h) for _ in range(REPEATS)], "plus": [fn(h) for _ in range(REPEATS)]} for h in SCALES]


def test_smooth_central_estimate_and_heldout_predictions_recover_known_derivatives():
    result = analyze_axis(point(), samples(lambda x: point(x, progress=2*x+3*x*x, risk=10+x+2*x*x)))
    assert result["status"] == "LOCAL_FD_ESTIMATE_CONSISTENT" and result["usable"]
    assert result["gradient"] == pytest.approx([2., 1.], abs=1e-10)
    assert len(result["central_estimates_by_scale"]) == 3
    assert len(result["samples"]) == 3 and not result["empirical_derivatives_verified"]


@pytest.mark.parametrize("fn", [lambda x: point(x, progress=abs(x)),
    lambda x: point(x, progress=float(x > 0))])
def test_kink_and_discontinuity_are_not_certified_derivatives(fn):
    result = analyze_axis(fn(0.), samples(fn))
    assert not result["usable"] and result["gradient"] is None
    assert "ONE_SIDED_INCONSISTENCY" in result["flags"]
    assert "HELDOUT_SCALE_PREDICTION_MISMATCH" in result["flags"]


def test_zero_response_is_explicitly_resolution_limited_not_imputed_gradient():
    result = analyze_axis(point(), samples(lambda x: point(0.)))
    assert "ZERO_OR_RESOLUTION_LIMITED_RESPONSE" in result["flags"]
    assert result["gradient"] is None and not result["usable"]
    assert all(v == [0., 0.] for v in result["central_estimates_by_scale"])


@pytest.mark.parametrize("failure", ["track", "context", "risk_unknown", "nonfinite", "stage_missing", "state",
    "unsupported", "unsafe", "repeat", "stage_switch", "missing_sign", "scale"])
def test_invalid_or_unstable_measurement_keeps_signed_samples_and_blocks_export(failure):
    data = samples(); bad = data[0]["plus"][0]
    if failure == "track": bad["tracks"][0]["track_id"] = 9
    if failure == "context": bad["tracks"][0]["radius"] = 2.
    if failure == "risk_unknown": bad["risk_proxy"] = [None]
    if failure == "nonfinite": bad["executed_progress"] = float('nan')
    if failure == "stage_missing": bad["stage_actions"].pop("obstacle_ttc")
    if failure == "state": bad["snapshot_fingerprint"] = "other"
    if failure == "unsupported": bad["supported"] = False
    if failure == "unsafe": bad["hard_safe"] = False
    if failure == "repeat": bad["executed_progress"] += 1e-15
    if failure == "stage_switch": bad["safety_evidence"]["stage_checks"]["initial_guard"]["shielded_axes"] = [[True, False]]
    if failure == "missing_sign": data[0]["plus"].pop()
    if failure == "scale": data[0]["h"] = .01
    result = analyze_axis(point(), data)
    assert not result["usable"] and result["gradient"] is None and result["samples"]
    if failure == "repeat": assert "REPEATABILITY_MISMATCH" in result["flags"]
    if failure == "stage_switch": assert "STAGE_SWITCH" in result["flags"]


def bindings():
    return {"admission_sha256": "a"*64, "registry_sha256": "b"*64, "checkpoint_sha256": "c"*64}


def test_full_synthetic_measurement_has_six_dimensional_contract_and_conflict_counts():
    registry = [{"key": "1:0", "state_index": 0}, {"key": "2:0", "state_index": 1}]
    baselines = {row["key"]: point() for row in registry}
    before = deepcopy(baselines)
    calls = []
    def evaluate(key, axis, delta):
        calls.append((key, axis, delta))
        slope = 1 if key == "1:0" else -1
        return point(delta, risk=10+slope*delta)
    result = measure_cohort(registry, baselines, evaluate, bindings())
    assert result["status"] == "SUBSPACE_ESTIMATES_REQUIRE_REVIEW"
    assert len(calls) == result["shadow_calls"] == max_calls(2) == 148
    assert baselines == before
    payload = result["parameter_sensitivities"]
    assert payload["parameter_count"] == 6 and len(payload["parameter_order"]) == 6
    assert all(len(r["gradient"]) == 6 for r in payload["progress"]+payload["risk"])
    assert payload["parameterization"]["effective_up_dimension"] == 34
    assert len(payload["parameterization"]["unmeasured_up_flat_indices"]) == 28
    assert result["sensitivity_contract"]["status"] == "DECLARED_PARAMETER_SENSITIVITIES_VALIDATED"
    assert not result["sensitivity_contract"]["empirical_derivatives_verified"]
    assert all(row["risk"]["opposed_signs_observed"] for row in result["response_sign_conflicts"])
    assert not result["actual_update_accepted"] and not result["direction_generated"]


def test_one_failed_state_kept_and_all_remaining_fixed_samples_still_measured():
    registry = [{"key": "1:0", "state_index": 0}, {"key": "2:0", "state_index": 1}]
    def evaluate(key, axis, delta):
        if key == "1:0" and axis == 0:
            raise RuntimeError("synthetic evaluation failure")
        return point(delta)
    result = measure_cohort(registry, {r["key"]: point() for r in registry}, evaluate, bindings())
    assert result["status"] == "MEASUREMENT_INVALID" and len(result["states"]) == 2
    assert result["shadow_calls"] == max_calls(2)
    assert result["parameter_sensitivities"] is None
    assert result["sensitivity_contract"]["status"] == "PARAMETER_SENSITIVITIES_UNAVAILABLE"
    assert result["states"][0]["axes"][0]["samples"][0]["minus"][0]["evaluation_error"]


@pytest.mark.parametrize("failure", ["baseline_repeat", "historical_replay"])
def test_unreproducible_baseline_blocks_whole_export(failure):
    registry = [{"key": "1:0", "state_index": 0}]
    count = 0
    def evaluate(key, axis, delta):
        nonlocal count
        count += 1
        return point(delta, progress=.1 if failure == "baseline_repeat" and count == 2 else 2*delta)
    historical = point(progress=.1) if failure == "historical_replay" else point()
    result = measure_cohort(registry, {"1:0": historical}, evaluate, bindings())
    assert result["status"] == "MEASUREMENT_INVALID" and result["parameter_sensitivities"] is None
    assert result["shadow_calls"] == max_calls(1) and result["states"][0]["errors"]


def test_empty_cohort_cannot_become_a_valid_empty_gradient_payload():
    with pytest.raises(ValueError, match="fixed measurement cohort"):
        measure_cohort([], {}, lambda *a: point(), bindings())


def test_packet_selection_preserves_registry_order_and_does_not_touch_audit_packet():
    from scripts.diagnose_ppcsa_parameter_sensitivity import registry_packets
    class AuditSentinel:
        def __getitem__(self, key): raise AssertionError("Audit packet read")
    packet = {"layout_seed": 1, "snapshot": SimpleNamespace(state_fingerprint="env"), "cache": [], "cache_step": 0}
    data = {"records": [{"layout": 9, "state_index": 1}, {"layout": 1, "step": 0, "state_index": 0}],
            "states": [packet, AuditSentinel()]}
    registry = [{"key": "1:0", "state_index": 0, "state_fingerprint": "env", "cache_sha256": digest([]), "cache_step": 0}]
    assert registry_packets(data, registry, {"fit": [1], "calibration": [], "audit": [9]}, digest) == {"1:0": packet}
    registry[0]["cache_step"] = 1
    with pytest.raises(ValueError, match="binding differs"):
        registry_packets(data, registry, {"fit": [1], "calibration": [], "audit": [9]}, digest)


def test_virtual_coordinate_mapping_and_zero_up_are_verified_without_changing_source():
    import torch
    from src.algorithms.ppcsa_executed_value import zero_adapter
    from scripts.diagnose_ppcsa_parameter_sensitivity import perturb_adapter
    adapter = zero_adapter()
    before = deepcopy(adapter.state_dict())
    for axis, (name, index, flat) in enumerate(COORDINATES):
        copy = perturb_adapter(adapter, axis, -SCALES[1])
        delta = torch.cat([(copy.up.weight-adapter.up.weight).flatten(), (copy.up.bias-adapter.up.bias).flatten()])
        assert torch.count_nonzero(delta) == 1 and delta[flat].item() == pytest.approx(-SCALES[1])
        assert all(torch.equal(before[k], v) for k, v in adapter.state_dict().items())
    with pytest.raises(ValueError): perturb_adapter(adapter, 0, .01)
    with torch.no_grad(): adapter.up.bias[0] = .01
    with pytest.raises(ValueError, match="zero-up"): perturb_adapter(adapter, 0, SCALES[0])


def cli_fixture(tmp_path, monkeypatch):
    from tests.test_ppcsa_common_veto import cli_fixture as previous_fixture
    import scripts.diagnose_ppcsa_parameter_sensitivity as cli
    old, source, a, f, common_output = previous_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "ROOT", a.parents[2])
    monkeypatch.setattr(old, "IMPLEMENTATION_FILES", (source,))
    admission = json.loads(a.read_text())
    admission.update(hidden_dim=128, adapter_rank=16, adapter_scale=.15)
    a.write_text(json.dumps(admission))
    feasibility = json.loads(f.read_text())
    feasibility["source_admission_report_sha256"] = cli.file_sha(a)
    feasibility["input_file_sha256"][str(a)] = cli.file_sha(a)
    f.write_text(json.dumps(feasibility))
    old.main(["--admission-report", str(a), "--feasibility-report", str(f), "--output-dir", str(common_output)])
    c = common_output/"common_veto_report.json"
    return cli, a, f, c, common_output.parent/"sensitivity"


def test_cli_preflight_never_executes_model_dataset_or_measurement(tmp_path, monkeypatch):
    cli, a, f, c, output = cli_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "run_frozen_measurement", lambda *a: (_ for _ in ()).throw(AssertionError("preflight ran models")))
    before = {str(p): cli.file_sha(p) for p in (a, f, c)}
    result = cli.main(["--admission-report", str(a), "--feasibility-report", str(f), "--common-veto-report", str(c), "--output-dir", str(output), "--preflight-only"])
    assert not result["diagnostic_run"] and not output.exists()
    assert before == {p: cli.file_sha(p) for p in before}


def test_synthetic_cli_freezes_protocol_before_callback_and_retains_invalid_reports(tmp_path, monkeypatch):
    cli, a, f, c, output = cli_fixture(tmp_path, monkeypatch)
    def fake_run(admission, protected, destination, source_bindings):
        frozen = json.loads((destination/"measurement_protocol.json").read_text())
        assert frozen["protocol_sha256"] == digest(protocol())
        registry = admission["fixed_state_registry"]
        return measure_cohort(registry, {r["key"]: point() for r in registry},
            lambda key, axis, delta: point(delta, progress=abs(delta)), source_bindings)
    monkeypatch.setattr(cli, "run_frozen_measurement", fake_run)
    result = cli.main(["--admission-report", str(a), "--feasibility-report", str(f), "--common-veto-report", str(c), "--output-dir", str(output), "--run-frozen-protocol"])
    assert result["status"] == "MEASUREMENT_FIDELITY_UNRESOLVED"
    assert result["parameter_sensitivities"] is None
    assert set(p.name for p in output.iterdir()) == {"measurement_protocol.json", "parameter_sensitivity_report.json", "parameter_sensitivity_report.md"}


@pytest.mark.parametrize("bad", ["binding", "architecture", "common_source"])
def test_cli_broken_binding_stops_before_model_execution(tmp_path, monkeypatch, bad):
    cli, a, f, c, output = cli_fixture(tmp_path, monkeypatch)
    obj = json.loads(c.read_text())
    if bad == "binding": obj["source_admission_report_sha256"] = "0"*64
    if bad == "common_source": obj["implementation_sha256"] = {}
    if bad == "architecture": obj["fixed_state_registry"] = []
    c.write_text(json.dumps(obj))
    with pytest.raises(ValueError): cli.main(["--admission-report", str(a), "--feasibility-report", str(f), "--common-veto-report", str(c), "--output-dir", str(output), "--preflight-only"])
    assert not output.exists()
