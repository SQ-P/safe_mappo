from copy import deepcopy
from fractions import Fraction
import json

import pytest

from src.algorithms.ppcsa_direction_feasibility import solve_simplex,analyze_admission
from src.algorithms.ppcsa_update_acceptance import CANDIDATES,candidate_admission


def test_exact_positive_proxy_requires_all_three_weights_but_never_certifies_parameters():
    result = solve_simplex([[1.,-.1,0.],[-.1,1.,0.]],[[1.,1.,-1.]])
    assert result["status"] == "PROXY_POSITIVE_DIRECTION_UNVERIFIED"
    a = [Fraction(result["weights_exact"][name]) for name in CANDIDATES]
    assert sum(a) == 1 and min(a)>0
    assert a[0]-Fraction(1,10)*a[1]>=0 and -Fraction(1,10)*a[0]+a[1]>=0
    assert a[0]+a[1]-a[2]<=0
    assert result["objective_net_progress"] == pytest.approx(.45)
    assert result["max_constraint_violation_exact"] == "0"
    assert result["actual_nonzero_parameter_status"] == "UNKNOWN"
    assert result["parameter_norm_lower_bound"] == 0 and result["parameter_norm_analytic_upper_bound"] == .01
    assert result["real_shadow_validation"] is False and not result["actual_safety_certified"]


def test_tiny_common_risk_veto_is_not_erased_by_solver_tolerance():
    result = solve_simplex([[1.,1.,1.]],[[1e-25,2e-25,3e-25]],risk_ids=["track:tiny"])
    assert result["status"] == "PROXY_INFEASIBLE"
    assert result["universal_veto_constraints"] == ["track:tiny"]
    assert result["first_empty_intersection"] == "track:tiny"
    assert result["weights"] is None


def test_joint_conflicts_are_infeasible_without_an_individual_universal_veto():
    result = solve_simplex([[1.,-2.,0.],[-2.,1.,0.]],[[0.,0.,1.]])
    assert result["status"] == "PROXY_INFEASIBLE" and not result["universal_veto_constraints"]


def test_zero_response_and_degenerate_feasible_segment_are_not_positive_updates():
    for rows in ([[0.,0.,0.]],[[1.,-1.,0.],[-1.,1.,0.]]):
        result = solve_simplex(rows,[])
        assert result["status"] == "PROXY_NONPOSITIVE"
        assert result["objective_net_progress_exact"] == "0"
        assert result["actual_nonzero_parameter_status"] == "UNKNOWN"
        assert result["feasible_vertices_exact"]


@pytest.mark.parametrize("g,r",[([[float("nan"),0.,0.]],[]),([[1.,0.,0.]],[[0.,None,0.]]),([[1.,0.]],[]),([],[])])
def test_invalid_coefficients_are_not_solved_on_a_favorable_subset(g,r):
    assert solve_simplex(g,r)["status"] == "INVALID_EVIDENCE"


def test_strict_positive_proxy_is_not_replaced_by_a_meaningful_effect_threshold():
    result = solve_simplex([[1e-30,0.,0.]],[[1e-30,-1e-30,0.]])
    assert result["status"] == "PROXY_POSITIVE_DIRECTION_UNVERIFIED"
    assert Fraction(result["objective_net_progress_exact"])>0
    assert result["constraint_tolerance"] == 0


def admission_fixture():
    registry = [{"key":"121000:0","state_index":0},{"key":"121024:0","state_index":1}]
    progress = {"ordinary":[1.,-.1],"paired":[-.1,1.],"kinematic":[0.,0.]}
    risk = {"ordinary":1.,"paired":1.,"kinematic":-1.}
    candidates = {}
    for name in CANDIDATES:
        rows = []
        for i,fixed in enumerate(registry):
            base = {"supported":True,"hard_safe":True,"source_unchanged":True,"track_capture_unchanged":True,
                "snapshot_fingerprint":"same-"+fixed["key"],"executed_progress":0.,"risk_proxy":[1.],
                "tracks":[{"observer_id":0,"track_id":3,"velocity_valid":True,
                    "relative_position":[10.,0.],"relative_velocity":[0.,0.],"radius":1.}],
                "requested_actions":[[0.,0.]],"executed_actions":[[0.,0.]],"stage_actions":{}}
            candidate = {**deepcopy(base),"executed_progress":progress[name][i],"risk_proxy":[1.+risk[name]],
                "requested_actions":[[.01,0.]],"executed_actions":[[.005,0.]]}
            rows.append({**fixed,"layout":int(fixed["key"].split(':')[0]),"step":0,
                "replay_match":True,"baseline":base,"candidate":candidate})
        result = candidate_admission([r["key"] for r in registry],rows,[.01])
        candidates[name] = {**result,"raw_state_evidence":rows,"direction_metadata":{
            "aggregate_gradient_norm":1.,"parameter_step":.01,"parameter_step_search":False,
            "direction_sha256":name,"source":"Calibration only","teacher_keys":["121024:0"]}}
    return {"fixed_state_registry":registry,"candidates":candidates,"candidate_registry":list(CANDIDATES),
        "parameter_step":.01,"execution_errors":[],"mechanism_conclusion_available":True}


def test_attribution_reproduces_vetoes_keeps_tracks_and_does_not_change_evidence():
    source = admission_fixture(); before = deepcopy(source)
    result = analyze_admission(source)
    assert source == before
    assert all(a["original_rejection_reproduced"] for a in result["attribution"].values())
    assert result["fixed_state_count"] == 2 and len(result["state_details"]) == 2
    assert result["complete_progress_constraint_count"] == 2 and result["complete_risk_constraint_count"] == 2
    assert result["proxy"]["status"] == "PROXY_POSITIVE_DIRECTION_UNVERIFIED"
    assert len(result["recurring_risk_veto_tracks"]) == 2 # Track ids are layout-local, not globally merged.
    assert result["state_pass_patterns"]["counts"] == {"001":2}
    assert not result["actor_training_authorized"] and not result["shadow_validation_authorized"]
    detail = result["state_details"][0]["directions"]["ordinary"]
    assert detail["execution_to_request_change_ratio"] == .5
    assert detail["baseline_risk_vector"] == [1.] and detail["candidate_risk_vector"] == [2.]


@pytest.mark.parametrize("failure",["missing_state","unknown_risk","track_identity","missing_track_context","baseline_origin","nonfinite_progress"])
def test_bad_raw_evidence_fails_diagnostic_and_preserves_state_count(failure):
    source = admission_fixture()
    row = source["candidates"]["ordinary"]["raw_state_evidence"][0]
    if failure == "missing_state": source["candidates"]["ordinary"]["raw_state_evidence"].pop()
    if failure == "unknown_risk": row["candidate"]["risk_proxy"] = [None]
    if failure == "track_identity": row["candidate"]["tracks"][0]["track_id"] = 9
    if failure == "missing_track_context": row["candidate"]["tracks"][0].pop("radius")
    if failure == "baseline_origin": row["baseline"]["executed_progress"] = .001
    if failure == "nonfinite_progress": row["candidate"]["executed_progress"] = float('inf')
    result = analyze_admission(source)
    assert result["proxy"]["status"] == "INVALID_EVIDENCE"
    assert len(result["state_details"]) == result["fixed_state_count"] == 2
    assert result["evidence_errors"]


def test_synthetic_cli_preflight_does_not_solve_and_outputs_only_readonly_artifacts(tmp_path,monkeypatch):
    import scripts.diagnose_ppcsa_direction_feasibility as cli
    import src.algorithms.ppcsa_direction_feasibility as pure
    root = tmp_path/"project"; root.mkdir()
    monkeypatch.setattr(cli,"ROOT",root)
    d = root/"results/admission"; d.mkdir(parents=True)
    metadata = root/"results/phase1/report.json"; metadata.parent.mkdir(); metadata.write_text("{}")
    implementation = root/"src/source.txt"; implementation.parent.mkdir(); implementation.write_text("synthetic source metadata")
    parent = root/"results/parent/parent.stub"; parent.parent.mkdir(); parent.write_bytes(b"never loaded as model")
    report = admission_fixture()
    report.update(schema_version="PPCSASharedUpdateAdmissionV1",audit_used=False,used_partitions=["fit","calibration"],
        source_actor_unchanged=True,source_critics_unchanged=True,source_adapter_unchanged=True,source_artifacts_unchanged=True,
        fixed_state_registry_sha256=cli.digest(report["fixed_state_registry"]),
        input_file_sha256={str(metadata):cli.file_sha(metadata)},implementation_sha256={str(implementation):cli.file_sha(implementation)},
        source_sha256={},checkpoint=str(parent),checkpoint_sha256=cli.file_sha(parent),
        source_mechanism_report_sha256=cli.file_sha(metadata),
        partitions={"fit":[121000],"calibration":[121024],"audit":[121032]})
    path = d/"report.json"; path.write_text(json.dumps(report))
    protected = {str(p):cli.file_sha(p) for p in (path,metadata,implementation,parent)}
    output = root/"results/proxy"
    args = ["--admission-report",str(path),"--output-dir",str(output)]
    original = pure.solve_simplex
    monkeypatch.setattr(pure,"solve_simplex",lambda *a,**kw:(_ for _ in ()).throw(AssertionError("Preflight solved LP")))
    assert cli.main(args+["--preflight-only"])["diagnostic_run"] is False
    assert not output.exists()
    monkeypatch.setattr(pure,"solve_simplex",original)
    result = cli.main(args)
    assert result["status"] == "PROXY_POSITIVE_DIRECTION_UNVERIFIED"
    assert result["new_shadow_calls"] == 0 and not result["model_execution"] and not result["parameter_reconstruction"]
    assert result["source_artifacts_unchanged"] and (output/"direction_feasibility_report.json").exists()
    assert set(p.suffix for p in output.iterdir()) == {".json",".md"}
    assert protected == {p:cli.file_sha(p) for p in protected}
    with pytest.raises(ValueError,match="new and outside"):
        cli.main(args)
    invalid = deepcopy(report)
    invalid["candidates"]["ordinary"]["raw_state_evidence"][0]["candidate"]["risk_proxy"] = [float('nan')]
    invalid_path = d/"invalid.json"; invalid_path.write_text(json.dumps(invalid))
    invalid_out = root/"results/invalid_proxy"
    invalid_result = cli.main(["--admission-report",str(invalid_path),"--output-dir",str(invalid_out)])
    assert invalid_result["status"] == "INVALID_EVIDENCE"
    saved = json.loads((invalid_out/"direction_feasibility_report.json").read_text())
    assert saved["state_details"][0]["directions"]["ordinary"]["candidate_risk_vector"] == [{"invalid_numeric_value":"nan"}]
    implementation.write_text("changed source metadata")
    with pytest.raises(ValueError,match="hash mismatch"):
        cli.main(args[:-1]+[str(root/"results/fail"),"--preflight-only"])
