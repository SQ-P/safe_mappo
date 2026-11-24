from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import pytest
import torch

from src.algorithms.ppcsa_update_acceptance import candidate_admission,candidate_set_admission,CANDIDATES


def evidence(key="1:0",gain=.1):
    track = {"observer_id":0,"track_id":3,"relative_position":[10.,0.],
        "relative_velocity":[0.,0.],"radius":1.,"velocity_valid":True}
    base = {"supported":True,"hard_safe":True,"source_unchanged":True,"track_capture_unchanged":True,
        "snapshot_fingerprint":"same","tracks":[track],"risk_proxy":[0.],"executed_progress":0.}
    return {"key":key,"replay_match":True,"baseline":base,"candidate":{**deepcopy(base),"executed_progress":gain},
        "predicted_gain":100.}


@pytest.mark.parametrize("failure",["negative_progress","risk","unsafe","unsupported","unknown_risk","unknown_motion",
    "track_order","track_missing","snapshot","replay","nonfinite"])
def test_one_bad_state_vetoes_whole_update_even_if_net_or_prediction_is_positive(failure):
    rows = [evidence("1:0",100.),evidence("2:0",.1)]
    row = rows[1]
    if failure == "negative_progress": row["candidate"]["executed_progress"] = -.01
    if failure == "risk": row["candidate"]["risk_proxy"] = [.001]
    if failure == "unsafe": row["candidate"]["hard_safe"] = False
    if failure == "unsupported": row["candidate"]["supported"] = False
    if failure == "unknown_risk": row["candidate"]["risk_proxy"] = [None]
    if failure == "unknown_motion": row["candidate"]["tracks"][0]["velocity_valid"] = False
    if failure == "track_order": row["candidate"]["tracks"][0]["track_id"] = 4
    if failure == "track_missing": row["candidate"]["tracks"] = []
    if failure == "snapshot": row["candidate"]["snapshot_fingerprint"] = "other"
    if failure == "replay": row["replay_match"] = False
    if failure == "nonfinite": row["candidate"]["executed_progress"] = float("nan")
    original = deepcopy(rows)
    result = candidate_admission(["1:0","2:0"],rows,[.01,0.])
    assert not result["would_accept"] and result["effective_virtual_update"] == [0.,0.]
    assert result["evaluated_states"] == 2 and not result["states"][1]["passed"]
    assert rows[0] == original[0] and row["predicted_gain"] == original[1]["predicted_gain"]


def test_positive_flat_zero_and_missing_coverage_have_distinct_results():
    rows = [evidence("1:0",.1),evidence("2:0",0.)]
    result = candidate_admission(["1:0","2:0"],rows,[.01])
    assert result["would_accept"] and result["effective_virtual_update"] == [.01]
    zero = candidate_admission(["1:0"],[evidence(gain=0.)],[.01])
    assert not zero["would_accept"] and "no_strictly_positive_executed_improvement" in zero["candidate_rejection_reasons"]
    missing = candidate_admission(["1:0","2:0"],rows[:1],[.01])
    assert "missing_duplicate_or_unregistered_state" in missing["candidate_rejection_reasons"]
    duplicate = candidate_admission(["1:0"],[rows[0],rows[0]],[.01])
    assert not duplicate["would_accept"]
    no_update = candidate_admission(["1:0"],rows[:1],[0.])
    assert not no_update["would_accept"]
    rejected = {name:zero for name in CANDIDATES}
    final = candidate_set_admission(rejected,3)
    assert final["zero_update_returned"] and final["fallback_update"] == [0.,0.,0.]
    final = candidate_set_admission({name:result for name in CANDIDATES},3)
    assert len(final["would_accept_candidates"]) == 3 and final["selected_candidate"] is None
    assert final["all_virtual_copies_discarded"] and not final["actor_update_applied"]


def test_empty_observation_is_not_the_same_as_missing_or_reordered_risk():
    row = evidence()
    for value in (row["baseline"],row["candidate"]):
        value["tracks"],value["risk_proxy"] = [],[]
    assert candidate_admission(["1:0"],[row],[.01])["would_accept"]
    row["candidate"].pop("tracks")
    assert not candidate_admission(["1:0"],[row],[.01])["would_accept"]
    row = evidence()
    t = {**deepcopy(row["baseline"]["tracks"][0]),"track_id":5}
    for value in (row["baseline"],row["candidate"]):
        value["tracks"].append(deepcopy(t)); value["risk_proxy"].append(0.)
    row["candidate"]["tracks"].reverse()
    assert not candidate_admission(["1:0"],[row],[.01])["would_accept"]


def test_only_the_exact_warning_patch_is_compatible_with_historical_hash(tmp_path,monkeypatch):
    import scripts.diagnose_ppcsa_update_acceptance as cli
    monkeypatch.setattr(cli,"ROOT",tmp_path)
    path = tmp_path/"src/algorithms/ppcsa_mechanism_diagnostics.py"
    path.parent.mkdir(parents=True)
    old = ("header\n"+cli.OLD_WARNING_CHECK+"\nbody\n").encode()
    current = old.replace(cli.OLD_WARNING_CHECK.encode(),cli.NEW_WARNING_CHECK.encode())
    path.write_bytes(current)
    expected = hashlib.sha256(old).hexdigest()
    assert cli.historical_source_match(str(path.relative_to(tmp_path)).replace('\\','/'),expected) == "exact_except_detach_item_warning_check"
    path.write_bytes(current+b"unrelated change\n")
    with pytest.raises(ValueError,match="implementation drift"):
        cli.historical_source_match("src/algorithms/ppcsa_mechanism_diagnostics.py",expected)


def test_budget_zero_check_emits_no_scalar_conversion_warning():
    from src.algorithms.ppcsa_executed_value import zero_adapter
    from src.algorithms.ppcsa_mechanism_diagnostics import match_calibration_budget
    adapter = zero_adapter()
    direction = torch.zeros(sum(p.numel() for p in adapter.parameters())); direction[-1] = 1.
    with warnings.catch_warnings():
        warnings.simplefilter("error",UserWarning)
        step,result = match_calibration_budget(adapter,direction,torch.ones(1,3,128),torch.ones(1),.01)
    assert step>0 and result["status"] == "MATCHED_ON_CALIBRATION"


def test_restored_state_drift_cannot_be_marked_as_a_matching_snapshot(monkeypatch):
    import scripts.diagnose_ppcsa_update_acceptance as cli
    from tests.test_ppcsa_contracts import normal_environment
    from src.safety.ppcsa_one_step import capture_packet
    env = normal_environment()
    packet = capture_packet(env,23,torch.zeros(1,3,128))
    original = cli.restore_packet
    def drift(*args):
        hidden = original(*args)
        args[0].agents[0].position[0] += 1.
        return hidden
    monkeypatch.setattr(cli,"restore_packet",drift)
    with pytest.raises(RuntimeError,match="fingerprint mismatch"):
        cli.restore_checked(env,packet)


def test_synthetic_cli_replays_raw_tracks_and_rejects_atomically(tmp_path,monkeypatch):
    import scripts.diagnose_ppcsa_update_acceptance as cli
    from scripts.diagnose_ppcsa_shared_update import run_diagnostics
    from tests.test_ppcsa_contracts import normal_environment
    from src.algorithms.ppcsa_executed_value import ExecutedValueCritic
    from src.algorithms.ppcsa_protocol import file_sha
    from src.safety.ppcsa_one_step import capture_packet,evaluate_ppcsa_normal_shadow,jsonable
    env = normal_environment(); env.obstacles = []; env._reset_obstacle_tracks()
    data = {"states":[],"records":[]}
    for seed in (121000,121024):
        packet = capture_packet(env,seed,torch.zeros(1,3,128))
        packet.update(base_mu=torch.zeros(3,2),hidden_features=torch.ones(3,128),context=np.zeros(4),parity_passed=True,factual_supported=True,step=0)
        baseline = evaluate_ppcsa_normal_shadow(env,np.zeros((3,2))).payload()
        data["records"].append({"layout":seed,"step":0,"state_index":len(data["states"]),
            "best_local_delta":np.ones((3,2),dtype=np.float32)*.05,"candidates":[{"kind":"base","result":baseline}]})
        data["states"].append(packet)
    data["states"].append(None); data["records"].append({"layout":121032,"state_index":2})
    g1,crit,phase1 = (tmp_path/p for p in ("gate1","critics","phase1"))
    for d in (g1,crit,phase1): d.mkdir()
    (g1/"manifest.json").write_text("{}"); (crit/"manifest.json").write_text("{}")
    torch.save(data,g1/"opportunities.pt")
    parent = tmp_path/"parent.stub"; parent.write_bytes(b"synthetic parent; not loaded")
    source = {"partitions":{"fit":[121000],"calibration":[121024],"audit":[121032]},
        "checkpoint":str(parent),"checkpoint_sha256":file_sha(parent),"config_sha256":"synthetic","task_config_fingerprint":"synthetic","gate0_dir":str(tmp_path/"gate0")}
    models,artifacts = {},{}
    for name in ("ordinary","paired"):
        m = ExecutedValueCritic(4)
        with torch.no_grad():
            for p in m.parameters(): p.zero_()
        a = {"context_mean":torch.zeros(4),"context_scale":torch.ones(4),"target_mean":0.,"target_scale":1.,
            "context_dim":4,"objective":name,"partitions":source["partitions"],"state_dict":m.state_dict()}
        models[name],artifacts[name] = m.freeze(),a
        torch.save(a,crit/f"{name}.pt")
    mechanism = {**source,**run_diagnostics(data,source,models,artifacts,env,"cpu"),
        "schema_version":"PPCSASharedMechanismDiagnosticV1","used_partitions":["fit","calibration"],
        "source_actor_unchanged":True,"source_artifacts_unchanged":True,
        "diagnostic_source_sha256":{p:file_sha(cli.ROOT/p) for p in cli.PHASE1_SOURCES},
        "input_file_sha256":{str(p):file_sha(p) for d in (g1,crit) for p in d.iterdir()}}
    mp = phase1/"report.json"; mp.write_text(json.dumps(jsonable(mechanism)))
    protected = {str(p):file_sha(p) for d in (g1,crit,phase1) for p in d.iterdir()}
    monkeypatch.setattr(cli,"validate_inputs",lambda *_:(source,{}))
    monkeypatch.setattr(cli,"runtime",lambda *_:(_ for _ in ()).throw(AssertionError("Preflight executed runtime")))
    out = tmp_path/"admission"
    args = ["--gate1-dir",str(g1),"--critic-dir",str(crit),"--mechanism-report",str(mp),"--output-dir",str(out)]
    assert cli.main(args+["--preflight-only"])["diagnostic_run"] is False
    assert not out.exists()
    monkeypatch.setattr(cli,"runtime",lambda *_:(SimpleNamespace(actor=torch.nn.Linear(1,1)),env,{}))
    result = cli.main(args)
    assert result["source_actor_unchanged"] and result["source_critics_unchanged"] and result["source_artifacts_unchanged"]
    assert result["whole_set"]["selected_candidate"] is None
    for value in result["candidates"].values():
        assert value["evaluated_states"] == 2
        assert all("tracks" in e["baseline"] and "risk_proxy" in e["candidate"] for e in value["raw_state_evidence"])
    assert not list(out.rglob("*.pt"))
    # Replayed known obstacle identities are preserved in a separate controlled fixture.
    from src.envs.entities import CircularObstacle
    tracks_env = normal_environment()
    tracks_env.obstacles = [CircularObstacle(tracks_env.positions[0]+np.array([10.,0.]),1.,np.zeros(2),False)]
    tracks_env._reset_obstacle_tracks()
    tracked = cli.shadow_evidence(tracks_env,np.zeros((3,2)),None)
    assert tracked["tracks"] and len(tracked["tracks"]) == len(tracked["risk_proxy"])
    assert tracked["source_unchanged"] and tracked["track_capture_unchanged"]
    original_shadow = cli.shadow_evidence
    counter = [0]
    def veto_one_state(*a,**kw):
        counter[0] += 1
        result = original_shadow(*a,**kw)
        if counter[0]%4 == 2:
            result["hard_safe"] = False
        return result
    monkeypatch.setattr(cli,"shadow_evidence",veto_one_state)
    other = tmp_path/"vetoed"
    second = cli.main(args[:-1]+[str(other)])
    assert second["whole_set"]["zero_update_returned"]
    assert not any(second["whole_set"]["fallback_update"])
    assert all(not c["would_accept"] and c["evaluated_states"] == 2 for c in second["candidates"].values())
    assert protected == {str(p):file_sha(p) for d in (g1,crit,phase1) for p in d.iterdir()}
    with pytest.raises(ValueError,match="new and separate"):
        cli.main(args)
    (g1/"opportunities.pt").write_bytes(b"changed source data")
    with pytest.raises(ValueError,match="Historical input artifact changed"):
        cli.main(args[:-1]+[str(tmp_path/"never_created"),"--preflight-only"])
