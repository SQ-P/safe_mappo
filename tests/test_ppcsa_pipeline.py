"""Tiny synthetic CLI integration: no real checkpoint, layouts or training run."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.test_ppcsa_contracts import normal_environment
from src.algorithms.ppo.networks import RecurrentActor
from src.algorithms.ppcsa_protocol import PARENT_SHA,source_hashes,file_sha
from src.safety.ppcsa_one_step import PROTOCOL


@pytest.mark.parametrize("guard_requests_recovery",[False,True])
def test_gate0_gate1_fit_gate2_synthetic_cli_flow(tmp_path,monkeypatch,guard_requests_recovery):
    import scripts.audit_ppcsa_gate0 as gate0
    import scripts.audit_ppcsa_gate1 as gate1
    import scripts.fit_ppcsa_executed_value as value
    import scripts.audit_ppcsa_gate2 as gate2
    parent = tmp_path/"synthetic_parent.pt"
    parent.write_bytes(b"unit test parent stub, never loaded as a model")
    panel_path = tmp_path/"synthetic_panel.json"
    panel = {"protocol":PROTOCOL,"layout_seeds":[121000,121024,121032],
             "partitions":{"fit":[121000],"calibration":[121024],"audit":[121032]}}
    panel_path.write_text(json.dumps(panel))
    def fixture_runtime(checkpoint,output,device):
        env = normal_environment()
        original = env.reset
        def reset(*args,**kwargs):
            original(seed=23)
            env.obstacles = []
            env._reset_obstacle_tracks()
            return env.build_local_observations(),{}
        env.reset = reset
        actor = RecurrentActor(52,128)
        with torch.no_grad():
            for p in actor.parameters():
                p.zero_()
        policy = SimpleNamespace(actor_modules=lambda:(actor,),initial_hidden=lambda n,d:torch.zeros(1,n,128))
        trainer = SimpleNamespace(actor=policy,device=torch.device("cpu"),normalize_local_observations=lambda x:np.asarray(x,dtype=np.float32),
            config={"safety_execution":{"recovery":{"maximum_steps":20,"maximum_attempts":1}}})
        provenance = {"protocol":PROTOCOL,"checkpoint":str(parent),"checkpoint_sha256":PARENT_SHA,
            "task_config_fingerprint":"synthetic","config_sha256":"synthetic","source_sha256":source_hashes()}
        return trainer,env,provenance
    monkeypatch.setattr(gate0,"validate_panel",lambda path:panel)
    monkeypatch.setattr(gate0,"runtime",fixture_runtime)
    monkeypatch.setattr(gate1,"runtime",fixture_runtime)
    monkeypatch.setattr(gate2,"runtime",fixture_runtime)
    actual_hash = file_sha
    mock_hash = lambda path:PARENT_SHA if Path(path)==parent else actual_hash(path)
    monkeypatch.setattr(gate0,"file_sha",mock_hash)
    monkeypatch.setattr(gate2,"file_sha",mock_hash)
    g0,g1,model,g2 = (tmp_path/name for name in ("gate0","gate1","value","gate2"))
    first = gate0.main(["--checkpoint",str(parent),"--panel",str(panel_path),"--output-dir",str(g0),
                       "--max-steps","1","--states-per-layout","1"])
    assert first["status"] == "PASS"
    if guard_requests_recovery:
        import src.safety.ppcsa_one_step as shadow_api
        original_capture = shadow_api._capture
        def recovery_capture(actions,branch,checks):
            checks["initial_guard"]["recovery_required"] = True
            return original_capture(actions,branch,checks)
        monkeypatch.setattr(shadow_api,"_capture",recovery_capture)
    opportunities = gate1.main(["--gate0-dir",str(g0),"--output-dir",str(g1)])
    if guard_requests_recovery:
        data = torch.load(g1/"opportunities.pt",weights_only=False)
        assert not data["examples"]
        assert opportunities["coverage_counts"]["positive_local"] == 0
        assert opportunities["coverage_counts"]["positive_reference"] == 0
        assert opportunities["rejection_counts_nonexclusive"]["initial_guard_recovery_required"]>0
        assert all(r["best_local_delta"] is None for r in data["records"])
        return
    assert opportunities["coverage_counts"]["positive_local"]>0
    # Review is mandatory before even a synthetic fit through the CLI.
    with pytest.raises(RuntimeError,match="reviewed"):
        value.main(["--gate1-dir",str(g1),"--output-dir",str(model),"--epochs","1"])
    (g1/"gate1_review.json").write_text(json.dumps({"schema_version":"PPCSAGate1ReviewV1",
        "report_sha256":actual_hash(g1/"ppcsa_opportunity_report.json"),"decision":"approve_critic_diagnostic","rationale":"synthetic test only"}))
    fitted = value.main(["--gate1-dir",str(g1),"--output-dir",str(model),"--epochs","2"])
    assert not fitted["actor_training_authorized"]
    result = gate2.main(["--gate1-dir",str(g1),"--critic-dir",str(model),"--output-dir",str(g2)])
    assert result["status"] == "REVIEW_REQUIRED" and result["actor_training_authorized"] is False
    assert result["audit_layouts"] == [121032]
    assert set(result["models"]) == {"ordinary","paired"}
    for row in result["models"].values():
        assert row["parameter_probe"]["efficacy_unproven"]
        assert "virtual_update_rejected" not in row["parameter_probe"]
    assert parent.read_bytes() == b"unit test parent stub, never loaded as a model"
