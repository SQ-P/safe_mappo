"""First-round launch readiness; never run real MAPPO updates in these tests."""
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.train_cross8u_160 as entry


def config():
    return entry.load_config(entry.CONFIG)


def test_first_round_matches_frozen_task_and_keeps_new_outputs():
    new = config()
    baseline = entry.load_config("configs/verification/cross8u_160/seed23.yaml")
    entry.training_contract(new)
    for key in ("env","reward","model","multi_agent","safety_execution","oracle_subgoal"):
        assert new[key] == baseline[key]
    assert new["train"]["total_updates"] == 400 and new["train"]["rollout_steps"] == 2048
    assert new["train"]["ppo_epochs"] == 4
    assert new["train"]["checkpoint_updates"] == new["evaluation"]["updates"] == [100,200,400]
    assert new["evaluation"]["seed_base"] == 240000 and new["evaluation"]["episodes"] == 10
    assert new["output"]["output_dir"] != baseline["output"]["output_dir"]
    assert new["verification_experiment"]["historical_weight_transfer"] is False
    assert new["verification_experiment"]["formal_evaluation_authorized"] is False


@pytest.mark.parametrize("section,key,value",[
    ("train","total_updates",401),("train","rollout_steps",128),
    ("evaluation","seed_base",241000),("env","num_agents",3),
    ("verification_experiment","historical_weight_transfer",True)])
def test_changed_task_or_budget_cannot_silently_start(section,key,value):
    new = config(); new[section][key] = value
    with pytest.raises(ValueError,match="frozen8U"): entry.training_contract(new)


def test_preflight_only_never_constructs_or_trains(monkeypatch):
    monkeypatch.setattr(entry,"preflight",lambda cfg:{"status":"READY_SEED23_U400"})
    monkeypatch.setattr(entry,"create_cross_trainer",lambda *a:pytest.fail("Preflight constructed a trainer"))
    assert entry.main(["--preflight-only"])["status"] == "READY_SEED23_U400"


@pytest.mark.parametrize("already_updated",[False,True])
def test_training_dispatch_uses_fresh_trainer_only_with_mock(monkeypatch,tmp_path,already_updated):
    calls = []
    trainer = SimpleNamespace(current_update=1 if already_updated else 0,output_dir=tmp_path,
        train=lambda:calls.append("fake_train_only") or {"updates_completed":400})
    monkeypatch.setattr(entry,"preflight",lambda cfg:{"output_dir":str(tmp_path)})
    monkeypatch.setattr(entry,"create_cross_trainer",lambda cfg,root:trainer)
    if already_updated:
        with pytest.raises(RuntimeError,match="start fresh"): entry.main([])
        assert not calls
    else:
        assert entry.main([])["updates_completed"] == 400 and calls == ["fake_train_only"]


def test_nonempty_output_stops_before_gpu_or_trainer(tmp_path,monkeypatch):
    (tmp_path/"evidence.txt").write_text("keep")
    monkeypatch.setattr(entry,"resolve",lambda p:tmp_path)
    with pytest.raises(FileExistsError,match="nonempty"): entry.preflight(config())
    assert (tmp_path/"evidence.txt").read_text() == "keep"


def test_history_drift_stops_before_trainer(monkeypatch):
    monkeypatch.setattr(entry,"sha",lambda p:"changed")
    with pytest.raises(RuntimeError,match="Protected environment/history changed"):
        entry.preflight(config())
