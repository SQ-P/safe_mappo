from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from src.safety.staged_execution.serial_conflict_audit import (
    STAGES, event_metrics, candidate_actions, screen_candidates, capture_prefix,
    audit_state, attach_factual_parity, summarize, guard_evidence)
from src.safety.staged_execution.contracts import ObservableObstacleV1


def synthetic(n=3):
    context = {"positions": [[10., 10.+i*10] for i in range(n)], "velocities": [[0., 0.]]*n,
        "goals": [[90., 10.+i*10] for i in range(n)], "subgoals": [[10., 90.]]*n,
        "dt": .5, "max_acceleration": 1., "max_speed": 2.}
    stages = {s: [[0., 0.]]*n for s in STAGES}
    return context, stages


@pytest.mark.parametrize("value,expected", [(.099999, False), (.1, True), (.100001, True)])
def test_e1_physical_threshold_is_per_agent_and_inclusive(value, expected):
    c, s = synthetic(); s["obstacle_ttc"] = [[value, 0.], [0., 0.], [0., 0.]]
    r = event_metrics(c, s, [], [])
    assert (0 in r["e1_material_agents"]) is expected
    assert r["nominal_delta_v_mps"][0] == pytest.approx(value*.5)


def test_tiny_direction_reversal_does_not_qualify_as_material():
    c, s = synthetic(); s["obstacle_ttc"] = [[1e-4, 0.]]*3; s["final_joint_guard"] = [[-1e-4, 0.]]*3
    assert not event_metrics(c, s, [], [])["e1_material_agents"]


@pytest.mark.parametrize("value,expected", [(.199999, False), (.2, True), (.200001, True)])
def test_e3_final_goal_threshold_and_local_goal_are_separate(value, expected):
    c, s = synthetic(); s["obstacle_ttc"] = [[value, 0.]]*3
    r = event_metrics(c, s, [], [])
    assert bool(r["e3_loss_agents"]) is expected
    assert r["subgoal_velocity_mps"]["ttc"][0] == 0


def test_saturation_and_undefined_goal_do_not_create_false_progress_loss():
    c, s = synthetic(); c["velocities"] = [[2., 0.]]*3
    s["obstacle_ttc"] = [[1., 0.]]*3; s["final_joint_guard"] = [[.5, 0.]]*3
    r = event_metrics(c, s, [], [])
    assert r["e1_material_agents"] and not r["e3_loss_agents"]
    assert r["actual_clipped_velocity_delta_mps"] == [0., 0., 0.]
    c["goals"] = c["positions"]
    assert event_metrics(c, s, [], [])["goal_direction_undefined_agents"] == [0, 1, 2]


def test_e2_is_only_same_state_rule_retrigger_and_preserves_track_identity():
    c, s = synthetic()
    risk = {"agent_id": 1, "track_id": 8, "ttc_seconds": 1., "predicted_clearance_m": -.2, "emergency": True}
    assert event_metrics(c, s, [], [risk])["e2_rule_retrigger"]
    assert event_metrics(c, s, [], [risk])["e2_new_ttc_risks"] == [risk]
    assert not event_metrics(c, s, [risk], [risk])["e2_rule_retrigger"]


def test_candidates_are_bounded_deduplicated_and_deterministic():
    c, s = synthetic(12)
    visible = tuple((ObservableObstacleV1(i, 1, (2., 1.), (-1., 0.), 1., True),) for i in range(12))
    before = deepcopy((c, s))
    first, coverage = candidate_actions(c, s, visible, list(range(12)))
    second, again = candidate_actions(c, s, visible, list(reversed(range(12))))
    assert len(first) <= 64 and coverage["budget_exhausted"] and not coverage["exhaustive_action_space"]
    assert all(v <= 8 for v in coverage["local_action_counts"].values())
    assert len({tuple(a.flatten()) for _, a in first}) == len(first)
    assert coverage == again and [n for n, _ in first] == [n for n, _ in second]
    assert all(np.array_equal(a, b) for (_, a), (_, b) in zip(first, second))
    assert (c, s) == before


def test_every_candidate_is_rechecked_as_a_whole_joint_action_and_unknowns_block_pass():
    class Fake:
        def _ttc_candidate_shadow(self, actions, **kw):
            assert actions.shape == (3, 2)
            return {"safe": True, "minimum_safety_margin_m": 1., "first_executed_actions": actions.copy()}
    c, s = synthetic(); visible = ((), (), ())
    rows, coverage = screen_candidates(Fake(), c, s, visible, [0, 1], ["INVALID_TRACK_VELOCITY"], False)
    assert all(r["screen_status"] == "EVIDENCE_INSUFFICIENT" for r in rows)
    assert coverage["no_candidate_found"] is None and not coverage["potential_opportunity_found"]
    assert coverage["screen_outcome"] == "CANDIDATES_UNRESOLVED"
    assert coverage["screen_status_counts"] == {"PASS": 0, "REJECT": 0, "UNKNOWN": len(rows), "PREEXISTING": 0}
    assert coverage["budget_status"] == "WITHIN_REGISTERED_BUDGET"
    rows, coverage = screen_candidates(Fake(), c, s, visible, [0], [], True)
    assert all(r["screen_status"] == "PREEXISTING_VIOLATION" for r in rows)
    assert coverage["no_candidate_found"] is None
    assert coverage["screen_status_counts"]["PREEXISTING"] == len(rows)


@pytest.mark.parametrize("safe", [True, False])
def test_candidate_pass_and_fully_rejected_registry_are_distinct(safe):
    class Fake:
        def _ttc_candidate_shadow(self, actions, **kw):
            return {"safe": safe, "minimum_safety_margin_m": 1. if safe else -1.,
                "first_executed_actions": actions.copy()}
    c, s = synthetic()
    rows, coverage = screen_candidates(Fake(), c, s, ((), (), ()), [0], [], False)
    assert coverage["no_candidate_found"] is (not safe)
    assert coverage["screen_outcome"] == ("CANDIDATE_SCREEN_PASS_FOUND" if safe else "NO_CANDIDATE_FOUND")
    assert coverage["screen_status_counts"]["PASS" if safe else "REJECT"] == len(rows)


def test_safe_prestate_is_not_reclassified_by_requested_action_or_projection_failure():
    from tests.test_ppcsa_contracts import normal_environment
    env = normal_environment()
    for agent, position in zip(env.agents, ([30., 30.], [60., 30.], [90., 30.])):
        agent.position = np.array(position)
        agent.velocity = np.zeros(2)
    initial = {"invariant_violations": 0, "pair_invariant_violations": 1,
        "pair_infeasible": True, "recovery_required": True, "minimum_pair_margin": -.5,
        "requested_pair_min_distance": 0.}
    result = guard_evidence(env, initial)
    assert not result["current_state"]["preexisting_violation"]
    assert result["requested_action_diagnostics"]["requested_pair_clearance_violation"]
    assert result["projection_residual_diagnostics"]["pair_infeasible"]
    env.agents[1].position = env.agents[0].position.copy()
    assert guard_evidence(env, {})["current_state"]["preexisting_violation"]


def test_safe_brake_audit_retains_nested_recovery_context_without_mutation():
    from tests.test_ppcsa_contracts import normal_environment
    from src.safety.ppcsa_one_step import integrity_token
    env = normal_environment()
    env._safety_mode = "SAFE_BRAKE"; env._safe_brake_active = True
    env.recovery_phase = "ESCAPE"; env._safe_recovery_elapsed_steps = 7
    env._star_recovery_affected_mask = np.array([True, False, True])
    env._safe_recovery_target = env.positions.copy()+1.
    before = integrity_token(env, None)
    row = audit_state(env, np.zeros((3, 2)), 23)
    assert row["evidence_status"] != "AUDIT_UNSUPPORTED", row.get("evidence_reasons")
    state = row["recovery_state"]
    assert state["mode"] == "SAFE_BRAKE" and state["brake_active"] is True
    assert state["phase"] == "ESCAPE" and state["affected_agent_mask"] == [True, False, True]
    assert state["scalars_and_counters"]["_safe_recovery_elapsed_steps"] == 7
    assert state["arrays_and_masks"]["_star_recovery_affected_mask"] == [True, False, True]
    assert state["targets_and_top_level_context"]["safe_recovery_target"] == env._safe_recovery_target.tolist()
    assert integrity_token(env, None) == before


def test_capture_and_same_state_audit_preserve_source_and_match_factual_old_chain():
    from tests.test_ppcsa_contracts import normal_environment
    from src.safety.ppcsa_one_step import integrity_token, digest
    env = normal_environment(); hidden = torch.zeros(1, 3, 128)
    raw = np.full((3, 2), .1, dtype=np.float32)
    before = integrity_token(env, hidden)
    captured = capture_prefix(env, raw, hidden)
    assert integrity_token(env, hidden) == before
    assert digest(captured) == digest(capture_prefix(env, raw, hidden))
    row = audit_state(env, raw, 23, hidden)
    assert row["source_unchanged"] and integrity_token(env, hidden) == before
    assert row["evidence_status"] != "AUDIT_UNSUPPORTED", row.get("evidence_reasons")
    from src.safety.ppcsa_one_step import jsonable
    from scripts.diagnose_ppcsa_direction_feasibility import json_safe
    json.dumps(json_safe(jsonable(row)), allow_nan=False)
    _, _, _, _, info = env.step(raw)
    attach_factual_parity(row, info)
    assert row["factual_parity"] == "PASS", row.get("factual_stage_parity")


def test_recovery_is_counted_and_unsupported_capture_does_not_disappear(monkeypatch):
    from tests.test_ppcsa_contracts import normal_environment
    import src.safety.staged_execution.serial_conflict_audit as core
    env = normal_environment(); env._safety_mode = "SAFE_BRAKE"
    monkeypatch.setattr(core, "capture_prefix", lambda *a: (_ for _ in ()).throw(ValueError("synthetic unsupported recovery")))
    row = core.audit_state(env, np.zeros((3, 2)), 23)
    assert row["evidence_status"] == "AUDIT_UNSUPPORTED" and row["recovery_mode"] == "SAFE_BRAKE"
    attach_factual_parity(row, {"action_stage_actions": {s: np.zeros((3, 2)) for s in STAGES}})
    assert not row["potential_progress_opportunity"] and row["factual_stages"]
    result = summarize([row], [23])
    assert result["audited_states"] == 1 and result["next_stage_decision"] == "INSUFFICIENT_MECHANISM_EVIDENCE"


def test_failed_factual_parity_excludes_apparent_opportunity():
    row = {"stages": {s: [[0., 0.]] for s in STAGES}, "evidence_status": "OBSERVABLE_SCREEN_ONLY",
        "prefix_deterministic": True, "source_unchanged": True, "preexisting_violation": False,
        "evidence_reasons": [], "candidate_coverage": {"potential_opportunity_found": True}}
    attach_factual_parity(row, {"action_stage_actions": {s: [[1., 0.]] for s in STAGES}})
    assert row["factual_parity"] == "FACTUAL_PARITY_FAILED" and not row["potential_progress_opportunity"]


def test_active_recovery_never_enters_stateless_candidate_screening(monkeypatch):
    from tests.test_ppcsa_contracts import normal_environment
    import src.safety.staged_execution.serial_conflict_audit as core
    env = normal_environment(); env._safety_mode = "SAFE_BRAKE"
    calls = []
    def forbidden(*args):
        calls.append(True)
        raise AssertionError("Recovery used as stateless candidates")
    monkeypatch.setattr(core, "screen_candidates", forbidden)
    row = core.audit_state(env, np.zeros((3, 2)), 23)
    assert not calls and row["recovery_mode"] == "SAFE_BRAKE"
    assert not row["mechanism_eligible"]


def test_cli_preflight_does_not_instantiate_or_run_replay(tmp_path, monkeypatch):
    import scripts.audit_serial_guard_ttc as cli
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "prepare", lambda *a: {"layout_seeds": list(range(121000, 121200))})
    monkeypatch.setattr(cli, "run_replay", lambda *a: (_ for _ in ()).throw(AssertionError("preflight replayed")))
    result = cli.main(["--config", "config.yaml", "--checkpoint", "model.pt", "--layout-seeds-file", "panel.json",
        "--output-dir", "new_output", "--preflight-only"])
    assert not result["diagnostic_run"] and not (tmp_path/"new_output").exists()


def test_summary_gate_is_review_only_and_negative_evidence_is_not_retuned():
    rows = [{"layout_seed": 121000+i//2, "e1_material_agents": [0], "e2_rule_retrigger": False,
        "e3_loss_agents": [0], "evidence_status": "OBSERVABLE_SCREEN_ONLY", "factual_parity": "PASS",
        "potential_progress_opportunity": True} for i in range(20)]
    rows += [{**rows[0], "layout_seed": seed, "potential_progress_opportunity": False} for seed in range(121010, 121200)]
    result = summarize(rows, list(range(121000, 121200)))
    assert result["next_stage_decision"] == "P1_CONTRACT_REVIEW_ONLY" and not result["p1_implementation_authorized"]
    assert summarize(rows[:19], []) ["next_stage_decision"] == "INSUFFICIENT_MECHANISM_EVIDENCE"
