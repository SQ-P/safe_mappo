from copy import deepcopy

import pytest

from src.diagnostics.paired_research_report import compare_paired_episodes
from src.utils.io_utils import save_jsonl
from scripts.compare_research_pairs import main


def rows():
    return [dict(train_seed=train_seed, seed=eval_seed, update=400, algorithm="ppo", collision_mode="terminal",
                 task_config_fingerprint="a", safe_success=False, reach_success=False, collision_count=0,
                 terminated=False, truncated=True)
            for train_seed in (23, 41, 59) for eval_seed in (41000, 41001)]


def test_episodes_are_not_treated_as_training_replicates():
    baseline, candidate = rows(), rows()
    for row in candidate:
        if row["seed"] == 41000:
            row.update(safe_success=True, reach_success=True, terminated=True, truncated=False)
    report = compare_paired_episodes(baseline, candidate)
    assert report["training_seed_count"] == 3
    assert report["estimates"]["cfms"]["paired_mean_difference"] == .5
    assert report["estimates"]["timeout_rate"]["paired_mean_difference"] == -.5
    assert "screening only" in report["warnings"][-1]
    assert report == compare_paired_episodes(baseline, candidate)


@pytest.mark.parametrize("problem", ["duplicate", "missing_seed", "eval_seed", "update", "task", "contradiction"])
def test_invalid_pairs_fail_closed(problem):
    baseline, candidate = rows(), rows()
    if problem == "duplicate":
        candidate.append(deepcopy(candidate[0]))
    elif problem == "missing_seed":
        candidate = [row for row in candidate if row["train_seed"] != 59]
    elif problem == "eval_seed":
        candidate[0]["seed"] = 42000
    elif problem == "update":
        candidate[0]["update"] = 600
    elif problem == "task":
        for row in candidate:
            row["task_config_fingerprint"] = "b"
    else:
        candidate[0]["safe_success"] = True
    with pytest.raises(ValueError):
        compare_paired_episodes(baseline, candidate)


def test_task_change_requires_explicit_declaration_and_one_seed_has_no_interval():
    baseline = rows()[:2]
    candidate = deepcopy(baseline)
    for row in candidate:
        row["task_config_fingerprint"] = "b"
    report = compare_paired_episodes(baseline, candidate, allow_task_difference=True)
    assert report["declared_task_difference_seeds"] == [23]
    assert report["estimates"]["cfms"]["seed_bootstrap_percentile_95_interval"] is None


def test_report_cli_writes_provenance_and_never_overwrites(tmp_path):
    source = tmp_path / "episodes.jsonl"
    save_jsonl(rows(), source)
    target = tmp_path / "report.json"
    args = ["--baseline-files", str(source), "--candidate-files", str(source), "--output", str(target)]
    report = main(args)
    assert target.is_file()
    assert report["source_files"]["baseline"] == [str(source.resolve())]
    assert report["estimates"]["cfms"]["paired_mean_difference"] == 0.
    with pytest.raises(ValueError, match="already exists"):
        main(args)


def test_confirmation_report_applies_preregistered_seed_level_gates(tmp_path):
    baseline = [
        dict(
            train_seed=train_seed, seed=eval_seed, update=600,
            algorithm="ppo", collision_mode="terminal",
            task_config_fingerprint="a", safe_success=False,
            reach_success=False, collision_count=1,
            boundary_collision_count=1, terminated=True, truncated=False,
        )
        for train_seed in (211, 223, 227, 229, 233)
        for eval_seed in (160000, 160001)
    ]
    candidate = deepcopy(baseline)
    for row in candidate:
        row.update(
            safe_success=True,
            reach_success=True,
                collision_count=0,
            terminated=True,
            truncated=False,
            boundary_collision_count=0,
        )
        row["task_config_fingerprint"] = "b"
    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    save_jsonl(baseline, baseline_path)
    save_jsonl(candidate, candidate_path)
    prereg = tmp_path / "preregister.json"
    prereg.write_text(
        '{"confirmation_training_seeds":[211,223,227,229,233],'
        '"evaluation_episodes":2,'
        '"timeout_noninferiority_margin_absolute":0.05}',
        encoding="utf-8",
    )
    report = main([
        "--baseline-files", str(baseline_path),
        "--candidate-files", str(candidate_path),
        "--allow-task-difference",
        "--preregistration", str(prereg),
        "--output", str(tmp_path / "confirmation.json"),
    ])
    assert report["confirmation_decision"]["all_gates_pass"]
    assert report["estimates"]["boundary_collision_rate"][
        "paired_mean_difference"
    ] == -1.0
