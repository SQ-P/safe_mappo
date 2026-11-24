"""Summarize paired evaluation JSONL files without rerunning or selecting models."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.paired_research_report import compare_paired_episodes
from src.utils.io_utils import read_jsonl, save_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-files", nargs="+", required=True)
    parser.add_argument("--candidate-files", nargs="+", required=True)
    parser.add_argument("--allow-task-difference", action="store_true")
    parser.add_argument("--preregistration")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    target = Path(args.output)
    if target.exists():
        raise ValueError("report already exists; choose a new output file")
    def read_files(paths):
        rows = []
        for path in paths:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
            rows.extend(read_jsonl(path))
        return rows
    report = compare_paired_episodes(read_files(args.baseline_files), read_files(args.candidate_files),
                                    allow_task_difference=args.allow_task_difference)
    report["source_files"] = {"baseline": [str(Path(path).resolve()) for path in args.baseline_files],
                              "candidate": [str(Path(path).resolve()) for path in args.candidate_files]}
    if args.preregistration:
        prereg_path = Path(args.preregistration)
        prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
        expected_seeds = sorted(
            int(value)
            for value in prereg.get("confirmation_training_seeds", [])
        )
        actual_seeds = sorted(
            int(row["train_seed"]) for row in report["seed_pairs"]
        )
        if expected_seeds != actual_seeds:
            raise ValueError(
                "paired report training seeds differ from preregistration"
            )
        expected_episodes = int(prereg.get("evaluation_episodes", -1))
        if any(
            int(row["evaluation_episodes_per_arm"]) != expected_episodes
            for row in report["seed_pairs"]
        ):
            raise ValueError(
                "paired report episode count differs from preregistration"
            )
        estimates = report["estimates"]
        timeout_margin = float(
            prereg["timeout_noninferiority_margin_absolute"]
        )
        gates = {
            "positive_mean_cfms": (
                estimates["cfms"]["paired_mean_difference"] > 0.0
            ),
            "positive_median_cfms": (
                estimates["cfms"]["paired_median_difference"] > 0.0
            ),
            "cfms_positive_in_at_least_four_of_five_seeds": (
                estimates["cfms"]["positive_seed_count"] >= 4
            ),
            "collision_nonworsening": (
                estimates["collision_rate"]["paired_mean_difference"] <= 0.0
            ),
            "boundary_collision_improves": (
                estimates["boundary_collision_rate"]["paired_mean_difference"] < 0.0
            ),
            "timeout_within_noninferiority_margin": (
                estimates["timeout_rate"]["paired_mean_difference"]
                <= timeout_margin
            ),
        }
        report["preregistration"] = {
            "path": str(prereg_path.resolve()),
            "sha256": hashlib.sha256(prereg_path.read_bytes()).hexdigest(),
        }
        report["confirmation_decision"] = {
            "gates": gates,
            "all_gates_pass": all(gates.values()),
            "timeout_noninferiority_margin_absolute": timeout_margin,
        }
    save_json(report, target)
    print(f"Compared {report['training_seed_count']} paired training seeds; descriptive report: {target}")
    return report


if __name__ == "__main__":
    main()
