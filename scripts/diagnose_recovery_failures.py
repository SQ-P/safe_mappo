"""Summarize recorded failure evidence without assigning unsupported causes."""
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.evaluate_recovery_study import read_rows, resolve
from src.utils.io_utils import save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError(output)
    rows = read_rows(resolve(args.episodes))
    failures = []
    for row in rows:
        if row["safe_success"]:
            continue
        events = row.get("safe_recovery_events", [])
        failures.append({"layout_seed": row["seed"], "reason": row["reason"],
            "dynamic_collision_count": row["dynamic_obstacle_collision_count"],
            "attribution": "unknown_requires_causal_trace",
            "evidence": {
                "F4_no_feasible_candidate_events": sum(e.get("event") == "robust_no_feasible_candidate" for e in events),
                "F5_projection_rejections": row.get("recovery_candidate_rejection_reason_counts", {}),
                "F6_recovery_failure_count": row.get("safe_recovery_failure_count", 0),
                "F6_persistent_recovery_timeout": row.get("persistent_safe_recovery_timeout", False),
            }})
    save_json({"schema_version": "RecoveryFailureEvidenceV1", "episodes": len(rows),
        "failures": failures, "interpretation": "Counters are evidence signals, not causal attribution. F1/F2/F3/F7 require matched stepwise observations; unknown is retained."}, output)


if __name__ == "__main__":
    main()
