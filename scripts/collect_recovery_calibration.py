"""Collect observable prediction residuals or fit/calibrate CUP-JR offline."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.evaluate_recovery_study import run_arm, resolve, read_rows, digest
from scripts.evaluate_safety_ablation import CHECKPOINT
from src.safety.staged_execution.uncertainty import fit_scale, fit_scale_v2, calibrate, scale, calibration_data_summary, feature_in_domain
from src.utils.io_utils import load_yaml, save_json


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["collect", "calibrate", "coverage"])
    p.add_argument("--panel", choices=["fit", "calibration", "validation"], default="fit")
    p.add_argument("--output-dir")
    p.add_argument("--checkpoint", default=CHECKPOINT)
    p.add_argument("--device", default="cpu")
    p.add_argument("--protocol", choices=["v1", "v2"], default="v1", help="collection executor protocol")
    p.add_argument("--fit-dir")
    p.add_argument("--calibration-dir")
    p.add_argument("--artifact")
    p.add_argument("--data-dir")
    p.add_argument("--target-coverage", type=float, default=.95)
    p.add_argument("--scale-model", choices=["legacy", "normalized_ridge_v2"], default="legacy")
    p.add_argument("--coverage-scope", choices=["strict", "observed_valid"], default="strict",
                   help="observed_valid is an explicit conditional development protocol; loss remains separately reported")
    args = p.parse_args(argv)
    if args.action == "collect":
        if not args.output_dir:
            p.error("collect requires --output-dir")
        panels = load_yaml(ROOT / "configs/experiments/recovery_seed23/panels.json")
        run_arm("R0", args.checkpoint, panels[args.panel], args.output_dir, args.device, collect=True, protocol=args.protocol)
    elif args.action == "calibrate":
        if not all((args.fit_dir, args.calibration_dir, args.artifact)):
            p.error("calibrate requires --fit-dir --calibration-dir --artifact")
        destination = resolve(args.artifact)
        if destination.exists():
            raise FileExistsError(destination)
        fit_dir, cal_dir = resolve(args.fit_dir), resolve(args.calibration_dir)
        fit = read_rows(fit_dir / "predictions.jsonl")
        calibration = read_rows(cal_dir / "predictions.jsonl")
        fit_seeds = sorted({r["layout_seed"] for r in fit})
        cal_seeds = sorted({r["layout_seed"] for r in calibration})
        if set(fit_seeds) & set(cal_seeds):
            raise ValueError("fit and calibration layouts overlap")
        panels = load_yaml(ROOT / "configs/experiments/recovery_seed23/panels.json")
        if fit_seeds != panels["fit"] or cal_seeds != panels["calibration"]:
            raise ValueError("fit/calibration data must match registered panels completely")
        provenance = [json.loads((directory / "provenance.json").read_text()) for directory in (fit_dir, cal_dir)]
        if provenance[0]["checkpoint_sha256"] != provenance[1]["checkpoint_sha256"]:
            raise ValueError("fit/calibration policy differs")
        if provenance[0]["runtime_task_fingerprint"] != provenance[1]["runtime_task_fingerprint"]:
            raise ValueError("fit/calibration executor contracts differ")
        adaptive = fit_scale_v2(fit) if args.scale_model == "normalized_ridge_v2" else fit_scale(fit)
        fixed = {"b": adaptive["b"], "c": [0., 0., 0.]}
        if args.scale_model == "normalized_ridge_v2":
            import numpy as np
            fixed = {"b": [max(.01, float(np.mean([r["prediction_error"] for r in fit
                if r["status"] == "observed" and r["velocity_valid"] and r["prediction_horizon"] == h])))
                for h in (1, 2, 3)], "c": [0., 0., 0.]}
        data_summary = calibration_data_summary(calibration)
        print("CALIBRATION_DATA: " + json.dumps(data_summary), flush=True)
        strict_gate_passed = True
        try:
            for model, adaptive_mode in ((adaptive, True), (fixed, False)):
                calibrate(calibration, model, args.target_coverage, adaptive=adaptive_mode)
        except ValueError:
            strict_gate_passed = False
        artifact = {"schema_version": "CUPJRCalibrationV1", "target_coverage": args.target_coverage,
                    "scale_model_version": args.scale_model,
                    "calibration_scope": args.coverage_scope,
                    "strict_gate_passed": strict_gate_passed,
                    "development_only": args.coverage_scope == "observed_valid",
                    "calibration_data_summary": data_summary,
                    "coverage_definition": "layout maximum over observed valid predictions only; track loss is separately reported" if args.coverage_scope == "observed_valid" else "layout maximum with valid track losses scored infinity",
                    "adaptive_model": adaptive, "fixed_model": fixed,
                    "adaptive_q": calibrate(calibration, adaptive, args.target_coverage, adaptive=True, scope=args.coverage_scope),
                    "fixed_q": calibrate(calibration, fixed, args.target_coverage, adaptive=False, scope=args.coverage_scope),
                    "fit_layouts": fit_seeds, "calibration_layouts": cal_seeds,
                    "checkpoint_sha256": provenance[0]["checkpoint_sha256"],
                    "fit_data_sha256": digest(fit), "calibration_data_sha256": digest(calibration)}
        save_json(artifact, destination)
        print(f"CALIBRATED: {destination}")
    else:
        if not all((args.artifact, args.data_dir, args.output_dir)):
            p.error("coverage requires --artifact --data-dir --output-dir")
        artifact = json.loads(resolve(args.artifact).read_text())
        rows = read_rows(resolve(args.data_dir) / "predictions.jsonl")
        if {r["layout_seed"] for r in rows} & set(artifact["fit_layouts"] + artifact["calibration_layouts"]):
            raise ValueError("coverage data is not independent")
        output = resolve(args.output_dir)
        if output.exists():
            raise FileExistsError(output)
        report = {}
        for adaptive in (False, True):
            groups, radii = {}, []
            strict_groups = {}
            for row in rows:
                eligible = row["velocity_valid"] and row["status"] != "episode_ended"
                if not eligible:
                    continue
                model = artifact["adaptive_model" if adaptive else "fixed_model"]
                radius = artifact["adaptive_q" if adaptive else "fixed_q"] * scale(model, row["prediction_horizon"], row["recent_residual_scale"] if adaptive else 0.)
                if adaptive and row.get("valid_residual_count", 0) < 2:
                    radius = max(radius, artifact["fixed_q"] * scale(artifact["fixed_model"], row["prediction_horizon"], 0.))
                radii.append(radius)
                covered = row["status"] == "observed" and row["prediction_error"] <= radius
                key = row["layout_seed"]
                strict_groups[key] = strict_groups.get(key, True) and covered
                if artifact.get("calibration_scope", "strict") == "strict" or row["status"] == "observed":
                    groups[key] = groups.get(key, True) and covered
            import numpy as np
            report["adaptive" if adaptive else "fixed"] = {"layout_group_coverage": float(np.mean(list(groups.values()))) if groups else None,
                "strict_layout_group_coverage_including_losses": float(np.mean(list(strict_groups.values()))) if strict_groups else None,
                "groups": len(groups), "mean_radius": float(np.mean(radii)) if radii else None,
                "p95_radius": float(np.quantile(radii, .95)) if radii else None}
            report["adaptive" if adaptive else "fixed"].update(
                radius_p50=float(np.quantile(radii, .5)) if radii else None,
                radius_p99=float(np.quantile(radii, .99)) if radii else None)
        report["missing_status_counts"] = {status: sum(r["status"] == status for r in rows) for status in {r["status"] for r in rows}}
        report["invalid_velocity_count"] = sum(not r["velocity_valid"] for r in rows)
        report["calibration_scope"] = artifact.get("calibration_scope", "strict")
        report["data_summary"] = calibration_data_summary(rows)
        report["adaptive_out_of_fit_range_count"] = sum(
            not feature_in_domain(artifact["adaptive_model"], r["recent_residual_scale"])
            for r in rows if r["velocity_valid"])
        report["radius_interpretation"] = "prediction-model radii; v2 online OOD/unknown tracks use physical reachable sets instead of extrapolation"
        save_json(report, output / "coverage.json")


if __name__ == "__main__":
    main()
