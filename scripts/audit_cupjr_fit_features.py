"""User-run Fit-only predictive diagnostics. Does not replace an online model."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from src.utils.io_utils import load_yaml, save_json
from scripts.evaluate_safety_ablation import CHECKPOINT


def causal_feature(row, origins, observations, dt, map_size):
    """Only query observed positions at time <= prediction origin, never target."""
    layout,agent,track,step = (row[k] for k in ("layout_seed","agent_id","obstacle_track_id","step"))
    prefix = (layout,agent,track)
    previous = origins.get((*prefix,step-1))
    position = observations.get((*prefix,step))
    old_position = observations.get((*prefix,step-1))
    if previous is None or not previous["velocity_valid"] or position is None or old_position is None:
        return None
    position,old_position = np.asarray(position),np.asarray(old_position)
    velocity = (position-old_position)/dt
    relative_change = np.linalg.norm(np.asarray(row["observed_relative_velocity"])-previous["observed_relative_velocity"])
    boundary_times = []
    for axis in range(2):
        if velocity[axis] > 0:
            boundary_times.append((map_size[axis]-position[axis])/velocity[axis])
        elif velocity[axis] < 0:
            boundary_times.append(-position[axis]/velocity[axis])
    if any(t < 0. for t in boundary_times):
        return None
    boundary_proximity = 1./(1.+min(boundary_times)) if boundary_times else 0.
    values = [float(row["recent_residual_scale"]),float(relative_change),float(boundary_proximity)]
    return values if np.isfinite(values).all() else None


def feature_records(rows, dt, map_size):
    origins,observations = {},{}
    for row in rows:
        key = tuple(row[k] for k in ("layout_seed","agent_id","obstacle_track_id","step"))
        if row["prediction_horizon"] == 1:
            origins[key] = row
        if row["status"] == "observed" and row.get("later_observed_obstacle_position") is not None:
            # The observation's availability time is origin+horizon. Lookup
            # below is at the NEW prediction origin, never its future target.
            observed_key = (*key[:3],key[3]+row["prediction_horizon"])
            observations[observed_key] = row["later_observed_obstacle_position"]
    records = []
    excluded = 0
    for row in rows:
        if row["status"] != "observed" or not row["velocity_valid"] or row["prediction_horizon"] not in {2,3}:
            continue
        feature = causal_feature(row,origins,observations,dt,map_size)
        if feature is None or not np.isfinite(row["prediction_error"]):
            excluded += 1
            continue
        records.append({"layout":row["layout_seed"],"horizon":row["prediction_horizon"],
                        "features":feature,"target":row["prediction_error"]})
    return records,excluded


def grouped_prediction_diagnostic(records):
    """Four disjoint Fit-layout folds; weighted OLS, no model selection/deployment."""
    layouts = sorted({r["layout"] for r in records})
    if len(layouts) < 4:
        raise ValueError("feature diagnosis needs at least four Fit layouts with complete causal records")
    folds = {layout:i%4 for i,layout in enumerate(layouts)}
    feature_sets = {"constant":[],"recent_residual":[0],"residual_plus_velocity_change":[0,1],
                    "residual_velocity_boundary":[0,1,2]}
    report = {}
    for horizon in (2,3):
        data = [r for r in records if r["horizon"] == horizon]
        if not data:
            raise ValueError("feature diagnosis requires both horizons 2 and 3")
        x = np.asarray([r["features"] for r in data])
        y = np.asarray([r["target"] for r in data])
        groups = np.asarray([r["layout"] for r in data])
        fold_ids = np.asarray([folds[g] for g in groups])
        counts = {g:int(sum(groups == g)) for g in set(groups)}
        weights = np.asarray([1./counts[g] for g in groups])
        horizon_report = {}
        for name,indices in feature_sets.items():
            by_layout = defaultdict(list)
            for fold in range(4):
                train,test = fold_ids != fold,fold_ids == fold
                if not np.any(train) or not np.any(test):
                    raise ValueError("every horizon must have records in all four Fit folds")
                selected = x[:,indices]
                mean = np.average(selected[train],axis=0,weights=weights[train])
                std = np.sqrt(np.average((selected[train]-mean)**2,axis=0,weights=weights[train]))
                std = np.where(std > 0.,std,1.)
                a = np.column_stack([np.ones(len(y)),(selected-mean)/std])
                coef = np.linalg.lstsq(a[train]*np.sqrt(weights[train,None]),y[train]*np.sqrt(weights[train]),rcond=None)[0]
                prediction = np.maximum(0.,a[test]@coef)
                for g,error in zip(groups[test],(y[test]-prediction)**2):
                    by_layout[int(g)].append(float(error))
            mse = {str(g):float(np.mean(values)) for g,values in sorted(by_layout.items())}
            horizon_report[name] = {"layout_balanced_mse":float(np.mean(list(mse.values()))),"per_layout_mse":mse}
        baseline = horizon_report["constant"]["layout_balanced_mse"]
        for values in horizon_report.values():
            values["mse_gain_over_constant"] = baseline-values["layout_balanced_mse"]
        report[str(horizon)] = horizon_report
    return {"schema_version":"CUPJRFitFeatureDiagnosticV1","horizons":report,"layouts":layouts,
        "records":len(records),"fold_assignment":folds,
        "features":["recent_residual_scale","observed_relative_velocity_change_norm","observed_boundary_motion_proximity"],
        "method":"four Fit-layout folds, layout-balanced OLS; train-fold normalization; nonnegative error predictions",
        "feature_selection_decision":None,"online_model_changed":False,"r3_authorized":False,
        "interpretation":"Fit-only exploratory predictive information; no Calibration/Validation/Final use; review consistency across layouts, not coefficients alone"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-data",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--checkpoint",default=CHECKPOINT)
    args = parser.parse_args(argv)
    resolve = lambda p:Path(p) if Path(p).is_absolute() else ROOT/p
    source,output,checkpoint = resolve(args.fit_data),resolve(args.output),resolve(args.checkpoint)
    if output.exists():
        raise FileExistsError(output)
    raw = source.read_bytes()
    rows = [json.loads(s) for s in raw.decode("utf-8-sig").splitlines() if s.strip()]
    registered = load_yaml(ROOT/"configs/experiments/recovery_seed23/panels.json")["fit"]
    if sorted({r["layout_seed"] for r in rows}) != registered:
        raise ValueError("only the complete registered Fit panel may be used for feature diagnosis")
    provenance_path = source.parent/"provenance.json"
    provenance = json.loads(provenance_path.read_text())
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if provenance["checkpoint_sha256"] != checkpoint_sha:
        raise ValueError("Fit checkpoint provenance mismatch")
    import torch
    config = torch.load(checkpoint,map_location="cpu",weights_only=False)["config"]
    records,excluded = feature_records(rows,float(config["env"]["dt"]),config["env"]["map_size"])
    report = grouped_prediction_diagnostic(records)
    report.update(fit_data_sha256=hashlib.sha256(raw).hexdigest(),checkpoint_sha256=checkpoint_sha,
        provenance_sha256=hashlib.sha256(provenance_path.read_bytes()).hexdigest(),excluded_incomplete_causal_records=excluded)
    save_json(report,output)
    print(f"FIT_FEATURES=REQUIRES_REVIEW; R3=UNCHANGED; DETAIL_LOG={output}")


if __name__ == "__main__":
    main()
