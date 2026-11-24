"""Observable-only prediction history and layout-group split calibration."""
from __future__ import annotations

from collections import defaultdict, deque
import math
import numpy as np

SCHEMA = "RecoveryUncertaintyV1"


def scale(model, horizon, recent):
    h = int(horizon) - 1
    feature = float(recent) / float(model.get("feature_scale", 1.))
    return max(1e-6, float(model["b"][h]) + float(model["c"][h]) * feature)


def feature_in_domain(model, recent):
    return 0. <= recent <= model.get("feature_max", float("inf"))


def _ridge_nonnegative(x, y, weights, penalty):
    """Exact two-variable constrained ridge, with unpenalized intercept."""
    weights = weights / weights.sum()
    a = np.column_stack([np.ones(len(x)), x])
    matrix = a.T @ (weights[:, None] * a) + np.diag([0., penalty])
    rhs = a.T @ (weights * y)
    unconstrained = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    candidates = [(max(.01, float(np.sum(weights*y))), 0.),
                  (.01, max(0., float(np.sum(weights*x*(y-.01))) /
                                 max(float(np.sum(weights*x*x)) + penalty, 1e-15)))]
    if unconstrained[0] >= .01 and unconstrained[1] >= 0.:
        candidates.append(tuple(unconstrained))
    return min(candidates, key=lambda bc: float(np.sum(weights*(y-bc[0]-bc[1]*x)**2) + penalty*bc[1]**2))


def fit_scale_v2(rows):
    """Normalized nonnegative ridge; penalty chosen using Fit layouts only."""
    valid = [r for r in rows if r["status"] == "observed" and r["velocity_valid"]]
    layouts = sorted({r["layout_seed"] for r in valid})
    if len(layouts) < 4:
        raise ValueError("v2 fitting needs at least four independent Fit layouts")
    features = np.asarray([r["recent_residual_scale"] for r in valid])
    feature_scale = max(float(np.quantile(features, .9)), 1e-3)
    model = {"version": "normalized_nonnegative_ridge_v2", "feature_scale": feature_scale,
             "feature_max": float(features.max()), "b": [], "c": [], "ridge_penalty": []}
    folds = {layout: index % 4 for index, layout in enumerate(layouts)}
    for horizon in (1, 2, 3):
        records = [r for r in valid if r["prediction_horizon"] == horizon]
        if not records:
            raise ValueError("Fit data must include every prediction horizon")
        x = np.asarray([r["recent_residual_scale"] / feature_scale for r in records])
        y = np.asarray([r["prediction_error"] for r in records])
        groups = np.asarray([r["layout_seed"] for r in records])
        counts = {g: int(np.count_nonzero(groups == g)) for g in set(groups)}
        weights = np.asarray([1./counts[g] for g in groups])
        fold_ids = np.asarray([folds[g] for g in groups])
        losses = []
        for penalty in (0., .001, .01, .1, 1., 10.):
            errors = []
            for fold in range(4):
                train, test = fold_ids != fold, fold_ids == fold
                if not np.any(train) or not np.any(test):
                    raise ValueError("each horizon requires observations in all Fit folds")
                # Normalizer is refitted inside each fold, avoiding held-out leakage.
                normalizer = max(float(np.quantile(x[train]*feature_scale, .9)), 1e-3)
                train_x = x[train]*feature_scale/normalizer
                test_x = x[test]*feature_scale/normalizer
                b, c = _ridge_nonnegative(train_x, y[train], weights[train], penalty)
                errors.append(float(np.average((y[test]-b-c*test_x)**2, weights=weights[test])))
            losses.append((float(np.mean(errors)), -penalty, penalty))
        penalty = min(losses)[2]
        b, c = _ridge_nonnegative(x, y, weights, penalty)
        model["b"].append(float(b))
        model["c"].append(float(c))
        model["ridge_penalty"].append(penalty)
    return model


def fit_scale(rows):
    """Small nonnegative linear scale fitted only on development layouts."""
    model = {"b": [], "c": []}
    for horizon in (1, 2, 3):
        valid = [r for r in rows if r.get("status") == "observed"
                 and r["velocity_valid"]
                 and r["prediction_horizon"] == horizon]
        if not valid:
            raise ValueError("fit data must contain all three horizons")
        x = np.asarray([r["recent_residual_scale"] for r in valid])
        y = np.asarray([r["prediction_error"] for r in valid])
        coefficients = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
        model["b"].append(max(0.01, float(coefficients[0])))
        model["c"].append(max(0., float(coefficients[1])))
    return model


def calibration_data_summary(rows):
    """Keep missing observations visible independently of residual coverage."""
    counts = defaultdict(int)
    layouts, observed_layouts, lost_layouts = set(), set(), set()
    valid_uncensored = 0
    loss_distances, closing_losses = [], 0
    for row in rows:
        layout = int(row["layout_seed"])
        layouts.add(layout)
        counts[row["status"]] += 1
        if not row["velocity_valid"]:
            counts["invalid_velocity"] += 1
            continue
        if row["status"] != "episode_ended":
            valid_uncensored += 1
        if row["status"] == "observed":
            observed_layouts.add(layout)
            counts["observed_valid"] += 1
        elif row["status"] == "track_lost":
            lost_layouts.add(layout)
            counts["lost_valid"] += 1
            relative = np.asarray(row.get("observed_relative_position", []))
            velocity = np.asarray(row.get("observed_relative_velocity", []))
            if relative.shape == (2,) and velocity.shape == (2,):
                loss_distances.append(float(np.linalg.norm(relative)))
                closing_losses += int(relative @ velocity < 0.)
    return {"records": len(rows), "status_counts": dict(counts),
            "layouts": len(layouts), "layouts_with_observed_valid": len(observed_layouts),
            "layouts_with_valid_track_loss": len(lost_layouts),
            "valid_uncensored_records": valid_uncensored,
            "lost_prediction_origin_distance_p50_m": float(np.median(loss_distances)) if loss_distances else None,
            "closing_valid_lost_prediction_count": closing_losses,
            "loss_statistics_unit": "prediction records; distances/motion at prediction origin, not inferred loss cause",
            "valid_track_loss_rate": counts["lost_valid"] / max(valid_uncensored, 1)}


def calibrate(rows, model, coverage, *, adaptive, scope="strict"):
    if not 0. < coverage < 1.:
        raise ValueError("coverage must lie strictly between zero and one")
    if scope not in {"strict", "observed_valid"}:
        raise ValueError("unknown calibration scope")
    groups = defaultdict(list)
    for row in rows:
        # Coverage is conditional on valid velocity and an uncensored horizon.
        # Track loss within that horizon counts as an infinite score.
        if not row["velocity_valid"] or row["status"] == "episode_ended":
            continue
        if scope == "observed_valid" and row["status"] != "observed":
            continue
        # Lost trajectories stay in the denominator and fail coverage closed.
        score = float("inf")
        if row["status"] == "observed" and row["velocity_valid"]:
            recent = row["recent_residual_scale"] if adaptive else 0.
            score = row["prediction_error"] / scale(model, row["prediction_horizon"], recent)
        groups[int(row["layout_seed"])].append(score)
    scores = sorted(max(values) for values in groups.values())
    rank = math.ceil((len(scores) + 1) * coverage)
    if not scores or rank > len(scores) or not math.isfinite(scores[rank - 1]):
        finite = sum(math.isfinite(value) for value in scores)
        raise ValueError(
            f"insufficient fully observed calibration groups for finite coverage: "
            f"scope={scope}, groups={len(scores)}, finite_groups={finite}, "
            f"required_rank={rank}, target={coverage}. "
            "Track loss is not a numeric prediction error. Keep the strict gate "
            "failed; use explicit --coverage-scope observed_valid only for "
            "conditional development calibration, not original-gate approval."
        )
    return float(scores[rank - 1])


class ObservablePredictionHistory:
    """Uses only snapshots observed after real time advances. No simulator API."""
    def __init__(self, dt, sink=None):
        self.dt = float(dt)
        self.sink = sink
        self.reset()

    def reset(self):
        self.pending = []
        self.residuals = defaultdict(lambda: deque(maxlen=8))
        self.age = defaultdict(int)
        self.last_step = None
        self.current = {}
        self.last_seen = {}
        self.latest_risk_events = []
        self.radius_events = []

    def recent(self, key):
        values = self.residuals[key]
        return float(np.median(values)) if values else 0.

    def observe(self, snapshot, layout_seed, *, terminal=False):
        if self.last_step == snapshot.step:
            return
        positions = np.asarray(snapshot.positions)
        velocities = np.asarray(snapshot.velocities)
        current = {}
        self.latest_risk_events = []
        for agent, tracks in enumerate(snapshot.visible_obstacles):
            for track in tracks:
                key = (agent, track.track_id)
                current[key] = (positions[agent] + track.relative_position,
                                velocities[agent] + track.relative_velocity, track)
                self.last_seen[key] = (positions[agent] + track.relative_position, track.radius, snapshot.step)
        previous_keys = set(self.current)
        reacquired = set(current) - previous_keys
        for key in set(self.age) - (set(current) & previous_keys):
            self.age.pop(key, None)
            self.residuals.pop(key, None)
        retained = []
        completed = []
        for row in self.pending:
            target = row["step"] + row["prediction_horizon"]
            key = (row["agent_id"], row["obstacle_track_id"])
            if key in previous_keys - set(current):
                lost = {**row, "status": "track_lost", "later_observed_obstacle_position": None,
                        "prediction_error": None}
                relative = np.asarray(row["observed_relative_position"])
                relative_velocity = np.asarray(row["observed_relative_velocity"])
                lost.update(loss_distance_m=float(np.linalg.norm(relative)),
                            loss_closing_dot=float(relative @ relative_velocity))
                self.latest_risk_events.append(lost)
                if self.sink is not None:
                    self.sink(lost)
                continue
            if target > snapshot.step and not terminal:
                retained.append(row)
                continue
            key = (row["agent_id"], row["obstacle_track_id"])
            value = current.get(key)
            observed = value is not None and key not in reacquired and target == snapshot.step
            row = dict(row)
            row["status"] = "observed" if observed else ("episode_ended" if target > snapshot.step else "track_lost")
            row["later_observed_obstacle_position"] = value[0].tolist() if observed else None
            row["prediction_error"] = float(np.linalg.norm(value[0] - row["predicted_obstacle_position"])) if observed else None
            completed.append(row)
            if observed and row["prediction_horizon"] == 1 and row["velocity_valid"]:
                self.residuals[key].append(row["prediction_error"])
            if self.sink is not None:
                self.sink(row)
        self.pending = retained
        self.current = current
        self.last_step = snapshot.step
        if terminal:
            return
        for key, (position, velocity, track) in current.items():
            self.age[key] += 1
            for horizon in (1, 2, 3):
                self.pending.append({
                    "schema_version": SCHEMA, "layout_seed": int(layout_seed),
                    "episode_id": int(layout_seed), "step": snapshot.step,
                    "agent_id": key[0], "obstacle_track_id": key[1],
                    "prediction_horizon": horizon,
                    "observed_relative_position": list(track.relative_position),
                    "observed_relative_velocity": list(track.relative_velocity),
                    "velocity_valid": track.velocity_valid,
                    "track_age": self.age[key],
                    "valid_residual_count": len(self.residuals[key]),
                    "recent_residual_scale": self.recent(key),
                    "predicted_obstacle_position": (position + velocity * self.dt * horizon).tolist(),
                })

    def radius(self, key, horizon, artifact, adaptive):
        value = self.current.get(key)
        if value is None or not value[2].velocity_valid:
            return float("inf")
        model = artifact["adaptive_model"] if adaptive else artifact["fixed_model"]
        q = artifact["adaptive_q"] if adaptive else artifact["fixed_q"]
        recent = self.recent(key) if adaptive else 0.
        # Cold starts use the larger calibrated global/adaptive radius.
        result = q * scale(model, horizon, recent)
        if len(self.residuals[key]) < 2:
            result = max(result, artifact["fixed_q"] * scale(artifact["fixed_model"], horizon, 0.))
        return result

    def radius_v2(self, key, horizon, artifact, adaptive, *, record=True):
        recent = self.recent(key)
        if adaptive and not feature_in_domain(artifact["adaptive_model"], recent):
            if record:
                self.radius_events.append({"key": key, "horizon": horizon, "mode": "ood_reachable", "recent": recent})
            return float("inf")
        result = self.radius(key, horizon, artifact, adaptive)
        if record:
            self.radius_events.append({"key": key, "horizon": horizon,
                                       "mode": "calibrated" if np.isfinite(result) else "unknown_velocity_reachable",
                                       "radius_m": float(result) if np.isfinite(result) else None})
        return result

    def lost_reachable(self, snapshot):
        """Keep last observations until reacquired; no unjustified expiry TTL."""
        return [(key[0], np.asarray(position) - np.asarray(snapshot.positions[key[0]]),
                 radius, (snapshot.step-step)*self.dt)
                for key, (position, radius, step) in self.last_seen.items()
                if key not in self.current]
