"""Frozen CPU six-coordinate sensitivity measurement. Real execution is user-only."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.review_ppcsa_common_veto import bind_inputs as bind_previous, unchanged
from scripts.diagnose_ppcsa_direction_feasibility import file_sha, json_safe
from src.algorithms.ppcsa_common_veto_diagnostics import validate_evidence
from src.algorithms.ppcsa_parameter_sensitivity import (
    COORDINATES, digest, protocol, parameterization, max_calls, measure_cohort)

IMPLEMENTATION_FILES = (Path(__file__).resolve(), ROOT/"src/algorithms/ppcsa_parameter_sensitivity.py")
ADAPTER_SHAPES = {"normalization.weight": [128], "normalization.bias": [128],
    "down.weight": [16, 128], "down.bias": [16], "up.weight": [2, 16], "up.bias": [2]}


def resolve(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT/path).resolve()


def bind_sources(admission_path, feasibility_path, common_path):
    admission, feasibility, protected = bind_previous(admission_path, feasibility_path)
    validation = validate_evidence(admission, feasibility)
    common = json.loads(common_path.read_text(encoding="utf-8"))
    if (common.get("schema_version") != "PPCSACommonVetoReviewV1"
            or common.get("status") != "COMMON_VETO_REVIEW_COMPLETE"
            or common.get("source_admission_report_sha256") != protected[str(admission_path)]
            or common.get("source_feasibility_report_sha256") != protected[str(feasibility_path)]
            or common.get("input_file_sha256") != protected
            or common.get("fixed_state_registry") != admission["fixed_state_registry"]
            or common.get("common_veto_constraints") != validation["common_veto_constraints"]
            or common.get("constraints") != validation["constraints"]
            or common.get("source_artifacts_unchanged") is not True
            or common.get("used_partitions") != ["fit", "calibration"]
            or common.get("parameter_sensitivities", {}).get("status") != "PARAMETER_SENSITIVITIES_UNAVAILABLE"):
        raise ValueError("Common-veto report provenance/coverage mismatch")
    implementations = common.get("implementation_sha256")
    if not isinstance(implementations, dict) or not implementations:
        raise ValueError("Missing common-veto implementation hashes")
    for name, expected in implementations.items():
        path = resolve(name); path.relative_to(ROOT.resolve())
        if file_sha(path) != expected:
            raise ValueError("Common-veto implementation drift")
        protected[str(path)] = expected
    protected[str(common_path)] = file_sha(common_path)
    if (admission.get("hidden_dim"), admission.get("adapter_rank"), admission.get("adapter_scale")) != (128, 16, .15):
        raise ValueError("Frozen Adapter architecture differs")
    bindings = {"admission_sha256": protected[str(admission_path)],
        "registry_sha256": admission["fixed_state_registry_sha256"], "checkpoint_sha256": admission["checkpoint_sha256"]}
    return admission, protected, validation, bindings


def perturb_adapter(adapter, axis, amount):
    """Virtual coordinate copy only; source remains bitwise frozen."""
    import torch
    if {k: list(p.shape) for k, p in adapter.named_parameters()} != ADAPTER_SHAPES:
        raise ValueError("Unexpected parameter ordering/dimensions")
    if torch.count_nonzero(adapter.up.weight).item() or torch.count_nonzero(adapter.up.bias).item():
        raise ValueError("Measurement requires the fixed zero-up initialization")
    if axis is not None and (type(axis) is not int or not 0 <= axis < len(COORDINATES)):
        raise ValueError("Unknown fixed coordinate")
    if (axis is None and amount != 0.) or (axis is not None and abs(amount) not in protocol()["scales"]):
        raise ValueError("Perturbation outside frozen signed scales")
    virtual = deepcopy(adapter)
    if axis is not None:
        name, index, _ = COORDINATES[axis]
        with torch.no_grad():
            dict(virtual.named_parameters())[name][index] += amount
    return virtual


def registry_packets(data, registry, partitions, snapshot_digest):
    """Do not inspect Audit packets or select a new favorable state cohort."""
    allowed = set(partitions["fit"]+partitions["calibration"])
    records = {}
    for record in data["records"]:
        if record["layout"] not in allowed:
            continue
        key = f"{record['layout']}:{record['step']}"
        if key in records:
            raise ValueError("Duplicate exposed state record")
        records[key] = record
    packets = {}
    for fixed in registry:
        record = records[fixed["key"]]
        if record["state_index"] != fixed["state_index"]:
            raise ValueError("Frozen state index differs")
        packet = data["states"][record["state_index"]]
        if (packet["layout_seed"] != record["layout"]
                or packet["snapshot"].state_fingerprint != fixed["state_fingerprint"]
                or snapshot_digest(packet["cache"]) != fixed["cache_sha256"]
                or packet["cache_step"] != fixed["cache_step"]):
            raise ValueError("Frozen packet state/cache binding differs")
        packets[fixed["key"]] = packet
    return packets


def run_frozen_measurement(admission, protected, output, bindings):
    # Deliberately lazy: preflight cannot load a checkpoint, dataset, Actor or env.
    import torch
    from src.algorithms.ppcsa_protocol import runtime
    from src.algorithms.ppcsa_executed_value import zero_adapter
    from scripts.diagnose_ppcsa_update_acceptance import restore_checked, shadow_evidence
    from src.safety.ppcsa_one_step import digest as snapshot_digest
    candidates = [Path(p) for p in protected if Path(p).name == "opportunities.pt"]
    if len(candidates) != 1:
        raise ValueError("Exactly one bound historical opportunity dataset required")
    torch.set_num_threads(1)
    trainer, env, provenance = runtime(admission["checkpoint"], output, "cpu")
    for field in ("checkpoint_sha256", "task_config_fingerprint", "config_sha256"):
        if provenance[field] != admission[field]:
            raise ValueError("Frozen runtime provenance differs")
    actor_before = {k: v.detach().cpu().clone() for k, v in trainer.actor.state_dict().items()}
    adapter = zero_adapter().cpu().eval()
    adapter_before = {k: v.detach().clone() for k, v in adapter.state_dict().items()}
    data = torch.load(candidates[0], map_location="cpu", weights_only=False)
    packets = registry_packets(data, admission["fixed_state_registry"], admission["partitions"], snapshot_digest)
    def evaluate(key, axis, delta):
        packet = packets[key]
        hidden = restore_checked(env, packet)
        features, mean = packet["hidden_features"].float().cpu(), packet["base_mu"].float().cpu()
        if features.shape != (3, 128) or mean.shape != (3, 2):
            raise ValueError("Frozen feature/action dimensions differ")
        virtual = perturb_adapter(adapter, axis, delta)
        with torch.no_grad():
            action = torch.tanh(mean+virtual(features)).numpy()
        result = shadow_evidence(env, action, hidden)
        actual_delta = 0.
        if axis is not None:
            name, index, _ = COORDINATES[axis]
            actual_delta = float(dict(virtual.named_parameters())[name][index].detach().item())
        result["parameter_probe"] = {"axis": axis, "requested_delta": delta,
            "actual_float32_delta": actual_delta, "denominator_scope": "preregistered nominal signed scale"}
        return result
    baselines = {r["key"]: r["baseline"] for r in admission["candidates"]["ordinary"]["raw_state_evidence"]}
    result = measure_cohort(admission["fixed_state_registry"], baselines, evaluate, bindings)
    if (any(not torch.equal(actor_before[k], v.detach().cpu()) for k, v in trainer.actor.state_dict().items())
            or any(not torch.equal(adapter_before[k], v) for k, v in adapter.state_dict().items())):
        raise RuntimeError("Frozen Actor or source Adapter changed")
    unchanged(protected)
    result.update(source_actor_unchanged=True, source_adapter_unchanged=True,
        adapter_initial_state_sha256=snapshot_digest(adapter_before), critic_execution=False,
        model_execution=True, new_collection=False, source_artifacts_unchanged=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission-report", required=True)
    parser.add_argument("--feasibility-report", required=True)
    parser.add_argument("--common-veto-report", required=True)
    parser.add_argument("--output-dir", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--run-frozen-protocol", action="store_true")
    args = parser.parse_args(argv)
    a, f, c, output = map(resolve, (args.admission_report, args.feasibility_report, args.common_veto_report, args.output_dir))
    for path in (a, f, c, output):
        path.relative_to(ROOT.resolve())
    admission, protected, validation, bindings = bind_sources(a, f, c)
    if output.exists() or any(output == Path(p).parent or Path(p).parent in output.parents for p in protected):
        raise ValueError("Output must be new and outside all historical source/artifact directories")
    if args.preflight_only:
        unchanged(protected)
        print(f"PREFLIGHT=PASS; STATES={validation['fixed_state_count']}; RISK_ROWS={validation['risk_constraint_count']}; MEASURED_DIM=6; EFFECTIVE_UP_DIM=34; MAX_SHADOW_CALLS={max_calls(validation['fixed_state_count'])}; PROTOCOL_SHA={digest(protocol())}; MODELS=NOT_EXECUTED; OUTPUT_CREATED=NO")
        return {"status": "READY", "diagnostic_run": False}
    tool_hashes = {str(p): file_sha(p) for p in IMPLEMENTATION_FILES}
    output.mkdir(parents=True)
    # Freeze exact protocol + implementation before the first real evaluation.
    (output/"measurement_protocol.json").write_text(json.dumps({"protocol": protocol(),
        "protocol_sha256": digest(protocol()), "parameterization_sha256": digest(parameterization()),
        "input_file_sha256": protected, "implementation_sha256": tool_hashes}, indent=2), encoding="utf-8")
    result = run_frozen_measurement(admission, protected, output, bindings)
    unchanged(protected)
    if any(file_sha(p) != sha for p, sha in tool_hashes.items()):
        raise RuntimeError("Measurement implementation changed during run")
    result.update(schema_version="PPCSAParameterSensitivityDiagnosticV1", input_file_sha256=protected,
        implementation_sha256=tool_hashes, bindings=bindings, used_partitions=["fit", "calibration"], audit_used=False,
        actual_safety_certified=False, optimizer_executed=False)
    (output/"parameter_sensitivity_report.json").write_text(json.dumps(json_safe(result), indent=2, allow_nan=False), encoding="utf-8")
    summary = ["# PPCSA fixed-subspace parameter sensitivity", "", "Status: **"+result["status"]+"**.",
        "Measured 6 coordinates of the 34-dimensional up layer. Unmeasured derivatives are not zero-imputed.",
        f"Shadow calls: {result['shadow_calls']}; maximum: {result['maximum_shadow_calls']}.",
        "No optimizer, update acceptance, Actor training or independent efficacy validation.",
        "Repeated/scale consistency is numerical evidence, not true derivative certification.",
        "Full signed samples, errors and fidelity flags: `parameter_sensitivity_report.json`."]
    (output/"parameter_sensitivity_report.md").write_text("\n\n".join(summary)+"\n", encoding="utf-8")
    print(f"PARAMETER_SENSITIVITY={result['status']}; ACTOR_TRAINING=BLOCKED; DIRECTION=NOT_GENERATED; DETAIL_LOG={output/'parameter_sensitivity_report.json'}")
    return result


if __name__ == "__main__":
    main()
