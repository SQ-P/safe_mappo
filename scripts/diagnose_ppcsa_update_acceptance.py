"""User-run whole-update virtual admission, fixed cohort/directions/step; no training."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from src.algorithms.ppcsa_protocol import resolve,file_sha,new_output,runtime
from src.algorithms.ppcsa_executed_value import ExecutedValueCritic,zero_adapter,adapter_jacobian
from src.algorithms.ppcsa_mechanism_diagnostics import tangent_trace,layout_weights,virtual_adapter
from src.algorithms.ppcsa_update_acceptance import CANDIDATES,PARAMETER_STEP,candidate_admission,candidate_set_admission
from src.safety.ppcsa_one_step import restore_packet,cache_payload,integrity_token,evaluate_ppcsa_normal_shadow,digest,jsonable
from scripts.diagnose_ppcsa_shared_update import validate_inputs,selected_records,score_functions
from src.utils.io_utils import save_json

OLD_WARNING_CHECK = '    if float(adapter.up.weight.abs().max()) or float(adapter.up.bias.abs().max()):'
NEW_WARNING_CHECK = '    if adapter.up.weight.detach().abs().max().item() or adapter.up.bias.detach().abs().max().item():'
PHASE1_SOURCES = ("scripts/diagnose_ppcsa_shared_update.py","src/algorithms/ppcsa_mechanism_diagnostics.py",
    "src/algorithms/ppo/networks.py","scripts/fit_ppcsa_executed_value.py")


def key(record):
    return f"{record['layout']}:{record['step']}"


def historical_source_match(name,expected):
    path = ROOT/name
    if file_sha(path) == expected:
        return "exact"
    # Preserve the old report; permit ONLY the prescribed non-mathematical
    # scalar-check correction, never a general source-hash bypass.
    if name == "src/algorithms/ppcsa_mechanism_diagnostics.py":
        content = path.read_bytes()
        new,old = NEW_WARNING_CHECK.encode(),OLD_WARNING_CHECK.encode()
        if content.count(new) == 1 and hashlib.sha256(content.replace(new,old)).hexdigest() == expected:
            return "exact_except_detach_item_warning_check"
    raise ValueError("Phase1 implementation drift: "+name)


def bind_inputs(gate1,critic,mechanism_path):
    source,_ = validate_inputs(gate1,critic)
    mechanism = json.loads(mechanism_path.read_text(encoding="utf-8"))
    if mechanism.get("schema_version") != "PPCSASharedMechanismDiagnosticV1" or mechanism.get("partitions") != source["partitions"]:
        raise ValueError("Phase1 report/partition mismatch")
    for name in ("checkpoint_sha256","config_sha256","task_config_fingerprint"):
        if mechanism.get(name) != source[name]:
            raise ValueError("Phase1 parent/config mismatch")
    if file_sha(resolve(source["checkpoint"])) != source["checkpoint_sha256"]:
        raise ValueError("Parent checkpoint drift")
    if (mechanism.get("audit_used") is not False or mechanism.get("used_partitions") != ["fit","calibration"]
            or not all(mechanism.get(k) is True for k in ("source_actor_unchanged","source_adapter_unchanged","source_artifacts_unchanged"))):
        raise ValueError("Phase1 isolation/integrity evidence missing")
    recorded = mechanism.get("diagnostic_source_sha256",{})
    if set(recorded) != set(PHASE1_SOURCES):
        raise ValueError("Phase1 implementation binding missing")
    compatibility = {name:historical_source_match(name,recorded[name]) for name in PHASE1_SOURCES}
    for name,sha in mechanism["input_file_sha256"].items():
        if file_sha(name) != sha:
            raise ValueError("Historical input artifact changed: "+name)
    if (mechanism["input_file_sha256"].get(str(gate1/"manifest.json")) != file_sha(gate1/"manifest.json")
            or mechanism["input_file_sha256"].get(str(critic/"manifest.json")) != file_sha(critic/"manifest.json")):
        raise ValueError("Phase1 report belongs to different input directories")
    data = torch.load(gate1/"opportunities.pt",map_location="cpu",weights_only=False)
    cohort,_ = selected_records(data,source["partitions"])
    expected = [key(r) for r,p in cohort]
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("Empty/duplicate fixed cohort")
    registry = [{"key":key(r),"state_index":r["state_index"],"state_fingerprint":p["snapshot"].state_fingerprint,
        "cache_sha256":digest(p["cache"]),"cache_step":p["cache_step"]} for r,p in cohort]
    for name in CANDIDATES:
        traces = mechanism["models"][name]["state_traces"]
        if [key(t) for t in traces] != expected:
            raise ValueError("Phase1 cohort differs; do not silently drop states")
    teachers = [(r,p) for r,p in cohort if r["layout"] in source["partitions"]["calibration"] and r.get("best_local_delta") is not None]
    if not teachers or any(mechanism["models"][name]["calibration_states"] != len(teachers) for name in CANDIDATES):
        raise ValueError("Fixed Calibration teacher cohort differs")
    return source,mechanism,cohort,teachers,registry,compatibility


def restore_checked(env,packet):
    hidden = restore_packet(env,packet)
    snapshot = env.export_counterfactual_snapshot(hidden)
    expected = packet["snapshot"]
    if (snapshot.state_fingerprint != expected.state_fingerprint
            or digest(snapshot.environment_rng_state) != digest(expected.environment_rng_state)
            or digest(snapshot.actor_gru_state) != digest(expected.actor_gru_state)
            or digest(cache_payload(env)) != digest(packet["cache"])
            or env._obstacle_track_cache_step != packet["cache_step"]):
        raise RuntimeError("Replay state/cache/GRU/RNG fingerprint mismatch")
    return hidden


def observable_tracks(env):
    # The frozen Shadow uses this same CURRENT snapshot and nested order for
    # its flattened risk vector. Clone here so capture cannot mutate the source.
    snapshot = deepcopy(env)._safety_planning_snapshot()
    return [{"observer_id":int(track.observer_id),"track_id":int(track.track_id),
        "relative_position":list(track.relative_position),"relative_velocity":list(track.relative_velocity),
        "radius":float(track.radius),"velocity_valid":bool(track.velocity_valid)}
        for tracks in snapshot.visible_obstacles for track in tracks]


def shadow_evidence(env,action,hidden):
    before = integrity_token(env,hidden)
    tracks = observable_tracks(env)
    result = evaluate_ppcsa_normal_shadow(env,action,hidden).payload()
    after_tracks = observable_tracks(env)
    result.update(tracks=tracks,track_capture_unchanged=tracks == after_tracks,
        source_unchanged=before == integrity_token(env,hidden),
        risk_order_contract="observer-order then current visible-obstacle order")
    return result


def build_direction(name,teachers,source,mechanism,adapter,models,artifacts,env,device):
    gradients,active = [],[]
    for record,packet in teachers:
        restore_checked(env,packet)
        values = score_functions(env,packet,models,artifacts,device)
        mean = packet["base_mu"].to(device).float()
        features = packet["hidden_features"].to(device).float().detach()
        jacobian = adapter_jacobian(adapter,features)
        trace = tangent_trace(adapter,features,mean,values[name],jacobian)
        reference = torch.tanh(mean+torch.as_tensor(record["best_local_delta"],device=device,dtype=mean.dtype))
        qref = float(values[name](reference).detach())
        gate = qref>trace["score"]
        gradients.append(trace["parameter_gradient"]*gate)
        active.append(gate)
    layouts = [r["layout"] for r,p in teachers]
    weights = layout_weights(layouts,device)
    ascent = (weights[:,None]*torch.stack(gradients)).sum(0)
    norm = float(ascent.norm())
    historical = mechanism["models"][name]
    if sum(active) != historical["active_calibration_states"] or not np.isclose(norm,historical["aggregate_parameter_gradient_norm"],rtol=2e-5,atol=1e-10):
        raise RuntimeError("Phase1 direction reproduction differs; no direction replacement")
    unit = ascent/ascent.norm() if norm>0 else torch.zeros_like(ascent)
    return unit,{"source":"Calibration only","teacher_keys":[key(r) for r,p in teachers],
        "active_teachers":sum(active),"aggregate_gradient_norm":norm,"direction_sha256":digest(unit),
        "parameter_step":PARAMETER_STEP,"parameter_step_search":False,
        "reproduction_tolerance_note":"Gradient metadata float32/device roundoff only; risk/progress tolerance remains zero"}


def run_acceptance(cohort,teachers,registry,source,mechanism,models,artifacts,env,device):
    expected = [r["key"] for r in registry]
    adapter = zero_adapter().to(device)
    original = {k:v.detach().clone() for k,v in adapter.state_dict().items()}
    count = sum(p.numel() for p in adapter.parameters())
    results = {}
    errors = []
    calls = 0
    for name in CANDIDATES:
        construction_error = None
        try:
            direction,metadata = build_direction(name,teachers,source,mechanism,adapter,models,artifacts,env,device)
        except (ValueError,RuntimeError,KeyError) as error:
            direction = torch.zeros(count,device=device)
            construction_error = str(error)
            metadata = {"construction_error":construction_error,"parameter_step":PARAMETER_STEP}
            errors.append({"candidate":name,"stage":"direction_construction","error":construction_error})
        proposed = (PARAMETER_STEP*direction).detach().cpu().tolist()
        virtual = virtual_adapter(adapter,direction,PARAMETER_STEP)
        evidence = []
        for record,packet in cohort:
            row = {"key":key(record),"layout":record["layout"],"step":record["step"],
                "state_index":record["state_index"],"replay_match":False,"baseline":None,"candidate":None}
            try:
                hidden = restore_checked(env,packet)
                mean = packet["base_mu"].to(device).float()
                base_action = torch.tanh(mean).cpu().numpy()
                row["baseline"] = shadow_evidence(env,base_action,hidden)
                calls += 1
                stored = next(c["result"] for c in record["candidates"] if c["kind"] == "base")
                if not row["baseline"]["executed_actions"] or not np.allclose(row["baseline"]["executed_actions"],stored["executed_actions"],atol=1e-6,rtol=0.):
                    raise RuntimeError("Original baseline action replay differs")
                hidden = restore_checked(env,packet)
                with torch.no_grad():
                    residual = virtual(packet["hidden_features"].to(device).float())
                    action = torch.tanh(mean+residual).cpu().numpy()
                row["candidate"] = shadow_evidence(env,action,hidden)
                calls += 1
                row["replay_match"] = row["baseline"]["snapshot_fingerprint"] == row["candidate"]["snapshot_fingerprint"]
                row["pre_tanh_delta_l2"] = float(residual.norm())
                if construction_error:
                    row["evaluation_error"] = "Candidate construction failed: "+construction_error
            except (ValueError,RuntimeError,KeyError) as error:
                row["evaluation_error"] = str(error)
                errors.append({"candidate":name,"key":row["key"],"stage":"state_evaluation","error":str(error)})
            evidence.append(row)
        decision = candidate_admission(expected,evidence,proposed)
        # Retain raw baseline/candidate vectors and track identities, including
        # every failed state. Never compute admission on a favorable subset.
        results[name] = {**decision,"direction_metadata":metadata,"raw_state_evidence":evidence,
            "virtual_copy_discarded":True}
    if any(not torch.equal(original[k],v) for k,v in adapter.state_dict().items()):
        raise RuntimeError("Original Adapter parameters changed")
    return {"candidates":results,"whole_set":candidate_set_admission(results,count),
        "fixed_state_registry":registry,"fixed_state_registry_sha256":digest(registry),
        "shadow_calls":calls,"source_adapter_unchanged":True,"audit_used":False,
        "execution_errors":errors,"mechanism_conclusion_available":not errors}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate1-dir",required=True)
    parser.add_argument("--critic-dir",required=True)
    parser.add_argument("--mechanism-report",required=True)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--preflight-only",action="store_true")
    args = parser.parse_args(argv)
    gate1,critic,mechanism_path = map(lambda p:resolve(p).resolve(),(args.gate1_dir,args.critic_dir,args.mechanism_report))
    source,mechanism,cohort,teachers,registry,compatibility = bind_inputs(gate1,critic,mechanism_path)
    output = resolve(args.output_dir).resolve()
    protected = [gate1,critic,resolve(source["gate0_dir"]),gate1.parent/"gate2_gradient",mechanism_path.parent]
    if output.exists() or any(output == p.resolve() or p.resolve() in output.parents for p in protected):
        raise ValueError("Output must be new and separate from historical artifacts")
    if args.preflight_only:
        print(f"PREFLIGHT=PASS; FIXED_STATES={len(cohort)}; CALIBRATION_TEACHERS={len(teachers)}; CANDIDATES=ordinary,paired,kinematic; PARAMETER_STEP=0.01; MAX_SHADOW_CALLS={6*len(cohort)}; MODEL_EXECUTION=NO; OUTPUT_CREATED=NO")
        return {"status":"READY","fixed_states":len(cohort),"registry_sha256":digest(registry),"diagnostic_run":False}
    before = {str(p):file_sha(p) for d in protected for p in d.rglob("*") if p.is_file()}
    started = perf_counter()
    output = new_output(output)
    trainer,env,provenance = runtime(source["checkpoint"],output,args.device)
    actor_before = {k:v.detach().cpu().clone() for k,v in trainer.actor.state_dict().items()}
    models,artifacts,critic_before = {},{},{}
    for name in ("ordinary","paired"):
        artifact = torch.load(critic/f"{name}.pt",map_location="cpu",weights_only=False)
        if artifact["objective"] != name or artifact["partitions"] != source["partitions"]:
            raise ValueError("Frozen model contract mismatch")
        model = ExecutedValueCritic(artifact["context_dim"]).to(args.device)
        model.load_state_dict(artifact["state_dict"])
        models[name],artifacts[name] = model.freeze(),artifact
        critic_before[name] = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    result = run_acceptance(cohort,teachers,registry,source,mechanism,models,artifacts,env,args.device)
    if any(not torch.equal(actor_before[k],v.detach().cpu()) for k,v in trainer.actor.state_dict().items()):
        raise RuntimeError("Frozen Actor changed")
    if any(not torch.equal(critic_before[name][k],v.detach().cpu()) for name,m in models.items() for k,v in m.state_dict().items()):
        raise RuntimeError("Frozen Critic changed")
    after = {str(p):file_sha(p) for d in protected for p in d.rglob("*") if p.is_file()}
    if before != after or file_sha(resolve(source["checkpoint"])) != source["checkpoint_sha256"]:
        raise RuntimeError("Historical artifacts changed")
    report = {**provenance,**result,"schema_version":"PPCSASharedUpdateAdmissionV1",
        "status":result["whole_set"]["status"] if not result["execution_errors"] else "DIAGNOSTIC_ERROR",
        "parameter_step":PARAMETER_STEP,"candidate_registry":list(CANDIDATES),
        "source_mechanism_report_sha256":file_sha(mechanism_path),"phase1_source_compatibility":compatibility,
        "input_file_sha256":before,"implementation_sha256":{p:file_sha(ROOT/p) for p in (
            "src/algorithms/ppcsa_update_acceptance.py","scripts/diagnose_ppcsa_update_acceptance.py",*PHASE1_SOURCES)},
        "source_actor_unchanged":True,"source_critics_unchanged":True,"source_artifacts_unchanged":True,
        "actor_training_authorized":False,"pilot_authorized":False,"refit_authorized":False,"new_collection":False,
        "partitions":source["partitions"],"used_partitions":["fit","calibration"],"device":args.device,
        "wall_seconds":perf_counter()-started,"risk_tolerance":0.,"progress_tolerance":0.,
        "scope":"Whole-update virtual simulation admission on the fixed exposed Phase1 certified cohort; not independent validation or online safety guarantee"}
    save_json(jsonable(report),output/"shared_admission_report.json")
    print(f"ADMISSION={report['status']}; ACTOR_TRAINING=BLOCKED; DETAIL_LOG={output/'shared_admission_report.json'}")
    return report


if __name__ == "__main__":
    main()
