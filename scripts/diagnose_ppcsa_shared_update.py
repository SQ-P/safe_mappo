"""User-run Fit/Calibration mechanism probes; no fitting, Audit use or Actor update."""
import argparse
from collections import Counter
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from src.algorithms.ppcsa_protocol import resolve,read_manifest,file_sha,new_output,runtime,validate_panel,PANEL
from src.algorithms.ppcsa_executed_value import ExecutedValueCritic,zero_adapter,adapter_jacobian,bounded_probe,assess_probe
from src.algorithms.ppcsa_mechanism_diagnostics import (kinematic_value,conflict_summary,layout_weights,
    virtual_adapter,match_calibration_budget,tangent_trace,cosine)
from src.safety.ppcsa_one_step import restore_packet,evaluate_ppcsa_normal_shadow,jsonable
from scripts.fit_ppcsa_executed_value import predict
from src.utils.io_utils import save_json

EPSILONS = (.01,.05,.10)
OBJECTIVES = ("ordinary","paired","kinematic")


def validate_inputs(gate1,critic):
    source,fitted = read_manifest(gate1),read_manifest(critic)
    panel = validate_panel(resolve(PANEL))
    if source["partitions"] != panel["partitions"] or fitted["partitions"] != source["partitions"]:
        raise ValueError("Frozen partition contract differs")
    if fitted["gate1_manifest_sha256"] != file_sha(gate1/"manifest.json"):
        raise ValueError("Critic and Gate1 dataset differ")
    for key in ("checkpoint_sha256","config_sha256","task_config_fingerprint"):
        if fitted[key] != source[key]:
            raise ValueError("Frozen source contract differs")
    return source,fitted


def selected_records(data,partitions):
    allowed = set(partitions["fit"]+partitions["calibration"])
    rows = []
    excluded = Counter()
    for record in data["records"]:
        # Never access Audit packet/features/targets or use its outcomes.
        if record["layout"] not in allowed:
            continue
        packet = data["states"][record["state_index"]]
        if packet["layout_seed"] != record["layout"]:
            raise ValueError("State/layout mismatch")
        baseline = next(c["result"] for c in record["candidates"] if c["kind"] == "base")
        if not packet.get("parity_passed") or not packet.get("factual_supported") or not baseline["supported"] or not baseline["hard_safe"]:
            excluded.update(baseline["rejection_reasons"] or ["unsupported_baseline"])
            continue
        rows.append((record,packet))
    return rows,dict(excluded)


def score_functions(env,packet,models,artifacts,device):
    context = torch.as_tensor(packet["context"],device=device).float().unsqueeze(0)
    values = {name:(lambda action,m=model,a=artifacts[name]:predict(m,a,context,action.unsqueeze(0)).squeeze(0)) for name,model in models.items()}
    geometry = {"positions":env.positions.copy(),"velocities":env.velocities.copy(),"goals":env.goals.copy(),
        "dt":env.dt,"acceleration":env.agent_max_acceleration,"max_speed":env.agent_max_speed,
        "start_max":env._episode_start_max_goal_distance,"start_mean":env._episode_start_mean_goal_distance,
        "bottleneck":env.bottleneck_weight}
    values["kinematic"] = lambda action:kinematic_value(action,**geometry)
    return values


def run_diagnostics(data,source,models,artifacts,env,device):
    selected,excluded = selected_records(data,source["partitions"])
    excluded = Counter(excluded)
    if not selected:
        raise RuntimeError("No certified Fit/Calibration states")
    adapter = zero_adapter().to(device)
    original = {k:v.detach().clone() for k,v in adapter.state_dict().items()}
    prepared = []
    calls = 0
    for record,packet in selected:
        hidden = restore_packet(env,packet)
        mean = packet["base_mu"].to(device).float()
        action = torch.tanh(mean).cpu().numpy()
        baseline = evaluate_ppcsa_normal_shadow(env,action,hidden)
        calls += 1
        stored = next(c["result"] for c in record["candidates"] if c["kind"] == "base")
        if not baseline.supported or not baseline.hard_safe:
            excluded.update({"replayed_baseline_not_certified":1})
            continue
        if not np.allclose(baseline.executed_actions,stored["executed_actions"],atol=1e-6,rtol=0.):
            raise RuntimeError("Stored baseline replay changed; stop diagnosis")
        features = packet["hidden_features"].to(device).float().detach()
        values = score_functions(env,packet,models,artifacts,device)
        jacobian = adapter_jacobian(adapter,features)
        traces = {name:tangent_trace(adapter,features,mean,value,jacobian) for name,value in values.items()}
        reference = (torch.tanh(mean+torch.as_tensor(record["best_local_delta"],device=device,dtype=mean.dtype))
            if record.get("best_local_delta") is not None else None)
        for name,trace in traces.items():
            qref = float(values[name](reference).detach()) if reference is not None else None
            trace.update(reference_value=qref,gate_active=qref is not None and qref>trace["score"])
        goal = env.subgoals-env.positions
        goal /= np.maximum(np.linalg.norm(goal,axis=1,keepdims=True),1e-8)
        future = env.positions+np.clip(env.velocities+action*env.agent_max_acceleration*env.dt,-env.agent_max_speed,env.agent_max_speed)*env.dt
        distances = np.linalg.norm(future-env.goals,axis=1)
        prepared.append({"record":record,"packet":packet,"mean":mean,"features":features,"jacobian":jacobian,
            "values":values,"traces":traces,"baseline":baseline,"goal":goal,
            "kinematic_kinks":{"tied_bottleneck_agents":int((distances == distances.max()).sum()),
                "at_or_beyond_speed_clip_components":int((np.abs(env.velocities+action*env.agent_max_acceleration*env.dt)>=env.agent_max_speed).sum())}})
    calibration = [s for s in prepared if s["record"]["layout"] in source["partitions"]["calibration"] and s["record"].get("best_local_delta") is not None]
    if not calibration:
        raise RuntimeError("No certified Calibration teachers for shared direction; no replacement sampling")
    layouts = [s["record"]["layout"] for s in prepared]
    cal_layouts = [s["record"]["layout"] for s in calibration]
    weights = layout_weights(cal_layouts,device)
    features = torch.stack([s["features"] for s in calibration])
    output_models = {}
    parameter_probes = []
    for name in OBJECTIVES:
        cal_gradients = torch.stack([s["traces"][name]["parameter_gradient"]*s["traces"][name]["gate_active"] for s in calibration])
        ascent = (weights[:,None]*cal_gradients).sum(0)
        norm = float(ascent.norm())
        unit = ascent/ascent.norm() if norm>0 else torch.zeros_like(ascent)
        variants = [("legacy_parameter_step",.01,{"status":"FIXED_PARAMETER_NORM","parameter_step":.01})]
        for epsilon in EPSILONS:
            step,match = match_calibration_budget(adapter,unit,features,weights,epsilon)
            variants.append((f"calibration_rms_{epsilon}",step,match))
        state_traces = []
        for s in prepared:
            trace = s["traces"][name]
            induced = s["jacobian"]@unit
            state_traces.append({"layout":s["record"]["layout"],"step":s["record"]["step"],
                "role":"calibration" if s["record"]["layout"] in source["partitions"]["calibration"] else "fit",
                "score":trace["score"],"reference_value":trace["reference_value"],"gate_active":trace["gate_active"],
                "action_gradient":trace["gradient"],"action_gradient_norm":float(trace["gradient"].norm()),
                "parameter_gradient_norm":float(trace["parameter_gradient"].norm()),"layer_gradients":trace["layers"],
                "jacobian_frobenius_norm":float(s["jacobian"].norm()),"shared_unit_action_tangent_l2":float(induced.norm()),
                "chain_rule_max_error":trace["chain_rule_max_error"],
                "action_gradient_vs_shared_tangent_cosine":cosine(trace["gradient"].reshape(-1),induced),
                "shared_unit_first_order_score_change":float(trace["gradient"].reshape(-1)@induced),
                "kinematic_kinks":s["kinematic_kinks"]})
        for variant,step,match in variants:
            if step is None:
                parameter_probes.append({"objective":name,"variant":variant,"budget":match,"states":[]})
                continue
            virtual = virtual_adapter(adapter,unit,step)
            outcomes = []
            for s in prepared:
                packet,record = s["packet"],s["record"]
                hidden = restore_packet(env,packet)
                with torch.no_grad():
                    delta = virtual(s["features"])
                    requested = torch.tanh(s["mean"]+delta)
                    linear = (s["jacobian"]@(step*unit)).reshape_as(delta)
                    score_gain = float(s["values"][name](requested)-s["values"][name](torch.tanh(s["mean"])))
                candidate = evaluate_ppcsa_normal_shadow(env,requested.cpu().numpy(),hidden)
                calls += 1
                baseline = s["baseline"]
                executed_change = np.asarray(candidate.executed_actions)-np.asarray(baseline.executed_actions)
                outcomes.append({"layout":record["layout"],"step":record["step"],**assess_probe(baseline,candidate),
                    "rejection_reasons":candidate.rejection_reasons,"pre_tanh_delta_l2":float(delta.norm()),
                    "pre_tanh_delta_linf":float(delta.abs().max()),
                    "requested_action_change_l2":float((requested-torch.tanh(s["mean"])).norm()),
                    "executed_action_change_l2":float(np.linalg.norm(executed_change)),
                    "linearized_delta_l2":float(linear.norm()),"linearization_error_l2":float((delta-linear).norm()),
                    "predicted_score_gain":score_gain,
                    "first_order_score_gain":float(s["traces"][name]["gradient"].reshape(-1)@linear.reshape(-1))})
            parameter_probes.append({"objective":name,"variant":variant,"parameter_step":step,"budget":match,"states":outcomes})
        output_models[name] = {"gradient_source":"calibration only","calibration_layouts":sorted(set(cal_layouts)),
            "calibration_states":len(calibration),"active_calibration_states":sum(s["traces"][name]["gate_active"] for s in calibration),
            "aggregate_parameter_gradient_norm":norm,"state_traces":state_traces,
            "unmasked_gradient_geometry":conflict_summary([s["traces"][name]["parameter_gradient"] for s in prepared],layouts),
            "gated_calibration_geometry":conflict_summary(list(cal_gradients*weights[:,None]),cal_layouts)}
    action_probes = []
    for s in prepared:
        packet,record = s["packet"],s["record"]
        hidden = restore_packet(env,packet)
        generator = torch.Generator().manual_seed(20261121+record["state_index"])
        directions = {name:s["traces"][name]["gradient"] for name in OBJECTIVES}
        directions.update(random=torch.randn(s["mean"].shape,generator=generator).to(device),goal=torch.as_tensor(s["goal"],device=device))
        for epsilon in EPSILONS:
            for name,direction in directions.items():
                action,delta = bounded_probe(s["mean"],direction,epsilon)
                candidate = evaluate_ppcsa_normal_shadow(env,action.detach().cpu().numpy(),hidden)
                calls += 1
                action_probes.append({"layout":record["layout"],"step":record["step"],"direction":name,"epsilon":epsilon,
                    "delta_l2":float(delta.norm()),"zero_direction":float(direction.norm()) == 0.,
                    **assess_probe(s["baseline"],candidate),"rejection_reasons":candidate.rejection_reasons})
    if any(not torch.equal(original[k],v) for k,v in adapter.state_dict().items()):
        raise RuntimeError("Diagnostic changed original Adapter")
    return {"models":output_models,"parameter_probes":parameter_probes,"action_probes":action_probes,
        "analyzed_states":len(prepared),"analyzed_layouts":sorted(set(layouts)),"excluded_reasons_nonexclusive":dict(excluded),
        "shadow_calls":calls,"source_adapter_unchanged":True,"audit_used":False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate1-dir",required=True)
    parser.add_argument("--critic-dir",required=True)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--preflight-only",action="store_true")
    args = parser.parse_args(argv)
    gate1,critic = resolve(args.gate1_dir),resolve(args.critic_dir)
    source,fitted = validate_inputs(gate1,critic)
    output = resolve(args.output_dir).resolve()
    protected = [gate1,critic,resolve(source["gate0_dir"]),gate1.parent/"gate2_gradient"]
    if output.exists() or any(output == p.resolve() or p.resolve() in output.parents for p in protected):
        raise ValueError("Diagnostic output must be new and separate from historical artifacts")
    if args.preflight_only:
        data = torch.load(gate1/"opportunities.pt",map_location="cpu",weights_only=False)
        rows,_ = selected_records(data,source["partitions"])
        calibration = [r for r,p in rows if r["layout"] in source["partitions"]["calibration"] and r.get("best_local_delta") is not None]
        if not rows or not calibration:
            raise RuntimeError("No certified Fit/Calibration diagnostic support")
        print(f"PREFLIGHT=PASS; CERTIFIED_STATES={len(rows)}; CALIBRATION_TEACHERS={len(calibration)}; MAX_SHADOW_CALLS={28*len(rows)}; MODEL_EXECUTION=NO; OUTPUT_CREATED=NO")
        return {"status":"READY","diagnostic_run":False,"certified_states":len(rows)}
    before = {str(p):file_sha(p) for d in protected for p in d.rglob("*") if p.is_file()}
    started = perf_counter()
    data = torch.load(gate1/"opportunities.pt",map_location="cpu",weights_only=False)
    # Output is never an existing result directory. No Actor/Critic serialization.
    output = new_output(output)
    trainer,env,provenance = runtime(source["checkpoint"],output,args.device)
    actor_before = {k:v.detach().cpu().clone() for k,v in trainer.actor.state_dict().items()}
    models,artifacts = {},{}
    for name in ("ordinary","paired"):
        artifact = torch.load(critic/f"{name}.pt",map_location="cpu",weights_only=False)
        if artifact["objective"] != name or artifact["partitions"] != source["partitions"]:
            raise ValueError("Frozen model metadata differs")
        model = ExecutedValueCritic(artifact["context_dim"]).to(args.device)
        model.load_state_dict(artifact["state_dict"])
        models[name],artifacts[name] = model.freeze(),artifact
    result = run_diagnostics(data,source,models,artifacts,env,args.device)
    if any(not torch.equal(actor_before[k],v.detach().cpu()) for k,v in trainer.actor.state_dict().items()):
        raise RuntimeError("Diagnostic changed frozen Actor")
    after = {str(p):file_sha(p) for d in protected for p in d.rglob("*") if p.is_file()}
    if after != before or file_sha(resolve(source["checkpoint"])) != source["checkpoint_sha256"]:
        raise RuntimeError("Historical artifacts changed")
    report = {**provenance,**result,"schema_version":"PPCSASharedMechanismDiagnosticV1","status":"MECHANISM_REVIEW_REQUIRED",
        "input_file_sha256":before,"source_artifacts_unchanged":True,"source_actor_unchanged":True,
        "diagnostic_source_sha256":{name:file_sha(ROOT/name) for name in ("scripts/diagnose_ppcsa_shared_update.py","src/algorithms/ppcsa_mechanism_diagnostics.py","src/algorithms/ppo/networks.py","scripts/fit_ppcsa_executed_value.py")},
        "epsilons":list(EPSILONS),"device":args.device,"wall_seconds":perf_counter()-started,
        "partitions":source["partitions"],"used_partitions":["fit","calibration"],
        "actor_training_authorized":False,"refit_authorized":False,"pilot_authorized":False,"new_collection":False,
        "limitations":["Exposed Fit/Calibration mechanism exploration only; no independent validation or causal attribution.",
            "Direction and RMS scalar use Calibration only. No per-evaluation-state rescaling or favorable epsilon selection.",
            "RMS matching does not equalize each state's action norm. Report each actual norm; do not claim statewise matched parameter comparisons.",
            "Kinematic derivative is unprojected and piecewise smooth; Shadow supplies execution/safety evidence independently.",
            "Uncertifiable baselines excluded with reasons; this does not establish coverage or safety outside analyzed states.",
            "Reference branches and Critic weights frozen; virtual parameter changes always discarded. No saved Actor."]}
    save_json(jsonable(report),output/"shared_mechanism_report.json")
    print(f"MECHANISM=REVIEW_REQUIRED; ACTOR_TRAINING=BLOCKED; DETAIL_LOG={output/'shared_mechanism_report.json'}")
    return report


if __name__ == "__main__":
    main()
