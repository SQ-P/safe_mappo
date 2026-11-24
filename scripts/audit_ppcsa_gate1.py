"""User-run safe reference/reachable-action opportunity and shared Jacobian audit."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from src.algorithms.ppcsa_protocol import resolve,new_output,runtime,read_manifest,file_sha,write_manifest
from src.algorithms.ppcsa_executed_value import zero_adapter,adapter_jacobian
from src.safety.ppcsa_one_step import evaluate_ppcsa_normal_shadow,restore_packet,safe_improvement,NormalOneStepResult
from src.safety.counterfactual import registered_reference_actions
from src.utils.io_utils import save_json


def candidate_bank(mean,goal):
    base = torch.tanh(mean).numpy()
    bank = [("base",base,np.zeros_like(base))]
    for i,action in enumerate(registered_reference_actions(base,goal)):
        bank.append((f"reference_{i}",action,None))
    # Axis probes are in PRE-tanh mean space. They are only an action-space
    # upper envelope; the actual shared Adapter is audited separately.
    for i in range(base.size):
        for sign in (-1.,1.):
            delta = torch.zeros_like(mean)
            delta.reshape(-1)[i] = sign*.15
            bank.append((f"local_{i}_{sign:+g}",torch.tanh(mean+delta).numpy(),delta.numpy()))
    generator = np.random.default_rng(20261121)
    for name,direction in (("goal",goal),("random",generator.normal(size=base.shape))):
        delta = .15*np.asarray(direction,dtype=np.float32)/max(float(np.max(np.abs(direction))),1e-12)
        delta = np.clip(delta,-np.float32(.15),np.float32(.15))
        bank.append((f"local_joint_{name}",torch.tanh(mean+torch.as_tensor(delta)).numpy(),delta))
    return bank


def shared_representability(states,records,maximum=32):
    selected = []
    for low,high,limit in ((121000,121024,16),(121024,121032,8),(121032,121040,8)):
        seen = set()
        for row in records:
            if low<=row["layout"]<high and row.get("best_local_delta") is not None and row["layout"] not in seen:
                selected.append(row)
                seen.add(row["layout"])
            if len(seen)>=limit:
                break
    if not selected:
        return {"status":"NO_REACHABLE_TARGETS","claims_actual_update_improvement":False}
    adapter = zero_adapter()
    jacobians,targets,layouts = [],[],[]
    for row in selected:
        packet = states[row["state_index"]]
        jacobians.append(adapter_jacobian(adapter,packet["hidden_features"].float()).numpy())
        targets.append(np.asarray(row["best_local_delta"]).reshape(-1))
        layouts.extend([packet["layout_seed"]]*targets[-1].size)
    matrix,target = np.concatenate(jacobians),np.concatenate(targets)
    active_columns = np.any(matrix != 0.,axis=0)
    matrix = matrix[:,active_columns]
    unique = sorted(set(layouts))
    train = np.asarray(layouts)<121024
    if not train.any():
        return {"status":"NO_FIT_TARGETS","claims_actual_update_improvement":False}
    coefficients = np.linalg.lstsq(matrix[train],target[train],rcond=None)[0]
    errors = {str(layout):float(np.linalg.norm(matrix[np.asarray(layouts)==layout]@coefficients-target[np.asarray(layouts)==layout])/
        (np.linalg.norm(target[np.asarray(layouts)==layout])+1e-12)) for layout in unique if layout>=121024}
    task_gradients = np.stack([j.T@t for j,t in zip(jacobians,targets)])
    norms = np.linalg.norm(task_gradients,axis=1)
    cosines = (task_gradients@task_gradients.T)/(norms[:,None]*norms[None,:]+1e-12)
    pairs = cosines[np.triu_indices(len(selected),1)]
    return {"status":"DIAGNOSTIC_ONLY","states":len(selected),"layouts":unique,
        "active_parameter_columns":int(active_columns.sum()),"fit_layouts":[g for g in unique if g<121024],"heldout_layout_relative_error":errors,
        "conflicting_target_gradient_pair_fraction":float(np.mean(pairs<0)) if len(pairs) else None,
        "claims_actual_update_improvement":False,"note":"Unconstrained first-order Jacobian fit; not a finite realizability certificate or deployed update"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate0-dir",required=True)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--progress-deadzone",type=float,default=0.)
    args = parser.parse_args(argv)
    if args.progress_deadzone<0 or not np.isfinite(args.progress_deadzone):
        parser.error("deadzone must be finite and nonnegative")
    gate0 = resolve(args.gate0_dir)
    manifest = read_manifest(gate0)
    if manifest.get("status") != "PASS":
        raise RuntimeError("Gate 0 did not pass; label generation blocked")
    output = new_output(args.output_dir)
    _,env,provenance = runtime(manifest["checkpoint"],output,args.device)
    states = torch.load(gate0/"states.pt",map_location="cpu",weights_only=False)
    records,examples = [],[]
    rejected,coverage,by_layout = Counter(),Counter(),defaultdict(Counter)
    calls = 0
    started = perf_counter()
    for index,packet in enumerate(states):
        if not packet.get("parity_passed") or not packet.get("factual_supported"):
            rejected["gate0_unsupported_state"] += 1
            continue
        hidden = restore_packet(env,packet)
        baseline = evaluate_ppcsa_normal_shadow(env,packet["requested_action"],hidden)
        if not baseline.executed_actions or not np.allclose(baseline.executed_actions,packet["baseline"]["executed_actions"],atol=1e-6,rtol=0.):
            raise RuntimeError("Gate 0 state replay changed the baseline execution; labels blocked")
        mean = packet["base_mu"].float()
        direction = env.subgoals-env.positions
        goal = direction/np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-8)
        candidates = []
        for name,action,delta in candidate_bank(mean,goal):
            result = baseline if name == "base" else evaluate_ppcsa_normal_shadow(env,action,hidden)
            calls += name != "base"
            improved = safe_improvement(result,baseline,args.progress_deadzone)
            kind = "base" if name == "base" else ("local" if name.startswith("local_") else "reference")
            for reason in result.rejection_reasons:
                rejected[reason] += 1
            if result.supported and result.hard_safe:
                raw_next_velocity = np.clip(env.velocities+action*env.agent_max_acceleration*env.dt,-env.agent_max_speed,env.agent_max_speed)
                raw_next_position = env.positions+raw_next_velocity*env.dt
                heuristic = env._team_goal_potential(np.linalg.norm(raw_next_position-env.goals,axis=1))-env._team_goal_potential(np.linalg.norm(env.positions-env.goals,axis=1))
                examples.append({"state_index":index,"layout":packet["layout_seed"],"context":packet["context"],
                    "action":action,"target":result.executed_progress,"base_action":packet["requested_action"],
                    "base_target":baseline.executed_progress,"heuristic_target":heuristic,"kind":kind,"teacher_eligible":improved})
            candidates.append({"name":name,"kind":kind,"improved":improved,"delta":delta,"result":result.payload()})
        references = [c for c in candidates if c["kind"] == "reference"]
        locals_ = [c for c in candidates if c["kind"] == "local"]
        positive = [c for c in locals_ if c["improved"]]
        best = max(positive,key=lambda c:c["result"]["executed_progress"],default=None)
        flags = {"states":1,"safe_reference":any(c["result"]["hard_safe"] for c in references),
            "positive_reference":any(c["improved"] for c in references),"positive_local":bool(positive),
            "positive_single_uav_local":any(c["improved"] and not c["name"].startswith("local_joint_") for c in locals_),
            "positive_joint_local":any(c["improved"] and c["name"].startswith("local_joint_") for c in locals_),
            "near_failure_positive_local":bool(packet.get("near_failure_diagnostic_only")) and bool(positive),
            "near_failure_diagnostic_states":bool(packet.get("near_failure_diagnostic_only"))}
        coverage.update(flags)
        by_layout[str(packet["layout_seed"])].update(flags)
        changed = [c for c in locals_ if c["result"]["executed_actions"]]
        absorbed = sum(np.allclose(c["result"]["executed_actions"],baseline.executed_actions,atol=1e-7,rtol=0.) for c in changed)
        coverage["local_probes"] += len(changed)
        coverage["absorbed_local_probes"] += absorbed
        records.append({"state_index":index,"layout":packet["layout_seed"],"step":packet["step"],
            "near_failure_diagnostic_only":bool(packet.get("near_failure_diagnostic_only")),
            "candidates":candidates,"best_local_delta":best["delta"] if best else None})
    shared = shared_representability(states,records)
    torch.save({"states":states,"records":records,"examples":examples},output/"opportunities.pt")
    report = {**provenance,"gate":1,"gate0_dir":str(gate0.resolve()),"gate0_manifest_sha256":file_sha(gate0/"manifest.json"),
        "status":"REVIEW_REQUIRED","coverage_counts":dict(coverage),"per_layout_counts":{k:dict(v) for k,v in by_layout.items()},
        "source_sampled_states":len(states),"excluded_state_count":sum(1 for s in states if not s.get("parity_passed") or not s.get("factual_supported")),
        "rejection_counts_nonexclusive":dict(rejected),"shared_parameter_representability":shared,
        "partitions":manifest["partitions"],"progress_deadzone":args.progress_deadzone,
        "shadow_calls":calls+len(records),"wall_seconds":perf_counter()-started,"device":args.device,
        "minimum_meaningful_opportunity_threshold":None,"critic_fit_authorized":False,
        "note":"Positive means > configured deadzone, not a research success threshold; all opportunities require review"}
    save_json(report,output/"ppcsa_opportunity_report.json")
    write_manifest(output,report,["opportunities.pt","ppcsa_opportunity_report.json"])
    print(f"GATE1=REVIEW_REQUIRED; DETAIL_LOG={output/'ppcsa_opportunity_report.json'}")
    return report


if __name__ == "__main__":
    main()
