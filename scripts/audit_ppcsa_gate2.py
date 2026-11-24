"""User-run held-out action/shared-Adapter gradient diagnostics; no Actor writes."""
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
from src.algorithms.ppcsa_protocol import resolve,new_output,read_manifest,file_sha,runtime,write_manifest
from src.algorithms.ppcsa_executed_value import ExecutedValueCritic,executed_gap_loss,zero_adapter,parameter_probe,audit_action_directions,assess_probe
from src.safety.ppcsa_one_step import restore_packet,evaluate_ppcsa_normal_shadow
from scripts.fit_ppcsa_executed_value import predict
from src.utils.io_utils import save_json

EPSILONS = (.01,.05,.10)
PARAMETER_STEP = .01


def summarize_parameter_outcomes(rows):
    """Keep safety, risk and progress evidence distinct; no efficacy gate."""
    def summarize(selected):
        gains = [r["gain"] for r in selected if r["gain"] is not None]
        certified = [r["gain"] for r in selected if r["certifiable"] and r["risk_proxy_regression"] is False and r["gain"] is not None]
        counts = Counter(r["outcome"] for r in selected)
        return {"states":len(selected),"outcome_counts":dict(counts),
            "improved":counts["improved"],"worsened":counts["worsened"],
            "absorbed":counts["absorbed"],"flat_changed_action":counts["flat_changed_action"],
            "absorbed_or_flat":counts["absorbed"]+counts["flat_changed_action"],
            "uncertifiable":sum(not r["certifiable"] for r in selected),
            "hard_safety_regression_count":sum(r["hard_safety_regression"] for r in selected),
            "risk_proxy_regression_count":sum(r["risk_proxy_regression"] is True for r in selected),
            "risk_proxy_unknown_count":sum(r["risk_proxy_regression"] is None for r in selected),
            "paired_progress_count":len(gains),
            "paired_progress_net":float(sum(gains)) if gains else None,
            "paired_progress_mean":float(np.mean(gains)) if gains else None,
            "certified_nonworse_risk_progress_count":len(certified),
            "certified_nonworse_risk_progress_net":float(sum(certified)) if certified else None}
    total = summarize(rows)
    return {**total,"per_layout":{str(layout):summarize([r for r in rows if r["layout"] == layout]) for layout in sorted({r["layout"] for r in rows})},
        "hard_safety_regression":bool(total["hard_safety_regression_count"]),
        "risk_proxy_regression":bool(total["risk_proxy_regression_count"]),
        "efficacy_unproven":True,"efficacy_decision":"REVIEW_REQUIRED",
        "note":"Paired sums describe sampled states, not an efficacy acceptance criterion; no all-layout improvement rule"}


def calibration_update(model,artifact,adapter,states,records,calibration,device):
    selected = [r for r in records if r["layout"] in calibration and r.get("best_local_delta") is not None]
    if not selected:
        return None,{"status":"NO_CALIBRATION_REACHABLE_TEACHERS","gradient_norm":0.}
    features = torch.stack([states[r["state_index"]]["hidden_features"] for r in selected]).to(device).float()
    mean = torch.stack([states[r["state_index"]]["base_mu"] for r in selected]).to(device).float()
    context = torch.as_tensor(np.stack([states[r["state_index"]]["context"] for r in selected]),device=device).float()
    delta = torch.as_tensor(np.stack([r["best_local_delta"] for r in selected]),device=device).float()
    actions = torch.tanh(mean+adapter(features))
    reference = torch.tanh(mean+delta)
    current = predict(model,artifact,context,actions)
    ref_value = predict(model,artifact,context,reference).detach()
    counts = {g:sum(r["layout"] == g for r in selected) for g in set(r["layout"] for r in selected)}
    weights = torch.as_tensor([1./counts[r["layout"]] for r in selected],device=device)
    valid = torch.ones_like(current,dtype=torch.bool)
    loss = executed_gap_loss(current,ref_value,weights,valid)
    params = tuple(adapter.parameters())
    active = ((ref_value-current).detach()>0).float()*weights
    parts = []
    for agent in range(3):
        mixed = torch.cat([actions[:,:agent].detach(),actions[:,agent:agent+1],actions[:,agent+1:].detach()],dim=1)
        value = -(active*predict(model,artifact,context,mixed)).sum()/weights.sum().clamp(min=1.)
        grads = torch.autograd.grad(value,params,retain_graph=True,allow_unused=True)
        parts.append(torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(params,grads)]))
    parts = torch.stack(parts)
    norms = parts.norm(dim=1)
    cosine = (parts@parts.T)/(norms[:,None]*norms[None,:]).clamp(min=1e-12)
    virtual,norm,_ = parameter_probe(adapter,loss,PARAMETER_STEP)
    return virtual,{"status":"VIRTUAL_DIAGNOSTIC_ONLY","gradient_norm":norm,
        "calibration_states":len(selected),"calibration_layouts":sorted(set(r["layout"] for r in selected)),
        "per_agent_gradient_cosine":cosine.detach().cpu().tolist(),
        "gradient_cancellation_ratio":float(parts.sum(0).norm()/norms.sum().clamp(min=1e-12)),
        "parameter_step":PARAMETER_STEP,"loss":float(loss.detach()),"parent_policy_modified":False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate1-dir",required=True)
    parser.add_argument("--critic-dir",required=True)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--device",default="cpu")
    args = parser.parse_args(argv)
    gate1,critic_dir = resolve(args.gate1_dir),resolve(args.critic_dir)
    source = read_manifest(gate1)
    fitted = read_manifest(critic_dir)
    if fitted.get("gate1_manifest_sha256") != file_sha(gate1/"manifest.json"):
        raise ValueError("Critic does not match this Gate 1 dataset")
    data = torch.load(gate1/"opportunities.pt",map_location="cpu",weights_only=False)
    states = data["states"]
    heldout = [(i,s) for i,s in enumerate(states) if s["layout_seed"] in source["partitions"]["audit"] and s.get("parity_passed") and s.get("factual_supported")]
    if not heldout:
        raise RuntimeError("No disjoint held-out supported states; gradient certification blocked")
    output = new_output(args.output_dir)
    trainer,env,provenance = runtime(source["checkpoint"],output,args.device)
    started = perf_counter()
    all_models = {}
    calls = 0
    for objective in ("ordinary","paired"):
        artifact = torch.load(critic_dir/f"{objective}.pt",map_location="cpu",weights_only=False)
        if artifact["objective"] != objective or artifact["partitions"] != source["partitions"]:
            raise ValueError("executed-value model contract differs")
        model = ExecutedValueCritic(artifact["context_dim"]).to(args.device)
        model.load_state_dict(artifact["state_dict"])
        model.freeze()
        adapter = zero_adapter().to(args.device)
        before = {k:v.detach().clone() for k,v in adapter.state_dict().items()}
        virtual,parameter_report = calibration_update(model,artifact,adapter,states,data["records"],source["partitions"]["calibration"],args.device)
        action_reports,parameter_reports = [],[]
        for index,packet in heldout:
            hidden = restore_packet(env,packet)
            mean = packet["base_mu"].to(args.device).float()
            context = torch.as_tensor(packet["context"],device=args.device).float().unsqueeze(0)
            def value(action):
                return predict(model,artifact,context,action.unsqueeze(0)).squeeze(0)
            def evaluate(action):
                nonlocal calls
                calls += 1
                return evaluate_ppcsa_normal_shadow(env,action,hidden)
            goal = env.subgoals-env.positions
            goal = goal/np.maximum(np.linalg.norm(goal,axis=1,keepdims=True),1e-8)
            for epsilon in EPSILONS:
                for row in audit_action_directions(mean,value,evaluate,goal,epsilon,seed=20261121+index):
                    action_reports.append({"layout":packet["layout_seed"],"step":packet["step"],**row})
            if virtual is not None:
                with torch.no_grad():
                    features = packet["hidden_features"].to(args.device).float()
                    residual = virtual(features)
                    proposed_action = torch.tanh(mean+residual)
                    base_action = torch.tanh(mean)
                    sigma = trainer.actor.actor_modules()[0]._effective_action_log_std().detach().exp()
                    kl = .5*(residual/sigma).square().sum(-1)
                baseline = evaluate(base_action.cpu().numpy())
                candidate = evaluate(proposed_action.cpu().numpy())
                parameter_reports.append({"layout":packet["layout_seed"],"step":packet["step"],
                    **assess_probe(baseline,candidate),"new_hard_violation":baseline.hard_safe and not candidate.hard_safe,
                    "pre_tanh_residual_max":float(residual.abs().max()),"analytic_fixed_std_kl_mean":float(kl.mean()),
                    "rejection_reasons":candidate.rejection_reasons})
        if any(not torch.equal(before[k],v) for k,v in adapter.state_dict().items()):
            raise RuntimeError("virtual parameter probe changed the source Adapter")
        parameter_report.update(heldout_outcomes=parameter_reports,
            **summarize_parameter_outcomes(parameter_reports),
            virtual_update_always_discarded=True)
        summaries = {name:dict(Counter(r["outcome"] for r in action_reports if r["direction"] == name)) for name in ("critic","random","goal")}
        all_models[objective] = {"action_direction_counts":summaries,"action_probes":action_reports,
            "predicted_positive_but_absorbed":sum(r["predicted_gain"]>0 and r["outcome"] == "absorbed" for r in action_reports if r["direction"] == "critic"),
            "parameter_probe":parameter_report,"actor_training_authorized":False}
    if file_sha(resolve(source["checkpoint"])) != source["checkpoint_sha256"]:
        raise RuntimeError("PPCSA diagnostic modified parent checkpoint")
    report = {**provenance,"gate":2,"status":"REVIEW_REQUIRED","models":all_models,
        "gate1_manifest_sha256":file_sha(gate1/"manifest.json"),"critic_manifest_sha256":file_sha(critic_dir/"manifest.json"),
        "audit_layouts":sorted(set(s["layout_seed"] for _,s in heldout)),"partitions":source["partitions"],
        "probe_epsilons":list(EPSILONS),"parameter_step":PARAMETER_STEP,
        "shadow_calls":calls,"wall_seconds":perf_counter()-started,"device":args.device,
        "success_thresholds":None,"pilot_authorized":False,"actor_training_authorized":False,
        "note":"One-step, held-out development evidence; all temporary parameter changes discarded; no PPO/Actor update or closed-loop guarantee"}
    save_json(report,output/"ppcsa_gradient_report.json")
    write_manifest(output,report,["ppcsa_gradient_report.json"])
    print(f"GATE2=REVIEW_REQUIRED; ACTOR_TRAINING=BLOCKED; DETAIL_LOG={output/'ppcsa_gradient_report.json'}")
    return report


if __name__ == "__main__":
    main()
