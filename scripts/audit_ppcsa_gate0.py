"""User-run frozen-parent Gate 0 collection/parity diagnostic; no training."""
import argparse
from collections import Counter
from pathlib import Path
import sys
from time import perf_counter

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from src.algorithms.ppcsa_protocol import PARENT,PANEL,resolve,file_sha,validate_panel,new_output,runtime,actor_state,context_vector,write_manifest
from src.safety.ppcsa_one_step import evaluate_ppcsa_normal_shadow,capture_packet,normal_support,factual_parity
from src.utils.io_utils import save_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",default=PARENT)
    parser.add_argument("--panel",default=PANEL)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--states-per-layout",type=int,default=4)
    parser.add_argument("--sample-stride",type=int,default=20)
    parser.add_argument("--max-steps",type=int,default=80)
    args = parser.parse_args(argv)
    if min(args.states_per_layout,args.sample_stride,args.max_steps)<=0:
        parser.error("sampling counts must be positive")
    panel_path = resolve(args.panel)
    panel = validate_panel(panel_path)
    output = new_output(args.output_dir)
    trainer,env,provenance = runtime(args.checkpoint,output,args.device)
    started = perf_counter()
    states,parities = [],[]
    excluded = Counter()
    episode_summaries = []
    for seed in panel["layout_seeds"]:
        obs,_ = env.reset(seed=seed)
        hidden = trainer.actor.initial_hidden(3,trainer.device)
        episode_states = []
        done = False
        info = {}
        for step in range(min(args.max_steps,env.max_episode_steps)):
            action,mean,features,next_hidden = actor_state(trainer,obs,hidden)
            sample = step%args.sample_stride == 0 and len(episode_states)<args.states_per_layout
            for reason in normal_support(env):
                excluded[reason] += 1
            shadow = None
            if sample:
                packet = capture_packet(env,seed,hidden.detach().cpu())
                packet.update(step=step,base_mu=mean,hidden_features=features,
                    context=context_vector(trainer,env,obs),requested_action=action)
                shadow = evaluate_ppcsa_normal_shadow(env,action,hidden)
                packet["baseline"] = shadow.payload()
                episode_states.append(packet)
            obs,_,terminated,truncated,info = env.step(action)
            hidden = next_hidden
            done = bool(terminated or truncated)
            if shadow is not None and shadow.executed_actions:
                passed,evidence = factual_parity(shadow,env,info)
                parities.append({"layout":seed,"step":step,"passed":passed,**evidence})
                packet["parity_passed"] = passed
                packet["factual_supported"] = not normal_support(env) and not done and not info.get("collision_count",0)
                if not passed:
                    break
            elif shadow is not None:
                packet.update(parity_passed=False,factual_supported=False)
            if done:
                break
        failure_step = env.step_count if info.get("collision_count",0) or (done and not info.get("safe_success",False)) else None
        for packet in episode_states:
            packet["near_failure_diagnostic_only"] = failure_step is not None and 0<=failure_step-packet["step"]<=10
        states.extend(episode_states)
        episode_summaries.append({"layout":seed,"steps":env.step_count,"sampled":len(episode_states),
            "terminated":done,"sampling_capped":not done,"collision_count":info.get("collision_count",0)})
        if any(not p["passed"] for p in parities):
            break
    if file_sha(resolve(args.checkpoint)) != provenance["checkpoint_sha256"]:
        raise RuntimeError("read-only PPCSA diagnostic changed the frozen parent")
    passed = bool(parities) and all(p["passed"] for p in parities) and len(episode_summaries)==len(panel["layout_seeds"])
    report = {**provenance,"gate":0,"status":"PASS" if passed else "FAIL",
        "scope":"development-only sampled NORMAL one-step parity; no long-term safety or efficacy claim",
        "panel_sha256":file_sha(panel_path),"partitions":panel["partitions"],
        "parity_checks":parities,"episode_summaries":episode_summaries,"excluded_step_reasons":dict(excluded),
        "states":len(states),"wall_seconds":perf_counter()-started,"device":args.device,
        "shadow_calls":sum(bool(s["baseline"]["executed_actions"]) for s in states),
        "sampling":{"states_per_layout":args.states_per_layout,"stride":args.sample_stride,"max_steps":args.max_steps}}
    torch.save(states,output/"states.pt")
    save_json(report,output/"gate0_report.json")
    write_manifest(output,report,["states.pt","gate0_report.json"])
    print(f"GATE0={report['status']}; DETAIL_LOG={output/'gate0_report.json'}")
    if not passed:
        raise RuntimeError("Gate 0 parity failed; Gate 1/2 label generation is blocked")
    return report


if __name__ == "__main__":
    main()
