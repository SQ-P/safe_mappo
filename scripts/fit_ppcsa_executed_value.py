"""User-only offline ordinary/paired executed-value fits after Gate 1 review."""
import argparse
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from src.algorithms.ppcsa_protocol import resolve,new_output,read_manifest,file_sha,require_gate1_review,write_manifest
from src.algorithms.ppcsa_executed_value import ExecutedValueCritic
from src.utils.io_utils import save_json


def predict(model,artifact,context,action):
    mean = torch.as_tensor(artifact["context_mean"],device=context.device)
    scale = torch.as_tensor(artifact["context_scale"],device=context.device)
    return model((context-mean)/scale,action)*artifact["target_scale"]+artifact["target_mean"]


def fit_models(examples,partitions,*,epochs=30,device="cpu"):
    if epochs<=0:
        raise ValueError("epochs must be positive")
    train = [e for e in examples if e["layout"] in partitions["fit"]]
    calibration = [e for e in examples if e["layout"] in partitions["calibration"]]
    if not train or not calibration:
        raise ValueError("PPCSA fitting requires safe examples in disjoint Fit and calibration layouts")
    if set(partitions["fit"])&set(partitions["calibration"]) or (set(partitions["fit"]+partitions["calibration"])&set(partitions["audit"])):
        raise ValueError("PPCSA layout leakage")
    x = torch.as_tensor(np.stack([e["context"] for e in train]),dtype=torch.float32,device=device)
    action = torch.as_tensor(np.stack([e["action"] for e in train]),dtype=torch.float32,device=device)
    base = torch.as_tensor(np.stack([e["base_action"] for e in train]),dtype=torch.float32,device=device)
    y = torch.as_tensor([e["target"] for e in train],dtype=torch.float32,device=device)
    ybase = torch.as_tensor([e["base_target"] for e in train],dtype=torch.float32,device=device)
    counts = {g:sum(e["layout"] == g for e in train) for g in set(e["layout"] for e in train)}
    weights = torch.as_tensor([1./counts[e["layout"]] for e in train],device=device)
    weights /= weights.sum()
    mean = x.mean(0)
    scale = x.std(0,unbiased=False).clamp(min=1e-6)
    target_mean,target_scale = float((weights*y).sum()),float(y.std(unbiased=False).clamp(min=1e-8))
    artifact = {"context_dim":x.shape[1],"context_mean":mean.cpu(),"context_scale":scale.cpu(),
        "target_mean":target_mean,"target_scale":target_scale,"hidden_dim":128,"partitions":partitions,
        "epochs":epochs,"learning_rate":.001,"seed":20261121,"actor_training_authorized":False}
    models,diagnostics = {},{}
    for objective in ("ordinary","paired"):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261121)
            model = ExecutedValueCritic(x.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(),lr=.001)
        normalized = (x-mean)/scale
        target = (y-target_mean)/target_scale
        for _ in range(epochs):
            optimizer.zero_grad()
            pred = model(normalized,action)
            loss = (weights*(pred-target).square()).sum()
            if objective == "paired":
                gap = model(normalized,action)-model(normalized,base)
                loss = loss+(weights*(gap-(y-ybase)/target_scale).square()).sum()
            loss.backward()
            optimizer.step()
        model.freeze()
        errors = []
        gains = []
        constant,heuristic = [],[]
        with torch.no_grad():
            for e in calibration:
                context = torch.as_tensor(e["context"],dtype=torch.float32,device=device).unsqueeze(0)
                a = torch.as_tensor(e["action"],dtype=torch.float32,device=device).unsqueeze(0)
                b = torch.as_tensor(e["base_action"],dtype=torch.float32,device=device).unsqueeze(0)
                q = float(predict(model,artifact,context,a))
                qb = float(predict(model,artifact,context,b))
                errors.append((e["layout"],(q-e["target"])**2))
                if e["target"] != e["base_target"]:
                    gains.append((q-qb)*(e["target"]-e["base_target"])>0.)
                constant.append((e["layout"],(target_mean-e["target"])**2))
                heuristic.append((e["layout"],(e["heuristic_target"]-e["target"])**2))
        def grouped(values):
            return float(np.mean([np.mean([v for k,v in values if k == g]) for g in set(k for k,v in values)]))
        diagnostics[objective] = {"calibration_layout_balanced_mse":grouped(errors),"constant_mse":grouped(constant),
            "unprojected_kinematic_mse":grouped(heuristic),"pair_direction_agreement":float(np.mean(gains)) if gains else None,
            "non_tied_comparisons":len(gains),
            "note":"Prediction metrics do not certify gradients; all candidate rows from a layout stay together"}
        models[objective] = {**artifact,"objective":objective,"state_dict":{k:v.cpu() for k,v in model.state_dict().items()}}
    return models,diagnostics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate1-dir",required=True)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--epochs",type=int,default=30)
    parser.add_argument("--device",default="cpu")
    args = parser.parse_args(argv)
    gate1 = resolve(args.gate1_dir)
    manifest = read_manifest(gate1)
    review = require_gate1_review(gate1,file_sha(gate1/"ppcsa_opportunity_report.json"))
    data = torch.load(gate1/"opportunities.pt",map_location="cpu",weights_only=False)
    if not manifest.get("coverage_counts",{}).get("positive_local",0):
        raise RuntimeError("No safe reachable opportunities; Critic fitting blocked")
    output = new_output(args.output_dir)
    started = perf_counter()
    models,metrics = fit_models(data["examples"],manifest["partitions"],epochs=args.epochs,device=args.device)
    for name,model in models.items():
        torch.save(model,output/f"{name}.pt")
    report = {**{k:manifest[k] for k in ("checkpoint","checkpoint_sha256","config_sha256","task_config_fingerprint","partitions","source_sha256")},
        "gate":2,"status":"GRADIENT_AUDIT_REQUIRED","gate1_manifest_sha256":file_sha(gate1/"manifest.json"),
        "gate1_review":review,"prediction_metrics":metrics,"epochs":args.epochs,"device":args.device,
        "wall_seconds":perf_counter()-started,"actor_training_authorized":False}
    save_json(report,output/"value_report.json")
    write_manifest(output,report,["ordinary.pt","paired.pt","value_report.json"])
    print(f"VALUE_FIT=DIAGNOSTIC_ONLY; NEXT=GRADIENT_AUDIT; DETAIL_LOG={output/'value_report.json'}")
    return report


if __name__ == "__main__":
    main()
