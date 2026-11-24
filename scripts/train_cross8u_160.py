"""User-started first round: fresh Cross8U MAPPO, Seed23, exactly400 updates."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from scripts.run_cross8u_160 import load_config, create_cross_trainer, resolve, sha

CONFIG = "configs/verification/cross8u_160/train_seed23_u400.yaml"


def training_contract(config):
    expected = load_config("configs/verification/cross8u_160/seed23.yaml")
    expected["train"].update(total_updates=400,checkpoint_updates=[100,200,400])
    expected["evaluation"].update(updates=[100,200,400],save_episode_records=True)
    expected["output"] = {
        "output_dir":"results/verification/cross8u_160/train/seed_23_u400_v1",
        "checkpoint_dir":"results/verification/cross8u_160/train/seed_23_u400_v1/checkpoints",
        "reject_nonempty":True}
    expected["verification_experiment"].update(scope="FIRST_ROUND_SEED23_U400_USER_RUN",
        training_authorized=True,formal_evaluation_authorized=False,
        training_update_limit=400,execution_owner="user")
    if config != expected:
        raise ValueError("First-round config must match the frozen8U snapshot plus Seed23/400-update overrides")


def preflight(config):
    """Readiness only: no trainer, environment rollout, checkpoint load or output."""
    import torch
    training_contract(config)
    folder = ROOT/"results/verification/cross8u_160/p2_implementation_v1"
    review = json.loads((folder/"web_review.json").read_text())
    if review.get("state") != "DONE" or review.get("review") != "ACCEPTED":
        raise RuntimeError("P2 accepted evidence is required")
    protected = json.loads((folder/"source_integrity_before.json").read_text())["protected_file_sha256"]
    changed = [p for p,digest in protected.items() if not (ROOT/p).is_file() or sha(ROOT/p) != digest]
    if changed:
        raise RuntimeError("Protected environment/history changed: "+str(changed))
    protocol = json.loads((ROOT/"results/verification/cross8u_160/smoke/p2_seed23_v1/smoke_protocol.json").read_text())
    if any(sha(ROOT/p) != digest for p,digest in protocol["source_sha256"].items()):
        raise RuntimeError("P2-tested environment/construction entry changed")
    output = resolve(config["output"]["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("First-round output is nonempty; refusing overwrite or implicit resume: "+str(output))
    if not torch.cuda.is_available():
        raise RuntimeError("Use the project's CUDA-ready Python environment")
    # Tiny compatibility test only: the base Python has an unsupported GPU build.
    sample = torch.ones((2,2),device="cuda")
    if not bool(torch.isfinite(sample@sample).all().item()):
        raise RuntimeError("CUDA compatibility test failed")
    torch.cuda.synchronize()
    return {"status":"READY_SEED23_U400","device":torch.cuda.get_device_name(0),
        "python":sys.executable,"torch":torch.__version__,"training_updates":400,
        "rollout_steps":2048,"ppo_epochs":4,"evaluation_updates":[100,200,400],
        "evaluation_episodes_per_update":10,"evaluation_seed_base":240000,
        "evaluation_max_steps":500,"formal_layouts_executed":0,
        "fresh_initialization":True,"historical_weight_transfer":False,
        "protected_files_unchanged":len(protected),"output_dir":str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default=CONFIG)
    parser.add_argument("--preflight-only",action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    ready = preflight(config)
    if args.preflight_only:
        print(json.dumps({**ready,"training_executed":False,"output_created":False},ensure_ascii=False))
        return ready
    print("TRAINING=STARTING; SEED=23; UPDATES=400; FRESH=YES; FORMAL_PANEL=NOT_RUN; OUTPUT="+ready["output_dir"],flush=True)
    trainer = create_cross_trainer(deepcopy(config),ROOT)
    if trainer.current_update != 0:
        raise RuntimeError("First round must start fresh, never from the P2 smoke checkpoint")
    summary = trainer.train()
    print("TRAINING=COMPLETE; UPDATES="+str(summary["updates_completed"])+"; DETAIL_LOG="+str(trainer.output_dir/"train_summary.json"),flush=True)
    return summary


if __name__ == "__main__":
    main()
