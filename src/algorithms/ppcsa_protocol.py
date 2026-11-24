"""Frozen, development-only PPCSA IO and parent contracts."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.safety.ppcsa_one_step import PROTOCOL, digest
from src.utils.io_utils import load_yaml, save_json

ROOT = Path(__file__).resolve().parents[2]
PARENT = "results/paper/standard_mappo_3u_8s2d_legacy_execution_v1/seed_23/checkpoints/update_0800.pt"
PARENT_SHA = "087448d2e036f51e450ae927f9c841b156e0b87ed28ec16ee98de1427bfecb02"
PANEL = "configs/evaluation/ppcsa_dev_121000_121039.json"
SOURCES = ("src/envs/cooperative_uav_env.py","src/safety/ppcsa_one_step.py","src/algorithms/ppcsa_executed_value.py","src/algorithms/ppcsa_protocol.py")


def source_hashes():
    return {name:file_sha(ROOT/name) for name in SOURCES}


def resolve(path):
    value = Path(path)
    return value if value.is_absolute() else ROOT/value


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_panel(path):
    panel = json.loads(Path(path).read_text())
    expected = list(range(121000,121040))
    parts = panel.get("partitions",{})
    if (panel.get("schema_version") != "PPCSADevelopmentPanelV1" or panel.get("protocol") != PROTOCOL
            or panel.get("layout_seeds") != expected
            or parts != {"fit":expected[:24],"calibration":expected[24:32],"audit":expected[32:]}):
        raise ValueError("PPCSA requires the frozen exposed development panel and disjoint layout partitions; formal/final panels are sealed")
    return panel


def validate_parent(saved,sha):
    cfg = saved["config"]
    safety,env = cfg["safety_execution"],cfg["env"]
    if (sha != PARENT_SHA or cfg.get("algorithm") != "mappo" or cfg.get("seed") != 23
            or saved.get("update") != 800 or saved.get("obs_dim") != 52
            or saved.get("hidden_dim") != 128 or saved.get("num_agents") != 3
            or env.get("obstacle_slots") != 6 or env.get("static_obstacle_num") != 8 or env.get("dynamic_obstacle_num") != 2
            or env.get("reward_mode") != "team_v4_subgoal"
            or safety.get("safety_state_machine_version") != "star_scoped_recovery_v4"
            or safety["obstacle_ttc"].get("version") != "causal_ttc_progress_projection_v2"
            or safety.get("goal_boundary_corridor",{}).get("enabled",False)
            or not all(safety.get(k,{}).get("enabled") for k in ("initial_joint_guard","obstacle_ttc","final_joint_guard"))):
        raise ValueError("PPCSA parent must be the pinned seed23 u800 52D V4/TTCv2 contract; no V5.4/corridor substitution")


def new_output(path):
    output = resolve(path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    return output


def runtime(checkpoint,output,device="cpu"):
    from src.algorithms.factory import create_trainer
    path = resolve(checkpoint)
    saved = torch.load(path,map_location="cpu",weights_only=False)
    sha = file_sha(path)
    validate_parent(saved,sha)
    config = deepcopy(saved["config"])
    config["device"] = device
    config["output"] = {"output_dir":str(Path(output)/"runtime"),"checkpoint_dir":str(Path(output)/"runtime/checkpoints"),"reject_nonempty":True}
    trainer = create_trainer(config,project_root=ROOT)
    trainer.load_checkpoint(path)
    if trainer.task_config_fingerprint() != saved["task_config_fingerprint"]:
        raise ValueError("PPCSA must not change the checkpoint task/reward/executor contract")
    trainer.actor.eval()
    for parameter in trainer.actor.parameters():
        parameter.requires_grad_(False)
    env = trainer._base_env()
    env.set_runtime_telemetry_level("full")
    return trainer,env,{"protocol":PROTOCOL,"checkpoint":str(path.resolve()),"checkpoint_sha256":sha,
        "task_config_fingerprint":saved["task_config_fingerprint"],"config_sha256":digest(saved["config"]),
        "obs_dim":52,"hidden_dim":128,"adapter_rank":16,"adapter_scale":.15,
        "source_sha256":source_hashes(),
        "actor_training_authorized":False,"formal_validation_authorized":False}


def actor_state(trainer,observations,hidden):
    actor = trainer.actor.actor_modules()[0]
    local = torch.as_tensor(trainer.normalize_local_observations(observations),device=trainer.device)
    with torch.no_grad():
        encoded = actor.encoder(local.float()).unsqueeze(0)
        recurrent,next_hidden = actor.gru(encoded,hidden.to(trainer.device).float())
        features = recurrent.squeeze(0)
        mean = actor.action_mean(features)
        action = torch.tanh(mean)
    return action.cpu().numpy(),mean.cpu(),features.cpu(),next_hidden.detach()


def context_vector(trainer,env,observations):
    cfg = trainer.config["safety_execution"]["recovery"]
    return np.concatenate([trainer.normalize_local_observations(observations).reshape(-1),np.asarray([
        float(env._safety_mode == "NORMAL"),env.step_count/env.max_episode_steps,
        env._safe_recovery_sequence_steps/max(1,cfg["maximum_steps"]),
        env._safe_recovery_attempt_index/max(1,cfg["maximum_attempts"])],dtype=np.float32)]).astype(np.float32)


def write_manifest(output,payload,files):
    payload = {**payload,"protocol":PROTOCOL,"file_sha256":{name:file_sha(Path(output)/name) for name in files},
               "actor_training_authorized":False,"formal_validation_authorized":False}
    save_json(payload,Path(output)/"manifest.json")
    return payload


def read_manifest(directory):
    directory = resolve(directory)
    manifest = json.loads((directory/"manifest.json").read_text())
    if manifest.get("protocol") != PROTOCOL or manifest.get("checkpoint_sha256") != PARENT_SHA:
        raise ValueError("PPCSA artifact protocol/parent mismatch")
    for name,sha in manifest["file_sha256"].items():
        if Path(name).name != name or file_sha(directory/name) != sha:
            raise ValueError("PPCSA artifact hash mismatch")
    if set(manifest.get("source_sha256",{})) != set(SOURCES):
        raise ValueError("PPCSA artifact lacks the frozen implementation hashes")
    for name,sha in manifest["source_sha256"].items():
        if not name.startswith("src/") or ".." in Path(name).parts or file_sha(ROOT/name)!=sha:
            raise ValueError("PPCSA implementation changed since data creation; repeat the required gate")
    return manifest


def require_gate1_review(directory,report_sha):
    path = Path(directory)/"gate1_review.json"
    if not path.exists():
        raise RuntimeError("Gate 1 opportunity evidence must be reviewed before any real Critic fit; gate1_review.json is absent")
    review = json.loads(path.read_text())
    if (review.get("schema_version") != "PPCSAGate1ReviewV1" or review.get("report_sha256") != report_sha
            or review.get("decision") != "approve_critic_diagnostic"
            or not str(review.get("rationale","")).strip()):
        raise RuntimeError("Gate 1 review does not approve this exact opportunity report")
    return review
