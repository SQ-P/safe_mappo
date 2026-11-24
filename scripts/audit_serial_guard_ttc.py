"""User-run P0 development replay of the unchanged serial Guard/TTC executor."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.safety.staged_execution.serial_conflict_audit import protocol

PARENT_SHA = "087448d2e036f51e450ae927f9c841b156e0b87ed28ec16ee98de1427bfecb02"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT/path).resolve()


def source_hashes(config, checkpoint, panel):
    files = [config, checkpoint, panel, Path(__file__).resolve()]
    files += list((ROOT/"src").rglob("*.py")) + list((ROOT/"configs").rglob("*.yaml"))
    for path in (ROOT/"results/ablations/safety_seed23").rglob("*.json"):
        files.append(path)
    return {str(p.resolve()): sha(p) for p in sorted(set(files))}


def prepare(config_path, checkpoint, panel, output):
    from src.utils.io_utils import load_yaml
    if sha(checkpoint) != PARENT_SHA:
        raise ValueError("P0 requires the pinned seed23 update0800 legacy checkpoint")
    config = load_yaml(config_path)
    safety = config["safety_execution"]
    if (config.get("algorithm") != "mappo" or config.get("seed") != 23
            or config["env"].get("num_agents") != 3 or config["env"].get("static_obstacle_num") != 8
            or config["env"].get("dynamic_obstacle_num") != 2
            or not config.get("oracle_subgoal", {}).get("enabled")
            or safety.get("version") != "staged_joint_guard_ttc_v1"
            or safety.get("safety_state_machine_version") != "star_scoped_recovery_v4"
            or safety.get("obstacle_ttc", {}).get("version") != "causal_ttc_progress_projection_v2"
            or not safety.get("recovery_enabled")
            or any(not safety.get(s, {}).get("enabled") for s in ("initial_joint_guard", "obstacle_ttc", "final_joint_guard"))):
        raise ValueError("Requested config does not identify the frozen full serial baseline")
    seeds = json.loads(panel.read_text(encoding="utf-8"))
    if seeds.get("layout_seeds") != list(range(121000, 121200)):
        raise ValueError("P0 requires the already exposed 200-layout development panel")
    if output.exists() or any(parent in output.parents or output == parent for parent in (
            ROOT/"src", ROOT/"configs", ROOT/"results/paper", ROOT/"results/ablations", checkpoint.parent)):
        raise ValueError("Output must be new and separate from sources and historical results")
    # Inspect checkpoint metadata only; never instantiate or execute a trainer in preflight.
    import torch
    from src.algorithms.ppcsa_protocol import validate_parent
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_parent(saved, PARENT_SHA)
    effective = saved["config"]
    for key in ("num_agents", "static_obstacle_num", "dynamic_obstacle_num", "reward_mode", "observation_schema_version"):
        if config["env"].get(key) != effective["env"].get(key):
            raise ValueError("Requested config/checkpoint environment identity differs: "+key)
    return {"config": str(config_path), "checkpoint": str(checkpoint), "panel": str(panel),
        "layout_seeds": seeds["layout_seeds"], "checkpoint_task_fingerprint": saved["task_config_fingerprint"],
        "configuration_source": "checkpoint configuration is authoritative; supplied config cross-checks identity",
        "input_sha256": source_hashes(config_path, checkpoint, panel), "protocol": protocol()}


def write(path, value):
    from src.safety.ppcsa_one_step import jsonable
    from scripts.diagnose_ppcsa_direction_feasibility import json_safe
    Path(path).write_text(json.dumps(json_safe(jsonable(value)), indent=2, allow_nan=False), encoding="utf-8")


def run_replay(bound, output):
    import torch
    from src.algorithms.ppcsa_protocol import runtime
    from src.safety.ppcsa_one_step import jsonable
    from scripts.diagnose_ppcsa_direction_feasibility import json_safe
    from src.safety.staged_execution.serial_conflict_audit import audit_state, attach_factual_parity, summarize, preserve_rng
    trainer, env, provenance = runtime(bound["checkpoint"], output, "cpu")
    if provenance["task_config_fingerprint"] != bound["checkpoint_task_fingerprint"]:
        raise ValueError("Runtime task configuration differs from frozen checkpoint")
    env.set_runtime_telemetry_level("full")
    actor_before = {k: v.detach().cpu().clone() for k, v in trainer.actor.state_dict().items()}
    critic_before = {k: v.detach().cpu().clone() for k, v in trainer.critic.state_dict().items()}
    rows, completed, counts = [], [], []
    event_path = output/"conflict_events.jsonl"
    with event_path.open("x", encoding="utf-8") as stream:
        for seed in bound["layout_seeds"]:
            obs, _ = trainer.env.reset(seed=seed)
            hidden = trainer.actor.initial_hidden(trainer.policy_rows, trainer.device)
            episode_steps = 0
            for _ in range(trainer.env.max_episode_steps):
                with torch.no_grad():
                    output_actor = trainer.actor.forward_step(torch.as_tensor(trainer.policy_observations(obs),
                        dtype=torch.float32, device=trainer.device), hidden, True)
                raw = output_actor.action.cpu().numpy().reshape(env.num_agents, 2)
                row = audit_state(env, raw, seed, output_actor.hidden)
                obs, _, terminated, truncated, info = trainer.env.step(raw.copy())
                row = attach_factual_parity(row, info)
                rows.append({k: row[k] for k in ("layout_seed", "e1_material_agents", "e2_rule_retrigger", "e3_loss_agents",
                    "evidence_status", "factual_parity", "potential_progress_opportunity")})
                # Retain every state, including non-events and unsupported Recovery.
                stream.write(json.dumps(json_safe(jsonable(row)), allow_nan=False)+"\n")
                hidden = output_actor.hidden
                episode_steps += 1
                if terminated or truncated:
                    break
            completed.append(seed); counts.append({"layout_seed": seed, "steps": episode_steps})
    if any(not torch.equal(actor_before[k], v.detach().cpu()) for k, v in trainer.actor.state_dict().items()):
        raise RuntimeError("Frozen policy parameters changed")
    if any(not torch.equal(critic_before[k], v.detach().cpu()) for k, v in trainer.critic.state_dict().items()):
        raise RuntimeError("Frozen value parameters changed")
    result = summarize(rows, completed)
    if completed != bound["layout_seeds"]:
        raise RuntimeError("Incomplete frozen development panel")
    result.update(provenance=provenance, episode_coverage=counts, policy_parameters_unchanged=True, critic_parameters_unchanged=True,
        config_semantics_unchanged=True, full_replay_run=True, candidate_actions_never_installed=True,
        source_integrity_unchanged=True, scope="P0 exposed development mechanism audit only")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--layout-seeds-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    config, checkpoint, panel, output = map(resolve, (args.config, args.checkpoint, args.layout_seeds_file, args.output_dir))
    for path in (config, checkpoint, panel, output): path.relative_to(ROOT.resolve())
    bound = prepare(config, checkpoint, panel, output)
    if args.preflight_only:
        print(f"PREFLIGHT=PASS; LAYOUTS=200; CHECKPOINT=SEED23_U800; MAX_JOINT_CANDIDATES=64; TRAINER=NOT_CREATED; REPLAY=NOT_RUN; OUTPUT_CREATED=NO")
        return {"status": "READY", "diagnostic_run": False}
    output.mkdir(parents=True)
    write(output/"source_integrity_before.json", bound)
    result = run_replay(bound, output)
    after = source_hashes(config, checkpoint, panel)
    write(output/"source_integrity_after.json", {"input_sha256": after, "unchanged": after == bound["input_sha256"]})
    if after != bound["input_sha256"]:
        raise RuntimeError("P0 modified protected source/checkpoint/config/historical artifacts")
    write(output/"serial_conflict_report.json", result)
    (output/"serial_conflict_report.md").write_text("# Serial Guard/TTC P0 audit\n\n"+
        f"Decision: **{result['next_stage_decision']}**.\n\n"+
        f"Audited states: {result['audited_states']}; potential opportunity states: {result['potential_opportunity_states']}.\n\n"+
        "Observable heuristic evidence only; not a safety certificate or closed-loop superiority result.\n"+
        "P1/P2/P3 and training remain unauthorized. Full state records: conflict_events.jsonl.\n", encoding="utf-8")
    print(f"SERIAL_AUDIT={result['next_stage_decision']}; TRAINING=BLOCKED; DETAIL_LOG={output/'serial_conflict_report.json'}")
    return result


if __name__ == "__main__":
    main()
