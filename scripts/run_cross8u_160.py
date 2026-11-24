"""Independent 8U/160 verification entry: P0/P1 and one bounded P2 smoke."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.envs.cross8u_160_env import Cross8U160Env, STARTS, GOALS, SCENARIO
from src.envs.cooperative_uav_env import CooperativeUAVEnv
from src.utils.io_utils import load_yaml, to_jsonable

CONFIG_DIR = ROOT/"configs/verification/cross8u_160"
OUTPUT_ROOT = ROOT/"results/verification/cross8u_160"
_INJECTION_LOCK = threading.Lock()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024*1024), b""):
            value.update(chunk)
    return value.hexdigest()


def resolve(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT/path).resolve()


def environment_config(config):
    env = deepcopy(config["env"])
    for key in ("reward", "oracle_subgoal", "subgoal_selector", "safety_execution", "runtime"):
        if key in config:
            env[key] = deepcopy(config[key])
    return env


def load_config(path):
    path = resolve(path)
    path.relative_to(CONFIG_DIR.resolve())
    config = load_yaml(path)
    verification = config.get("verification_experiment", {})
    if (config.get("algorithm") != "mappo" or config.get("seed") != 23
            or verification.get("environment_class") != "src.envs.cross8u_160_env.Cross8U160Env"
            or verification.get("fresh_initialization_required") is not True
            or verification.get("historical_weight_transfer") is not False):
        raise ValueError("Independent fresh MAPPO configuration required")
    output = resolve(config["output"]["output_dir"])
    output.relative_to(OUTPUT_ROOT.resolve())
    checkpoint = resolve(config["output"]["checkpoint_dir"])
    checkpoint.relative_to(output)
    if config["output"].get("reject_nonempty") is not True:
        raise ValueError("Independent outputs must reject nonempty paths")
    return config


@contextmanager
def isolated_environment_injection():
    """Synchronous single-process construction only; always restore the symbol."""
    import src.algorithms.ppo.trainer as trainer_module
    if threading.current_thread() is not threading.main_thread() or threading.active_count() != 1:
        raise RuntimeError("Cross8U injection requires a single Python thread")
    if not _INJECTION_LOCK.acquire(blocking=False):
        raise RuntimeError("Cross8U construction injection is already active")
    original = trainer_module.CooperativeUAVEnv
    try:
        if original is not CooperativeUAVEnv:
            raise RuntimeError("Unexpected preexisting environment injection")
        trainer_module.CooperativeUAVEnv = Cross8U160Env
        yield
    finally:
        trainer_module.CooperativeUAVEnv = original
        _INJECTION_LOCK.release()


def create_cross_trainer(config, project_root=ROOT):
    """Construct fresh untrained MAPPO for compatibility checks; no checkpoint IO."""
    from src.algorithms.factory import create_trainer
    if config.get("algorithm") != "mappo" or config["env"].get("verification_scenario") != SCENARIO:
        raise ValueError("Only the independent cross8u configuration is allowed")
    with isolated_environment_injection():
        trainer = create_trainer(deepcopy(config), project_root=project_root)
    if not isinstance(trainer._base_env(), Cross8U160Env):
        raise RuntimeError("Trainer failed to retain the independent environment")
    if trainer.num_agents != 8 or trainer.obs_dim != 80 or trainer.global_state_dim != 640:
        raise RuntimeError("Cross8U model dimensions differ")
    return trainer


def read_panels():
    development = json.loads((CONFIG_DIR/"development_layouts.json").read_text())
    formal = json.loads((CONFIG_DIR/"formal_layouts.json").read_text())
    for panel in (development, formal):
        if panel.get("scenario") != SCENARIO or panel.get("schema_version") != "Cross8U160LayoutPanelV1":
            raise ValueError("Wrong verification seed panel")
    dev, heldout = development["layout_seeds"], formal["layout_seeds"]
    if len(dev) != 40 or len(heldout) != 200 or len(set(dev+heldout)) != 240:
        raise ValueError("Development/formal panels must be unique, disjoint and 40/200 layouts")
    return dev, heldout


def layout_diagnostics(env):
    starts, goals = env.positions.copy(), env.goals.copy()
    minimum_agent = min(float(np.linalg.norm(a.position-b.position)-a.radius-b.radius)
        for i,a in enumerate(env.agents) for b in env.agents[i+1:])
    minimum_pair = min(float(np.linalg.norm(a.position-b.position)-a.radius-b.radius)
        for i,a in enumerate(env.obstacles) for b in env.obstacles[i+1:])
    minimum_start = min(float(np.linalg.norm(o.position-p)-o.radius-env.agent_radius) for o in env.obstacles for p in starts)
    minimum_goal = min(float(np.linalg.norm(o.position-p)-o.radius-env.goal_radius) for o in env.obstacles for p in goals)
    passed = (np.array_equal(starts,STARTS) and np.array_equal(goals,GOALS)
        and len(env.obstacles) == 10 and sum(o.dynamic for o in env.obstacles) == 6
        and minimum_agent > 0 and min(minimum_pair,minimum_start,minimum_goal) >= env.obstacle_spawn_clearance
        and env.layout_generation["routes"]["all_routes_exist"])
    return {"passed": bool(passed), "generation": deepcopy(env.layout_generation),
        "minimum_initial_uav_surface_gap_m": minimum_agent, "minimum_obstacle_surface_gap_m": minimum_pair,
        "minimum_start_surface_gap_m": minimum_start, "minimum_goal_region_gap_m": minimum_goal,
        "obstacles": [{"position":o.position.tolist(), "radius":o.radius, "velocity":o.velocity.tolist(), "dynamic":o.dynamic} for o in env.obstacles]}


def validate_development(config, steps=3):
    """Bounded P1 reset/short-step checks, not policy efficacy or a training run."""
    seeds, _ = read_panels()
    env = Cross8U160Env(environment_config(config))
    records = []
    for seed in seeds:
        observation, _ = env.reset(seed=seed)
        record = {"seed":seed, **layout_diagnostics(env), "policies":{}}
        original = [(o.position.copy(),o.velocity.copy(),o.radius) for o in env.obstacles]
        repeat, _ = env.reset(seed=seed)
        deterministic = np.array_equal(observation,repeat) and all(np.array_equal(p,o.position) and np.array_equal(v,o.velocity) and r == o.radius for (p,v,r),o in zip(original,env.obstacles))
        record["deterministic_reset"] = deterministic
        for policy in ("random", "goal_acceleration"):
            observation, _ = env.reset(seed=seed)
            rng = np.random.default_rng(seed)
            executed = 0
            for _ in range(steps):
                if policy == "random":
                    action = rng.uniform(-1,1,size=(8,2)).astype(np.float32)
                else:
                    direction = env.goals-env.positions
                    direction /= np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-8)
                    action = np.clip((2*direction-env.velocities)/(env.agent_max_acceleration*env.dt),-1,1)
                observation, reward, terminated, truncated, info = env.step(action)
                if (observation.shape != (8,80) or env.get_global_state().shape != (640,)
                        or not np.isfinite(observation).all() or not np.isfinite(reward).all()):
                    raise RuntimeError("Nonfinite or wrong-dimensional P1 transition")
                executed += 1
                if terminated or truncated:
                    break
            record["policies"][policy] = {"executed_steps": executed, "collision_count": int(info["collision_count"]),
                "terminated": bool(terminated), "truncated": bool(truncated)}
        if not record["passed"] or not deterministic:
            raise RuntimeError("Development initialization/geometry/determinism gate failed")
        records.append(record)
    return {"schema_version":"Cross8U160P1ReportV1", "stage":"P0_P1", "status":"P1_ENVIRONMENT_CHECKS_PASS",
        "layouts":records, "development_layout_count":40, "formal_layouts_executed":0,
        "start_positions":STARTS.tolist(), "goal_positions":GOALS.tolist(),
        "num_agents":8, "obs_dim":80, "global_state_dim":640, "action_shape":[8,2],
        "total_short_steps":sum(p["executed_steps"] for row in records for p in row["policies"].values()),
        "maximum_generation_attempts":max(row["generation"]["attempts"] for row in records),
        "obstacle_statistics":{"static_instances":160, "dynamic_instances":240,
            "radius_min_m":min(o["radius"] for row in records for o in row["obstacles"]),
            "radius_max_m":max(o["radius"] for row in records for o in row["obstacles"]),
            "minimum_start_surface_gap_m":min(row["minimum_start_surface_gap_m"] for row in records),
            "minimum_obstacle_surface_gap_m":min(row["minimum_obstacle_surface_gap_m"] for row in records),
            "minimum_goal_region_gap_m":min(row["minimum_goal_region_gap_m"] for row in records)},
        "training_updates":0, "checkpoint_loaded":False, "p2_smoke_executed":False,
        "limitations":["Initial geometric reachability is not dynamic/joint feasibility or learned success.",
            "Original nearest-teammate observation and x-only legacy telemetry remain unchanged; use cross8u_directional for both groups.",
            "P2/P3 are separate subsequent stages; this readiness result demonstrates no policy efficacy."]}


def smoke_config_contract(config):
    """Freeze every smoke override; never alter the full-training snapshot."""
    expected = load_config("configs/verification/cross8u_160/seed23.yaml")
    expected["device"] = "cpu"
    expected["train"].update(total_updates=1,rollout_steps=128,ppo_epochs=1,
        checkpoint_updates=[1],save_every=1,save_periodic_checkpoints=False,log_every=1)
    expected["evaluation"].update(episodes=1,interval_updates=1,seed_base=240000,
        updates=[1],save_episode_records=True)
    expected["verification_experiment"].update(scope="P2_SMOKE_ONLY",training_authorized=True,
        p2_update_limit=1,central_probe_seeds=[240000,240001],central_probe_step_limit=200,p3_authorized=False)
    expected["output"] = config["output"]
    if config != expected:
        raise ValueError("P2 requires the exact one-update/128-step/one-epoch/one-evaluation smoke snapshot")
    return {"updates":1,"rollout_steps":128,"ppo_epochs":1,"evaluation_episodes":1,
        "evaluation_seed":240000,"evaluation_max_steps":500,"central_probe_seeds":[240000,240001],
        "central_probe_max_steps":200,"p3_authorized":False}


def central_probes(config, output):
    env = Cross8U160Env(environment_config(config))
    results = []
    counters = ("joint_guard_intervention_count", "initial_joint_guard_intervention_steps",
        "final_joint_guard_residual_count", "obstacle_ttc_projection_count",
        "obstacle_ttc_fallback_count", "safe_recover_entry_count", "safe_recover_step_count",
        "recovery_candidate_fixed_point_streak", "collision_count", "static_obstacle_collision_count",
        "dynamic_obstacle_collision_count", "inter_agent_collision_count", "boundary_collision_count")
    with (output/"central_trajectories.jsonl").open("x",encoding="utf-8") as trace:
        for seed in (240000,240001):
            env.reset(seed=seed)
            entered = np.zeros(8,dtype=bool); outside_dynamic = False
            for index in range(200):
                direction = env.goals-env.positions
                direction /= np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-8)
                action = np.clip((2*direction-env.velocities)/(env.agent_max_acceleration*env.dt),-1,1).astype(np.float32)
                observation,reward,terminated,truncated,info = env.step(action)
                if (observation.shape != (8,80) or action.shape != (8,2) or not np.isfinite(action).all()
                        or not np.isfinite(observation).all() or not np.isfinite(reward).all()
                        or not np.isfinite(env.positions).all() or not np.isfinite(env.velocities).all()
                        or not np.isfinite(env.get_global_state()).all()):
                    raise FloatingPointError("Invalid central-probe state/action/reward")
                for stage_action in info.get("action_stage_actions",{}).values():
                    if np.asarray(stage_action).shape != (8,2) or not np.isfinite(stage_action).all():
                        raise FloatingPointError("Invalid central-probe execution-stage action")
                entered |= np.all((env.positions >= 45)&(env.positions <= 115),axis=1)
                outside = any(o.dynamic and bool(np.any(o.position < 45) or np.any(o.position > 115)) for o in env.obstacles)
                outside_dynamic |= outside
                events = {}
                for name in counters:
                    value = info.get(name,getattr(env,name,0))
                    events[name] = len(value) if isinstance(value,(list,tuple)) else int(value)
                row = {"seed":seed,"step":index+1,"positions":env.positions.tolist(),"velocities":env.velocities.tolist(),
                    "requested_action":action.tolist(),"stage_actions":info.get("action_stage_actions",{}),
                    "entered_central_ever":entered.tolist(),"dynamic_outside_spawn_region":outside,
                    "events":events,"directional":info["cross8u_directional"],
                    "first_collision_step":info.get("first_collision_step"),"first_collision_type":info.get("first_collision_type"),
                    "reason":str(info.get("reason","unknown")),"terminated":bool(terminated),"truncated":bool(truncated)}
                trace.write(json.dumps(diagnostic_json(row),allow_nan=False)+"\n")
                if terminated or truncated:
                    break
            results.append({"seed":seed,"steps":index+1,"agents_entered_central":int(entered.sum()),
                "entered_central_mask":entered.tolist(),"dynamic_left_initial_spawn_region":outside_dynamic,
                "final_events":events,"first_collision_step":info.get("first_collision_step"),
                "first_collision_type":info.get("first_collision_type"),"terminal_reason":str(info.get("reason","unknown")),
                "terminated":bool(terminated),"truncated":bool(truncated),"finite_state_checks_passed":True})
    return {"policy":"fixed goal-acceleration diagnostic, not trained or tuned", "seeds":results,
        "max_steps_per_seed":200,"total_steps":sum(r["steps"] for r in results),
        "central_interaction_observed":any(r["agents_entered_central"] for r in results),
        "scope":"Collisions or nonarrival may occur; this checks execution/numeric stability, not safety efficacy."}


def diagnostic_json(value):
    """Keep undefined legacy diagnostics explicit; essential numerics are checked separately."""
    value = to_jsonable(value)
    if isinstance(value,float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("+Infinity" if value > 0 else "-Infinity")
    if isinstance(value,dict):
        return {k:diagnostic_json(v) for k,v in value.items()}
    if isinstance(value,list):
        return [diagnostic_json(v) for v in value]
    return value


def tensor_tree_equal(left,right):
    import torch
    if isinstance(left,torch.Tensor):
        return isinstance(right,torch.Tensor) and torch.equal(left.cpu(),right.cpu())
    if isinstance(left,dict):
        return isinstance(right,dict) and left.keys() == right.keys() and all(tensor_tree_equal(v,right[k]) for k,v in left.items())
    if isinstance(left,(list,tuple)):
        return isinstance(right,type(left)) and len(left) == len(right) and all(tensor_tree_equal(a,b) for a,b in zip(left,right))
    return left == right


@contextmanager
def gradient_audit(trainer, evidence):
    """Observe actual post-clipping gradients without changing update rules."""
    import torch
    originals = []
    optimizers = [("actor",o) for o in trainer.actor_optimizer.optimizers]+[("critic",trainer.critic_optimizer)]
    try:
        for role, optimizer in optimizers:
            original = optimizer.step
            originals.append((optimizer,original))
            def checked(*args,_role=role,_optimizer=optimizer,_step=original,**kwargs):
                parameters = [p for group in _optimizer.param_groups for p in group["params"]]
                gradients = [p.grad for p in parameters if p.grad is not None]
                if not gradients or any(not bool(torch.isfinite(g).all()) for g in gradients):
                    raise FloatingPointError("Missing/nonfinite "+_role+" optimizer gradients")
                norm = math.sqrt(sum(float(g.detach().double().square().sum()) for g in gradients))
                result = _step(*args,**kwargs)
                if any(not bool(torch.isfinite(p).all()) for p in parameters):
                    raise FloatingPointError("Nonfinite "+_role+" parameters after step")
                evidence.append({"role":_role,"post_clip_gradient_l2":norm,"gradient_tensors":len(gradients),"parameters_finite":True})
                return result
            optimizer.step = checked
        yield
    finally:
        for optimizer,original in originals:
            optimizer.step = original


def parameter_delta(module, before):
    return math.sqrt(sum(float((p.detach().cpu().double()-before[name].double()).square().sum())
        for name,p in module.named_parameters()))


def strict_smoke_reload(trainer, checkpoint, config, output):
    import torch
    payload = torch.load(checkpoint,map_location="cpu",weights_only=False)
    for key,value in {"algorithm":"mappo","update":1,"num_agents":8,"obs_dim":80,
            "global_state_dim":640,"action_dim":2,"policy_rows":8}.items():
        if payload.get(key) != value:
            raise RuntimeError("Smoke checkpoint metadata differs: "+key)
    if payload.get("task_config_fingerprint") != trainer.task_config_fingerprint():
        raise RuntimeError("Smoke checkpoint task fingerprint differs")
    for name,actual in (("actor",trainer.actor.state_dict()),("critic",trainer.critic.state_dict()),
            ("actor_optimizer",trainer.actor_optimizer.state_dict()),("critic_optimizer",trainer.critic_optimizer.state_dict())):
        if not tensor_tree_equal(actual,payload[name]):
            raise RuntimeError("Saved checkpoint differs from trained state: "+name)
    restored_config = deepcopy(config)
    restored_config["output"] = {"output_dir":str(output/"reload"),"checkpoint_dir":str(output/"reload/checkpoints"),"reject_nonempty":True}
    restored = create_cross_trainer(restored_config,ROOT)
    restored.load_checkpoint(checkpoint)  # Strict defaults: no task or research-line override.
    for name,actual in (("actor",restored.actor.state_dict()),("critic",restored.critic.state_dict()),
            ("actor_optimizer",restored.actor_optimizer.state_dict()),("critic_optimizer",restored.critic_optimizer.state_dict())):
        if not tensor_tree_equal(actual,payload[name]):
            raise RuntimeError("Strict reloaded checkpoint state differs: "+name)
    if restored.current_update != 1 or restored.task_config_fingerprint() != trainer.task_config_fingerprint():
        raise RuntimeError("Reloaded update/fingerprint differs")
    observation,_ = restored.env.reset(seed=240000)
    with torch.no_grad():
        action = restored.actor.forward_step(torch.tensor(restored.normalize_local_observations(observation)),
            restored.actor.initial_hidden(8),deterministic=True).action
        value = restored.critic(torch.tensor(restored.normalize_global_state(restored.env.get_global_state())).unsqueeze(0))
    if action.shape != (8,2) or not torch.isfinite(action).all() or not torch.isfinite(value).all():
        raise FloatingPointError("Reloaded forward pass invalid")
    return {"strict_reload":True,"actor_equal":True,"critic_equal":True,"actor_optimizer_equal":True,
        "critic_optimizer_equal":True,"update":1,"task_config_fingerprint":restored.task_config_fingerprint(),
        "checkpoint_sha256":sha(checkpoint),"forward_action_shape":[8,2],"forward_finite":True}


def smoke_train_once(config, output):
    """One real update only. Tests replace the trainer; no automatic retry."""
    import torch
    trainer = create_cross_trainer(config,ROOT)
    if trainer.current_update != 0:
        raise RuntimeError("Smoke must start from a fresh model")
    before = {role:{name:p.detach().cpu().clone() for name,p in module.named_parameters()}
        for role,module in (("actor",trainer.actor),("critic",trainer.critic))}
    gradients, evaluations = [], []
    original_evaluate = trainer.evaluate
    def bounded_evaluate(*args,**kwargs):
        if args or kwargs.get("episodes") != 1 or kwargs.get("seed_base") != 240000 or len(evaluations):
            raise RuntimeError("Smoke evaluation exceeded fixed one-development-episode budget")
        evaluations.append({"episodes":1,"seed":240000,"max_steps":500})
        return original_evaluate(**kwargs)
    trainer.evaluate = bounded_evaluate
    try:
        with gradient_audit(trainer,gradients):
            summary = trainer.train()
    finally:
        trainer.evaluate = original_evaluate
    if trainer.current_update != 1 or summary.get("updates_completed") != 1 or summary.get("environment_steps") != 128 or len(evaluations) != 1:
        raise RuntimeError("Smoke update/rollout/evaluation count differs")
    for field in ("policy_loss","value_loss","actor_grad_norm","critic_grad_norm"):
        if not isinstance(summary.get(field),(float,int)) or not math.isfinite(summary[field]):
            raise FloatingPointError("Nonfinite smoke metric: "+field)
    changes = {role:parameter_delta(module,before[role]) for role,module in (("actor",trainer.actor),("critic",trainer.critic))}
    if any(value <= 0 or not math.isfinite(value) for value in changes.values()) or {g["role"] for g in gradients} != {"actor","critic"}:
        raise RuntimeError("Both Actor and Critic must have effective finite updates")
    episodes = [json.loads(line) for line in (trainer.output_dir/"evaluation_episodes.jsonl").read_text().splitlines() if line.strip()]
    if len(episodes) != 1 or episodes[0]["seed"] != 240000 or not 1 <= episodes[0]["episode_length"] <= 500:
        raise RuntimeError("Saved smoke evaluation exceeded development budget")
    checkpoint = trainer.checkpoint_dir/"latest.pt"
    reload = strict_smoke_reload(trainer,checkpoint,config,output)
    return {"summary":summary,"parameter_delta_l2":changes,"gradient_steps":gradients,
        "evaluations":evaluations,"evaluation_episode":episodes[0],"reload":reload,
        "checkpoint":str(checkpoint),"training_updates":1,"rollout_steps":128}


def run_smoke(config, output):
    import torch
    limits = smoke_config_contract(config)
    output = resolve(output)
    output.relative_to((OUTPUT_ROOT/"smoke").resolve())
    if output.exists():
        raise FileExistsError("Use a new isolated smoke directory; no automatic overwrite or resume")
    manifest = ROOT/"results/verification/cross8u_160/p2_implementation_v1/source_integrity_before.json"
    protected = json.loads(manifest.read_text())["protected_file_sha256"]
    def verify_history():
        changed = [p for p,digest in protected.items() if not (ROOT/p).is_file() or sha(ROOT/p) != digest]
        if changed:
            raise RuntimeError("Protected historical/frozen8U files changed: "+str(changed))
    verify_history()
    runtime_config = deepcopy(config)
    runtime_config["output"] = {"output_dir":str(output/"train"),"checkpoint_dir":str(output/"train/checkpoints"),"reject_nonempty":True}
    output.mkdir(parents=True)
    (output/"smoke_protocol.json").write_text(json.dumps({"limits":limits,"config":runtime_config,
        "protected_file_sha256":protected,"source_sha256":{name:sha(ROOT/name) for name in
            ("scripts/run_cross8u_160.py","src/envs/cross8u_160_env.py")}},indent=2),encoding="utf-8")
    try:
        torch.set_num_threads(1)
        print("P2=RUNNING_CENTRAL_PROBES; SEEDS=240000,240001; MAX_STEPS_PER_SEED=200",flush=True)
        probes = central_probes(runtime_config,output)
        print("P2=RUNNING_ONE_UPDATE; ROLLOUT_STEPS=128; EVAL_EPISODES=1; DEVICE=cpu",flush=True)
        training = smoke_train_once(runtime_config,output)
        verify_history()
        report = {"schema_version":"Cross8U160P2SmokeV1","status":"P2_SMOKE_PASS","limits":limits,
            "central_probes":probes,"training":training,"protected_files_unchanged":True,
            "protected_file_count":len(protected),"formal_layouts_executed":0,"historical_checkpoint_loaded":False,
            "p3_authorized":False,"config_sha256":sha(CONFIG_DIR/"smoke_seed23.yaml"),
            "scope":"Training/checkpoint/execution compatibility only; no safety efficacy or success-rate claim."}
        report["diagnostic_nonfinite_encoding"] = "Undefined legacy telemetry is tagged NaN/+Infinity/-Infinity; states, losses, gradients and parameters must be finite."
        (output/"p2_smoke_report.json").write_text(json.dumps(diagnostic_json(report),indent=2,allow_nan=False),encoding="utf-8")
        print(f"P2=P2_SMOKE_PASS; UPDATES=1; RELOAD=PASS; P3=NOT_STARTED; DETAIL_LOG={output/'p2_smoke_report.json'}",flush=True)
        return report
    except Exception as error:
        failure = {"status":"P2_SMOKE_FAILED","error":str(error),"traceback":traceback.format_exc(),
            "automatic_retry":False,"p3_authorized":False}
        (output/"p2_failure.json").write_text(json.dumps(failure,indent=2),encoding="utf-8")
        print(f"P2=P2_SMOKE_FAILED; RETRY=BLOCKED; DETAIL_LOG={output/'p2_failure.json'}",flush=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/verification/cross8u_160/seed23.yaml")
    parser.add_argument("--mode", choices=("preflight","validate","smoke","train","evaluate"), default="preflight")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.mode in ("train","evaluate"):
        raise RuntimeError("This P0/P1 revision does not open P2 training smoke or P3 training/evaluation")
    config = load_config(args.config)
    if args.mode == "smoke":
        limits = smoke_config_contract(config)
        if args.preflight_only:
            print("P2_PREFLIGHT=PASS; UPDATES=1; ROLLOUT=128; EPOCHS=1; EVAL_EPISODES=1; CENTRAL_SEEDS=2; MODELS=NOT_EXECUTED; OUTPUT_CREATED=NO")
            return {"status":"READY_P2","limits":limits,"training_updates":0}
        return run_smoke(config,args.output_dir or "results/verification/cross8u_160/smoke/p2_seed23_v1")
    if args.preflight_only:
        raise ValueError("--preflight-only is reserved for --mode smoke")
    dev, formal = read_panels()
    env = Cross8U160Env(environment_config(config))
    if args.mode == "preflight":
        print(f"PREFLIGHT=PASS; MAP=160x160; UAV=8; OBSTACLES=4S+6D; OBS=80; GLOBAL=640; DEV={len(dev)}; FORMAL_SEALED={len(formal)}; TRAINING=NOT_RUN; OUTPUT_CREATED=NO")
        return {"status":"READY_P0_P1", "training_updates":0}
    output = resolve(args.output_dir or "results/verification/cross8u_160/preflight/p1_v1")
    output.relative_to(OUTPUT_ROOT.resolve())
    if output.exists():
        raise FileExistsError("Use a new independent validation output directory")
    result = validate_development(config)
    result["config_sha256"] = sha(resolve(args.config))
    result["source_sha256"] = {str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__).resolve(),ROOT/"src/envs/cross8u_160_env.py")}
    output.mkdir(parents=True)
    (output/"p1_environment_report.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(f"P1={result['status']}; DEV_LAYOUTS=40; SHORT_STEPS={result['total_short_steps']}; P2=NOT_RUN; DETAIL_LOG={output/'p1_environment_report.json'}")
    return result


if __name__ == "__main__":
    main()
