# Independent 160 x 160 cross-navigation verification: P0/P1/P2

This is a separate scenario, not a replacement for the existing 3U experiment.
Only new files under the verification namespace are added. No old environment,
algorithm, reward, safety executor, YAML, checkpoint or historical result changes.

## Frozen scenario

Eight UAVs (radius1.5), map160x160, max component speed2.0, acceleration1.0,
dt0.5, goal radius3, horizon500, detection radius30 and ten tracked obstacle slots.
The first four start at `(12,30/60/100/130)` and have targets at
`(148,130/100/60/30)`. The other four start at `(30/60/100/130,12)` and target
`(130/100/60/30,148)`. Starts and goals are explicit 2D coordinates, with no jitter.

Four static and six dynamic circles (radius uniform2..5) are generated with
centres in `[45,115]^2`. The original non-overlap sampler enforces radius-aware
separation from starts, goal regions and other obstacles. Dynamic velocities
retain the original componentwise uniform[-1,1] distribution and full-map
reflection: the central region constrains initial spawning, not future motion.

The new subclass changes initialization and adds direction-neutral telemetry;
all reward/Guard/TTC/Recovery step logic is inherited unchanged. A conservative
2m four-connected free-cell check inflates every initial obstacle by UAV radius,
2m clearance and cell half-diagonal, and requires a route for every start/goal.
Layout resampling is bounded at100 layouts. This proves only initial geometric
passability at the stated discretization; it is not a joint multi-UAV or moving-
obstacle safety guarantee and does not prove policy success.

## Independent configuration and compatibility

`base.yaml` and `seed23.yaml` are complete expanded snapshots, without `extends`
links to the active or old experiment YAMLs. They copy the current executable
Standard MAPPO seed23 reward, model, training hyperparameters, Oracle and safety
configuration. Only scenario geometry/sensing/horizon, the development evaluation
seed base, isolated output paths and explicit verification metadata change.
`source_snapshot.json` records the derivation. Future main YAML edits do not
implicitly update these snapshots. The subclass still uses shared source logic;
future changes to that logic require another explicit compatibility review.

Development layouts:240000..240039 (40). Formal layouts:241000..241199 (200),
reserved and not executed in P0/P1. First prospective training seed23; seeds41/67
are later repetitions requiring their own snapshots. All models must be freshly
initialized for the new task. Original 3U weights cannot be assumed compatible.

Local observation remains `10 + 10*7 = 80`; global critic state is640;
shared Actor action width2 with eight policy rows (joint action8x2), GRU1x8x128.
Only the nearest teammate is observed, as before; this is a known eight-UAV
information limitation. Original x-only progress telemetry is retained for
compatibility but is unsuitable for the vertical group. Use
`info['cross8u_directional']` for both crossing groups; reward semantics are unchanged.

`run_cross8u_160.py` temporarily binds the original PPO trainer module's
environment symbol only during synchronous factory construction. It rejects
Python-thread concurrency and unexpected prior bindings, and always restores
the original symbol in `finally`. The constructed trainer retains its own
Cross8U environment afterward. No trainer/factory source files change. This is
a single-process verification adapter, not general vector-environment injection.
Use this dedicated entry only: the existing generic training/evaluation scripts
do not select the new environment class from YAML.

## Stages and commands

P0/P1 implement configuration, deterministic initialization, geometric checks,
40 development layouts with short random and goal-acceleration steps, and fresh
untrained MAPPO initialization/forward compatibility. No optimizer update,
checkpoint training/resume, full episode evaluation or formal panel is run.
The generic `train`/`evaluate` CLI modes remain blocked. The separately reviewed
P2 entry below allows exactly one bounded training-update smoke + save/restore;
P3 full training and200-layout evaluation are not authorized.

Read-only configuration/dimension preflight (no output or training):

```powershell
Set-Location E:\Code\diagnostic\mappo_safe
python -B scripts\run_cross8u_160.py --mode preflight
```

Bounded development environment validation (40 layouts, 3 steps per policy/layout,
maximum240 transitions; no training). Use a new output directory:

```powershell
python -B scripts\run_cross8u_160.py --mode validate --output-dir results\verification\cross8u_160\preflight\p1_user_v1
```

`p1_environment_report.json` contains initial coordinates, obstacle/radius/velocity
statistics, route checks, layout regeneration counts, reset determinism and short
transition results. P1 pass is environment readiness, not success-rate evidence.
Short-step collisions/termination are recorded rather than hidden as policy claims.

The implementation report and before/after SHA inventory live separately under
`results/verification/cross8u_160/implementation_v1`. Existing checkpoints,
source/tests/configuration files and historical formal summaries are protected.

## P2: one bounded training/checkpoint smoke

Task: `CROSS8U_160_4S6D_SMOKE_V1`. `smoke_seed23.yaml` is a new complete
snapshot; frozen `base.yaml`/`seed23.yaml` are unchanged. The entry compares the
entire smoke snapshot with the formal snapshot plus explicit P2 overrides before
constructing a model. Budget: CPU, Seed23, exactly1 update, rollout128, PPO1 epoch.
The existing trainer automatically evaluates on its final update: this is limited
to one development episode, seed240000, horizon500. A second evaluation is blocked.

Two fixed goal-acceleration probes run first: development seeds240000 and240001,
at most200 steps each, stopping on environment termination. Their JSONL trace
records positions, requested/stage actions, direction-neutral progress, central
region entry, dynamic obstacles leaving the initial spawn region, Guard/TTC/
Recovery counters and collision/termination reasons. This policy is not tuned;
collision, nonarrival or early termination is diagnostic evidence, not a software
failure or permission to adjust the task. State/action/reward numerics must be finite.

The real update records post-clipping gradients at every Actor/Critic optimizer
step and checks finite losses, gradients, parameters and nonzero parameter changes.
Only the new8U `latest.pt` is loaded into a fresh trainer, with strict default task
and research-line checks. Actor/Critic and optimizer states are compared exactly;
update1, task fingerprint, obs80/global640/8 policy rows and finite forward8x2
are required. No old3U weights are read. Undefined legacy diagnostic fields are
explicitly tagged `NaN`/`+Infinity`/`-Infinity` in the P2 report; essential learning
and state numerics are checked separately and cannot use these tags to pass.

Read-only P2 preflight (no model, trajectories or output created):

```powershell
python -B scripts\run_cross8u_160.py --mode smoke --config configs\verification\cross8u_160\smoke_seed23.yaml --preflight-only
```

The authorized single execution command (already executed once if its report
exists; do not repeat automatically):

```powershell
python -B scripts\run_cross8u_160.py --mode smoke --config configs\verification\cross8u_160\smoke_seed23.yaml
```

Results: `results/verification/cross8u_160/smoke/p2_seed23_v1/`;
`smoke_protocol.json` freezes the config, limits and protected hashes;
`central_trajectories.jsonl` contains raw diagnostic transitions;
`p2_smoke_report.json` includes finite/effective update and strict reload evidence.
An existing output path is refused. A failure writes `p2_failure.json` and stops
without retry, scaling or P3. P2 implementation/tests/command outputs and before/
after SHA inventories are retained separately in `p2_implementation_v1`.
Temporary synthetic test outputs are removed after their command outputs are
recorded. Real smoke checkpoints and raw traces are retained as audit evidence.

P2 pass demonstrates training/checkpoint/execution compatibility only. It proves
neither safety efficacy nor convergence or task success. The40 development and200
formal panels are not evaluated by this stage; only the two named development
probes and one development evaluation are allowed. P3 requires a separate plan.

### Executed P2 result

`smoke/p2_seed23_v1/p2_smoke_report.json`: `P2_SMOKE_PASS`. Exactly1 real
update/128 rollout steps/1 PPO epoch, one development evaluation (70 steps,
terminated by inter-agent collision), and strict new-checkpoint reload passed.
Actor/Critic parameter delta L2:0.09569654/0.05412734, with finite losses,
gradients and updated parameters. Both diagnostic seeds brought all8 agents
into the central region; they terminated on dynamic/static obstacle collisions
at steps80/64. Guard/TTC/Recovery were exercised; these collisions are retained
and show why this readiness result must not be interpreted as safety efficacy.
Dynamic obstacles left their initial central spawn region in both probes.
55 non-training/synthetic regression tests passed,1 training test was deselected.
All314 protected files (including the301 historical baseline files) stayed
unchanged; formal panel and P3 were not started. See
`p2_implementation_v1/implementation_report.md` for the command/evidence ledger.

## User-requested first round: Seed23, 400 updates (prepared, not executed)

Following the P2 acceptance, the user requested a launch command for a first
Seed23 training round of400 PPO updates. The environment modification and P2
training/checkpoint compatibility are complete. This subsequent request permits
preparing that training entry; Codex has not launched it. Prior P2 statements
about P3 not being authorized describe the P2 stage boundary. Formal200-layout
evaluation remains outside this first-round command.

New complete snapshot: `configs/verification/cross8u_160/train_seed23_u400.yaml`.
Dedicated entry: `scripts/train_cross8u_160.py`, reusing the P2-tested environment
construction without changing or opening the generic P0/P1 `train`/`evaluate`
modes. The frozen `seed23.yaml` still has its original800-update settings and
is not the new400-update launch config. Do not use the smoke command for training.

The first round starts from fresh weights, not the one-update P2 checkpoint.
It uses CUDA, Seed23,400 updates,2048 rollout steps/update and4 PPO epochs;
reward, model, Guard/TTC/Recovery and other training hyperparameters remain
as frozen. The600-update entropy anneal schedule is retained, not rescaled.
Checkpoints and10-episode development evaluations occur at updates100/200/400.
Evaluation seeds240000..240009 are inside the development panel; horizon500,
and per-episode records are saved. No formal200-layout evaluation is requested.

Use the verified project Python: the default Anaconda base Python's Torch
does not support this RTX5060. The existing project interpreter has Torch2.11
CUDA13.0 with sm_120 support and passed a tiny CUDA compatibility calculation.
Preflight constructs no trainer and does not run any environment transitions,
training updates or evaluation. It does not create the training output directory.

Optional readiness check:

```powershell
Set-Location E:\Code\diagnostic\mappo_safe
$Py = 'D:\Anaconda\envs\env\fcrm-mapf\python.exe'
& $Py -B scripts\train_cross8u_160.py --preflight-only
```

Start the first400-update experiment (the user runs this):

```powershell
Set-Location E:\Code\diagnostic\mappo_safe
$Py = 'D:\Anaconda\envs\env\fcrm-mapf\python.exe'
& $Py -B scripts\train_cross8u_160.py
```

Output: `results/verification/cross8u_160/train/seed_23_u400_v1/`.
Checkpoints: `checkpoints/latest.pt`, periodic `update_0100.pt`,
`update_0200.pt`, `update_0400.pt`, plus automatically selected best checkpoints.
Metrics/logs: `train_metrics.jsonl`, `evaluation_episodes.jsonl`, `train.log`
and `train_summary.json`. Existing nonempty outputs are refused; no implicit
resume, historical weight transfer or overwrite. Readiness-only evidence is
stored separately in `seed23_u400_readiness_v1` and is not training evidence.

P2 collision evidence remains unresolved evidence about safety performance;
environment completion and training readiness do not establish learned success.
