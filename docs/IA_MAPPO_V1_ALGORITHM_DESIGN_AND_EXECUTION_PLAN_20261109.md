# IA-MAPPO V1 algorithm design and execution plan

## Research boundary

`ia_mappo_v1` studies whether a low-level actor can reduce avoidable staged
safety intervention under CTDE without reducing task performance. Frozen
Oracle subgoals and the shared staged safety executor are experimental
conditions, not contributions. FACS/FACS-STAR high-level planning and
Projection-aware V1/V2 losses are historical negative results and may not be
used for initialization, tuning, or training.

The first environment is 3 UAV, 6 static obstacles, and 1 dynamic obstacle.
The 8S+2D task remains out of scope until R5 passes.

## Implemented contracts

- Clean Standard config: `configs/active/ia_mappo_v1_3u_6s1d_standard.yaml`.
- Counterfactual config: `configs/experiments/ia_mappo_v1_3u_6s1d_counterfactual.yaml`.
- Scalar direct-baseline config:
  `configs/experiments/ia_mappo_v1_3u_6s1d_scalar_material.yaml`.
- Safety telemetry: `CausalSafetySnapshotV1` and `StageInterventionTraceV1`.
- Fixed action chain: actor request → Initial Joint Guard → obstacle TTC →
  Final Joint Guard → executed.
- Modification thresholds: Any action L2 > 1e-7; Material Δv ≥ 0.05 m/s;
  Large Δv ≥ 0.25 m/s; direction reversal cosine < 0.
- Recovery, SAFE_BRAKE, and terminal transitions remain observable telemetry
  but are ineligible as actor-risk supervision.
- Shared bounded recovery is selected by `safety_execution.recovery_enabled`;
  it has no high-level selector dependency.
- Shadow panel: raw, HOLD, all-agent braking, four goal-direction scales, and
  24 fixed-seed Sobol joint actions.
- Shadow validity: each unreached UAV loses at most 0.10 m/s goal-direction
  velocity relative to raw execution and may not become negative.
- Stage cost: `Material × (1 + clip(Δv / 0.25, 0, 1))`; total is the equal
  mean across the three safety stages.
- Critic: twin independently initialized 2×256 SiLU+LayerNorm encoders with
  three material-classification and three severity heads.
- Critic training loss: frequency-weighted BCE plus Material-only Huber.
- Actor penalty: frozen twin critic, conservative maximum risk, and λ in
  `{0.003, 0.01, 0.03}`. PPO likelihood remains based on sampled raw action
  and stored old log-probability.
- The scalar Material baseline has its own twin scalar-critic artifact schema;
  it cannot accidentally load the stage-wise artifact.
- `evaluate_shadow_action_v1` executes the real Initial Guard → TTC → Final
  Guard stack and verifies exact simulator/RNG/recovery-state invariance.
- Layout-grouped AUROC/AUPRC/ECE/recall gates and the formal two-level paired
  bootstrap are executable utilities, not spreadsheet-only conventions.

## Data schemas

Every training example must carry a `layout_id`; no random step split is
allowed. Dataset writers must store:

- train/validation/test layout lists and their SHA256;
- snapshot schema and trace schema versions;
- per-head positive/negative counts before fitting;
- whether Recovery, SAFE_BRAKE, or terminal exclusion removed the row;
- raw and best valid shadow cost, best candidate index, and immutable snapshot
  digest before/after shadow evaluation;
- feature contract: local/global state, bounded joint mean action, causal
  safety context, and current safety margin.

Temperature calibration is fitted only on validation layouts. Test layouts are
opened once after model/calibration freezing.

## Stage gates

### R0 — archived history (complete)

- FACS code/config/runner/test/document entry points moved to
  `archive/paused_facs_20261109/` with a hash manifest and dry-run restore.
- The manifest currently contains 708 unique, hash-verified moved files.
- Historical results remain in place and are marked `HISTORICAL_ONLY.md`.
- A separate ledger records 578 historical checkpoint/gate artifacts
  (888,631,925 bytes) without moving them.
- The 320-episode panel was analyzed without retraining. The update-200 focus
  contains 16 timeout→success rescues and 10 success→failure harms.
- The old P4 failure remains final; it is not re-gated.

### R1 — Clean Standard and telemetry (implementation complete; training pending)

1. Run focused tests and a one-update CPU smoke. This is complete: the IA
   suite passes 21 tests, the retained capability suite passes 41 tests, and
   the complete active test suite passes 277 tests with 1 skipped. The
   train/evaluation smoke both complete.
2. Freeze the clean config, observation schema, Oracle config, and safety stack
   hashes in an R1 manifest. This is complete in
   `results/ia_mappo_v1/r1_frozen_contract.json`; the manifest covers both
   preregistered seed configs and all shared safety implementation files.
3. Fresh-init seed 23 and 41 to u200. Never pass `--resume` and never copy an
   old normalizer.
4. Evaluate the same 40 layouts at u100/u200 and confirm 15 stage records per
   step for the 3-UAV task.

Do not start the u200 runs until the smoke preserves action, reward, termination
reason, and checkpoint values when telemetry serialization is disabled.

The launch wrapper `scripts/run_ia_mappo_r1_standard.ps1` fails closed on a
non-empty output or an unsupported CUDA architecture. GPU utilization is
recorded for provenance but is not a launch gate. The verified VS Code
environment `D:\Anaconda\envs\env\fcrm-mapf` uses PyTorch 2.11.0+cu130 and
includes `sm_120`; its R1 preflight returns `training_ready=true`. No u200
process was started by the implementation work. The first user-launched u200
runs under `standard/` were later invalidated because the trainer omitted the
top-level `safety_execution` contract when constructing the environment. The
runtime contract is now fail-closed and corrected fresh-init runs use
`standard_runtimefix_v1/`.

### R2 — 40-layout learnability pilot

Use grouped five-fold cross-validation. Compare local+single-action,
global+joint-action, global+safety-context+joint-action, and a simple
safety-margin/TTC heuristic. Each trained head requires at least 100 positive
and 100 negative training events plus 20 positive and 20 negative held-out
events.

Pass all of: AUROC bootstrap 95% lower bound > 0.65; AUPRC/prevalence lift
lower bound > 1.0 and point estimate ≥ 1.5; ECE ≤ 0.10; Material recall ≥ 0.80
at precision ≥ 0.50. If only global passes, implement Observation V3 teammate
mean/max pooling, fresh-init Standard, and repeat R2. If neither passes, stop.

### R3 — counterfactual critic

Freeze the selected Standard checkpoint. Collect 200 independent layouts with
immutable shadows and layout-isolated train/validation/test partitions. Omit a
stage head when event support is insufficient. Certify discrimination,
calibration, cross-layout stability, and snapshot invariance before allowing
actor gradients.

### R4 — two-seed pilot

Fresh-init Standard, scalar Material-critic baseline, and stage-wise
counterfactual IA-MAPPO for seeds 23/41. Select the largest λ whose risk
gradient is at most 5% of the PPO policy gradient on a separate calibration
batch, then freeze λ. Check u100 and u200 on the identical 40 layouts.

Advance only when pooled Material modification falls by at least 5 pp,
Initial Guard decreases for both seeds, CFMS/Progress do not fall, no failure
guardrail increases, pair/boundary collisions are zero, and both seeds agree at
the same update. Otherwise stop; do not add updates or retune λ post hoc.

### R5 — formal experiment

Run 3 algorithms × 5 fresh training seeds × 200 shared evaluation layouts.
Use a two-level paired bootstrap stratified by train seed and layout. Require:
Material modification point estimate ≤ -5 pp with CI upper bound < 0;
ΔCFMS CI lower bound ≥ -2.5 pp; ΔProgress lower bound ≥ -0.02;
ΔCollision/ΔTimeout upper bounds ≤ +2.5 pp; zero pair/boundary collision.

Claim overall superiority only if task performance also improves. Otherwise,
the maximum supported claim is improved execution consistency.

## Required commands before R1 training

```powershell
python -m pytest -q --basetemp .pytest_tmp_ia tests/ia_mappo_v1
python scripts/train_ppo.py --config configs/experiments/ia_mappo_v1_smoke_cpu.yaml --output-dir results/ia_mappo_v1/smoke_cpu
```

The smoke output is disposable and is not part of the R1 seed evidence.
