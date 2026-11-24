# PPCSA: reachable local improvement after a frozen safety executor

## Current scope

This implements Gate 0-2 diagnostic tools, not a trained PPCSA Actor. No Actor
training or independent formal validation is authorized. CSA and CUP-JR remain
separate protocols and their artifacts/results are not modified. Their negative
results do not establish either success or failure of this new hypothesis.

Parent: `results/paper/standard_mappo_3u_8s2d_legacy_execution_v1/seed_23/checkpoints/update_0800.pt`.
Pinned SHA256: `087448d2e036f51e450ae927f9c841b156e0b87ed28ec16ee98de1427bfecb02`.
The loader requires seed 23, update 800, 3 UAVs, 52 observation features, 6 obstacle
slots, 8 static + 2 dynamic obstacles, trained V4 Recovery and causal TTC progress
projection v2. Reward, Oracle, observation scaling, weights and all Guard/Recovery
settings come from the checkpoint. No V5.4 or Goal Corridor substitution is allowed.
The task fingerprint and implementation/data hashes must match at every gate.

The exposed development panel is registered in
`configs/evaluation/ppcsa_dev_121000_121039.json`:

| Role | Layouts | Purpose |
| --- | --- | --- |
| Fit | 121000-121023 | Fit-only value models and normalizers |
| Calibration | 121024-121031 | Prediction diagnostics, virtual parameter direction |
| Held-out audit | 121032-121039 | Action and shared-parameter Shadow direction checks |

All candidates/time steps from a layout remain in that role. The 40 layouts are
development data, not independent formal evidence. CSA formal 120000-120199 and
CUP-JR Final 125000-125199 remain sealed. No script accepts arbitrary layout lists.

## Gate 0: accurate, immutable NORMAL one-step control

The old `evaluate_shadow_action_v2()` is preserved. Its generic V5.4 envelope
does not reproduce all of V4's progress-aware TTC candidate selection and must
not be relabeled as PPCSA evidence.

`evaluate_ppcsa_normal_shadow()` copies the environment and runs the actual V4
control prefix, including its TTCv2 selection. A default-absent diagnostic hook
captures stage outputs before UAV motion/collision/Oracle advancement. Obstacle
step methods on that clone are disabled; actual future obstacle positions are
never supplied to the prediction. The source environment is never restored or
mutated. Full fingerprint, track history/cache, source RNG and input GRU state
are compared before/after the shadow.

The factual path continues to advance obstacles and execute the full environment
step. Gate 0 compares all three safety-stage actions, final executed action and
the comparable next UAV positions with the shadow. Physical predictions follow
the same float32 component clipping/integration as `UAVState.apply_acceleration`.
Parity tolerance is 1e-6 absolute, zero relative tolerance; this is numerical
integration tolerance, not safety or research improvement slack. Collision-contact
interpolation makes next positions noncomparable and is explicitly recorded.
Any action/stage/comparable-position mismatch fails Gate 0 and blocks labels.

Supported input is NORMAL, nonterminal, no active Recovery target/incident or
Safe Brake. A branch that activates an unsupported overlay is not a teacher.
Initial/Final Guard requests (`recovery_required`, `pair_infeasible`, or nonzero
`pair_invariant_violations`) also invalidate NORMAL support and teacher/value-label
eligibility, with separate reasons even when the projected geometry is safe.
Factual baseline post-step Recovery, termination or collisions are excluded from
Gate 1 and counted. Prediction covers the current control prefix and one-step
kinematics; it does not simulate future Recovery transitions or long-term return.

Default sampling is four predeclared steps per layout (stride 20, max 80 real
steps per layout). Counts include unsupported states and capped episodes. There
is no replacement sampling to hide difficult states. Failure-proximity tags are
post-hoc diagnostics only, not Critic/Actor inputs. Sampling limits are reported
and must not be changed after inspecting results to manufacture coverage.

## Gate 1: evidence before fitting

One-step value is the checkpoint environment's final-goal potential difference:

```
g1(x,a) = env._team_goal_potential(next_goal_distances_after_S1(x,a))
          - env._team_goal_potential(current_goal_distances)
```

The existing normalized mean/bottleneck potential is reused without adding a
new tunable team weight. Per-agent distance gains and executed next velocities
are saved. This is a one-step task-progress proxy, not a task-completion value.

Physical and risk evidence are independent from intervention magnitude and old
`reference_valid`. All original stage checks must pass. Walls and UAV pairs must
pass swept geometry, and observed obstacles must pass a conservative whole-step
physical-speed reachable-ball check. The public component-speed bound is used;
no simulator true future velocity enters this check. Unknown track motion and
full observation slots (incomplete visibility cannot be certified) prevent
positive-teacher certification and are explicitly counted. This conservative
support rule may reject many states; that is evidence to report, not a reason
to silently weaken it.

Risk proxies are per observed track: nonnegative predicted penetration over the
frozen TTC horizon under observable constant velocity. A positive teacher must
have no larger risk proxy on any track and higher g1 than the supported baseline.
Action modification probabilities/deltas are diagnostics, not physical risk.
Default deadzone is zero to describe mathematical positive opportunity only;
it is not a meaningful-effect threshold. Any nonzero development deadzone must
be selected before running and is saved in provenance.

The candidate bank contains the original six deterministic references, single-
UAV pre-tanh coordinate probes and two joint goal/random probes. Every local
probe is clipped inside `tanh(mu0 + delta)`, max absolute delta <= 0.15. This
action envelope is not proof of realizability by one shared rank-16 Adapter.
Reference, single-agent and joint opportunity counts, projection absorption,
per-layout coverage, failure-proximity retention and overlapping rejection reasons
are reported separately.

Shared-parameter expressibility uses the actual zero-initialized rank-16 Adapter
Jacobian on frozen GRU features. A small, predeclared subset is used (up to 16
Fit, 8 calibration, 8 audit layouts, one target per layout). Coefficients are fit
only on Fit Jacobian rows and evaluated on held-out layout targets. This is an
unconstrained first-order upper-bound diagnostic, not a finite shared-parameter
improvement certificate. Target-gradient conflicts are reported.

Gate 1 always reports REVIEW_REQUIRED. With no safe local opportunity, Critic
fitting is blocked even if a review file is supplied. Meaningful opportunity
coverage/effect thresholds and cost adequacy require research review; no automatic
research-success threshold is invented.

## Gate 2: prediction and real direction checks

The separate `ExecutedValueCritic` predicts g1 from normalized joint local
observations/observable control context and requested joint action. It does not
reuse intervention-probability outputs. Ordinary MSE and paired-value-difference
supervision are trained with identical seed, architecture, Fit layouts and update
budget. The fixed default is 30 full-batch Adam steps, lr 0.001; both models are
reported, not selected using held-out audit outcomes. Fit-only feature/target
normalizers and layout-balanced losses are used. Calibration diagnostics include
a constant value baseline, unprojected-kinematic heuristic, MSE and non-tied
pair-direction agreement. None certify a useful gradient.

The correctly signed objective is:

```
Lgap = sum(w * relu(stop_gradient(q_reference) - q_current - delta_g))
       / max(1, sum(w))
```

Critic parameters and reference/weights are frozen; the current action retains
its gradient. Invalid references produce zero signal. In the same active sample
set this action gradient equals direct executed-value maximization. Unit tests
make that equivalence explicit; the reference is not claimed to create a new
direction by itself.

Action probes use registered pre-tanh L2 step sizes 0.01, 0.05, 0.10, all within
the Adapter envelope. Critic, random and goal directions have equal pre-tanh
perturbation norm in each comparison. On held-out audit layouts, the actual
frozen NORMAL Shadow is rerun to classify safe improvement, worsening, flat or
absorbed execution, unknown risk, and unsupported/unsafe outcomes. All scales
are reported; no favorable scale is cherry-picked after the audit.

A virtual shared Adapter direction is computed from calibration-state Gap loss,
not held-out outcomes. A temporary copy takes one normalized parameter step of
0.01; audit layouts measure actual Shadow changes, hard-safety regressions,
bounded residuals and fixed-variance Gaussian KL. Per-UAV parameter-gradient
cosines and cancellation are saved. Original Adapter parameters and parent
checkpoint are verified unchanged. All virtual updates are discarded, including
apparently successful ones. This is not PPO or Actor training and cannot authorize
deployment.

Virtual-update reports separate hard-safety regression, risk-proxy regression,
absorbed/flat outcomes and uncertifiable cases. They retain per-layout counts and
paired net/mean progress, with certified risk-nonworse progress reported separately
from all available progress pairs. Zero progress is not a hard-safety failure.
Mixed improvements/worsening are preserved without an all-layout improvement rule
or a net-benefit threshold. Efficacy remains unproven pending research review.

Gate 2 always reports REVIEW_REQUIRED and actor_training_authorized=false.
Failure to improve the actual Shadow, unsafe shared changes, predominantly
absorbed gradients, or no advantage over simpler matched baselines are stop
signals. Exact meaningful gain/coverage/direction/safety thresholds remain research
decisions, not constants invented by this implementation. A later Gate 3 Pilot
requires an explicit new decision; it is not exposed in these scripts.

## User-side sequence

Run in the same Python environment as the registered legacy parent, from
`E:\Code\diagnostic\mappo_safe`. CPU is the diagnostic default; `--device cuda`
may be added to each command. Codex has not run these commands on real data.
Every destination must be new; existing experiments/checkpoints are preserved.

```powershell
# First establish label validity. Stop unless GATE0=PASS.
python scripts\audit_ppcsa_gate0.py --panel configs\evaluation\ppcsa_dev_121000_121039.json --output-dir results\ppcsa_seed23\gate0

# Then measure the actual opportunity distribution; no fitting in this step.
python scripts\audit_ppcsa_gate1.py --gate0-dir results\ppcsa_seed23\gate0 --output-dir results\ppcsa_seed23\gate1
```

Review `gate0/gate0_report.json` and `gate1/ppcsa_opportunity_report.json` before
proceeding. An evidence-bound `gate1/gate1_review.json` is required before fitting:

```
schema_version: PPCSAGate1ReviewV1
report_sha256: SHA256 of this exact ppcsa_opportunity_report.json
decision: approve_critic_diagnostic
rationale: a nonempty, evidence-based opportunity/support/cost decision
```

This review file is not generated automatically. A later research review can
create it only after inspecting the real Gate 1 report. Its presence does not
authorize any Actor update. The following commands are implemented but remain
blocked until that exact report is approved:

```powershell
python scripts\fit_ppcsa_executed_value.py --gate1-dir results\ppcsa_seed23\gate1 --output-dir results\ppcsa_seed23\gate2_value
python scripts\audit_ppcsa_gate2.py --gate1-dir results\ppcsa_seed23\gate1 --critic-dir results\ppcsa_seed23\gate2_value --output-dir results\ppcsa_seed23\gate2_gradient
```

Gate 0 saves `states.pt`, parent/config/implementation/panel hashes and sampled
parity details. Gate 1 saves `opportunities.pt` plus opportunity/Jacobian reports.
Value fitting saves only two independent Critic artifacts, never Actor weights.
Gate 2 saves `ppcsa_gradient_report.json` and source hashes. Exact replay packets
contain private simulator state for restoring development snapshots; that state
is never part of Critic features. Predicted labels use only observable tracks
and the captured control prefix. Real future outcomes are post-hoc diagnostics.

## Test hygiene

Tests use synthetic models and isolated fixtures, not the real parent weights,
real fitting, dataset collection or large rollouts. Run with bytecode/cache writes
disabled and one named workspace scratch directory. Archive stdout/stderr and
exit status in the C2C execution record, then remove verified test scratch/caches.
Preserve source, actual experiment results, model checkpoints, gate reports and
the execution record. All cleanup targets must resolve within the named workspace.

## Existing-data Gate 1 effect audit

Before the fitting decision, summarize the saved opportunities without rerunning
an environment or fitting a model:

```powershell
python scripts\audit_ppcsa_gate1_effects.py --gate1-dir results\ppcsa_seed23\gate1 --output-dir results\ppcsa_seed23\gate1_effect_audit
```

The destination must be new and outside the original Gate 0/1 directories.
`effect_audit.json` and `effect_audit.md` report candidate/state/layout units,
gain distributions, baseline certification, eligible samples by frozen partition,
and numerical precision references. All original Gate 0/1 file hashes are checked
before and after reading. ULP comparisons describe numerical scale, not a formal
error bound or a meaningful-effect threshold. Audit layouts are descriptive only;
fit decisions and hyperparameters must not be selected using their outcomes.
The audit itself never creates a fitting approval or trains any model.

## Existing-result Gate 2 paired layout review

After the fixed gradient audit, review its saved rows without another Shadow
call, model forward pass, fit or parameter update:

```powershell
python scripts\review_ppcsa_gate2_layouts.py --gate2-dir results\ppcsa_seed23\gate2_gradient --gate1-dir results\ppcsa_seed23\gate1 --critic-dir results\ppcsa_seed23\gate2_value --output-dir results\ppcsa_seed23\gate2_layout_review
```

The new, separate destination contains `paired_layout_review.json` and `.md`.
Comparisons join identical layout/state/epsilon keys and retain all fixed scales.
Report certification coverage and risk discordance before interpreting paired
progress on the common certifiable, risk-nonworse subset. Equal-layout means do
not impute missing layouts as zero. These outcome-conditioned, descriptive
comparisons neither establish statistical significance nor authorize selecting
an epsilon, refitting, or entering Gate 3. The missing kinematic-gradient control
and the absence of a causal diagnosis of shared-parameter limitations remain
explicit limitations. Source artifacts and manifests are preserved.

Phase 1 shared-parameter mechanism diagnostics, including the matched kinematic
gradient and Calibration-only budget mapping, are specified in
`docs/PPCSA_SHARED_MECHANISM_DIAGNOSTICS.md`. They are separate user-run
Fit/Calibration probes and do not reopen Actor training or formal validation.

The next virtual whole-update admission stage is specified in
`docs/PPCSA_SHARED_UPDATE_ACCEPTANCE.md`. It fixes the Phase1 certified state
cohort and three legacy-step candidates; any state-level safety, track-risk or
executed-progress veto rejects the entire shared update. A virtual admission
never installs parameters or grants Gate3 authorization.

Evidence-only rejection attribution and three-direction endpoint-secant LP
diagnostics are documented in `docs/PPCSA_DIRECTION_FEASIBILITY.md`. They retain
all admission states/tracks and do not reconstruct or execute a parameter update.
Proxy feasibility is distinct from actual nonzero parameters and real safety;
new Shadow validation remains a separately reviewed phase.

Following a fixed-convex-family `PROXY_INFEASIBLE` result, common-veto evidence
attribution and the future parameter-sensitivity input contract are specified in
`docs/PPCSA_COMMON_VETO_DIAGNOSTICS.md`. This read-only stage preserves all
states/tracks and original admission conditions. It implements no optimizer,
measures no parameter derivatives, and does not authorize Gate3 or new Shadow.

Phase A parameter-sensitivity measurement is specified separately in
`docs/PPCSA_PARAMETER_SENSITIVITY.md`: an explicit six-coordinate subspace,
fixed signed difference scales, repeatability and held-out-scale numerical
checks, complete state/track coverage and no automatic direction acceptance.
Codex authorization covers implementation, synthetic tests and read-only
preflight only. Real frozen Shadow measurement is user-run and does not
authorize Actor training, parameter optimization or formal validation.

Existing Phase A signed samples can be classified without new measurement using
`docs/PPCSA_SENSITIVITY_QUALITY.md`. This audit separates observed flatness,
small nonzero responses and overlapping instability flags, preserves all
constraints and reproduces original export vetoes. It changes no threshold,
measurement outcome or gradient export authorization.
