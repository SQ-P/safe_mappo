# CUP-JR implementation and experiment commands

Implementation status: code and unit checks only. No CUP-JR collection,
calibration, validation, or final experiments were run by Codex.

R0 is the existing V5.4 selector. R1 keeps its exact candidate eligibility and
uses progress-tolerance selection. R2 adds all-track robust geometry and calibrated global
horizon-dependent radii. R3 adds state-dependent residual scales and its own
layout-group calibration quantile. All four use the original candidate set,
Guard/TTC chain, incident budgets and V5.4 release checks.
R1-R3 retain V5.4's existing positive margin-gain and nonzero projected-action
admission rules. R2/R3 additionally apply the robust geometry filter. Removing
those legacy admission rules is a separate ablation, not part of this study.

The checkpoint was trained with V4 execution. The evaluation-only loader
permits exactly the V5.4 state-machine version and the goal corridor required
by V5.4. Other task/model/reward/safety changes are rejected. This is a frozen
policy executor intervention, not a V5.4-trained policy or an A4 reproduction.
R0 fixed-state equivalence is tested; rollout-level parity remains an experiment.

Development defaults: 3-step horizon, progress tie tolerance 0.05 m, target
coverage 0.95. These are implementation defaults, not validated parameters.
Freeze the settings after validation before accessing the final panel.

Panels are disjoint from the existing configs/evaluation registry:

| Split | Layout seeds | Use |
| --- | --- | --- |
| Fit | 122000–122039 | Fit nonnegative linear residual scale |
| Calibration | 123000–123039 | Freeze fixed and adaptive quantiles |
| Validation | 124000–124039 | Coverage and R0–R3 diagnostics |
| Final | 125000–125199 | Paired final comparison after freezing |

Run from PowerShell in `E:\Code\diagnostic\mappo_safe`.
Every output directory must be new. Existing artifacts are never overwritten.

## 1. Collect observable-only fit and calibration data

```powershell
python scripts\collect_recovery_calibration.py collect --panel fit --output-dir results\recovery_seed23\fit --device cuda
python scripts\collect_recovery_calibration.py collect --panel calibration --output-dir results\recovery_seed23\calibration --device cuda
```

Collection runs frozen-policy R0 episodes. It records all visible tracks at all
steps, not only recovery steps. A record stores its causal features at origin
time; later observations fill the residual only when real time advances.
Lost tracks and right-censored episode horizons remain explicitly recorded.
No simulator future position, obstacle true velocity or identity label is used
as a prediction feature. The track ID only joins observable history.

## 2. Fit and calibrate offline

```powershell
python scripts\collect_recovery_calibration.py calibrate --fit-dir results\recovery_seed23\fit --calibration-dir results\recovery_seed23\calibration --artifact results\recovery_seed23\calibration_v1.json --target-coverage 0.95
```

Each calibration layout contributes one maximum standardized residual across
its valid, uncensored prediction events and horizons. Lost tracks inside an
eligible horizon score infinity; insufficient finite groups stop calibration.
Invalid velocity rejects robust candidates online; fewer than two completed
valid residuals count as a cold start, which uses the larger
of fixed/adaptive calibrated radii. Track discontinuity clears old residuals.
Coverage is conditional on valid velocity
and uncensored horizons, not unconditional safety. Counts of exclusions/losses
must accompany any coverage report. A non-finite calibration result is G2 STOP;
do not replace it with an arbitrary finite radius to continue final tests.

### Explicit conditional development calibration after track-loss failure

The collected calibration panel contains 37/40 layouts with valid track loss.
Under the original strict maximum-per-layout rule these layouts score infinity;
the original G2 gate therefore fails. This is not a dependency or GPU error.
Existing fit/calibration observations can be reused without collecting again.

The following opt-in command defines a different, development-only target:
95% layout-group residual coverage **conditional on later observed, valid
predictions**. Track losses stay in the artifact's missingness report and in
the separate strict coverage metric. They are not claimed to be covered.

```powershell
python scripts\collect_recovery_calibration.py calibrate --fit-dir results\recovery_seed23\fit --calibration-dir results\recovery_seed23\calibration --artifact results\recovery_seed23\calibration_observed_dev_v1.json --target-coverage 0.95 --coverage-scope observed_valid
```

Use that artifact path for development coverage and validation commands only.
The original strict command remains the default. Conditional artifacts are
explicitly blocked from the original final-test entry. Advancing this revised
coverage target to a formal protocol requires a separate research decision.

## 3. Independent coverage and development comparison

```powershell
python scripts\collect_recovery_calibration.py collect --panel validation --output-dir results\recovery_seed23\coverage_data --device cuda
python scripts\collect_recovery_calibration.py coverage --artifact results\recovery_seed23\calibration_v1.json --data-dir results\recovery_seed23\coverage_data --output-dir results\recovery_seed23\coverage_report
python scripts\evaluate_recovery_study.py --panel validation --output-dir results\recovery_seed23\validation --calibration-artifact results\recovery_seed23\calibration_v1.json --device cuda
```

Inspect coverage, region sizes, rejected candidates and actual differences in
selected labels before moving on. Coverage under the R0 collection policy
does not establish coverage under R3's induced state distribution.
If prediction error is not supported as a dominant failure mechanism, or
R3 produces no useful difference versus R2, stop and reassess the hypothesis.

The evidence classifier preserves unknown causes rather than inventing them:

```powershell
python scripts\diagnose_recovery_failures.py --episodes results\recovery_seed23\validation\R3\episodes.jsonl --output results\recovery_seed23\validation\failure_evidence.json
```

F1/F2/F3/F7 causal attribution still requires matched stepwise failure traces;
the classifier reports F4/F5/F6 counters as supporting signals only.

## 4. Final test after freezing development decisions

The following example preregisters a 0.02 absolute collision-rate noninferiority
margin. Choose and justify that margin before viewing final results.

```powershell
python scripts\evaluate_recovery_study.py --panel final --output-dir results\recovery_seed23\final --calibration-artifact results\recovery_seed23\calibration_v1.json --noninferiority-delta 0.02 --device cuda
```

Outputs include per-arm episodes, summary, checkpoint/config/artifact hashes,
environment-step P95/P99 and recovery-selector (including shadow) P95/P99.
`paired_analysis.json` contains 10,000 paired bootstrap draws, success swaps,
and the explicit noninferiority decision. R3 versus R2 is the primary contrast.
The final panel remains unused until this command is launched by the user.
No result supports a publication claim until the corresponding evidence exists.

## V2 development revision after the first calibration analysis

The current-chat decision implements stages 1-3 only. R4/new candidates and
Final testing are outside this revision. Default `--protocol v1` preserves the
old entry. V2 must be selected explicitly and requires a new V2 artifact;
previous data/results/artifacts are not overwritten.

Problems addressed:

- Loss versus error: conditional residual coverage and strict loss-inclusive
  coverage are both reported. Loss counts, prediction-origin distance and
  approach direction remain separate; these signals do not prove why a track
  disappeared. Newly collected V2 data also saves loss event records.
- Scale instability: V2 normalizes the residual feature using Fit data and fits
  a nonnegative two-parameter ridge model. Penalty selection uses four grouped
  Fit-layout folds, each with its own normalizer, and layout-balanced losses.
  Calibration data never selects the penalty. No manual safety-radius cap is used.
- OOD: features exceeding the Fit maximum do not drive linear online
  extrapolation. The candidate checker uses a physical reachable ball instead.
- Unknown/lost tracks: a far nonintersecting reachable ball no longer rejects
  every candidate. A ball intersecting the swept path rejects the candidate.
  The bound is sqrt(2)*max(abs(dynamic_obstacle_velocity_range)), from the
  checkpoint's public component-speed limits; reflection preserves this bound.
  There is no read of simulator future state or actual obstacle velocity.
  Lost tracks retain their last observed center and age until reacquisition;
  there is no ungrounded expiry timer. This can remain conservative on long
  disappearances; V2 does not claim to solve perception loss or all deadlocks.
- Decisions: R3 records the fixed-R2 and adaptive-R3 choices on exactly the
  same candidate shadows and state. `selection_diagnostics.json` reports the
  executed-action disagreement rate (including candidate availability changes).
  `timing.json` reports both instrumented timings and timings excluding the
  additional R2 probe. These are R3-induced states, not separate paired rollouts.

Existing Fit/Calibration data can be reused for this offline refit because the
source frozen R0 controller and causal residual definition did not change.
No V2 experiment or real-data refitting was run by Codex.

```powershell
python scripts\collect_recovery_calibration.py calibrate --fit-dir results\recovery_seed23\fit --calibration-dir results\recovery_seed23\calibration --artifact results\recovery_seed23\calibration_observed_dev_v2.json --target-coverage 0.95 --coverage-scope observed_valid --scale-model normalized_ridge_v2
python scripts\collect_recovery_calibration.py collect --panel validation --protocol v2 --output-dir results\recovery_seed23\coverage_data_v2 --device cuda
python scripts\collect_recovery_calibration.py coverage --artifact results\recovery_seed23\calibration_observed_dev_v2.json --data-dir results\recovery_seed23\coverage_data_v2 --output-dir results\recovery_seed23\coverage_report_v2
python scripts\evaluate_recovery_study.py --protocol v2 --panel validation --output-dir results\recovery_seed23\validation_v2 --calibration-artifact results\recovery_seed23\calibration_observed_dev_v2.json --device cuda
```

Check conditional coverage, strict coverage, loss statistics, radius P50/P95/P99,
OOD calls, reachable-set rejection reasons, action disagreement, collisions,
CFMS and timeout before any next-stage research decision. The original strict
failure remains a failure. A finite conditional artifact is not Final approval;
the V2 Final entry is explicitly disabled pending a frozen reviewed protocol.

## 2026-11-21: Gate A only; V3 is not yet authorized

The current-chat audit rejects V2's efficacy claim. On the 40 already exposed
Validation layouts (124000-124039), R0/R1/R2/R3 CFMS is 32.5/25/5/5 percent
and collision rate is 15/30/47.5/47.5 percent. R3's slopes are all zero and
it disagrees with R2 on 0/235 same-state decisions. Conditional coverage is
97.5 percent; strict loss-inclusive coverage is 5 percent. These are different
claims. V2 does not establish safe, progress-preserving adaptive recovery.

The R2 candidate counters locate 362 legacy rejections, 305 additional
robust-or-stage rejections and 6 reachable-set rejections. They do not explain
which constraint caused each rejection or whether no-candidate caused a later
collision. Prior logs cannot retroactively recover unrecorded snapshots.

This revision implements diagnostics only. It does not change candidate
generation, admission, ranking, radius models, MAPPO weights, Guard, causal
TTCv2, V5.4 state transitions or safety budgets. V3 is deferred until Gate A
identifies a specific, safely repairable mechanism. No new fit, calibration,
collection or rollout was executed by Codex.

### User-run historical audit (no simulation)

```powershell
python scripts\audit_cupjr_gate_a.py --root results\recovery_seed23\validation_v2 --output results\recovery_seed23\gate_a_historical_report.json
```

The report hashes its input files and preserves unknown causes. It never
automatically approves V3, even if diagnostic traces are present. Output must
be a new file; existing artifacts are not overwritten.

### User-run diagnostic replay, then audit

Use the same Python environment as the completed V2 run. No refitting or
recollection of Fit/Calibration data is needed. This command replays the
already exposed Validation layouts for attribution, not new efficacy evidence:

```powershell
python scripts\evaluate_recovery_study.py --protocol v2 --panel validation --gate-a-diagnostics --calibration-artifact results\recovery_seed23\calibration_observed_dev_v2.json --output-dir results\recovery_seed23\gate_a_replay_v2 --device cuda
python scripts\audit_cupjr_gate_a.py --root results\recovery_seed23\gate_a_replay_v2 --output results\recovery_seed23\gate_a_detailed_report.json
```

Each arm writes `gate_a_decisions.jsonl`, including all candidate shadows,
causal planning snapshots, stage projections, legacy eligibility and its
reason, margin improvement, progress, requested/first executed commands,
fixed/adaptive uncertainty radii and constraint decomposition. R0-R3 shadow
choices share each decision's state and candidate predictions. No alternative
shadow is executed in the environment. Missing selection remains null; the
trace does not equate that with a proven brake-to-collision causal chain.

Constraint violations are nonexclusive: wall, teammate pair, visible obstacle,
lost track. Nominal and inflated margins are separate. A nominal constant-
velocity prediction is not ground-truth physical safety. Lost/unknown tracks
retain physical reachability bounds. `calibration_only_violation` excludes
violations due to reachable sets. An outward improving path can still fail
the unchanged robust screen, which is recorded without granting permission.

Initial margin at the observed time has zero future prediction error for a
visible track. `initial_first_future_ball_margin_m` separately describes
overlap with the first future uncertainty ball. `risk_margin_improving` compares
the terminal margin with that expanded initial diagnostic; the separate
`time_matched_risk_margin_improving` compares time-matched initial/terminal
uncertainty. Both are geometric evidence, not a recovery safety certificate.
Per-segment start margins reveal overlap under that segment's end-time ball.

Diagnostics do not append to the controller's radius statistics. Timings retain
instrumented totals and add entries excluding audit/probe overhead; these
subtractions are diagnostic estimates, not separate production benchmarks.

### Fixed innovation boundaries and stop rules

- Gate A: review same-state R0/R1 safety-margin choices, identify nominal
  versus inflation/reachability rejection, and inspect initial-overlap escape
  geometry. If no repairable mechanism can be attributed, do not develop V3.
- Gate B: at most one bounded V3 repair, after Gate A review, restricted to
  safety-first ranking, explicitly validated constrained recovery semantics,
  or causal observable features with demonstrated predictive information.
  Never relax physical safety constraints to manufacture CFMS gains.
- Preserve the frozen Actor, Guard/TTC, V5.4 executor, current three candidate
  directions and safety budgets. New planners/candidates or retraining change
  the innovation being tested and cannot rescue its original claim.
- Gate C: before V3 evaluation, register fresh development layouts disjoint
  from all existing Fit/Calibration/Validation/Final layouts. The exposed
  Validation panel is diagnostic/design data. Final 125000-125199 stays sealed.
- Freeze minimum meaningful CFMS/progress gain, collision noninferiority
  margin, statistical method and sample adequacy before seeing new outcomes.
  These scientific thresholds are currently unresolved; 0.02 is not silently
  accepted. Diagnostic commands neither require nor invent such thresholds.
- After the single justified repair, no meaningful adaptive action difference
  or independent benefit stops the adaptive claim. Persistent candidate loss
  without an admissible safe recovery stops this robust-screen route. Gains
  requiring relaxed safety, severe progress loss or an expanded innovation
  scope do not establish CUP-JR success. Inconclusive evidence is not proof of
  universal impossibility, and is not permission for unbounded iteration.

No immediate V3 experiment command exists: algorithm and calibration protocol
must follow Gate A evidence and review. Run the diagnostics above first.

## Bounded V3 after the completed Gate A replay

Gate A review now supports two targeted repairs: R1's progress-first order
chooses lower predicted margins in 45/47 changed decisions on R1 states; R2
rejects 311/343 legacy-eligible candidates, of which 298 show calibration-only
violations and 207 show expanded initial overlap plus terminal improvement.
These overlapping categories are design evidence, not 207 safe commands.
The no-candidate-to-collision causal chain remains unproven.

V3 is an explicit new protocol; old V1/V2 behavior and artifacts are preserved.
R0 remains V5.4. R1 changes ranking only. R2 retains the V2 fixed calibration
artifact and adds the three-mode certificate below. R3 deployment/retraining/
refitting is deferred pending Fit-only evidence. The Actor, Guard/TTC, existing
candidate directions, execution state machine and recovery budgets are unchanged.

### Exact safety ordering and separate recovery certificate

`robust_feasible` requires legacy admission and the original full calibrated
swept screen. It outranks `constrained_recovery`; both outrank `unavailable`.
R1 and full-robust R2 candidates sort lexicographically by legacy margin gain,
minimum margin, team progress, affected minimum progress, negative actor
deviation and deterministic label priority. Safety tie tolerance is exactly
zero. No epsilon is used to exchange safety for progress. The old 0.05 m
progress tie setting remains a V1/V2 parameter and is not V3 safety slack.

Constrained candidates must still satisfy original Legacy/Guard/TTC admission,
projected nonzero actions and minimum margin gain. For every observed obstacle
at current center z, and every lost obstacle at last center z of age tau,
the entire segment [p(k-1),p(k)] must avoid a ball with original safety radius
R plus v_max*(tau+k*dt). v_max is the public task speed bound, used even for
tracks that appear static. Wall, UAV-pair and nominal obstacle swept margins
must also remain nonnegative. Unknown/OOD/lost tracks cannot themselves justify
an uncertainty escape; they must pass their conservative physical balls.

For each valid observed track and each k, compare:

```
m_ref(k) = |predicted_obstacle(k) - original_UAV_position| - R - rho(k)
m_cmd(k) = |predicted_obstacle(k) - candidate_UAV_position(k)| - R - rho(k)
```

Reference and candidate use the same prediction time and calibration radius.
The stationary-position reference is geometric, not a certified brake action.
If m_ref(k)>=0, m_cmd(k) must remain >=0. If m_ref(k)<0, m_cmd(k) must be >=
m_ref(k). This is checked per track and time, never just as a global minimum.
At least one currently expanded-overlapping valid track must have strictly
positive terminal gain relative to that same-time reference. No finite radius
is reduced, and the Gate A `risk_margin_improving` flag is not an admission rule.
Nonfinite evidence and inconsistent shadow endpoints fail closed.

Constrained ranking prioritizes the worst terminal gain among initially
overlapping tracks, then physical margin, legacy gain and robust margin, before
progress. Detailed candidate certificates and rejection reasons are recorded.
Every V3 R2 persistent target command is recertified from the latest observation;
the initial certificate does not authorize later steps blindly. An unavailable
active command invokes the existing unavailable/budget handling. This adds a
safety gate and does not create a new Recovery state machine or extend budgets.

These certificates depend on the configured motion/observation model and apply
to the checked command/horizon. Passing a full robust screen is not a full
closed-loop calibrated coverage guarantee under the changed V3 state distribution.
Constrained mode explicitly has no full calibrated coverage claim.

### Current user choice: mechanism diagnostics only

The user explicitly paused formal effectiveness validation. The fresh registry
`configs/experiments/recovery_seed23/panels_v3_dev.json` reserves 126000-126039
for future Fit, 127000-127039 for future Calibration and 128000-128039 for future
development Validation, all disjoint from the old panels and one another.
Those panels have not been run; the V3 launcher currently rejects them. Final
125000-125199 remains sealed. Minimum meaningful CFMS/progress gain, safety
noninferiority margin and sample adequacy remain unresolved. No values are
invented and no efficacy/noninferiority pass is emitted.

Use the same Python environment as V2, from the project root. The following
commands have not been executed by Codex on real data or in the environment.

```powershell
# Fit-only exploratory feature validation; does not replace or train R3.
python scripts\audit_cupjr_fit_features.py --fit-data results\recovery_seed23\fit\predictions.jsonl --output results\recovery_seed23\v3_fit_feature_diagnostic.json

# V3 R0/R1/R2 mechanism replay on exposed design layouts, not fresh validation.
python scripts\evaluate_recovery_study.py --protocol v3 --panel validation --mechanism-only --arm R0 R1 R2 --calibration-artifact results\recovery_seed23\calibration_observed_dev_v2.json --output-dir results\recovery_seed23\v3_mechanism_validation --device cuda
```

The feature tool compares constant, recent-residual, velocity-history and
boundary-motion features on four disjoint Fit-layout folds for horizons 2/3.
It uses only features actually available in the stored data. World positions
are reconstructed from observations that matured at or before the new
prediction origin; future target positions/errors never enter its features.
Relative velocity change includes observer motion and is labeled accordingly.
Normalization is refit inside each training fold, and losses are balanced by
layout. OLS is used only as an exploratory predictor, with no online model
deployment, penalty search on held-out panels or forced nonzero adaptive slope.
Incomplete causal histories are counted/excluded. This command performs offline
cross-validation fits when the user runs it; Codex has run synthetic tests only.

Replay outputs include per-arm `selection_diagnostics.json`, detailed mode/
reason/certificate events in `episodes.jsonl`, provenance, timing and descriptive
R1-R0/R2-R1/R2-R0 paired comparisons in `paired_analysis.json`. R0's selection
count is recorded separately because its events retain the original format.
Instrumented step timing includes ongoing R2 certificate checks. Selector
timing alone excludes checks performed while following a persistent target.
All output destinations must be new; existing results are never overwritten.

Inspect candidate availability and each rejected certificate first. A reduced
rejection count alone is not a success claim. If the certificate cannot restore
useful commands without violating hard checks, stop this repair route. If R1
collapses to R0 under exact safety ordering, record that result rather than
tuning safety slack after seeing outcomes. Do not infer an adaptive contribution
from R2 changes. Formal validation requires a later explicit research decision
and frozen thresholds/sample/statistical protocol before fresh outcomes.
