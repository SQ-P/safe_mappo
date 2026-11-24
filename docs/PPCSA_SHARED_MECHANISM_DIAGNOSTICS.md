# Shared Adapter mechanism diagnostics: phase 1

This phase implements diagnostic tools only. The existing V4 parent, two frozen
Critics, Adapter architecture, Gap loss, Guard/TTC/Recovery and historical Gate
artifacts remain unchanged. No Actor training, new fit, rollout collection or
independent efficacy validation is authorized. A real diagnostic is user-run.

## Question and evidence chain

The prior parameter step 0.01 is a parameter-vector norm, whereas action probe
epsilon is a joint pre-tanh action norm. Their magnitudes cannot be compared
directly. This diagnostic distinguishes:

1. Each state's action derivative and zero/active reference Gap gate.
2. The actual shared Adapter Jacobian and action-to-parameter chain rule.
3. Per-layer active Jacobian columns and gradient norms at zero initialization.
4. Conflicting state gradients, layout-averaged gradients and cancellation.
5. The Calibration-aggregated parameter direction and its induced action change
   in every analyzed state.
6. First-order Jacobian prediction versus the finite virtual update.
7. Requested action change versus executed action change and Shadow progress,
   including absorption, risk regression and certification loss.

These are diagnostic associations. Neither a cosine nor a weak finite update
proves a particular causal failure mechanism or justifies changing architecture.

## Data and direction construction

Only the existing Fit 121000–121023 and Calibration 121024–121031 records are
analyzed. Audit records are skipped before packet/features/targets are accessed.
Loading the existing bundled `opportunities.pt` deserializes its container, but
no Audit content is used for model evaluation, direction construction, scaling
or selection. No formal/Final panel is opened.

The analyzed subset requires original factual parity/support and an observable
hard-safe baseline; excluded reasons remain explicit. The current data preflight
finds 47 such states. Unknown/uncertifiable states are not reclassified as safe.

The shared direction uses only Calibration states with stored safe positive
local teachers (currently 6 states from 3 layouts). The same positive Gap gate
and layout-balanced weights are applied separately for Ordinary, Paired and the
kinematic value. References and model weights are detached/frozen. Unmasked
state derivatives are reported separately from gated Calibration contributions.
Fit/Calibration are exposed mechanism-development evidence, not held-out proof.

## Matched kinematic control

The new kinematic value differentiates the checkpoint's unprojected one-step
position/velocity integration and final-goal potential. It uses the original
component speed clipping, dt, acceleration, start-distance normalization and
bottleneck weight. It does not differentiate through Guard/TTC or read true
future obstacle motion. Observable hard safety and risk are checked by the
unchanged NORMAL Shadow for every candidate independently.

Norm, maximum-distance, velocity-clipping and potential-clipping operations are
piecewise smooth. Tied bottleneck and speed-clipped components are reported;
derivatives at kinks use PyTorch subgradients. Finite-difference tests cover smooth
points and verify zero derivatives in fully speed-clipped components. Forward
agreement is checked against the existing NumPy heuristic with a numerical
tolerance, not an efficacy threshold.

All five action directions (Ordinary, Paired, kinematic, random, goal) use each
predeclared epsilon 0.01, 0.05 and 0.10. Nonzero directions receive the same joint
pre-tanh norm; zero derivatives remain zero and are explicitly flagged. All
scales remain in the report; none is selected by observed gains.

## Shared-parameter budget matching

For each value source, report both the old normalized parameter step 0.01 and
three virtual changes matched to Calibration joint-action RMS targets
0.01/0.05/0.10. The RMS is layout-weighted. Exactly one scalar per direction and
target is solved using Calibration features alone, never Shadow outcomes.
The same virtual parameter copy is then applied to every Fit/Calibration state.
There is no per-state rescaling.

The solver relies on the actual zero-output initialization: only the up layer
has a nonzero first-order derivative, and the residual magnitude is monotone
along that direction. It rejects a nonzero initialization or non-up tangent.
The bracket cap 1024 and 40 bisections are numerical limits; an unreachable
target is reported rather than relaxed. A zero gated direction stays zero.
The existing Adapter output bound max absolute residual <= 0.15 is preserved.

Calibration RMS matching does not guarantee equal action norms in individual
states. Each actual pre-tanh norm, requested/executed action change, predicted
score change and actual Shadow gain is saved. Parameter comparisons must retain
that qualification. Every virtual copy is discarded; no Actor weights are saved.

## Run after implementation review

Use the project's existing Python environment. The destination must be new.

```powershell
Set-Location E:\Code\diagnostic\mappo_safe
python scripts\diagnose_ppcsa_shared_update.py --gate1-dir results\ppcsa_seed23\gate1 --critic-dir results\ppcsa_seed23\gate2_value --output-dir results\ppcsa_seed23\shared_mechanism_v1 --device cpu
```

For a read-only readiness check, append `--preflight-only`. This validates hashes
and data support without model/environment execution or creating an output.
The present 47-state input permits at most 1,316 Shadow calls; the real run is
left to the user. The output is `shared_mechanism_report.json`; stdout provides
one concise status line and its path.

The report records input/implementation hashes and verifies unchanged source
artifacts, frozen Actor parameters and original Adapter parameters. Original
reports/checkpoints are preserved. Tests use synthetic models/environments and
isolated scratch, with logs recorded before cleanup.

## Decision after the report

Inspect budget mismatch, gating, state/layout conflicts, first-order error and
execution absorption before selecting a single targeted mechanism revision.
If matched kinematic directions explain the useful signal, the learned-Critic
contribution remains unsupported. The diagnostic does not automatically choose
a model, step, architecture or revision and does not authorize Gate 3.

Any revised-mechanism validation requires a separately reviewed, preregistered
independent development protocol. Current Audit exposure cannot be undone;
formal efficacy validation and sealed panels remain outside this phase.
