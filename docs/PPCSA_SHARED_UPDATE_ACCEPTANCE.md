# PPCSA Shared Update Admission V1

This is a virtual whole-update admission diagnostic driven by the existing
frozen NORMAL V4/TTCv2 Shadow. It neither installs an Adapter into the Actor nor
trains/refits a model. All parent/config/Guard/TTC/Recovery semantics and source
artifacts remain unchanged. Gate3, collection and formal panels remain blocked.

## Fixed cohort and candidates

The state registry is frozen before evaluation from Phase1's original certified
Fit/Calibration records. It must exactly match every model's Phase1 state-trace
list, including order. The current registry contains 47 states: 41 Fit and 6
Calibration. Original out-of-scope uncertifiable baselines are not claimed safe.
No Audit packet content is evaluated or used to construct directions.

Once frozen, no state can be removed or replaced. A replayed state that loses
support/certification remains in every candidate's evidence and vetoes the
whole update. Missing/duplicate coverage also vetoes admission.

Exactly three candidates are registered, without model selection or ranking:

| Candidate | Direction source | Parameter norm step |
|---|---|---:|
| Ordinary | Existing frozen Ordinary Critic | 0.01 |
| Paired | Existing frozen Paired Critic | 0.01 |
| Kinematic | Existing unprojected kinematic value control | 0.01 |

Each direction is rebuilt exactly as Phase1: six Calibration teachers, same
positive Gap gate, layout-balanced contributions, same zero-initialized shared
Adapter, normalized aggregate parameter gradient. Historical active-teacher
counts and gradient norms are checked for reproduction. The norm comparison
tolerance is float32/device metadata roundoff only, not risk/progress slack.
RMS variants are historical sensitivity evidence and are not candidates here.
There is no scale search, backtracking or per-state parameter update.

## Whole-update contract

For the same virtual parameter copy across ALL registered states:

1. Restore the original state/RNG/GRU/cache and verify its fingerprints.
2. Evaluate baseline and candidate on the same snapshot. Both must remain
   NORMAL-supported and hard-safe under the existing observable one-step checks.
3. Capture raw per-track risk values and observable track identities/context in
   the same observer/visible-track nested order used by the frozen Shadow.
   Identity, order, relative state, radius and velocity-validity must match.
   Unknown/missing/nonfinite risks or ambiguous correspondence reject admission.
4. Require every candidate track's risk <= its baseline risk (zero tolerance).
5. Require every state's actual executed progress gain >= 0 (zero tolerance).
6. Require at least one strictly positive gain and strictly positive full-cohort
   net executed gain. A zero parameter direction or all-zero gains do not pass.

Predicted Critic gain is never an acceptance criterion. High gains in other
states cannot cancel a state veto. Counts of risk, safety, replay and progress
rejections are nonexclusive; retain raw evidence and all reasons.

Any veto rejects the ENTIRE parameter candidate and returns an exact zero vector
as its effective virtual update. If all three candidates are rejected, the
overall result is `NO_ACCEPTABLE_SHARED_UPDATE` with an explicit all-zero
fallback. This is an admissible diagnostic outcome, not a software failure.
If several candidates pass, record all `would_accept=true` values without
automatically choosing a model. All virtual copies are discarded regardless.

Construction/evaluation exceptions also reject admission, but are separately
reported as `DIAGNOSTIC_ERROR`: they do not establish that correctly evaluated
directions lack an admissible update.

## Track evidence without modifying the Shadow

The wrapper reads the CURRENT planning snapshot on a clone before/after each
Shadow evaluation. It records `(observer_id, track_id)` and raw context along
with the returned risk vector, checking source and track immutability. The risk
vector correspondence follows the original Shadow's nested iteration order.
No true future obstacle motion is supplied. Empty known observed-track lists
are distinct from missing evidence; all original hard-safety support limitations
still apply. No online or unobserved-state safety guarantee is implied.

## Provenance and scalar-warning correction

The Phase1 report, manifests, data and model hashes are bound before evaluation.
Original input files, Actor, Critics and Adapter parameters are checked again
afterward. The new report preserves its fixed-state registry/hash and raw
baseline/candidate state evidence. No Actor/Adapter checkpoint is written.

The only Phase1 helper change is the zero-initialization scalar check using
`.detach().item()`. The historical report is preserved: source compatibility
requires either its exact recorded hash or an exact reverse substitution of
that single prescribed check producing the recorded hash. Any unrelated source
change fails. This does not waive any data/model/config checks.

## User-run commands

Use the original project Python environment and a fresh output directory.
Preflight reads hashes/cohort metadata only, with no model/Shadow execution and
no output directory creation:

```powershell
Set-Location E:\Code\diagnostic\mappo_safe
python -B scripts\diagnose_ppcsa_update_acceptance.py --gate1-dir results\ppcsa_seed23\gate1 --critic-dir results\ppcsa_seed23\gate2_value --mechanism-report results\ppcsa_seed23\shared_mechanism_v1\shared_mechanism_report.json --output-dir results\ppcsa_seed23\shared_admission_v1 --preflight-only
```

After implementation review, the user runs the real virtual diagnostic:

```powershell
python -B scripts\diagnose_ppcsa_update_acceptance.py --gate1-dir results\ppcsa_seed23\gate1 --critic-dir results\ppcsa_seed23\gate2_value --mechanism-report results\ppcsa_seed23\shared_mechanism_v1\shared_mechanism_report.json --output-dir results\ppcsa_seed23\shared_admission_v1 --device cpu
```

The fixed 47-state/three-candidate protocol needs at most 282 Shadow calls. All
states are retained after vetoes to preserve complete rejection evidence.
Output: `shared_admission_report.json`; stdout: concise admission status,
Actor-training boundary and detail path. No real diagnostic is run by Codex.

## Verification and research boundary

Synthetic tests cover predicted-positive/actual-negative updates despite a
positive aggregate, one unsafe state, one risk-worse state, unknown/missing or
reordered tracks, replay drift, zero gains, missing/duplicate states, atomic
zero fallback, multiple acceptable candidates without selection, warning-free
scalar checks and unchanged sources. Related PPCSA/MAPPO/CSA/CUP-JR regressions
remain required. Test logs are recorded before disposable scratch is removed.

Implementation acceptance does not require a candidate to pass. A virtual pass
only characterizes this exposed simulated development cohort. The 41 Fit states
participated in Critic fitting and the 6 Calibration states constructed the
directions. Independent efficacy/safety generalization is not established.
If all fixed candidates fail, preserve that conclusion instead of adjusting the
state set, step or constraints. Any subsequent mechanism revision or independent
development validation requires its own preregistered, reviewed protocol.
