# PPCSA Phase A: frozen executed parameter-sensitivity measurement

Implementation and synthetic tests only. Codex does not execute real model or
Shadow measurements. The user must separately choose to run the frozen protocol.
No optimizer, new direction acceptance, Actor training, Critic refit or rollout
collection is included. Original admission and all historical files remain fixed.

## Preregistered scope

Use the pinned CPU seed23/u800 checkpoint, frozen 52D V4/TTCv2 task, saved
Fit/Calibration packets, and all 47 admission states / 328 observed risk tracks.
No selection based on the eight common vetoes; no Audit packet inspection.
Use the deterministic diagnostic Adapter initialization (seed 20261121,
hidden128/rank16/scale0.15, zero `up` layer). Source Actor/Adapter bytes and source
file SHA values are checked before/after; candidate perturbations use copies.
Executed-value Critic artifacts are not loaded or evaluated, and Actor forward
passes are not needed: requests use the bound saved hidden features and pre-tanh
means. The existing parent-checkpoint runtime restores the trainer state,
including its MAPPO value weights, without calling that value model or fitting it.

At zero `up`, first-order support is in the 34 up-layer scalars. The **measured
parameterization is an explicit six-coordinate subspace**, ordered as follows:

| Coordinate | Full flattened up index (weight row-major, then bias) |
|---|---:|
| `up.weight[0,0]` | 0 |
| `up.weight[0,1]` | 1 |
| `up.weight[1,0]` | 16 |
| `up.weight[1,1]` | 17 |
| `up.bias[0]` | 32 |
| `up.bias[1]` | 33 |

The first two hidden-unit positions are chosen from source order, not response
data. Dead ReLU coordinates may produce unresolved zero responses; do not replace
them after looking at results. No derivatives of the other 28 up coordinates or
other layers are inferred or zero-imputed. The SHA of `parameterization()` binds
the coordinate order, full-layer shapes and mapping.

## Fixed finite differences and budget

For every state/coordinate, evaluate both parameter signs at `h=0.001`,
`0.0005` and `0.00025`. Repeat every signed evaluation twice and the baseline
twice. Maximum calls: `47 * (2 + 6 * 3 * 2 * 2) = 3478` NORMAL Shadow calls.
Restore state/cache/GRU/RNG parity before each call; capture the current track
identity/order/context and all factual execution stages. Do not advance or
collect new environment trajectories. Every state and scheduled evaluation is
retained, including failures; there is no favorable-state export.

The estimate is `(f(+h/2)-f(-h/2))/h`. Retain all three central estimates and
positive/negative one-sided slopes. The third scale is held out from estimating
the derivative and is used to predict signed endpoint responses. It is a
**numerical consistency check on the same development state**, not independent
sample validation or proof that a true derivative exists.
Raw samples record the requested parameter delta and actual float32-stored
delta. Difference denominators use the fixed nominal scales; the tiny storage
rounding is exposed rather than represented as an exact parameter displacement.

Fixed numerical diagnostic comparison of output changes `a,b`:

`abs(a-b) <= 1e-7 + 0.1 * max(abs(a),abs(b))`.

These preregistered values are numerical diagnostics, not physical error bounds,
safety margins, effect thresholds or admission slack. No tuning from veto data.
Original progress/risk admission tolerances remain exactly zero.

Block the whole usable sensitivity export when any condition holds:

- A state/sign/repeat is missing, unsupported, not hard safe, nonfinite, or has
  unknown risk, incomplete tracks, different track context/order or state token.
- Repeated payloads differ (exact comparison), or baseline replay disagrees with
  historical requested/executed actions, progress, risk or tracks.
- Forward/backward responses disagree, the first two central estimates disagree,
  or held-out signed-scale predictions fail the fixed numeric comparison.
- Discrete Guard flags or TTC intervention masks change. This is a conservative
  indicator, not exhaustive proof of every execution branch; hidden active-set
  changes can also appear as finite-difference inconsistencies.
- Any measured progress or risk response is at/below the `1e-7` absolute output
  floor. Such a zero or tiny response remains `ZERO_OR_RESOLUTION_LIMITED_RESPONSE`;
  it is not silently accepted as a precisely known zero gradient. This conservative
  rule can block export on risk plateaus and does not establish unlearnability.

Raw signed payloads, all estimates, invalid evidence and fidelity flags remain
in the report even when export is blocked. Report per-coordinate positive/
negative/zero response counts across dependent states/tracks descriptively;
these are neither a feasibility result nor a causal explanation of Guard/TTC.

## Outputs and interpretation

Before the first real evaluation write `measurement_protocol.json`, containing
the exact protocol/parameterization SHA, source bindings and tool hashes.
Outputs `parameter_sensitivity_report.json` and `.md` retain the complete fixed
registry, raw baseline/signed repeats, numerical checks and source provenance.
No real Adapter update/checkpoint is emitted.

Statuses:

| Status | Interpretation |
|---|---|
| `MEASUREMENT_INVALID` | At least one invalid/incomplete measurement or source replay |
| `MEASUREMENT_FIDELITY_UNRESOLVED` | Complete recordings but numerical, branch or resolution checks failed |
| `SUBSPACE_ESTIMATES_REQUIRE_REVIEW` | All fixed numerical checks passed for this six-coordinate subspace only |

Only the last case includes a `PPCSAParameterSensitivitiesV1` payload, validated
with the unchanged `validate_parameter_sensitivities` contract: `parameter_count=6`,
six explicit scalar names, full state/track row coverage, and source SHA bindings.
`empirical_derivatives_verified=false` remains mandatory. Contract validity does
not certify exact derivatives, nonlinear safety, performance, generalization or
full Adapter feasibility. No outcome automatically authorizes Phase B/C.

## Commands

Read-only preflight (no tensor/checkpoint loading, runtime, Shadow or outputs):

```powershell
Set-Location E:\Code\diagnostic\mappo_safe
python -B scripts\diagnose_ppcsa_parameter_sensitivity.py --admission-report results\ppcsa_seed23\shared_admission_v1\shared_admission_report.json --feasibility-report results\ppcsa_seed23\direction_feasibility_v1\direction_feasibility_report.json --common-veto-report results\ppcsa_seed23\common_veto_v1\common_veto_report.json --output-dir results\ppcsa_seed23\parameter_sensitivity_v1 --preflight-only
```

After code review, **only when the user chooses to execute the real frozen
measurement**, use the explicit run mode. This performs at most 3478 Shadow
calls; Codex must not launch it under the implementation-only authorization.

```powershell
python -B scripts\diagnose_ppcsa_parameter_sensitivity.py --admission-report results\ppcsa_seed23\shared_admission_v1\shared_admission_report.json --feasibility-report results\ppcsa_seed23\direction_feasibility_v1\direction_feasibility_report.json --common-veto-report results\ppcsa_seed23\common_veto_v1\common_veto_report.json --output-dir results\ppcsa_seed23\parameter_sensitivity_v1 --run-frozen-protocol
```

Use a new output directory; never overwrite a prior diagnostic. No CLI basis,
scale, state-subset, device, effect threshold or optimizer tuning is exposed.
