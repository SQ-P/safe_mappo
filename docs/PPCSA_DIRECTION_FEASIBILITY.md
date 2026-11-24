# PPCSA rejection attribution and three-direction secant feasibility

## Scope

This implements phases A/B only: read existing shared-admission evidence,
reproduce rejection predicates, attribute recorded vetoes, and solve a fixed
three-weight endpoint-secant linear program. It does not reconstruct parameters,
execute Actor/Critic/Adapter/Shadow, collect states, refit, relax thresholds,
change admission logic or open Audit/formal panels. Phase C is not authorized.

The original frozen cohort is retained in full: currently 47 Fit/Calibration
states and all 328 observed state/track risk rows. Repeated candidates, steps
and tracks are not independent experiments. Track identity is layout-local:
the same numeric obstacle id across different layouts is not one physical track.

## Attribution contract

The tool recomputes the original state, risk, progress and whole-candidate vetoes
using the unchanged admission predicates. Candidate nonzero/zero metadata is
used only to reproduce the source decision; no placeholder scalar is represented
as an original parameter vector. Reproduction mismatch invalidates the proxy.

Outputs retain every state, original baseline/candidate risk vector, track
identity/context, negative progress magnitude, positive risk deltas, recurring
layout-local veto tracks, state pass-pattern overlap, and requested/executed
action-change norms by safety stage. These quantities describe recorded responses;
they do not identify a causal failure mechanism.

Float32 value-storage ULP references quantify potential numerical sensitivity.
They are not full integration/norm/risk error bounds, statistical thresholds or
grounds for ignoring small negative gains. Original progress/risk tolerance stays
exactly zero. Missing, nonfinite or unaligned evidence yields INVALID_EVIDENCE;
no favorable state/track subset is solved. Nonfinite values are retained as
explicit invalid-value tags in the output JSON, never usable coefficients.

## Endpoint-secant proxy

Let k index Ordinary, Paired and Kinematic, each generated previously with
normalized direction and parameter step 0.01. For state i and track j:

```
G[i,k] = recorded_candidate_progress[i,k] - recorded_baseline_progress[i]
R[i,j,k] = recorded_candidate_risk[i,j,k] - recorded_baseline_risk[i,j]

maximize sum_i sum_k alpha[k] * G[i,k]
subject to:
  alpha[k] >= 0; sum_k alpha[k] = 1
  sum_k alpha[k] * G[i,k] >= 0        for EVERY state
  sum_k alpha[k] * R[i,j,k] <= 0      for EVERY observed track
```

These are finite-step endpoint secants, not measured derivatives, new mixed
Shadow outcomes or a local safety certificate. Baselines must be identical
across all candidates, and track identity/order/context must align fully.
Categorical hard-safety predicates are not certified for mixtures by the LP.
Any proposed mixture would still require the unchanged whole-update admission
and new Shadow validation under a separately authorized protocol.

## Exact solver and result meanings

Three weights live on a two-dimensional simplex. The fixed LP is solved by
exact rational half-plane clipping of the simplex polygon, followed by exact
linear-objective evaluation at all remaining vertices. This is not a grid,
random candidate search, epsilon sweep, parameter step search or backtracking.
Degenerate feasible segments/points remain valid. Deterministic ties use the
lexicographically greatest exact optimum vertex.

Each rational coefficient is the decimal round-trip representation of the
recorded float delta. The LP has zero numerical constraint tolerance. The report
retains all exact coefficient rows, exact weights/objective, maximum violation
(zero for feasible witnesses), and an empty-intersection witness for infeasible
problems. Infeasible models have no feasible-point residual; it is recorded as
not applicable rather than invented. Exact arithmetic is only a statement about
the recorded proxy, not precision or correctness of the physical measurements.

| Status | Allowed interpretation |
|---|---|
| PROXY_INFEASIBLE | This fixed simplex endpoint-secant model has no feasible point |
| PROXY_NONPOSITIVE | Proxy constraints are feasible but no strictly positive full-cohort proxy net gain exists |
| PROXY_POSITIVE_DIRECTION_UNVERIFIED | Proxy-positive weights found; actual parameter direction and execution unverified |
| INVALID_EVIDENCE | Missing, inconsistent, nonfinite or unaligned evidence; no subset conclusion |

No meaningful-effect threshold substitutes for strictly positive proxy net gain.
Tiny positive proxy gains are reported numerically without efficacy claims.

The norm bound is analytical only:

```
||0.01 * sum_k alpha[k] d[k]|| <= 0.01
```

Original nonzero direction vectors were not retained in rejected admission
outputs. Their SHA/norm metadata and zero effective-update fallbacks cannot
recover those vectors. Nonzero weights therefore do not prove a nonzero mixed
parameter update: cancellation can yield a zero lower bound. Every result keeps
`actual_nonzero_parameter_status=UNKNOWN`; no status is FEASIBLE_SHARED_UPDATE.

## User commands

Use a new output directory. Preflight reads evidence and checks bindings,
coverage, reproduction and finite coefficients without solving the LP or
creating a result directory:

```powershell
Set-Location E:\Code\diagnostic\mappo_safe
python -B scripts\diagnose_ppcsa_direction_feasibility.py --admission-report results\ppcsa_seed23\shared_admission_v1\shared_admission_report.json --output-dir results\ppcsa_seed23\direction_feasibility_v1 --preflight-only
```

After code review, the user runs the read-only attribution/LP diagnostic:

```powershell
python -B scripts\diagnose_ppcsa_direction_feasibility.py --admission-report results\ppcsa_seed23\shared_admission_v1\shared_admission_report.json --output-dir results\ppcsa_seed23\direction_feasibility_v1
```

Outputs: `direction_feasibility_report.json` with full attribution/exact LP
evidence and `direction_feasibility_report.md` for reading. Stdout is one status
line plus the detail path. Input report, referenced historical artifacts,
implementation hashes and checkpoint bytes are verified without loading models;
all source hashes are checked again before writing separate results.

## Review boundary

Weights are selected on exposed Fit/Calibration evidence; they are not independent
validation. No actual candidate vector or Actor checkpoint is emitted. A proxy
positive result is only a proposal to freeze and review before separately
authorizing direction reconstruction and new Shadow evaluation. Proxy failure
does not establish impossibility in the full Adapter parameter space. Existing
admission code and all zero-tolerance conditions remain unchanged.
