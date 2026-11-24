# PPCSA common-veto attribution and future parameter-sensitivity contract

The `PROXY_INFEASIBLE` result rules out the nonnegative convex family of the
three fixed **endpoint responses**. It does not rule out the full Adapter
parameter space, and does not measure a derivative of the executed map.
Stop fixed convex mixing; retain the original strict whole-update admission.

This stage implements a read-only evidence review and a pure data contract.
It neither implements an optimizer nor reconstructs a parameter direction.

## Common-veto review

Inputs are the original Shared Admission V1 and Direction Secant Feasibility V1
JSON reports. Bind their SHA values, historical input/checkpoint/source hashes,
and historical feasibility implementation hashes. Reproduce the complete fixed
registry, candidate order, every progress/risk coefficient and the exact common
veto list from raw admission evidence. No states or tracks may be dropped.

For the current seed-23 report, coverage is 47 states, 47 progress constraints,
328 risk constraints, and eight common vetoes (one progress, seven risk).
Counts are verified against the bound inputs rather than substituted for them.

For each common veto record state/layout/step, observer/track identity when
applicable, the three exact constraint coefficients, baseline/candidate progress
and risk, full track context, hard-safety evidence, and complete action arrays.
Record both within-pipeline stage changes and baseline-to-candidate changes at
Actor raw, recovery, corridor, Initial Guard, TTC, Final Guard and execution.
Vector changes, L2 norms and execution/request ratios are descriptive only.
Track identity remains layout-local; repeated states/directions are dependent.

All causal mechanism labels remain `UNIDENTIFIED`. An action difference at TTC
does not establish that TTC caused risk regression. The saved finite precision
does not provide a physical error bound. Risk/progress tolerances remain zero.
Missing/nonfinite action stages, incomplete tracks, divergent baselines,
rewritten constraints or broken source bindings stop the review with an error.
The environment-state registry fingerprint and the composite Shadow fingerprint
have different scopes. Each is preserved and compared within its own domain;
they are not required to equal each other.

The real historical evidence does not retain measured parameter sensitivities.
Every current review therefore reports `PARAMETER_SENSITIVITIES_UNAVAILABLE`.
No derivative is inferred from a three-element endpoint-response vector.

## Future input contract, implemented as pure validation only

`validate_parameter_sensitivities` validates declarations, shape, finite values,
full ordered state/track coverage and source bindings. It never verifies empirical
derivative accuracy and never authorizes measuring derivatives or optimizing.

Required payload fields:

| Field | Contract |
|---|---|
| `schema_version` | `PPCSAParameterSensitivitiesV1` |
| `quantity` | `executed_parameter_derivatives` |
| `method` | `autograd_of_executed_map` or `parameter_finite_difference` |
| `endpoint_secants_used` | `false` |
| `bindings` | Exact expected admission, registry and checkpoint SHA bindings |
| `measurement_protocol_sha256` | Frozen measurement protocol SHA256 |
| `parameterization_sha256` | Declared shared parameterization SHA256 |
| `parameter_order` | Ordered unique names with positive integer shape dimensions |
| `parameter_count` | Total scalar count in that same parameterization |
| `progress` | Every ordered `progress:layout:step` ID, each with a full finite `gradient` |
| `risk` | Every ordered `risk:layout:step:observer:track` ID, each with a full finite `gradient` |

No payload yields `PARAMETER_SENSITIVITIES_UNAVAILABLE`; invalid payloads yield
`INVALID_PARAMETER_SENSITIVITIES`. A conforming payload yields only
`DECLARED_PARAMETER_SENSITIVITIES_VALIDATED`, with
`empirical_derivatives_verified=false`. In particular, declarations cannot prove
that autodiff traversed the real non-smooth execution map correctly. That needs a
separately reviewed measurement protocol and numerical fidelity evidence.

The future mathematical problem, **not solved here**, would maximize
`sum_i w_i grad_theta(G_i) dot update` with prespecified weights, while requiring
every state progress response >= 0, every track risk response <= 0, a nonzero
shared parameter vector, strict positive net progress, and norm <= 0.01.
Categorical hard safety and nonlinear executed outcomes still require separate
real evaluation under unchanged admission rules.

`check_declared_linear_direction` is a pure synthetic witness checker: it validates
the contract then checks a caller-supplied vector exactly; it does not search,
generate, emit or install a real update. Zero, negative progress, positive risk,
budget excess, and positive-but-unverified responses are tested. A synthetic
example shows that all three old responses can worsen risk while another
parameter-space direction improves progress and reduces risk. It is a logical
counterexample, not evidence that this project's Adapter has such a direction.

## Commands

Preflight verifies sources and complete input reproduction without the new
common-veto attribution stage, LP solve, models or output creation:

```powershell
Set-Location E:\Code\diagnostic\mappo_safe
python -B scripts\review_ppcsa_common_veto.py --admission-report results\ppcsa_seed23\shared_admission_v1\shared_admission_report.json --feasibility-report results\ppcsa_seed23\direction_feasibility_v1\direction_feasibility_report.json --output-dir results\ppcsa_seed23\common_veto_v1 --preflight-only
```

After code review, the user runs the full offline evidence review:

```powershell
python -B scripts\review_ppcsa_common_veto.py --admission-report results\ppcsa_seed23\shared_admission_v1\shared_admission_report.json --feasibility-report results\ppcsa_seed23\direction_feasibility_v1\direction_feasibility_report.json --output-dir results\ppcsa_seed23\common_veto_v1
```

The output directory must be new and separate from bound historical sources.
Outputs are `common_veto_report.json` and `.md`; stdout gives status and detail
path. Original artifacts are checked before/after reading and never overwritten.

This is not Gate 3. Actor/Critic/Adapter/Shadow execution, refitting, training,
new rollout collection, parameter search, negative-weight search, threshold
changes and Audit/formal panel use remain outside this stage. No performance
gain, true shared update feasibility, or independent validation is claimed.
