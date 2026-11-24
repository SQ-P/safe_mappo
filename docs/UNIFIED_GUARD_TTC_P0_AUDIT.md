# P0: serial Guard/TTC mechanism conflict audit

Only this P0 tool is implemented. The normal executor, Recovery state machine,
Actor/Critic, reward, observations, configurations, StageInterventionTraceV1,
checkpoints and historical results are unchanged. P1 contracts, P2 integration,
P3 paired evaluation and training require a later decision; no automatic advance.

## Frozen input and scope

Use Seed23 update0800, checkpoint SHA256
`087448d2e036f51e450ae927f9c841b156e0b87ed28ec16ee98de1427bfecb02`,
3U/8S+2D, 52D observations, Subgoal enabled, old staged Joint Guard/TTC v2 and
STAR scoped Recovery v4. The checkpoint configuration is authoritative; the
supplied experiment YAML cross-checks baseline identity. No safety override is
applied. The already used layouts 121000–121199 are **exposed development**,
not a fresh independent validation panel.

Preflight hashes the checkpoint, source modules, configuration files and historical
safety-ablation JSON artifacts, reads checkpoint metadata and validates the exact
panel. It does not create a trainer, execute policy/Shadow/replay or write output.

## Fixed event definitions

- E1: for any UAV, `norm(A_final-A_ttc) * max_acceleration * dt >= 0.05 m/s`.
  This uses the registered material physical threshold, not enum ordering:
  a tiny direction reversal alone must not qualify. Record action delta and
  clipped actual-model velocity delta separately.
- E2: on the same pre-control observable tracks, the old TTC rule returns no
  triggered risks for `A_ttc` and at least one for `A_final`. Keep added UAV/
  track IDs, TTC and clearance. Empty risks do not mean all-obstacle safety.
- E3: the decrease in clipped predicted velocity toward the **final goal** is
  at least 0.10 m/s for a UAV. This adopts the existing headroom diagnostic value
  as a P0 preregistered research threshold, not a runtime safety margin. Goals
  within 1e-8 m have undefined direction. Subgoal velocities are separate.

E3 is a descriptive same-state action contrast, not proof Final Guard made an
unnecessary change. The TTC action may violate pair/boundary constraints.
Thresholds, scales and budgets cannot be tuned from the audit outcomes.

## Same-state capture and factual parity

For each user-run old-policy replay step, clone the pre-action environment and
reuse the existing diagnostic prefix hook. Capture Actor, Recovery overlay,
Initial Guard, TTC, Final Guard and executed actions, plus Guard/Fallback checks.
The old prefix is run twice from independent clones for determinism; no audit
candidate is installed in the factual control path. The clone follows the old
control prefix exactly; candidate screening consumes only the pre-action
observable snapshot, never future simulator obstacle truth.

Environment snapshot, obstacle cache, Recovery state, Actor GRU and global Python,
NumPy and Torch RNG are verified unchanged by each audit. Candidate evaluations
start from independent copies. The user-run factual old-policy replay intentionally
advances its environment **after** the read-only audit; this is not an isolation
failure. Capture must match all factual stage actions exactly. A mismatch is
`FACTUAL_PARITY_FAILED` and excludes that state from mechanism opportunities.

Recovery states remain counted as `RECOVERY_ACTIVE`; unsupported captures retain
their actual factual stage trace, identity and error as `AUDIT_UNSUPPORTED`.
There is no silent removal of unsupported or failed states. This does not extend
PPCSA's NORMAL-only evaluator to Recovery: only its general checkpoint loader,
snapshot integrity helpers and the existing clone hook are reused.

Recovery context is extracted from nested snapshot scalars and arrays: mode,
brake status, phase, affected-agent masks, counters and recovery targets are
retained for both supported and unsupported captures.

## Finite candidate coverage, not a new verifier

On implicated event states retain old-stage joint actions, deterministic joint
braking, and up to 8 existing registered local TTC/headroom actions per implicated
UAV. Priority is raw, hold/brake, goal-aligned, then fixed action lexicographic
order and labels. Add one-UAV substitutions and synchronized local ranks.
Deduplicate at 10 decimals; evaluate at most 64 joint actions. Any discarded
registered joints produce `CANDIDATE_BUDGET_EXHAUSTED`. The action space is never
claimed exhaustive, even if this finite registry fits the budget.

Re-evaluate the whole joint matrix with the existing observable three-step TTC
shadow on a separate copy; independently inspect first-step Guard reprojection.
An internally rewritten candidate is rejected as a candidate preservation
opportunity. This is diagnostic reuse of the existing heuristic, not P1's
unified pure verifier or a proof of global safety.

Output `OBSERVABLE_SCREEN_PASS` only for a non-reprojected candidate with a
passing legacy screen and nonnegative recorded margin, in NORMAL with no known
evidence gap or existing invariant violation. Invalid track velocities, full
observation slots, absence of observable tracks, and a detected discrepancy
between unclipped TTC acceleration and actual clipped velocity remain
`EVIDENCE_INSUFFICIENT`. Only action-independent current geometry and discrete
braking violations trigger `PREEXISTING_VIOLATION`. Requested-action pair
clearance violations and Guard projection residuals are recorded separately;
projection failure alone does not establish a preexisting violation.
Candidate-specific first-step
clipping mismatches are also unknown. Later non-smooth/model differences and
unobserved obstacle behavior remain limitations of the legacy screen.

Candidate coverage records separate PASS, REJECT, UNKNOWN and PREEXISTING counts.
`NO_CANDIDATE_FOUND` requires the evaluated finite registry to be fully rejected
with sufficient evidence. Unknown candidates without a pass yield
`CANDIDATES_UNRESOLVED` and `no_candidate_found=null`; preexisting violations
also remain separate. Budget exhaustion is retained independently in every
screened result. No candidate found is not physical infeasibility.
Do not classify a safety correction as definitely mandatory merely because the
finite registry found no alternative. All apparent alternatives remain potential,
requiring the future P1 contract before safety claims.

A potential opportunity must pass this observable screen, have no per-UAV final-
goal velocity regression, and gain >=0.10 m/s for at least one UAV versus Final
Guard. Undefined goal directions make the opportunity unresolved. Exact factual
parity, deterministic capture, complete evidence and no preexisting violation
are required. Even a positive result remains a candidate for further checking.

## Outputs, gate and commands

New separate directory only:

- `serial_conflict_report.json` / `.md`: denominator, event/evidence/parity counts,
  opportunity state/layout counts and a review-only next-stage decision.
- `conflict_events.jsonl`: **every audited step**, including non-events, failures,
  Recovery, local state, observable tracks, stages, candidate records and coverage.
- `source_integrity_before.json` / `_after.json`: source/checkpoint/config and
  historical artifact hashes; any change stops the run.

Gate: complete all 200 layouts, no factual parity failure, at least 20 eligible
potential-opportunity states covering at least 10 layouts. Passing only proposes
`P1_CONTRACT_REVIEW_ONLY`; it never authorizes P1 implementation. Otherwise retain
negative/unknown evidence and return `INSUFFICIENT_MECHANISM_EVIDENCE`, without
expanding candidates or changing thresholds to obtain a pass.

Preflight (Codex may run):

```powershell
Set-Location E:\Code\diagnostic\mappo_safe
$Py = 'D:\Anaconda\envs\env\fcrm-mapf\python.exe'
& $Py -B scripts/audit_serial_guard_ttc.py --config configs/experiments/standard_mappo_3u_8s2d_seed23.yaml --checkpoint results/paper/standard_mappo_3u_8s2d_legacy_execution_v1/seed_23/checkpoints/update_0800.pt --layout-seeds-file configs/evaluation/safety_ablation_seed23_200.json --output-dir results/verification/unified_joint_ttc_v1/p0_serial_audit --preflight-only
```

Only after P0 code review, the user may launch the actual 200-layout replay:

```powershell
& $Py -B scripts/audit_serial_guard_ttc.py --config configs/experiments/standard_mappo_3u_8s2d_seed23.yaml --checkpoint results/paper/standard_mappo_3u_8s2d_legacy_execution_v1/seed_23/checkpoints/update_0800.pt --layout-seeds-file configs/evaluation/safety_ablation_seed23_200.json --output-dir results/verification/unified_joint_ttc_v1/p0_serial_audit
```

The full replay can be expensive because each state needs isolated legacy-prefix
copies and event candidates need whole-joint checks. It is not run by Codex under
the implementation authorization. Neither training nor checkpoint updates occur.
