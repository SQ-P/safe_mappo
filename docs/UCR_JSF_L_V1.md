# UCR-JSF-L V1

UCR-JSF-L V1 is a new versioned joint safety executor. It does not modify or
call the legacy `Initial Guard -> obstacle TTC -> Final Guard` implementation.
The environment dispatches to exactly one executor version before the normal
physics, reward and termination code.

## Frozen task contract

- MAPPO, three UAVs, eight static and two dynamic obstacles.
- Six tracked-obstacle slots and 52 local observation values.
- `oracle_subgoal.enabled`, `generation_enabled` and `recovery_enabled` are all
  false. The policy still observes the original fixed-width schema.
- Map, dynamics, reward, Actor and Critic contracts are unchanged.

## Decision contract

Every cycle starts from one pre-action team state. Up to eight deterministic
local actions per agent are combined into at most 64 joint candidates. Each
joint candidate is evaluated by the same boundary, swept pair-distance and
observer-scoped obstacle constraints. Only model-admissible candidates enter progress
optimization, and exactly one final joint action is returned.

The progress score includes final-goal progress and tangential escape only when
an observed obstacle envelope intersects the local final-goal corridor,
straggler priority, selected-action passing-side commitment, controlled side
switching and a predicted goal-region exit penalty. Modes are stateful:
`NORMAL`, `STALL_ESCAPE`, and `SAFETY_RECOVERY`.

Invalid velocity and short causal track loss use a 2 m/s reachable envelope;
possible boundary reflection expands that envelope. The model never reads
future simulator motion. Reports distinguish `MODEL_ADMISSIBLE`,
`MODEL_REJECTED`, and `EVIDENCE_INSUFFICIENT`. Invalid velocity or short loss
covered by a reachable envelope is bounded uncertainty; expired lost tracks
and saturated observation slots are insufficient evidence. Insufficient
evidence is recorded without being converted into a hidden all-team brake.
`model_admissible` means that no physical violation was found by this
registered observable model, not that it has a formal safety certificate.

Four of the 64 total candidate evaluations are reserved for emergency actions.
If normal candidates are rejected, braking, risk-directed motion and lateral
escape are compared by residual physical risk. Backup braking recomputes the
brake command and holds zero velocity instead of applying constant acceleration
until the predicted UAV reverses. Residual risk is recorded and search failure
is never called physical infeasibility.

The wall contract preserves the existing 3.0 m surface clearance plus the
registered 0.25 m invariant buffer. This prevents a hidden reduction in safety
margin from being attributed to the new selector.

## Telemetry

`JointSafetyTraceV1` records the selected candidate, modes, candidate counts,
constraint margins, unknown evidence, stall state and passing-side memory. It
does not emit `StageInterventionTraceV1` or reinterpret legacy stage metrics.
Evaluation also reports stall-step ratio, passing-side flips, liveness-recovery
success rate, goal exits, candidate-unknown rate and joint-intervention rate.
Legacy stage metrics are explicitly marked not applicable for UCR-JSF-L.
Candidate telemetry separates observed-constraint passes, fully
`MODEL_ADMISSIBLE` candidates, `EVIDENCE_INSUFFICIENT` candidates and explicit
model rejections, and records the selected candidate's evidence status.
Passing-side commitment and flip accounting use the predicted next
displacement under the registered velocity-first dynamics, not acceleration
direction.

## Research boundary

The implementation and bounded tests establish code-path and contract
readiness only. They do not show that collisions or timeouts decrease, and do
not claim zero-collision safety or liveness. Training, 200-layout replay and
full evaluation are deliberately not run during implementation.

## Strict frozen-checkpoint evaluation

`scripts/evaluate_ppo.py --allow-safety-executor-transition` accepts only a
stored `staged_joint_guard_ttc_v1` task contract changing to
`ucr_jsf_l_v1`. Every non-safety field must match exactly, and the checkpoint
fingerprint and file hash are verified. This flag cannot be combined with a
generic task override and a loaded transition cannot enter `train()`. A
read-only load of the retained 52D Seed-23 OFF U800 checkpoint is covered by
the bounded integration tests; the checkpoint timestamp and contents are not
modified. V4/V5 spelling compatibility is confined to this registered
evaluation-only transition; the ordinary task contract remains V4.
