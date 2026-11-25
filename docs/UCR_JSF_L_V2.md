# UCR-JSF-L V2

UCR-JSF-L V2 is an evaluation-only, versioned safety executor. It preserves the V1 unified constraint evaluator, dynamics and recovery controller while replacing V1 candidate search and liveness choice code with an independent implementation under `src/safety/ucr_jsf_l_v2/`.

## Frozen boundary

- Environment: 3 UAV, 8 static obstacles, 2 dynamic obstacles, 6 tracked-obstacle slots and 52-dimensional observations.
- Oracle subgoals: disabled.
- Reward, dynamics, observation schema, MAPPO Actor/Critic and checkpoint: unchanged.
- Runtime: one `joint_safety_selection` stage.
- Transition: `staged_joint_guard_ttc_v1 -> ucr_jsf_l_v2` is evaluation-only.
- V1 and the legacy staged executor remain separate and unchanged.

## Canonical decision contract

The executor derives an agent-id-free physical key from urgency, current state, goal, requested action, sorted local obstacle evidence and versioned liveness/reachability memories. Equal keys form an explicit degeneracy group. A degeneracy group may use only symmetric local ranks; stable input order is not accepted as a physical tie-break.

`agent_id` remains an array and memory address only. It does not affect search priority, passing side, candidate ranking or final selection. Joint-action signatures are expressed in physical canonical order.

Progress, tangential progress, arrival-hold penalty and Actor deviation are accumulated in canonical order with `math.fsum`. Candidate IDs are telemetry only.

## Candidate budget

| Layer | Contract |
|---|---:|
| Heterogeneous anchors | at most 8 |
| Single-agent deviations | every non-baseline local rank |
| Pair search | 15 base slots, 5 for each canonical pair |
| Triple search | 16 base slots |
| Normal evaluations | at most 60 |
| Emergency reserve | at most 4 |
| Total evaluations | at most 64 |

Unused anchor or deduplication space is reassigned to remaining pair candidates and then triple candidates without reducing the base pair/triple allocation. Emergency generation is breadth-first: all-stop first, one geometrically valid Away action per physical canonical UAV next, then Tangent actions only if reserve remains.

## Passing-side cascade

For a head-on blockage, V2 applies this order:

1. non-zero goal/obstacle cross product;
2. predicted dynamic-obstacle clearance using obstacle-minus-agent relative velocity;
3. map-boundary free space;
4. predicted teammate clearance;
5. existing lateral velocity;
6. global `+1` convention.

Passing-side memory is committed from selected physical displacement, not merely from an acceleration label.

## Final selection

Normal candidates use:

`(-progress, actor_deviation, -minimum_effective_margin, canonical_action_signature)`

Emergency candidates use:

`(residual_risk, actor_deviation, -minimum_effective_margin, canonical_action_signature)`

Neither path contains `candidate_id`.

## Telemetry

`JointSafetyTraceV2` records the selected canonical signature, deterministic scores, per-layer counts, single-agent local-rank coverage, pair/triple counts, emergency physical-agent coverage and the existing constraint/recovery evidence fields.

## Authorization boundary

This implementation authorizes unit, integration, compile and read-only checkpoint-load tests only. It does not authorize training, a 200-episode replay, checkpoint mutation, parameter tuning or V3 work.
