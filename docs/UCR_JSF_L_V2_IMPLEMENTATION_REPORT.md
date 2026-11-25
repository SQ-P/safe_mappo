# UCR-JSF-L V2 Implementation Report

Date: 2026-11-25

Status: implementation and bounded code acceptance passed. No training or 200-episode replay was run.

## Implemented scope

- Added the independent `src/safety/ucr_jsf_l_v2/` package: ordering, contracts, candidates, liveness, reachability and executor.
- Added the exact V2 experiment configuration and `JointSafetyTraceV2` contract.
- Registered an explicit V2 environment branch and the evaluation-only staged-to-V2 checkpoint transition while retaining the V1 branches.
- Added V2 config guards for the frozen environment, Oracle-off state, telemetry and 64-evaluation budget.

## Permutation acceptance

The test matrix executes every one of the six permutations for all three recovery modes: `NORMAL`, `STALL_ESCAPE` and `SAFETY_RECOVERY`. It permutes positions, velocities, goals, requested actions, obstacle observers, previous distances, stall counts, passing-side ownership, reachability tracks and lost-track evidence.

After mapping back to physical UAV identity, the selected actions, progress score, Actor deviation, effective safety margins, canonical action signature and canonical candidate ID match. Explicit symmetric-degeneracy tests reject unequal local ranks and do not use stable input order.

## Candidate coverage fixture

The saturated three-blocker fixture produced:

| Item | Observed |
|---|---:|
| Local candidates per physical UAV | 7 / 7 / 7 |
| Local non-baseline ranks covered | 6 / 6 / 6 |
| Heterogeneous/team anchors after deduplication | 5 |
| Single-agent deviations | 18 |
| Base pair candidates | 15 |
| Pair allocation | 5 / 5 / 5 |
| Pair refill candidates | 6 |
| Base triple candidates | 16 |
| Normal candidates | 60 |
| Emergency candidates | 4 |
| Total bounded evaluations | 64 |

The emergency IDs were `emergency:all_stop_hold`, `emergency:c000:away`, `emergency:c001:away` and `emergency:c002:away`. Across all six input permutations, each Away action followed the same physical UAV.

## Tie and liveness acceptance

- Normal and emergency tie tests select by canonical action signature when all preceding values tie, independent of candidate list order and candidate ID.
- Static/dynamic head-on, boundary, teammate, lateral-velocity and fully symmetric passing-side cases pass.
- Passing-side commitment follows actual predicted displacement and retains flip accounting.
- Source checks confirm the V2 liveness implementation has no agent parity fallback.

## Test results

- V2 focused suite: `21 passed`.
- V2 + V1 + cooperative environment regression: `73 passed`.
- `python -B -m compileall -q src/safety/ucr_jsf_l_v2`: passed.
- V2 Seed23 U800 checkpoint transition/load: passed on CPU in read-only mode.

These are implementation and compatibility results, not evidence that V2 improves task success, collision rate or timeout rate.

## Protected artifact verification

The implementation-before and implementation-after SHA256 manifests match for all 12 protected artifacts: the V1 package files, V1 config, V1 document and Seed23 U800 checkpoint.

Key hashes:

- Seed23 U800 checkpoint: `138BC86B4C3CABA4B53401CD68CFE020EE01AFBC0017ADED17D01C51E1F6846C`
- V1 config: `9B6191262BAA47FA0BC6B02A72AA84E43B97649F7CFB3C3C711017E527BD5686`
- V1 document: `034519225EBB363488E43019BDADD193E8C224507B1B2127FB1DCC724FD4F463`

Result: `PROTECTED_SHA256=UNCHANGED`.
