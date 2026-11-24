# Project scope: paper-ready MAPPO safety alignment

## Active research question

Can a competent recurrent Standard MAPPO policy be made safer under dense multi-UAV obstacle interactions by learning only a small, zero-initialized actor adapter from counterfactual intervention advantage, without changing the task, reward, observation contract, or frozen safety executor?

## Active method chain

1. Train or load a competent Standard MAPPO parent.
2. Freeze the parent policy and staged safety executor.
3. Collect factual actions and counterfactual reference actions under identical layouts.
4. Learn an intervention-advantage critic that estimates relative task/safety value.
5. Add a zero-initialized residual adapter to the actor mean and gate its updates with the certified advantage signal.
6. Evaluate with paired seeds/layouts against the same Standard MAPPO parent.

The adapter, CSA objective, confidence gates, Lagrangian baseline, checkpoint provenance, frozen panels and formal paired-report gate are implemented. They are research infrastructure, not positive empirical evidence: no clean 8S+2D three-seed formal matrix has been run yet.

## Paper positioning

This is a Technique Paper. The contribution is a minimal safety-alignment mechanism added to an existing MAPPO problem, not a new benchmark or a new environment setting.

## Evidence boundary

- Retained 3U+8S+2D checkpoint and independent-layout evaluation establish single-seed parent capability only.
- The historical 200-layout result is not evidence for the proposed adapter.
- Clean seeds 23/41/67 and the single-use paired 120000--120199 panel are required before statistical paper claims.
- Historical FACS/STAR and Projection-aware failures are retained only as compact provenance where directly relevant; their implementations are not active dependencies.

## Retained code boundary

- `src/algorithms/ppo`, `src/algorithms/mappo`, and shared multi-agent components.
- Environment, planning, frozen safety execution, utilities, and rendering.
- Intervention datasets, counterfactual advantage critic/certification, and paired reporting.
- Minimal training/evaluation/data/critic/certification scripts.
- Focused tests for MAPPO, environment contracts, safety contracts, and intervention advantage.

## Implemented CSA boundary

- Six-reference bank: HOLD, team brake and four goal-direction scales.
- Twin R4.4 probability critic with reference stop-gradient, action-support, eligibility and 0.05 deadzone gates.
- Rank-16, scale-0.15, zero-initialized residual Adapter; frozen Actor backbone by default.
- Parent/critic SHA256, frozen-parameter list and Adapter/optimizer state in checkpoints.
- Full-Actor, No-Counterfactual and No-Confidence-Gate ablations.
- Formal two-level paired bootstrap and all preregistered non-inferiority gates.

Exact execution order and current evidence gaps are in `docs/CSA_MAPPO_IMPLEMENTATION_AND_PROTOCOL.md`.

## Removed boundary

- HAPPO comparison code.
- Archived FACS/STAR selectors and orchestration.
- Projection-aware executed-action imitation implementations.
- LLM diagnosis/self-refinement machinery except the stable reward wrapper required by trainers.
- Generated configs, caches, obsolete reports, and superseded experimental outputs.

The complete pre-cleanup source and historical artifacts remain recoverable from `E:\Code\diagnostic\ppo_env_3D`.
