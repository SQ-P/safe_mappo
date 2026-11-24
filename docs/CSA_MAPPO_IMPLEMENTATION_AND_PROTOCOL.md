# CSA-MAPPO implementation and formal protocol

> **2026-11-17 protocol amendment.** The original clean 8-slot/u400 parent
> protocol below is paused. The user requested new 6-slot, progress-aware
> safety-execution Standard runs for seeds 23/41/67 with an 800-update budget.
> Do not execute the critic, CSA, Lagrangian or formal-test stages until the
> three new parents exist, a common parent update is frozen, and the critic is
> rebuilt and certified against the new task fingerprint.

## Current status

Implemented and unit-tested:

- zero-initialized rank-16 residual Adapter on the recurrent Actor mean;
- frozen-backbone CSA mode and full-Actor ablation mode;
- six registered same-state reference actions and immutable safety-stack evaluation;
- frozen certified R4.4 Twin Critic, reference stop-gradient, support/deadzone/eligibility gates;
- MAPPO-Lagrangian baseline initialized from the matching Standard u400 parent;
- parent and critic SHA256, alignment contract, frozen parameter list, Adapter and optimizer resume state;
- per-episode avoidable/material intervention rates;
- seed-to-layout hierarchical paired bootstrap and fail-closed paper claim gates.

Not completed and therefore not evidence:

- clean Standard u400 checkpoints for seeds 23, 41 and 67 under the exact 8-slot contract;
- fresh 118000--118199 critic certification;
- seed-23 fixed-rollout lambda calibration;
- nine formal training runs and the single-use 120000--120199 test;
- empirical values for Tables 1--2 or Figures 1--3.

The historical 8S+2D checkpoint has a six-slot/legacy selector contract. It must not be copied into the formal table or used as a CSA parent.

## Fixed method contract

CSA uses only the Adapter and centralized Value Critic as trainable components. The encoder, GRU, original action head and `log_std` are frozen. The Adapter is:

```text
delta = 0.15 * tanh(W_up(ReLU(W_down(LayerNorm(h)))))
mu = mu_base + delta
```

`W_up` and its bias start at zero. Its parameter count is checked to be below 3% of the original Actor. Deployment uses the Actor, Adapter and unchanged staged safety executor only; the Twin Critic and reference search are training/evaluation instrumentation.

The CSA loss compares the Actor action with six references (HOLD, brake, goal scales 0.25/0.50/0.75/1.00). The reference branch is detached. Recovery, SAFE_BRAKE, terminal, unsupported-action and no-valid-reference rows contribute zero alignment gradient.

## Frozen panels

| Role | Layouts | Use |
|---|---:|---|
| Critic development | 117000--117199 | 120/20/20/40 train/calibration/validation/test |
| Fresh certification | 118000--118199 | One shot; no fitting or protocol changes |
| Actor validation | 119000--119039 | Seed 23 only |
| Formal test | 120000--120199 | Same 200 layouts for every method and training seed; one shot |

The exact files are in `configs/evaluation/`. The frozen decision contract is `configs/evaluation/csa_mappo_formal_protocol.json`.

## Execution order

1. Train the three clean Standard parents:

```powershell
python scripts/train_ppo.py --config configs/active/standard_mappo_3u_8s2d.yaml
python scripts/train_ppo.py --config configs/experiments/standard_mappo_3u_8s2d_seed41.yaml
python scripts/train_ppo.py --config configs/experiments/standard_mappo_3u_8s2d_seed67.yaml
```

2. Build/train/certify the R4.4 critic on the frozen 117000 and 118000 panels. Actor training is forbidden if fresh certification reports `certified=false`.

3. Evaluate the seed-23 Standard parent on the 40-layout actor-validation panel with safety-alignment metrics enabled. Freeze the Lagrangian budget:

```powershell
python scripts/freeze_lagrangian_budget.py `
  --parent-validation <seed23_validation_episode_records.jsonl> `
  --output results/paper/mappo_lagrangian/cost_budget_seed23.json
```

4. On a fixed seed-23 rollout, record the unscaled CSA/PPO gradient-norm ratio, then freeze lambda:

```powershell
python scripts/calibrate_csa_lambda.py `
  --diagnostics <fixed_rollout_gradient_diagnostics.json> `
  --output results/paper/csa_mappo/lambda_calibration_seed23.json
```

The helper chooses the smallest value in `{0.01, 0.03, 0.10, 0.30}` whose scaled ratio lies in `[0.05, 0.10]`; it fails if none qualifies.

5. Run the three CSA configs and three Lagrangian configs in `configs/experiments/`. Run the three seed-23 ablations with the same parent and 100-update budget.

6. After every checkpoint and protocol choice is frozen, evaluate all formal methods once on `configs/evaluation/csa_formal_test_120000_120199.json`, writing one non-overwriting episode JSONL per arm.

7. Produce the paper gate report:

```powershell
python scripts/compare_csa_formal.py `
  --standard-files <standard_seed23.jsonl> <standard_seed41.jsonl> <standard_seed67.jsonl> `
  --candidate-files <csa_seed23.jsonl> <csa_seed41.jsonl> <csa_seed67.jsonl> `
  --output results/paper/formal/csa_vs_standard.json
```

The report rejects missing/duplicate layouts, mismatched task fingerprints, unexpected training seeds and overwritten output. It resamples training seeds and then paired layouts 10,000 times. The claim is authorized only when every registered efficacy and non-inferiority gate passes.

## Paper artifacts

- Table 1 consumes the formal Standard/Lagrangian/CSA episode rows.
- Table 2 consumes the three seed-23 ablation rows and must be labeled mechanism evidence.
- Figure 1 uses Standard safety-stage traces from the formal task.
- Figure 2 diagrams frozen Twin Critic/reference bank, confidence gate and Adapter.
- Figure 3 uses the paired CFMS--avoidable-intervention estimates plus training telemetry.
- The appendix reports fresh critic certification, per-seed results, 4S/6S generalization and failure cases.
