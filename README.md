# safe_mappo

Standard recurrent MAPPO for cooperative UAV navigation. The active task uses three UAVs, eight static obstacles and two dynamic obstacles. Oracle subgoals are enabled; the high-level selector is disabled. The safety executor applies initial Joint Guard, obstacle TTC, final Joint Guard, and scoped Recovery.

## Main configuration

- `configs/active/standard_mappo_3u_8s2d.yaml`
- `configs/experiments/standard_mappo_3u_8s2d_seed{23,41,67}.yaml`

The active `research_line` label is retained for checkpoint compatibility; the configured algorithm is Standard MAPPO.

Set `oracle_subgoal.generation_enabled: false` to use each UAV's final goal as its subgoal and skip Oracle subgoal generation and replanning. The default is `true`. A checkpoint trained with a different value has a different task contract.

## Commands

```powershell
python scripts/train_ppo.py --config configs/active/standard_mappo_3u_8s2d.yaml
python scripts/evaluate_ppo.py --config configs/active/standard_mappo_3u_8s2d.yaml --checkpoint <checkpoint.pt>
python scripts/render_ppo.py --config configs/active/standard_mappo_3u_8s2d.yaml --checkpoint <checkpoint.pt>
python -m pytest -q tests/test_mappo.py tests/paper/test_standard_legacy_execution_config.py
```

`results/` contains experiment outputs and is ignored by Git. Cleanup of tracked source files does not alter those outputs. See `docs/PROJECT_SCOPE.md` for the current source boundary.
