# Current project scope

The supported entry point is recurrent Standard MAPPO for the 3 UAV, 8 static obstacle, 2 dynamic obstacle task in `configs/active/standard_mappo_3u_8s2d.yaml`. The three seed files in `configs/experiments/standard_mappo_3u_8s2d_seed{23,41,67}.yaml` inherit this task contract.

## Active execution

- `algorithm: mappo`; shared actor and centralized team critic.
- Oracle subgoal planning enabled; high-level subgoal selector disabled.
- `oracle_subgoal.generation_enabled` controls subgoal generation. When false, the final goals occupy the unchanged subgoal observation slots and no replanning occurs.
- Staged safety execution enabled: initial Joint Guard, obstacle TTC, final Joint Guard, plus scoped safety Recovery.
- Training, evaluation, and rendering entry points default to the active Standard MAPPO config. Explicit legacy configs are still present pending the tracked-file cleanup review.

The historical `research_line` label remains in the active YAML for checkpoint compatibility. The MAPPO trainer, actor network and update formulas remain unchanged.

## Cleanup status

The standalone historical research configurations, scripts, tests, and independent modules identified in the cleanup review have been removed. The baseline contract tests, safety panels, and paired evaluation tools remain available. The core trainers and environment still contain branches and imports for older research methods. Those branches must be separated before their supporting modules can be removed safely.

`README.md` has been rewritten for the Standard MAPPO baseline and its previous merge state was resolved.
