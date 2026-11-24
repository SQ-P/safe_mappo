# Original100x100 task: MAPPO + safety, no intermediate subgoal generation

User-started Seed23/400-update experiment. Prepared only; no training run by Codex.

Complete independent snapshot:
`configs/experiments/mappo_safety_no_subgoal_100_seed23_u400.yaml`.
Derived from the resolved original `standard_mappo_3u_8s2d_seed23.yaml`, preserving
100x100 map,3 UAVs,8 static+2 dynamic obstacles,300-step horizon, start-Y jitter2,
detection25 and6 tracked obstacle slots (obs52/global156). Original and160x160
configurations, training scripts, environment and algorithm sources are unchanged.

`oracle_subgoal.enabled=false` stops initial generation and all replanning.
`oracle_subgoal.recovery_enabled=false` also disables Oracle-specific recovery
target generation; the high-level selector remains disabled. The existing
environment copies final goals into its target slots when Oracle is disabled:
Actor targets are final goals, with no intermediate waypoint generator.

Shared safety is unchanged: initial Guard, obstacle TTC, final Guard and safety
Recovery remain enabled. Safety Recovery is separate from Oracle recovery and
may still select escape actions as part of the safety executor.

The reward mode/coefficients remain the original `team_v4_subgoal` snapshot.
With target slots equal to final goals, the legacy `subgoal_progress_reward`
field measures progress toward final goals, plus the unchanged best-completion
term. Its legacy name does not mean a planner ran. No reward redesign, sparse-only
reward, algorithm modification or network input-width change is introduced.

Training: fresh MAPPO, CUDA,Seed23,400 updates,2048 rollout steps,4 PPO epochs.
Entropy anneal600 and other original training settings remain unchanged.
Checkpoints/evaluations at100/200/400;10 development episodes each using the
original evaluation seed base111000, with per-episode records saved.
No automatic formal-panel evaluation or historical checkpoint loading.

Run in PowerShell:

```powershell
Set-Location E:\Code\diagnostic\mappo_safe
$Py = 'D:\Anaconda\envs\env\fcrm-mapf\python.exe'
& $Py -B scripts\train_ppo.py --config configs\experiments\mappo_safety_no_subgoal_100_seed23_u400.yaml
```

Output: `results/ablations/mappo_safety_no_subgoal_100/seed_23_u400_v1/`.
Periodic checkpoints: `checkpoints/update_0100.pt`, `update_0200.pt`,
`update_0400.pt`, plus `latest.pt` and automatically selected best checkpoints.
Existing nonempty output is rejected. Omit `--resume` and task override flags:
this experiment starts from fresh weights under a changed target contract.

Readiness evidence:
`results/ablations/mappo_safety_no_subgoal_100/readiness_seed23_u400_v1/`.
29 tests passed,1 real-training checkpoint test deselected. Disabled-planner
checks patch Oracle generation to fail if called and inspect22 short zero-action
steps across2 seeds; final-goal target slots remain equal and replanning counts
stay zero. Fresh CPU MAPPO initialization/forward and CLI dispatch were checked;
the latter uses a fake trainer. No optimizer update or experimental evaluation
was run. Temporary synthetic outputs were removed, command/test evidence retained.

This is a separate no-planner condition. Changed target inputs and removed
intermediate guidance produce a different task fingerprint; compare results
as an ablation, not as a continuation of either historical or160x160 training.
