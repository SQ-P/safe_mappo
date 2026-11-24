# IA-MAPPO V1 当前项目说明与实验启动指南

> 2026-11-10 更新：R3.1 已证明动作条件 TTC 修复有效，但 Initial Guard 的 validation 阈值贴近 recall 下限，测试 recall=0.7886。活动阶段为 R3.2：独立 calibration/threshold-selection layouts、layout-bootstrap 稳健阈值、严重度基线认证和全新 62000–62199 布局。详见 `IA_MAPPO_R3_2_ROBUST_CERTIFICATION_AND_EXECUTION_20261110.md`。

更新时间：2026-11-09

## 1. 当前研究边界

本项目当前研究线为 `ia_mappo_v1`。目标是在冻结 Oracle 高层子目标和共享安全执行层的条件下，研究低层 MAPPO Actor 能否减少可避免的安全介入，同时不降低任务完成率、进度和安全性。

以下内容不再属于活动研究线：

- FACS/FACS-STAR 高层子目标选择；
- Projection-aware MAPPO V1/V2 的 executed-action imitation loss；
- 旧 selector、candidate response、competence trainer 和专用 runner；
- 从旧 Projection/FACS checkpoint、optimizer 或 normalizer 继续训练。

旧实现只作为历史负结果保存在 `archive/paused_facs_20261109/`；历史 `results/` 保持原位并标记为 `HISTORICAL_ONLY.md`。

## 2. 已完成状态

| 阶段 | 状态 | 当前证据 |
| --- | --- | --- |
| R0 归档与历史归因 | 完成 | 708 个归档文件通过 SHA256/大小/路径验证；320 episode、160 对配对归因完成 |
| R1 接口、遥测和 Clean Standard | 实现完成 | 阶段级 trace、因果快照、真实安全栈 shadow、fresh-init 配置和启动/评估脚本已完成 |
| R1 seed 23/41 u200 | 已完成 | 修复后的 `standard_runtimefix_v1/` 已完成训练与固定 40-layout 评估 |
| R2 可学习性 Pilot | 旧 per-agent V3 已失败并冻结 | 相同 Global 输入匹配不同 UAV 标签，不能进入 R3 |
| R2R Team-level 修订 | GateFix V2 已通过 | Primary decision 为 `advance_r3_flat_and_agent`，允许进入 R3 |
| R3 V1 Counterfactual Critic | 已失败并冻结 | TTC head 在固定 0.5 阈值下 recall=0.4548，未达到 0.80；旧 51000–51199 已打开 |
| R3.1 Counterfactual Critic | 已失败并冻结 | TTC 通过；Initial recall=0.7886，暴露阈值边界脆弱性 |
| R3.2 Robust Certification | 修复完成，待执行 | 独立 calibration/threshold partitions、layout-bootstrap 下界、严重度基线和全新 62000–62199 布局 |
| R4 双 Seed Pilot | R3.2 已审计通过；执行前因果特征、scalar 基线、独立 λ 校准、六 cell runner 与 Gate 已实现 | 可执行 `run_ia_mappo_r4.ps1 -Phase Prepare`；尚未产生正式 R4 结果 |
| R5 正式实验 | 两级 paired bootstrap gate 已实现 | 必须等待 R4 通过 |

本轮 R3.2 相关 IA-MAPPO V1 测试结果为 `49 passed`，CPU read-only preflight 为 PASS。31-candidate shadow、动作条件 TTC、阶段成本守恒、四分区布局隔离、layout-bootstrap 阈值、严重度基线、训练认证 smoke 和真实 transition 非干扰测试已通过；工程检查仍不构成性能证据。

## 3. 当前目录与关键文件

```text
configs/
├── active/
│   └── ia_mappo_v1_3u_6s1d_standard.yaml
├── experiments/
│   ├── ia_mappo_v1_3u_6s1d_standard_seed41.yaml
│   ├── ia_mappo_v1_3u_6s1d_scalar_material.yaml
│   ├── ia_mappo_v1_3u_6s1d_counterfactual.yaml
│   └── ia_mappo_v1_smoke_cpu.yaml
└── evaluation/
    └── ia_mappo_v1_r1_40_layouts.json

src/
├── safety/                         # 公共安全合同、shadow 与分阶段执行层
├── algorithms/intervention_critic.py
├── algorithms/intervention_data.py
├── algorithms/intervention_evaluation.py
├── algorithms/ia_mappo_gates.py
└── envs/cooperative_uav_env.py

scripts/
├── prepare_ia_mappo_r1.py
├── run_ia_mappo_r1_standard.ps1
├── evaluate_ia_mappo_r1_standard.ps1
├── collect_ia_mappo_r2_dataset.py
├── run_ia_mappo_r2_collection.ps1
├── run_ia_mappo_r2_learnability.py
├── run_ia_mappo_r2_learnability.ps1
├── summarize_ia_mappo_r2_learnability.py
├── run_ia_mappo_r2_v2_pipeline.ps1
├── derive_ia_mappo_r2r_team_dataset.py
├── run_ia_mappo_r2r_team_learnability.py
├── summarize_ia_mappo_r2r.py
├── run_ia_mappo_r2r_team_pipeline.ps1
└── analyze_ia_mappo_r0_history.py
```

历史研究总结的 MarkItDown 提取副本位于 `docs/PROJECTION_AWARE_MAPPO_PRIOR_SUMMARY_SOURCE.md`。它是研究输入材料，不是当前实现规范。

## 4. 当前算法合同

固定执行链：

```text
actor_request
→ initial_joint_guard
→ obstacle_ttc
→ final_joint_guard
→ executed
```

每个 UAV、每个阶段、每一步均记录输入/输出动作、方向余弦、修改原因、安全余量、eligibility 和修改级别。阈值为：

- Any：动作任一分量差异大于 `1e-7`，仅用于兼容诊断；
- Material：`Δv ≥ 0.05 m/s`，当前动力学下约等于动作 L2 差异 `≥ 0.10`；
- Large：`Δv ≥ 0.25 m/s`，约等于动作 L2 差异 `≥ 0.50`；
- 方向反转：before/after cosine `< 0`。

Recovery、SAFE_BRAKE 和 terminal transition 会被记录，但不作为 Actor 风险监督样本。

Shadow panel 固定包含原动作、HOLD、全体制动、四档目标方向动作和 24 个固定种子的 Sobol 联合扰动，共 31 个候选。所有候选经过同一 Initial Guard、TTC 和 Final Guard；执行前后环境、障碍 track、RNG、GRU 与 Recovery 状态必须完全一致。

## 5. 运行环境要求

正式配置使用 CUDA。VS Code 当前环境为 `D:\Anaconda\envs\env\fcrm-mapf`：Python 3.10.20、PyTorch 2.11.0+cu130，编译架构包含 RTX 5060 所需的 `sm_120`。GPU 当前占用率只记录到 manifest，不作为阻断条件。

R3.2 runner 默认固定使用该环境的 `python.exe`。Preflight 现在同时核对解释器、Torch/CUDA 版本、GPU compute capability、compiled arch list，并执行真实 CUDA kernel；误用只支持到 `sm_90` 的 base `D:\Anaconda\python.exe` 会在采集前 fail-closed。

在 PowerShell 中设置解释器：

```powershell
$Py = 'D:\Anaconda\envs\env\fcrm-mapf\python.exe'
& $Py -c "import torch; print(torch.__version__); print(torch.cuda.get_arch_list()); print(torch.cuda.get_device_capability(0))"
```

实测 `torch.cuda.get_arch_list()` 包含 `sm_120`，设备能力为 `(12, 0)`，R1 预检返回 `training_ready=true`。该环境未安装 SciPy，但活动源码、脚本和测试没有 SciPy 依赖，不影响当前研究线。

## 6. R1 训练启动命令

### 6.1 只做预检，不启动训练

```powershell
Set-Location 'E:\Code\diagnostic\ppo_env_3D'
$Py = 'D:\Anaconda\envs\env\fcrm-mapf\python.exe'
& $Py scripts/prepare_ia_mappo_r1.py --require-ready
```

预检将更新 `results/ia_mappo_v1/r1_frozen_contract.json`，并检查：

- CUDA 架构兼容；
- seed 必须严格为 23、41；
- 两组必须为 200 updates；
- intervention-aware loss 必须禁用；
- Oracle、观测 schema 和安全栈必须一致；
- 正式输出目录必须为空或不存在。

预检不检查 GPU 占用率。

### 6.2 顺序启动 seed 23 与 seed 41

```powershell
Set-Location 'E:\Code\diagnostic\ppo_env_3D'
& .\scripts\run_ia_mappo_r1_standard.ps1 `
    -PythonExecutable 'D:\Anaconda\envs\env\fcrm-mapf\python.exe'
```

该脚本先运行 fail-closed 预检，然后顺序训练 seed 23 和 seed 41。脚本不会传入 `--resume`，也不会允许旧 selector 或 task override。

### 6.3 分别启动两个 seed

如需分别在两个终端或调度器中运行：

```powershell
Set-Location 'E:\Code\diagnostic\ppo_env_3D'
$Py = 'D:\Anaconda\envs\env\fcrm-mapf\python.exe'

# Seed 23
& $Py scripts/train_ppo.py `
    --config configs/active/ia_mappo_v1_3u_6s1d_standard.yaml

# Seed 41
& $Py scripts/train_ppo.py `
    --config configs/experiments/ia_mappo_v1_3u_6s1d_standard_seed41.yaml
```

不要添加 `--resume`，不要复制旧 checkpoint、optimizer 或 normalizer。正式输出为：

```text
results/ia_mappo_v1/3u_6s1d/standard_runtimefix_v1/seed_23/
results/ia_mappo_v1/3u_6s1d/standard_runtimefix_v1/seed_41/
```

计划保存 `update_0100.pt`、`update_0200.pt` 和 `latest.pt`。

## 7. R1 固定 40-layout 评估

训练完成后，一次评估两个 seed 的 u100/u200：

```powershell
Set-Location 'E:\Code\diagnostic\ppo_env_3D'
& .\scripts\evaluate_ia_mappo_r1_standard.ps1 `
    -PythonExecutable 'D:\Anaconda\envs\env\fcrm-mapf\python.exe'
```

评估固定使用 `configs/evaluation/ia_mappo_v1_r1_40_layouts.json` 中的 40 个 layout seed，保证 seed 23/41 和 u100/u200 完全配对。输出位于：

```text
results/ia_mappo_v1/3u_6s1d/evaluation/standard_runtimefix_v1/
├── seed_23/update_0100/
├── seed_23/update_0200/
├── seed_41/update_0100/
└── seed_41/update_0200/
```

每个目录包含 episode records、summary、运行日志和 checkpoint/config provenance。评估脚本拒绝覆盖已有输出。

单独评估示例：

```powershell
& $Py scripts/evaluate_ppo.py `
    --config configs/active/ia_mappo_v1_3u_6s1d_standard.yaml `
    --checkpoint results/ia_mappo_v1/3u_6s1d/standard_runtimefix_v1/seed_23/checkpoints/update_0200.pt `
    --layout-seeds-file configs/evaluation/ia_mappo_v1_r1_40_layouts.json `
    --device cuda `
    --run-output results/ia_mappo_v1/3u_6s1d/evaluation/standard_runtimefix_v1/seed_23/update_0200 `
    --episode-records results/ia_mappo_v1/3u_6s1d/evaluation/standard_runtimefix_v1/seed_23/update_0200/episodes.jsonl `
    --summary-output results/ia_mappo_v1/3u_6s1d/evaluation/standard_runtimefix_v1/seed_23/update_0200/summary.json `
    --provenance-output results/ia_mappo_v1/3u_6s1d/evaluation/standard_runtimefix_v1/seed_23/update_0200/provenance.json
```

## 8. 诊断与验证命令

重新生成 R0 历史配对归因：

```powershell
& $Py scripts/analyze_ia_mappo_r0_history.py
```

运行 IA 定向测试：

```powershell
& $Py -m pytest -q tests/ia_mappo_v1
```

运行全部活动测试：

```powershell
& $Py -m pytest -q
```

归档恢复映射 dry-run：

```powershell
& .\archive\paused_facs_20261109\restore.ps1
```

不要添加 `--apply`，除非确实决定恢复全部历史文件。

## 9. R2 V2 数据采集与五折可学习性 Gate

R2 采集使用已经完成的 u200 checkpoint 和同一组 40 layouts。seed 41 是主数据，seed 23 只作为敏感性数据；两者分别生成 folds 和支持量报告，不应把同一 layout 的重复轨迹当作独立泛化样本。

旧的 `IAMAPPOR2DatasetV1` 将执行后 UAV 位置与执行前 obstacle track 混合到 `causal_safety_context`，不能用于 Global+Context 比较，已在两个旧结果根写入 `INVALID_CAUSAL_CONTEXT.md`。V2 context 完全由执行前 `CausalSafetySnapshotV1` 计算；五折入口会拒绝 V1。

完整执行 V2 重采集、两个 seed 的五折评估和非合并汇总：

```powershell
Set-Location 'E:\Code\diagnostic\ppo_env_3D'
$Py = 'D:\Anaconda\envs\env\fcrm-mapf\python.exe'
& .\scripts\run_ia_mappo_r2_v2_pipeline.ps1 -PythonExecutable $Py
```

输出目录为：

```text
results/ia_mappo_v1/3u_6s1d/r2_learnability/
├── seed_41_u200_contextfix_v2/
│   ├── dataset/events.npz
│   ├── dataset/snapshots.jsonl
│   ├── dataset/episodes.jsonl
│   ├── dataset/folds.json
│   ├── dataset/support.json
│   ├── dataset/manifest.json
│   ├── evaluation_summary.json
│   └── provenance.json
└── seed_23_u200_sensitivity_contextfix_v2/
```

采集器拒绝覆盖已有目录。每个 agent-step 完整保存五个 trace stage；其中 Initial Guard、Obstacle TTC、Final Guard 三个可训练阶段另外保存物理 `Δv`、方向余弦、原因、eligibility、Material/Large/反转标签。数据同时包含 raw/normalized local observation、global state、joint bounded mean action、因果 safety context、safety margin 和仅由当前可观测障碍 track 计算的 TTC。Recovery、SAFE_BRAKE 和终止 step 保留在数据中，但 `stage_eligible=false`。

`support.json` 对固定五折逐 head 检查：训练部分至少 100 个正例和 100 个负例，held-out 部分至少 20 个正例和 20 个负例。支持量不足的 head 不得进入后续 critic 训练。

现有 V1 支持量证据表明两个 seed 都只有 Initial Guard 和 Obstacle TTC 通过，Final Guard 均未通过。因此 V2 五折只训练前两个 head，Final Guard 输出 `unsupported`。不得合并 seed 23/41 补足 Final Guard。

五折比较四类输入：Local+单机动作、Global+联合动作、Global+Context+联合动作、margin/TTC 线性启发式。每轮使用 3 个 folds 拟合、1 个 fold 做 temperature calibration、1 个 fold 测试。所有 Gate 使用 layout bootstrap。

五折输出位于：

```text
results/ia_mappo_v1/3u_6s1d/r2_learnability_eval/
├── seed_41_u200_contextfix_v2/report.json
├── seed_23_u200_sensitivity_contextfix_v2/report.json
└── combined_contextfix_v2.json
```

`combined_contextfix_v2.json` 只用 seed 41 作预注册可行性决策，seed 23 作为不合并的敏感性结果。决策规则为：Local 与 Global+Context 均通过则进入 R3；只有 Global+Context 通过则实现 Observation V3 并重做 R2；Global+Context 失败则停止 Actor 风险正则路线。

后续审计确认 `combined_contextfix_v2.json` 的 weighted-BCE logits 未移除 class-prior offset，旧 ECE、阈值指标和 combined decision 已标记无效。当前只需在冻结的 V2 数据上运行：

```powershell
& .\scripts\run_ia_mappo_r2_learnability.ps1 -PythonExecutable $Py
```

`combined_calibrationfix_v3.json` 最终仍为 `stop_ia_mappo_actor_risk`。旧 per-agent 目标已冻结，不得继续加 epoch、改 fold 或调阈值。

## 10. R2R Team-level 修订

R2R 从现有 V2 快照离线派生 `IAMAPPOR2TeamDatasetV3`，不重新运行环境。每个 snapshot 只保留一条 team row，标签固定为阶段内 any-UAV Material 事件，连续严重度固定为 per-UAV cost 的 mean。action-conditioned TTC 使用 Actor raw action，所有输入保持执行前因果性。

四组比较为 Team heuristic、Flat Global MLP、Agent Encoder mean pooling、Agent Encoder mean/max pooling + safety context。Final Guard 继续 unsupported，启发式不参与晋级。

原始 V1 报告已经冻结在 `r2r_team_eval/`。其 TTC head 在 primary seed 41 的三种学习模型上均为 5/5 folds 通过，但 Initial Guard 因高 prevalence 下不可达的 AUPRC lift 条件而失败。不得覆盖或直接修改旧 JSON。

GateFix V2 复用现有 TeamDatasetV3，只重新训练相同模型并使用版本化评估合同。新 AUPRC Gate 为 `(AP - prevalence) / (1 - prevalence)`，点估计至少 0.30、layout-bootstrap 95% 下界严格大于 0。其他 Gate、fold、模型和训练设置不变。

GateFix V2 一键执行：

```powershell
Set-Location 'E:\Code\diagnostic\ppo_env_3D'
$Py = 'D:\Anaconda\envs\env\fcrm-mapf\python.exe'
& .\scripts\run_ia_mappo_r2r_gatefix_v2.ps1 -PythonExecutable $Py -Device cuda
```

结果读取：

```powershell
Get-Content results/ia_mappo_v1/3u_6s1d/r2r_team_eval_gatefix_v2/combined.json
```

完整修改合同与单独运行方式见 `docs/IA_MAPPO_R2R_GATEFIX_V2_20261109.md`。

只有 `advance_r3_flat_and_agent` 或 `advance_r3_agent_encoder` 允许进入 R3。`inspect_agent_encoder_do_not_enter_r3` 和 `stop_ia_mappo_actor_risk` 都必须停在 R2R。

## 11. R3–R5 的启动限制

当前不要直接使用以下两个配置训练 Actor：

```text
configs/experiments/ia_mappo_v1_3u_6s1d_scalar_material.yaml
configs/experiments/ia_mappo_v1_3u_6s1d_counterfactual.yaml
```

它们的 critic artifact 路径故意保持为空，活动 trainer 会 fail-closed。R2R GateFix V2 已允许进入 R3；当前必须先执行：

1. `scripts/preflight_ia_mappo_r3.py`；
2. `scripts/run_ia_mappo_r3_2.ps1` 完成全新 200-layout Shadow 数据、Twin Critic 训练与稳健认证；
3. R3.2 已满足 `certified_for_r4_lambda_calibration=true`；下一步执行独立 λ 梯度校准，只有生成 actor-certified V2 artifact 后才允许 R4 Actor 训练。

R3 V1/R3.1 文档只保留为历史协议。当前完整合同和命令见 `docs/IA_MAPPO_R3_2_ROBUST_CERTIFICATION_AND_EXECUTION_20261110.md`。

R4 入口已经实现：独立 on-policy λ calibration batch、风险梯度/PPO 梯度比例检查、三组 fresh-init 双 seed 训练、统一 40-layout paired evaluation 与 R4 Gate。执行与审计说明见 `docs/IA_MAPPO_R3_2_AUDIT_AND_R4_IMPLEMENTATION_20261110.md`。

R4 首次 scalar seed 23 在 update 85 后暴露 TTC 判别式切触点的 NaN 反向传播；已改为直接 raw-action 风险梯度 + detached action-conditioned TTC context，并强制重新 λ 校准。旧证据完整保留，重新执行步骤见 `docs/IA_MAPPO_R4_NAN_GRADIENT_FIX_20261110.md`。

在这些入口完成并通过 gate 前，不应手工填入伪造 artifact 路径，也不应绕过配置 guard。

## 12. 结果解释边界

- R0 是历史只读归因，不重新判断旧 P4。
- R1 smoke 只证明工程链路可执行，不证明算法有效。
- R1 Standard 是后续共同基线，不包含 IA 风险正则。
- 40-layout R2 只决定可学习性，不作为论文正式泛化证据。
- 介入率下降但 CFMS/Progress 下降，或 Collision/Timeout 增加，均不能宣称算法优于 Standard。
- 只有任务表现也改善时，才允许声称整体算法优于 Standard；否则只能声称执行一致性改善。
