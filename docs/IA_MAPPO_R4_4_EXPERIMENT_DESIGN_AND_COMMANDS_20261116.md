# IA-MAPPO R4.4：Advantage-Gated Critic 实验设计与启动指令

日期：2026-11-16  
研究线：`advantage_gated_ia_mappo_v1`  
环境：3U + 6S + 1D  
状态：实现与定向测试完成，实验尚未启动

## 1. 本轮整改结论

R4.4 不再修补 positive-severity head。Actor-facing 风险改为 Initial Guard 与 TTC 的可避免介入概率，并增加同状态原动作/参考动作排序监督：

```text
执行前因果状态 + raw joint action
        ↓
Twin probability Critic
        ↓
p_initial, p_ttc
        ↓
raw risk - best valid reference risk - deadzone
```

R4.4 只认证 Critic 信号，不训练 Actor、不校准 `lambda_I`、不启动 Canary。只有 R4.4B 通过后，R4.5 才能校准 PPO/risk 梯度比例与 `lambda_I`。

## 2. 对原计划的必要数据修正

旧 R4.3.5/R4.3.6 数据保存了 raw action、`best_shadow_index` 和 Shadow 成本，但没有保存最佳参考动作及其 action-conditioned TTC 特征。因此旧文件可以用于 absolute probability 诊断，不能提供可审计的 raw/reference pairwise 输入。

整改后的 R4.4A 不打开新布局，而是在已经打开的 R4.3.2 200-layout panel 上重新执行六个冻结 policy anchors，新增以下因果字段：

- 六个确定性 reference joint actions：HOLD、全体制动、四档 goal-direction；
- 每个 reference 的 action-conditioned TTC、warning、emergency、predicted clearance；
- 每个 reference 的三阶段成本、总成本与推进约束有效性；
- 原动作与 reference 共用的 snapshot/environment/GRU/RNG 不变性证据。

24 个 Sobol 动作仍只用于离线 avoidable-cost 标签，不进入在线 reference bank。

## 3. R4.4 模型合同

模型使用共享 UAV encoder：

```text
每个 UAV：
[local global-state slice,
 raw/reference action,
 per-agent margins,
 action-conditioned TTC/warning/emergency/clearance]
        ↓ shared 256-256 encoder
mean pooling + max pooling
        + causal team context + current margin
        ↓ team 256-256 encoder
Initial probability head + TTC probability head
```

固定规则：

- Final Guard head 禁用；
- severity 继续保留在原始数据中，但不参与阻断判定；
- 两个 Critic 独立初始化；
- 单 Critic 使用 temperature calibration；
- Actor-facing 评估使用两个 Critic 概率的逐阶段最大值；
- reference 分支在未来 Actor 接入时必须 `stop_gradient`；
- OOD、Recovery、SAFE_BRAKE、terminal row 不允许传播风险梯度。

训练目标：

```text
L = source-balanced weighted BCE(y_avoid, p)
  + alpha_pair * softplus(-(logit_raw - logit_better_reference))
```

预注册 `alpha_pair=0.25`。同时训练 `alpha_pair=0` 的 absolute-probability 对照。proposed 模型的 pooled pairwise accuracy 允许相对 absolute 对照最多下降 0.02；其他主要 Gate 必须全部通过。

## 4. R4.4A：开发筛查

### 数据与拆分

- 六个 anchors：warmup u100 seed 23/41、Standard u150 seed 23/41、历史 IA u150 seed 23/41；
- 使用已打开的 R4.3.2 200 layouts；
- 120 train / 20 calibration / 20 validation / 40 development-test；
- layout 隔离，不随机拆 step；
- 所有结果标记 `development_only`，不得作为正式泛化证据。

### 阻断 Gate

每个单 Critic、Initial/TTC：

- test 正负样本各不少于 100；
- AUROC layout-bootstrap 95% 下界 `>0.65`；
- AUPRC lift 点估计 `>=1.5`，95% 下界 `>1.0`；
- ECE `<=0.10`；
- Brier、NLL 优于 prevalence baseline。

Pairwise：

- pooled accuracy layout-bootstrap 95% 下界 `>0.65`；
- 六个 policy cell 均 `>=0.60`，任何 cell 不得 `<0.50`；
- snapshot、environment、GRU/RNG 指纹保持不变。

Twin conservative total：

- AUROC 95% 下界 `>0.65`；
- Spearman 95% 下界 `>0.30`；
- risk-bin 与 layout calibration error 均 `<=0.10`；
- action OOD rate `<=10%`。

梯度接口：

- Critic 参数冻结且无 `.grad`；
- action gradient 有限；
- active row 非零梯度比例 `>=90%`；
- 小步负风险梯度使预测风险下降的比例 `>=95%`。

R4.4A 通过后生成 `frozen_protocol.json`；失败则 fresh panel 保持关闭。

## 5. R4.4B：一次性 fresh certification

- 唯一 fresh panel：layout seed `75000–75199`；
- 六个 anchors 使用完全相同的 200 layouts；
- 网络、权重、temperature、`alpha_pair`、deadzone、Gate、bootstrap seed 全部冻结；
- 禁止在 fresh 数据上重训练、重新校准、选择阈值或修改协议；
- 只运行一次；失败后不允许打开第二个 fresh panel。

R4.4B 全部通过只授权进入 R4.5；仍不直接授权 Actor 训练。

## 6. 已实现文件

- `src/safety/counterfactual.py`：返回完整 Shadow panel 成本与 validity；
- `src/algorithms/intervention_r3.py`：记录六 reference 的动作、TTC 特征与成本；
- `src/algorithms/intervention_advantage.py`：probability-only structured Twin Critic、pair loss、detached reference advantage；
- `src/algorithms/intervention_advantage_certification.py`：数据构建、训练、校准、Gate、protocol freeze；
- `scripts/build_ia_mappo_r4_4_advantage_dataset.py`；
- `scripts/train_ia_mappo_r4_4_advantage_critic.py`；
- `scripts/certify_ia_mappo_r4_4_advantage_critic.py`；
- `scripts/run_ia_mappo_r4_4.ps1`；
- `configs/evaluation/ia_mappo_v1_r4_4_fresh_certification_200_layouts.json`；
- `tests/test_intervention_advantage.py`；
- `tests/test_intervention_advantage_certification.py`。

历史 R4.x 文件、结果和 artifact 均未覆盖。

## 7. 实验启动指令

在当前 VS Code PowerShell 环境中执行：

```powershell
cd E:\Code\diagnostic\ppo_env_3D
$Python = "D:\Anaconda\envs\env\fcrm-mapf\python.exe"
```

### 7.1 先运行 R4.4A Development

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_ia_mappo_r4_4.ps1 `
  -Phase Development `
  -PythonExecutable $Python `
  -Device cuda `
  -Epochs 40 `
  -BatchSize 1024 `
  -BootstrapReplicates 1000 `
  -AlphaPair 0.25
```

CUDA 预检若报告当前 PyTorch 不支持 RTX 5060 的 `sm_120`，不要继续用该 CUDA build；可直接改为：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_ia_mappo_r4_4.ps1 `
  -Phase Development `
  -PythonExecutable $Python `
  -Device cpu `
  -Epochs 40 `
  -BatchSize 1024 `
  -BootstrapReplicates 1000 `
  -AlphaPair 0.25
```

查看 R4.4A 决策：

```powershell
Get-Content .\results\ia_mappo_v1\3u_6s1d\r4_4a_advantage_development\model_comparison\development_decision.json
```

只有文件中出现：

```json
"ready_for_fresh_certification": true
```

才执行下一步。

### 7.2 通过后运行 R4.4B Fresh Certification

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_ia_mappo_r4_4.ps1 `
  -Phase FreshCertification `
  -PythonExecutable $Python `
  -Device cuda `
  -BatchSize 1024
```

若 CUDA build 不兼容，同样将 `-Device cuda` 改为 `-Device cpu`。

查看最终决策：

```powershell
Get-Content .\results\ia_mappo_v1\3u_6s1d\r4_4b_advantage_fresh_certification\certification\decision.json
```

只有：

```json
"ready_for_r4_5": true
```

才进入 R4.5 lambda/PPO-gradient calibration。

### 7.3 中断后的分阶段恢复

```powershell
# R4.4A
.\scripts\run_ia_mappo_r4_4.ps1 -Phase CollectDevelopment -PythonExecutable $Python -Device cuda
.\scripts\run_ia_mappo_r4_4.ps1 -Phase BuildDevelopment -PythonExecutable $Python -Device cuda
.\scripts\run_ia_mappo_r4_4.ps1 -Phase TrainDevelopment -PythonExecutable $Python -Device cuda

# 仅在 R4.4A PASS 后
.\scripts\run_ia_mappo_r4_4.ps1 -Phase CollectFreshCertification -PythonExecutable $Python -Device cuda
.\scripts\run_ia_mappo_r4_4.ps1 -Phase BuildFreshCertification -PythonExecutable $Python -Device cuda
.\scripts\run_ia_mappo_r4_4.ps1 -Phase CertifyFresh -PythonExecutable $Python -Device cuda
```

runner 对已完整输出会跳过；对不完整目录会停止并保留现场，不自动覆盖。

## 8. 输出位置

```text
results/ia_mappo_v1/3u_6s1d/
├── r4_4a_advantage_development/
│   ├── reference_collection/
│   ├── advantage_dataset/
│   └── model_comparison/
│       ├── absolute_probability/
│       ├── advantage_pairwise/
│       ├── development_decision.json
│       └── frozen_protocol.json
└── r4_4b_advantage_fresh_certification/
    ├── reference_collection/
    ├── advantage_dataset/
    └── certification/
        ├── fresh_certification.json
        └── decision.json
```

## 9. 当前验证状态

- 新增 Python 文件静态编译通过；
- PowerShell runner 语法解析通过；
- fresh panel schema、数量和历史 panel 隔离检查通过；
- R4.4 与既有 counterfactual/critic 定向测试共 21 项通过；
- 未启动数据采集、Critic 训练或 fresh certification。
