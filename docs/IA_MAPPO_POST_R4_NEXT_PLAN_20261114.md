# IA-MAPPO：R4 失败后的下一阶段算法与实验计划

日期：2026-11-14  
适用项目：`ppo_env_3D` / `ia_mappo_v1` / `3U + 6S + 1D`  
计划状态：设计冻结草案，尚未授权新 Actor 训练

## 1. 决策摘要

R4–R4.3.6 的结果支持以下三个不同结论，不能再合并解释：

1. **安全链中的可避免介入事件是可预测的。** 现有 Twin Critic 对团队总风险仍具有稳定的 AUROC、Spearman 和风险分箱校准能力。
2. **Material 正样本的条件严重度不可稳定预测。** 当前 `probability × positive severity` 分解中的 severity head 无法在所有 policy/seed/layout 上稳定优于常数基线；R4.3.6 的 layout Group-DRO 也没有找到合格验证候选。
3. **即使介入率下降，也不代表任务表现改善。** R4、R4.1、R4.2.1 均出现过 Material Modification 下降但 CFMS、Timeout、到达率或跨 seed 一致性恶化。

因此，下一阶段不再继续 R4.3.7 severity 调参，而是建立新的研究支线：

```text
advantage_gated_ia_mappo_v1
```

核心修改限定为：

```text
从“预测并最小化绝对介入严重度”
改为“预测 Actor 动作相对可行参考动作的可避免介入概率，
仅在超过风险死区时向 Actor 提供保守风险梯度”。
```

这一修改保留 R3 的因果快照、Shadow Action、Initial/TTC 阶段归因和 Twin Critic，不恢复 FACS，不修改 Oracle 子目标，不绕过现有安全执行层。

---

## 2. 当前证据边界

### 2.1 已成立的证据

- R3.2 的不可变 Shadow 数据合同和 `stage_avoidable_costs` 定义继续有效。
- Initial Guard 与 TTC 的 Material 事件分类可以学习。
- Twin conservative total risk 在 R4.3.6 六个 policy/seed cell 中保持有效排序。
- Critic 参数冻结、动作路径风险梯度和 OOD support mask 已有实现基础。
- Final Guard 仍然缺乏支持量，继续标记为 unsupported。

### 2.2 已失败或未成立的声明

- 不能声明 positive severity 可跨布局稳定预测。
- 不能声明 Group-DRO 能修复严重度泛化。
- 不能声明降低 Material Modification 会自然提升任务成功率。
- 不能将 R4.3.6 candidate 用于 lambda 校准、Actor 训练或 Canary。
- 尚无证据允许进入 R5 正式实验。

### 2.3 冻结的失败证据

以下结果原地保留，不覆盖、不重新解释：

- `results/ia_mappo_v1/3u_6s1d/r4_pilot/`
- `results/ia_mappo_v1/3u_6s1d/r4_1_pilot/`
- `results/ia_mappo_v1/3u_6s1d/r4_2_1_pilot/`
- `results/ia_mappo_v1/3u_6s1d/r4_3_3_expected_risk/`
- `results/ia_mappo_v1/3u_6s1d/r4_3_4_joint_severity/`
- `results/ia_mappo_v1/3u_6s1d/r4_3_5_robust_severity/`
- `results/ia_mappo_v1/3u_6s1d/r4_3_6_layout_dro_validation/`

R4.3.6 的 candidate 永久保留 `diagnostic_only: true`。

---

## 3. 与现有论文的对应关系

### 3.1 SAILR：使用相对安全优势，而不是绝对风险

[Safe Reinforcement Learning Using Advantage-Based Intervention](https://proceedings.mlr.press/v139/wagener21a.html) 使用相对于 backup policy 的 safety advantage 决定是否介入。其关键思想是：动作只有在比参考策略更不安全时才应被惩罚；直接使用绝对 safety Q 容易过度保守。

对本项目的借鉴：

- 不再让 Actor 持续最小化绝对期望介入成本；
- 将 Actor 动作与同状态的可行参考动作比较；
- 只惩罚“相对参考动作额外增加的可避免介入风险”；
- 使用风险死区，避免将已经足够安全的动作继续推向 HOLD/制动。

### 3.2 SQRL / Conservative Safety Critics：预测概率比精确严重度更稳健

[Learning to be Safe: Deep RL with a Safety Critic](https://arxiv.org/abs/2010.14603) 使用 Safety Critic 预测未来失败概率，并据此约束动作；[Conservative Safety Critics for Exploration](https://arxiv.org/abs/2010.14497) 强调对安全风险采用保守估计。

对本项目的借鉴：

- Actor-facing 主输出改为 calibrated probability；
- 保留两个独立 Critic，并使用二者的较大风险；
- severity 仅作为诊断和论文附加分析，不作为 Actor artifact 的发布阻断项；
- OOD 时不传播风险梯度。

### 3.3 Recovery RL：任务优化与安全恢复职责分离

[Recovery RL](https://arxiv.org/abs/2010.15920) 将 task policy 与 recovery policy 分开，避免单一策略同时承担任务和恢复职责。

本项目已有 JointSafetyGuardV2、TTC、Final Guard 和 Recovery，因此不新增学习式 recovery policy，但保留其职责分离原则：

- MAPPO 继续优化任务回报；
- 安全执行层继续负责最终动作修正；
- Intervention Critic 只用于降低“可避免的策略—安全层不一致”，不得替代执行安全层。

### 3.4 EGPO：干预应作为约束，而不是无限减小的奖励项

[Safe Driving via Expert Guided Policy Optimization](https://openreview.net/pdf?id=KnOYrZf17CQ) 使用 guardian intervention cost 和 Lagrangian 方法限制介入频率，而不是简单把介入惩罚无限压低。

对本项目的借鉴：

- 新 Actor penalty 设置风险死区/预算；
- 第一轮仍使用冻结的注册 lambda，避免同时引入自适应乘子；
- PID-Lagrangian 只作为后续对照基线，不进入 R4.4 主算法。

### 3.5 CPO、PID-Lagrangian 与 Lyapunov Projection：作为正式基线

- [Constrained Policy Optimization](https://proceedings.mlr.press/v70/achiam17a.html) 将奖励最大化与成本约束分开；
- [PID Lagrangian](https://proceedings.mlr.press/v119/stooke20a.html) 缓解普通 Lagrangian 的振荡和超调；
- [Safe Policy Learning for Continuous Control](https://proceedings.mlr.press/v155/chow21a.html) 使用状态相关 Lyapunov 约束投影动作或参数。

这些方法说明“风险预算/约束”比固定加权和更适合安全目标，但本项目当前应先验证新的 intervention-advantage 信号是否可学，之后再决定是否加入自适应约束优化。

### 3.6 不建议直接照搬的方法

- 不直接照搬 Recovery RL 的第二策略：项目已有确定性安全执行层，新增 recovery policy 会混淆贡献。
- 不直接使用 CPO 替换 MAPPO：改动过大，且无法回答当前失败究竟来自标签还是优化器。
- 不继续使用最坏 4-layout fold 的 Group-DRO 作为模型选择：当前最坏 fold 支持量过小，导致模型选择被单个布局主导。
- 不将 distributional severity 作为下一阶段主方案：当前首先缺失的是稳定可识别的 severity 信号，而不是输出分布参数化能力。

---

## 4. 新算法：Advantage-Gated Intervention Critic

### 4.1 保持不变的输入合同

对每个唯一 team-step 使用：

```text
执行前 global observable state
+ Actor raw joint bounded action
+ causal safety context
+ current safety margins
+ per-UAV action-conditioned raw TTC
```

禁止使用未来 simulator truth。所有 reference action 查询必须基于同一个不可变因果快照。

### 4.2 监督标签

继续从完整安全链计算：

```text
C_raw(s)  = Actor raw action 在阶段 s 的 team-stage cost
C_ref(s)  = 满足推进约束的固定 Shadow 候选中的最低 team-stage cost
DeltaC(s) = max(0, C_raw(s) - C_ref(s))
```

支持阶段：

```text
s ∈ {initial_joint_guard, obstacle_ttc}
```

主要分类标签：

```text
y_avoid(s) = 1[DeltaC(s) > 1e-8]
```

保留 `DeltaC(s)`、原 positive severity 和 total avoidable cost 作为诊断字段，但不再要求 severity head 在每个 policy/seed cell 中显著优于常数中位数。

### 4.3 Critic 输出

每个 Twin member 输出：

```text
p_initial(x, a) = P(Initial 存在可避免 Material intervention | x, a)
p_ttc(x, a)     = P(TTC 存在可避免 Material intervention | x, a)
```

保守团队风险：

```text
p_s^max(x,a) = max(p_s^critic1(x,a), p_s^critic2(x,a))
```

Critic 结构继续使用共享 agent encoder、mean/max pooling、team encoder 和独立阶段 head。Final head 禁用。

### 4.4 Reference action bank

Actor 更新时只使用确定性、低成本 reference bank：

```text
HOLD
全体制动
四档 goal-direction joint action
```

不在在线 Actor 更新中运行 24 个 Sobol Shadow；Sobol 仅用于离线标签生成。

reference 必须满足原推进约束：

- 未到达 UAV 的 goal-direction velocity 不得比 raw action 低超过 `0.10 m/s`；
- 不得变为负推进；
- reference TTC 特征必须使用 reference action 重新计算；
- reference 选择和 reference risk 均 `stop_gradient`。

### 4.5 Actor-facing intervention advantage

对每个支持阶段：

```text
B_s(x) = stop_gradient(min_{a_ref in valid_reference_bank} p_s^max(x, a_ref))
A_s(x,a_actor) = relu(p_s^max(x,a_actor) - B_s(x) - eta_s)
R_adv = mean(A_initial, A_ttc)
```

Actor 目标：

```text
L_actor = L_PPO - beta * H + lambda_I * R_adv
```

固定规则：

- `eta_s` 只允许在 development layouts 上选择一次，fresh certification 前冻结；
- 风险梯度只传回 Actor raw bounded mean action；
- 不向 `log_std` 传播风险梯度；
- Critic 和 reference branch 参数完全冻结；
- OOD row、Recovery、SAFE_BRAKE 和 terminal transition 不传播 Actor 风险梯度；
- lambda 仍从 `{0.003, 0.01, 0.03}` 中选择满足风险梯度/PPO 梯度不超过 5% 的最大值；
- 保留现有参数级冲突监测，但不把它当作任务效果的替代证据。

### 4.6 训练损失

主要损失：

```text
L_cls = source-balanced weighted BCE(y_avoid, p)
```

增加 pairwise ordering 辅助损失：

```text
L_pair = softplus(-(r_raw - r_better_shadow) / tau)
```

其中 `better_shadow` 必须在真实完整安全链上具有更低 `DeltaC`，且满足推进约束。

总 Critic 损失：

```text
L_critic = L_cls + alpha_pair * L_pair
```

`alpha_pair` 只在 development 数据中选择；严重度 Huber 可以保留为非阻断 auxiliary loss，但默认第一版关闭，以避免再次由弱 severity 信号主导 encoder。

---

## 5. 修订后的认证原则

### 5.1 不再使用的阻断条件

删除以下 blocking Gate：

```text
每个 policy cell
× 每个 Critic
× 每个阶段
× MAE/Huber
的 layout-bootstrap 95% 下界必须全部 > 0
```

删除理由不是“为了让实验通过”，而是该条件认证的是内部条件严重度分解，不是 Actor 实际消费的 intervention-advantage probability。

severity 指标继续输出，不得从报告中删除。

### 5.2 主要概率认证

对 Initial 和 TTC、每个单 Critic：

- 训练与测试按 layout 隔离；
- AUROC layout-bootstrap 95% 下界 `> 0.65`；
- AUPRC lift 点估计 `>= 1.5× prevalence`；
- AUPRC lift 95% 下界 `> 1.0× prevalence`；
- ECE `<= 0.10`；
- Brier 与 NLL 均优于 prevalence baseline；
- 每个阶段测试集至少 100 个正样本和 100 个负样本。

### 5.3 Pairwise action-ordering Gate

该 Gate 直接检查 Critic 是否能给 Actor 提供正确动作方向：

```text
raw action risk > causally better valid-shadow risk
```

要求：

- pooled pairwise accuracy 的 layout-bootstrap 95% 下界 `> 0.65`；
- 六个 policy/seed cell 的点估计均 `>= 0.60`；
- 不允许任何 cell 的点估计 `< 0.50`；
- raw/shadow 输入只允许动作及 action-conditioned TTC 不同，状态快照必须逐项相等。

### 5.4 Actor-facing total risk Gate

对 `max(Critic1, Critic2)`：

- total avoidable-event AUROC 95% 下界 `> 0.65`；
- Spearman 95% 下界 `> 0.30`；
- risk-bin calibration error `<= 0.10`；
- layout mean absolute bias `<= 0.10`；
- 输出有限、非负；
- sampled-action OOD rate `<= 10%`。

Twin max 不接受 Brier/NLL/ECE 的概率认证；这些仍由两个单 Critic 分别完成。

### 5.5 梯度可用性 Gate

在冻结 Actor anchors 上检查：

- Critic 参数 `.grad` 全为空；
- Actor risk gradient 100% 有限；
- eligible active rows 中非零梯度比例 `>= 90%`；
- 小步沿负风险梯度移动后，预测风险下降比例 `>= 95%`；
- raw-versus-shadow pairwise 排序不能因梯度接口与离线接口差异而改变；
- 选定 lambda 在全部 anchors 上的风险/PPO 梯度比 `<= 5%`。

若概率认证通过但 pairwise 或梯度 Gate 失败，则停止，不允许 Actor 训练。

---

## 6. 分阶段执行计划

## R4.4A：开发数据上的目标重构

目的：只判断新目标是否可学，不训练 Actor，不打开新测试 panel。

数据：

- 使用已经打开的 R4.3.5/R4.3.6 layouts；
- 明确标记为 development-only；
- 不再把这些 layouts 用作论文测试或 fresh certification。

任务：

1. 从现有不可变 Shadow 记录派生 `y_avoid` 和 pairwise raw/better-shadow 样本；
2. 验证 team-step 唯一性和 snapshot equality；
3. 训练 probability-only Twin Critic；
4. 比较以下三个预注册模型：
   - 当前 `probability × severity`，仅作历史参考；
   - absolute avoidable-event probability；
   - proposed advantage-gated probability + pairwise loss；
5. 仅按第 5 节的 probability、pairwise、total-risk 和 gradient Gate 决策。

通过条件：proposed 模型全部通过，且相对 absolute probability 在 pairwise accuracy 上非劣；否则停止该设计。

输出建议：

```text
results/ia_mappo_v1/3u_6s1d/r4_4a_advantage_development/
```

## R4.4B：冻结协议与独立 fresh certification

只有 R4.4A 通过才执行。

固定：

- 网络结构；
- feature contract；
- reference bank；
- `eta_initial`、`eta_ttc`；
- `alpha_pair`；
- 温度校准方法；
- 全部 Gate；
- bootstrap seed 和 layout 分层方法。

采集：

- 200 个未被 R4.3.x 使用的独立 layouts；
- 六个冻结 policy/seed anchors 使用相同 layouts；
- 采集后只允许按预注册协议运行一次；
- fresh panel 不允许重新训练、选阈值或改温度。

输出建议：

```text
results/ia_mappo_v1/3u_6s1d/r4_4b_advantage_fresh_certification/
```

决策：

```text
全部通过
→ 生成 diagnostic candidate
→ 进入 R4.5 lambda/gradient calibration

任一主要 Gate 失败
→ 停止 advantage-gated IA-MAPPO
→ 不再打开第二个 fresh panel
```

## R4.5：冻结 Critic 与 lambda 校准

只在 R4.4B 通过后执行。

校准 anchors：

- Standard u100，seed 23/41；
- Standard u150，seed 23/41；
- 历史 IA u150，seed 23/41，只作 support stress anchor。

生成新 artifact schema：

```text
TwinInterventionAdvantageCriticArtifactV1
```

artifact 必须记录：

- 数据 manifest SHA256；
- 模型、normalizer、temperature SHA256；
- reference bank 和 progress filter 版本；
- selected lambda；
- 所有 anchor 梯度报告；
- `severity_blocking: false`；
- `final_guard_supported: false`。

## R4.6：双 seed Canary

三组 fresh-init：

1. Standard MAPPO + 冻结公共安全执行层；
2. 当前 absolute expected-cost IA-MAPPO，作为历史机制基线；
3. Advantage-Gated IA-MAPPO。

训练 seed：23、41。

阶段：

```text
u0–u100：风险关闭，执行 parity Gate
u101–u150：风险开启，执行 Canary Gate
u151–u200：仅在 u150 通过后继续
```

u100 parity：

- Standard 与新 IA 的 Actor、task Critic、optimizer、normalizer 完全一致；
- 评估指标一致；
- intervention penalty 为 0；
- 不允许提前激活风险。

u150 Canary 通过标准：

- pooled Material Total Modification 至少下降 5 pp；
- Initial Guard 两个 seed 均不增加；
- 两个 seed 的 Material 方向一致；
- `Delta CFMS >= -2.5 pp`；
- `Delta Progress >= -0.02`；
- Collision、Timeout 上升均不超过 2.5 pp；
- Pair/Boundary collision 为 0；
- 任一 UAV arrival rate 下降不超过 5 pp；
- Persistent Brake、Persistent Recovery、Fixed Point 不增加；
- OOD rate `<= 10%`；
- 风险梯度始终有限，p95 梯度比 `<= 5%`。

未通过即停止，不延长 updates、不替换 seed、不重新选择 lambda。

## R5：正式实验

只有 R4.6 u200 通过后恢复 R5：

- 3 组算法 × 5 个训练 seed × 200 个相同评估 layouts；
- 训练 seed 和 layout 两级 paired bootstrap；
- 主要比较为 Advantage-Gated IA 与 Standard；
- absolute expected-cost IA 作为机制消融；
- EGPO-style/PID-Lagrangian 可作为附加基线，但不得替换 Standard 主基线。

论文正式声明分级：

```text
只降低 Material，任务指标非劣
→ 声称 execution consistency improvement

同时改善 CFMS/Progress 或 Timeout
→ 才允许声称整体算法优于 Standard

只通过 Critic 认证、Canary 未通过
→ 只能报告 risk-modeling result，不得声明 IA-MAPPO 有效
```

---

## 7. 必要消融

至少包含：

1. 无 pairwise loss；
2. 无 reference subtraction，仅 absolute probability；
3. 无风险死区 `eta=0`；
4. single critic 与 twin max；
5. probability-only 与旧 `probability × severity`；
6. Initial-only 与 Initial+TTC；
7. 无 OOD gradient mask；
8. 无参数级任务冲突保护。

消融只在主算法通过 R4.6 后运行；不能用消融结果反向调整正式主 Gate。

---

## 8. 计划实现文件

建议新增而不是覆盖历史实现：

```text
src/algorithms/intervention_advantage.py
src/algorithms/intervention_advantage_certification.py
scripts/build_ia_mappo_r4_4_advantage_dataset.py
scripts/train_ia_mappo_r4_4_advantage_critic.py
scripts/certify_ia_mappo_r4_4_advantage_critic.py
scripts/calibrate_ia_mappo_r4_5_actor.py
scripts/run_ia_mappo_r4_4.ps1
scripts/run_ia_mappo_r4_6.ps1
configs/experiments/ia_mappo_v1_3u_6s1d_advantage_gated.yaml
tests/test_intervention_advantage.py
tests/test_intervention_advantage_certification.py
```

活动代码不得删除旧 R4.x 结果读取能力，但旧 artifact schema 不得被新训练模式接受。

---

## 9. 测试与验收

### 数据测试

- 每个 snapshot 只有一条 team row；
- raw/reference 仅动作和 action-conditioned TTC 不同；
- 环境、障碍 track、GRU、RNG、Recovery 状态前后完全相等；
- pairwise 标签和完整安全链重新执行结果一致；
- layout split 无 step 泄漏；
- fresh certification layout 不出现在任何历史 manifest。

### 模型测试

- temperature calibration 只使用 calibration layouts；
- severity 不参与 blocking decision；
- Twin member 独立初始化；
- reference branch 完全 detach；
- Critic 参数冻结时无参数梯度；
- Actor action 路径风险梯度非零且有限；
- sampled raw action 和 old log-prob 仍用于 PPO ratio；
- reference action 绝不替换真实 rollout action。

### 决策测试

- development-only candidate 无法被 Actor trainer 加载；
- fresh certification 失败后禁止 calibration；
- calibration 失败后禁止生成训练配置；
- Canary 失败后禁止 u200；
- 所有失败报告保留完整原因，不允许 `if/elif` 隐藏并发失败项。

---

## 10. 预计工期与停止规则

按每周 15–25 个有效工时：

| 阶段 | 预计时间 | 是否需要新环境采集 |
|---|---:|---|
| R4.4A 数据派生、模型与开发 Gate | 2–3 周 | 否 |
| R4.4B 200-layout fresh certification | 2–3 周 | 是 |
| R4.5 梯度与 lambda 校准 | 1 周 | 否 |
| R4.6 双 seed Canary | 2–4 周 | 是 |
| R5 正式实验 | 8–12 周 | 是 |

硬停止条件：

1. R4.4A pairwise Gate 失败：停止，不采集 fresh panel；
2. R4.4B fresh certification 失败：停止该算法设计；
3. R4.5 梯度方向或 support Gate 失败：停止，不训练 Actor；
4. R4.6 任一任务保持 Gate 失败：停止，不进入 R5；
5. 不允许通过增加测试 panel、删除失败 seed、延长 updates 或事后放宽阈值恢复路线。

---

## 11. 最终推荐

下一步推荐执行顺序为：

```text
冻结 R4.3.6
    ↓
R4.4A：在旧开发 layouts 上验证 advantage probability + pairwise ordering
    ↓ 仅在通过时
冻结协议、模型和全部阈值
    ↓
R4.4B：一次性 200-layout independent certification
    ↓ 仅在通过时
R4.5：冻结 Critic、校准 lambda 与 Actor 梯度
    ↓ 仅在通过时
R4.6：fresh-init 双 seed Canary
    ↓ 仅在任务与安全 Gate 同时通过时
R5 正式实验
```

该路线的论文定位应是：

> 利用完整安全执行链产生的同状态反事实，学习相对于可行参考动作的可避免介入概率，并通过保守、阈值化的 intervention-advantage 正则减少 MAPPO 与安全执行层之间的策略不一致。

它不声称替代安全层，也不声称给出形式化安全保证。
