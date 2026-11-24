# Projection-aware MAPPO：重新设计与后续研究方案

根据压缩包中的 Projection-aware MAPPO V1、V2 Mean 实现、实验配置和 2026 年 11 月 6 日 P4 失败报告，我建议：

保留 Projection-aware MAPPO 的研究问题，但停止沿用 V1/V2 的 executed-action imitation 设计，重新设计一个以安全层介入风险为学习对象的 MAPPO 算法。

这不是继续开发 Projection-aware MAPPO V3，也不是重新启动已经失败的 FACS-LSP。

我建议建立一条独立研究路线，暂定名称：

Intervention-Aware MAPPO（IA-MAPPO，安全介入感知多智能体近端策略优化）

核心思想从：

> 让 Actor 学习模仿安全层最终执行的动作。

转变为：

> 让 Actor 学习识别哪些动作会触发不必要或高代价的安全层介入，并在保持任务收益和安全性的前提下，主动减少这类介入。

两种方法的差别很大。

前一种是在学习安全层给出的动作结果；后一种是在学习安全层为什么需要修改动作，以及如何减少这种修改。

我认为后一种更值得作为下一阶段的研究方向。

# 一、先总结 Projection-aware MAPPO 已经做了什么

压缩包中实际上有两代设计，不能混为一谈。

## 1.1 V1：Executed-action NLL

Standard MAPPO 的执行过程是：

Standard MAPPO

Actor 根据观测输出动作：

\\[ a\_{\text{request}}=\tanh(z) \\]

Joint Guard → TTC → Final Guard

\\[ a\_{\text{exec}}=S(s,a\_{\text{request}}) \\]

环境真正执行 \\(a\_{\text{exec}}\\)

普通 PPO 仍然根据 Actor 实际采样的动作计算：

\\[ L\_{\text{PPO}} \\]

V1 发现 Actor 不知道安全层最后执行了什么，于是增加：

\\[ z\_{\text{exec}} = \operatorname{atanh}(a\_{\text{exec}}) \\]

\\[ L\_{\text{total}} = L\_{\text{PPO}} + \lambda\_{\text{exec}}L\_{\text{NLL}} \\]

其中 NLL 让 Actor 的动作分布拟合安全层最终执行动作。

V1 的问题：NLL 可以通过增大策略方差降低。

实验中：

| 指标                  | Standard | Projection V1 |
| ------------------- | -------- | ------------- |
| Action Std（Seed 23） | 0.426    | 0.893         |
| Action Std（Seed 41） | 0.436    | 0.890         |
| u800 CFMS           | 92.75%   | 88.50%        |
| Total Modification  | 96.33%   | 96.48%        |

结果表明，辅助损失下降了，但安全层修改率没有改善，CFMS 反而下降。

因此 V1 的失败原因不是研究动机错误，而是损失函数存在方差捷径。

## 1.2 V2 Mean：只模仿 Actor 均值动作

V2 针对 V1 的问题进行了修改：

\\[ a\_{\text{mean}}=\tanh(\mu\_\theta) \\]

\\[ L\_{\text{exec}} = \operatorname{SmoothL1} (a\_{\text{mean}},\operatorname{stopgrad}(a\_{\text{exec}})) \\]

总损失：

\\[ L\_{\text{total}} = L\_{\text{PPO}} + \lambda\_{\text{exec}}L\_{\text{exec}} \\]

它解决了一个明确问题：

- 辅助损失不再向 `log_std` 传播梯度；
- PPO ratio 仍然使用原始采样动作和 old log-prob；
- 只对符合资格且发生干预的样本进行辅助监督。

但它仍然存在根本性缺陷：

Actor 学习的是接近最终执行动作，而不是学习进入安全执行层的可行控制区域。

V2 的 P4 失败正好证明了这一点。

# 二、为什么 V2 Mean 仍然不够？

我认为有四个主要问题。

## 2.1 问题一：最终执行动作不是稳定的理想监督目标

当前最终执行动作受到多个机制影响：

- Recovery Overlay；
- Initial JointSafetyGuard；
- TTC Projection；
- Final JointSafetyGuard；
- Safe Brake / HOLD 等状态机制。

因此：

\\[ a\_{\text{exec}} = S(s,a\_{\text{request}},q) \\]

其中 \\(q\\) 代表安全状态机及执行上下文。

同一个请求动作，在不同团队状态、不同动态障碍位置、不同恢复状态下，可能得到不同的最终动作。

这意味着 V2 在学习一个由整个执行系统共同决定的目标。

但 Actor 不一定能从自己的局部观测中推断出所有这些因素。

## 2.2 问题二：动作接近不代表安全介入减少

V2 u200 的结果尤其说明这个问题。

### V2 Mean u200 关键结果

- `+0.252`：Seed 23 的 Executed/Raw P50 改善；
- `+0.255`：Seed 41 的 Executed/Raw P50 改善；
- `1.56 pp`：Total Modification 改善；
- `约 95%`：剩余 Total Modification。

这说明 Actor 均值确实更接近执行动作，但安全层依然几乎每一步都在修改动作。

因此：

\\[ \text{Action Alignment} \neq \text{Intervention Reduction} \\]

这是 V2 最重要的设计缺陷。

## 2.3 问题三：安全层介入是多阶段的，但 V2 把它压缩成了一个动作差值

例如：

| 安全执行阶段              | 修改原因            |
| ------------------- | --------------- |
| Initial Joint Guard | UAV 间距或边界约束     |
| TTC                 | 动态或静态障碍物碰撞风险    |
| Final Joint Guard   | TTC 修改后出现新的联合风险 |
| Recovery            | 已经进入异常状态，需要恢复   |
| Safe Brake          | 当前状态下需要制动       |

这些修改的性质不同。

不能把它们全部变成：

\\[ \\|a\_{\text{exec}}-a\_{\text{request}}\\| \\]

然后让 Actor 统一模仿。

例如：

- Guard 对请求动作做轻微修正，不一定意味着严重风险；
- TTC 修改可能意味着真实碰撞风险；
- Recovery 可能已经不是 Actor 正常控制的状态；
- Final Guard 可能是在修复前面 TTC 修改产生的联合冲突。

它们应该分别研究。

## 2.4 问题四：Actor 看不到安全层的全部决策信息

这是我从源码中发现的一个值得优先解决的问题。

当前 `build_local_observations()` 的基础特征包括：

- 自身位置；
- 自身速度；
- 最终目标；
- 局部子目标；
- 最近一架队友的相对位置；
- 已跟踪障碍物的位置和相对速度。

但是 `JointSafetyGuardV2` 使用的是整个团队的位置和速度，检查所有 UAV 之间的预测轨迹。

当前 Actor 的局部观测没有显式提供全部队友的相对速度，也没有完整表达其他队友的相互关系。

因此：

\\[ o_i \not\Rightarrow \text{Complete Joint Safety Context} \\]

即使 V2 训练得再充分，Actor 也可能无法从当前局部观测中推断出 Guard 为什么修改了某个动作。

这不是单纯增加训练步数能够解决的问题。

# 三、我建议的新设计：Intervention-Aware MAPPO

这里是我建议重点研究的算法。

它与 V2 Mean 最大的区别是：

不再要求 Actor 输出的动作接近安全层最终执行的动作，而是让 Actor 通过学习安全介入风险，主动减少容易触发安全层修改的动作。

## 3.1 先重新定义算法的研究目标

新算法不应该以：

\\[ \min \\|a\_{\text{mean}}-a\_{\text{exec}}\\| \\]

作为核心目标。

而应该关注：

\\[ \min P(\text{Material Intervention}\mid s,a) \\]

但这里不能简单地要求所有安全层介入都消失。

因为安全层介入有时是必要的。

例如：

- UAV 已经处于障碍物附近，必须减速；
- 两架 UAV 正在相互接近，必须调整方向；
- TTC 发现即将发生碰撞，必须紧急避让。

如果 Actor 为了减少介入而学会一直减速，虽然动作修改率可能下降，但任务完成率也会下降。

因此，真正需要优化的是：

在保持任务推进和安全性的前提下，减少可以通过策略本身避免的、有实质影响的安全层介入。

这是新算法最重要的设计原则。

## 3.2 第一项改进：把一个最终动作差值拆成多个阶段的介入标签

当前环境已经记录了：

- `recovery_overlay`
- `initial_joint_guard`
- `obstacle_ttc`
- `final_joint_guard`
- `total`

但目前的 Projection-aware 训练接口仍然主要使用最终的 `executed_action` 和单一 `intervention_mask`。

我建议下一版建立完整的阶段监督数据。

首先定义：

\\[ a_t^{(0)}=a_t^{request} \\]

经过第一个执行阶段：

\\[ a_t^{(1)}=S_1(s_t,a_t^{(0)}) \\]

再经过第二个阶段：

\\[ a_t^{(2)}=S_2(s_t,a_t^{(1)}) \\]

依次得到最终动作：

\\[ a_t^{(K)}=a_t^{exec} \\]

每个阶段单独记录：

\\[ \Delta_t^{(k)} = a_t^{(k)}-a_t^{(k-1)} \\]

这样就能知道究竟是谁修改了动作。

### 修改率必须重新定义

目前代码使用：

\\[ \\|a\_{\text{exec}}-a\_{\text{request}}\\|>10^{-7} \\]

判断是否发生修改。

这个阈值可以保留为历史 Any Modification 指标，但不适合成为新算法唯一的优化目标。

我建议建立三级统计：

| 指标                    | 定义         | 用途     |
| --------------------- | ---------- | ------ |
| Any Modification      | 任意分量发生数值修改 | 工程诊断   |
| Material Modification | 超过物理意义阈值   | 核心优化指标 |
| Large Modification    | 超过较大控制影响阈值 | 严重介入指标 |

物理阈值应根据动作缩放、最大加速度和控制周期推导。

例如：

\\[ \Delta v=a\_{\max}\cdot dt\cdot\\|\Delta a\\| \\]

如果动作修改对应的速度变化低于预先规定的控制分辨率，就不应该与大幅避障动作使用同一种统计口径。

此外，还要记录方向变化：

\\[ C\_{\text{direction}} = \frac{a\_{\text{before}}\cdot a\_{\text{after}}} {\\|a\_{\text{before}}\\|\\|a\_{\text{after}}\\|+\epsilon} \\]

因为动作幅值变化和动作方向被反转是两种不同的问题。

### 每个 UAV 必须独立记录

当前代码存在一个需要优先修正的地方：

`execution_stage` 是团队级单一阶段标签，Trainer 会将该标签映射到所有 Agent。

`execution_target_eligible` 也使用团队级的 Recovery、Safe Brake 等状态决定是否有资格训练。

这可能造成：

一架 UAV 进入 Scoped Recovery 后，其他没有进入 Recovery 的 UAV 也被一起排除。

因此新版本应该使用：

\\[ M\_{i,k,t} \\]

表示第 \\(i\\) 架 UAV 在第 \\(k\\) 个执行阶段、第 \\(t\\) 步是否符合监督条件。

这样才能避免团队级标签污染单机样本。

## 3.3 第二项改进：增加 Intervention Critic（安全介入预测器）

这是整个新算法最重要的模块。

我建议先不要直接让 Actor 增加新的辅助损失，而是先训练一个独立的 Intervention Critic。

它不是负责控制 UAV，而是负责预测：

给定当前环境状态和 MAPPO 请求动作，安全执行层可能会产生什么类型、什么程度的介入？

输入：

\\[ (x_t,\mathbf a_t^{request},q_t) \\]

其中：

- \\(x_t\\)：团队全局状态；
- \\(\mathbf a_t^{request}\\)：三架 UAV 的联合请求动作；
- \\(q_t\\)：当前安全执行状态和必要的历史上下文。

输出：

\\[ \begin{aligned} P_G &= P(\text{Material Initial Guard})\\\ P_T &= P(\text{Material TTC})\\\ P_F &= P(\text{Material Final Guard}) \end{aligned} \\]

同时预测对应的修改严重程度：

\\[ D_G,D_T,D_F \\]

这样形成阶段级的风险预测。

## Intervention Critic 架构

输入

- Global State
- Joint Requested Actions
- Causal Obstacle Tracks
- Safety State Context

Shared Encoder

提取团队状态与联合动作特征。

Guard Head

介入概率

修改程度

TTC Head

介入概率

修改程度

Final Guard Head

介入概率

修改程度

第一版暂时不把 Recovery 和 Safe Brake 作为 Actor 需要模仿的训练目标。

因为这些状态下，Actor 可能已经不再拥有实际控制权。

它们仍然需要被完整记录，作为任务活性和安全状态机的独立评价指标。

### 为什么使用全局状态，而不是只使用 Actor 局部观测？

因为 JointSafetyGuardV2 是联合安全执行器。

它需要知道：

- 所有 UAV 的位置；
- 所有 UAV 的速度；
- 所有 UAV 的请求动作；
- UAV 之间的相对距离；
- 地图边界和安全余量。

单架 UAV 的局部观测并不包含全部这些信息。

当前 MAPPO 的 Centralized Critic 已经使用团队全局状态。

因此，可以采用 CTDE 思路：

- MAPPO Actor 仍然使用局部观测；
- Intervention Critic 训练时使用团队全局状态；
- Intervention Critic 只参与训练，不参与最终部署；
- 原有确定性安全执行层仍然负责最终动作安全。

这既保留了分布式 Actor，也避免要求每架 UAV 在部署时额外运行一个全局风险预测网络。

但有一个重要前提：必须先验证全局状态预测器和局部观测预测器的差异。如果局部观测缺少的信息对介入判断至关重要，中央 Critic 只能提供训练期指导，不能保证 Actor 能针对每个局部状态准确避免介入。

## 3.4 第三项改进：从模仿执行动作改成惩罚介入风险

这是新算法与 V2 Mean 最本质的区别。

原来的 V2：

\\[ L\_{\text{exec}} = \operatorname{SmoothL1} (a\_{\text{mean}},a\_{\text{exec}}) \\]

新的 IA-MAPPO：

\\[ L\_{\text{intervention}} = \mathbb E\_{\pi\_\theta} \left[ \hat C\_{\text{intervention}} (x_t,\mathbf a_t) \right] \\]

其中：

\\[ \hat C\_{\text{intervention}} = w_G P_GD_G+ w_T P_TD_T+ w_F P_FD_F \\]

总损失：

\\[ \boxed{ L\_{\text{IA-MAPPO}} = L\_{\text{PPO}} -\beta H(\pi\_\theta) +\lambda_I L\_{\text{intervention}} } \\]

各部分含义：

| 损失                | 作用          |
| ----------------- | ----------- |
| PPO Loss          | 保持任务奖励优化    |
| Entropy           | 保留策略探索      |
| Intervention Loss | 减少可避免的安全层介入 |

需要强调：

PPO 的 ratio 仍然只使用真正采样的 raw action 和原有 old log-prob。

安全层执行动作不再被拿来替代 PPO 的采样动作，也不再作为 Actor 的直接回归目标。

### 进一步解决 V2 的另一个问题：不要惩罚不可避免的介入

我建议再引入一个很重要的概念：

Progress-preserving Avoidable Intervention（保持任务推进条件下的可避免介入）。

对于同一个环境状态，使用小规模 Shadow Action 集合，测试不同请求动作在同一安全栈下的介入成本。

定义：

\\[ C\_{\text{avoid}}(s,a) = \max \left( 0,\\, C\_{\text{int}}(s,a) - \min\_{a'\in\mathcal A\_{\text{cf}}} C\_{\text{int}}(s,a') \right) \\]

但这个最小值不能在任意动作集合上计算。

候选动作 \\(a'\\) 必须：

1. 通过当前安全执行合同；
2. 满足规定的目标方向推进要求；
3. 不依赖未来真实障碍物状态；
4. 与被比较动作处于相同环境状态和团队上下文。

否则，最小介入动作可能就是完全不动的 HOLD。

这样会重新引入 Timeout 问题。

因此，新算法真正优化的应当是：

\\[ L\_{\text{intervention}} = \mathbb E[C\_{\text{avoid}}] \\]

而不是无条件最小化所有安全层修改。

我建议先在 Trace-only 阶段验证这种可避免介入是否能够可靠定义，再决定是否把它加入第一版训练目标。不要在没有证据时同时增加多个辅助损失。

# 四、训练新算法之前，必须先解决数据和可观测性问题

我认为这是比设计网络更重要的一步。

如果训练标签本身不稳定，或者 Actor 看不到产生安全介入的关键状态，那么 Intervention Critic 再复杂也没有意义。

## 4.1 首先进行历史 320 个 Episode 的只读归因

P4 失败报告中已经有一组非常重要的数据：

| Standard → Projection V2             | 数量 |
| ------------------------------------ | -- |
| Timeout → Success                    | 16 |
| Success → Dynamic Obstacle Collision | 4  |
| Success → Timeout                    | 6  |

这说明 V2 不只是没有收益，而是改变了不同布局上的失败类型。

我建议先将这 26 个关键布局单独提取出来，逐步分析：

- 修改前后 Actor 请求动作；
- Initial Guard 修改；
- TTC 修改；
- Final Guard 修改；
- SAFE_BRAKE；
- Recovery；
- 首次碰撞时间；
- 任务推进程度；
- 最终失败原因。

目标是回答：

哪些安全介入是真正应该由 Actor 提前避免的？哪些是当前状态下必须发生的安全修正？

如果不先回答这个问题，就无法正确设计 Intervention Critic 的监督标签。

## 4.2 做一个 40-layout 的可学习性审计

这是我建议新算法的第一个正式研究实验。

不要立即训练 IA-MAPPO。

先收集 40 个独立布局上的完整执行轨迹，建立以下对照：

| 预测模型    | 输入                 | 验证目的             |
| ------- | ------------------ | ---------------- |
| Model A | Actor 局部观测 + 请求动作  | Actor 能否自己识别介入风险 |
| Model B | 团队全局状态 + 联合请求动作    | 中央预测器能否识别介入风险    |
| Model C | 全局状态 + 安全状态 + 联合动作 | 安全状态上下文是否提供额外信息  |

三个模型分别预测：

- Material Initial Guard；
- Material TTC；
- Material Final Guard。

重点评价：

- AUROC；
- AUPRC；
- Brier Score；
- ECE；
- 关键风险事件召回率；
- 跨布局泛化；
- 相邻时刻标签稳定性。

其中 AUPRC 很重要，因为某些执行阶段的 Material Intervention 可能是不平衡事件。

### 审计结果如何决定下一步？

情况 A

局部模型和全局模型都能准确预测。

可以直接研究 Intervention-aware Actor Loss。

情况 B

全局模型准确，局部模型不准确。

说明当前 Actor 观测存在信息不足。考虑增加少量队友相对速度和安全余量特征，建立新的 Observation Schema，并重新训练 Standard 与 IA 两组基线。

情况 C

两种模型都无法稳定预测。

说明当前介入标签可能存在强烈的状态依赖、阶段混合或不可预测性。停止学习式介入优化，不进入 Actor 训练。

我尤其建议重点检查当前的队友观测。

`_nearest_teammate_relative()` 只返回最近队友的相对位置，没有返回其相对速度，也没有完整提供所有队友的相对状态。

而 JointSafetyGuardV2 使用全部 UAV 的位置和速度。

这意味着 Observation 与 Safety Constraint 之间可能存在信息缺口。

如果审计确认这个缺口是主要原因，就应该先解决观测问题，而不是让网络继续拟合不可预测的标签。

## 4.3 修正训练数据的四个问题

在开始训练 Intervention Critic 之前，我建议完成以下修改。

| 当前问题                  | 新设计                               |
| --------------------- | --------------------------------- |
| 单一最终执行动作标签            | 每个安全阶段独立记录输入、输出和动作差值              |
| `1e-7` 二元修改判断         | Any / Material / Large 三级指标       |
| 团队级 Eligibility       | UAV × Stage × Step 三级 Eligibility |
| 团队级单一 Execution Stage | 每架 UAV 的阶段级干预记录                   |

此外，所有 Shadow 数据都必须来自相同的因果环境快照。

不能在 Shadow 过程中修改：

- 真实环境位置；
- 动态障碍状态；
- GRU Hidden；
- 随机数状态；
- Recovery 状态机。

也不能用未来真实障碍物位置生成训练标签。

# 五、完整的 IA-MAPPO 应该怎样工作？

未来的算法流程应该是：

# Intervention-Aware MAPPO

建议研究架构

1\. Local Observation

每架 UAV 根据自己的观测和 GRU Hidden 生成动作分布。

2\. MAPPO Actor

生成实际有界请求动作：

\\[ a\_{\text{request}}=\tanh(z) \\]

3\. Intervention Critic（训练期）

预测各安全阶段的 Material Intervention Risk。

对当前策略动作提供可微的风险估计。

4\. MAPPO + Intervention-aware Optimization

PPO 主目标保持不变，增加经过约束的介入风险正则项。

5\. 原有确定性安全执行层

Initial Joint Guard → TTC → Final Joint Guard → Recovery。

6\. Environment Outcome

记录实际介入、碰撞、推进、Timeout 和 CFMS。

这里有一个重要区别：

Intervention Critic 只用于训练，不是在线安全控制器。

部署时：

- 只需要 MAPPO Actor；
- 原有安全层继续运行；
- 不需要增加 Intervention Critic 的推理负担；
- 不允许因为 Critic 预测风险较低就绕过 Guard 或 TTC。

## 5.1 如何避免 Intervention Critic 被 Actor 欺骗？

这是新设计必须考虑的另一个问题。

Actor 会根据风险预测器的梯度改变自己的动作。

如果 Intervention Critic 预测错误，Actor 可能找到预测器认为安全、但实际 Guard 会修改的动作。

因此需要：

1. Intervention Critic 使用独立训练、验证、测试布局。
2. Actor 更新时冻结 Critic 参数，但保留对动作输入的梯度。
3. 使用多个独立模型或不确定性估计，避免高不确定性区域提供过强梯度。
4. 对超出训练数据分布的动作不使用激进的风险奖励或惩罚。
5. 每次真实 rollout 都用确定性 Guard/TTC 的实际介入结果验证 Critic 预测。

尤其是第五点：

模型预测的介入率不是实际介入率。

新算法的最终评价必须来自真实执行。

## 5.2 训练目标还需要一个防止 Timeout 的约束

我建议不要简单地使用一个固定的 \\(\lambda\_{\text{exec}}\\)。

V1/V2 已经说明，仅通过辅助梯度与 PPO 梯度的比例限制，无法保证行为目标正确。

可以把训练目标写成约束优化：

\\[ \max\_\theta J\_{\text{task}}(\theta) \\]

满足：

\\[ \mathbb E[C\_{\text{avoidable}}]\leq \kappa \\]

同时保持：

\\[ J\_{\text{progress}}\geq P\_{\min} \\]

其中：

- \\(J\_{\text{task}}\\)：原有 MAPPO 任务目标；
- \\(C\_{\text{avoidable}}\\)：可避免的实质介入成本；
- \\(\kappa\\)：预注册的介入预算；
- \\(P\_{\min}\\)：最低任务推进要求。

可以使用缓慢更新的拉格朗日乘子控制介入惩罚：

\\[ \lambda\_{t+1} = \operatorname{clip} \left[ \lambda_t+\eta(C_t-\kappa), 0,\lambda\_{\max} \right] \\]

但这一机制也需要单独验证稳定性。

我建议第一版先完成介入风险预测和固定权重的小规模 Pilot，再研究自适应乘子，不要一开始同时引入复杂的双层优化。

# 六、后续实验应该如何安排？

我建议把新研究分成六个阶段。

注意：以下是我建议建立的新实验路线，不是当前压缩包已经完成的实验。

| 阶段 | 研究内容                            | 通过后进入  |
| -- | ------------------------------- | ------ |
| R0 | 冻结 V1/V2 失败结果，完成 320 Episode 归因 | R1     |
| R1 | 修正阶段级介入遥测                       | R2     |
| R2 | 40-layout 可学习性审计                | R3     |
| R3 | 训练独立 Intervention Critic        | R4     |
| R4 | IA-MAPPO 两 Seed Pilot           | R5     |
| R5 | 五 Seed 正式对照实验                   | 论文结果评估 |

## 6.1 R0：冻结原有失败结果

必须保留：

- 4S+1D Projection V1 失败；
- 6S+1D Projection V2 Mean P4 失败；
- 原始 Standard Parent；
- 两个 Seed 的全部结果；
- 原始 Gate；
- Checkpoint、Normalizer 和 SHA。

不要将 IA-MAPPO 伪装成 V2 的一个小版本。

它应该有独立的配置、结果目录、Checkpoint 和实验协议。

建议选择 3U/6S+1D 作为第一阶段实验环境，因为 V2 Mean 已经在这个环境中完成了 P4 对照。

但 Standard 和新 IA-MAPPO 必须在新协议下重新 fresh-init，不能继续从 V2 的 u200 Checkpoint 接着训练。

## 6.2 R1：先修正数据接口

这是必须完成的工程任务：

- per-agent stage delta；
- per-stage eligibility；
- Any/Material/Large 修改率；
- 安全约束余量；
- Intervention Reason；
- Causal Safety Context。

不能先训练，然后再补充归因数据。

## 6.3 R2：40-layout 可学习性审计

这是第一道真正的算法 Gate。

我建议要求：

- Material Guard 和 TTC 事件有足够样本；
- 局部与全局预测器的差异能够解释；
- 独立布局上的预测显著优于基线；
- 预测概率具备合理校准；
- 不依赖 Simulator Truth；
- 关键介入事件的召回率满足预注册要求。

如果这一阶段不通过，就暂停 IA-MAPPO，不训练 Actor。

## 6.4 R3：训练 Intervention Critic

只有 R2 通过才训练。

这里不需要重新训练 MAPPO。

先冻结一个新的 Standard MAPPO Parent，采集：

\\[ (x_t,\mathbf a_t^{request}) \rightarrow (y_G,y_T,y_F,d_G,d_T,d_F) \\]

然后训练 Intervention Critic。

重点是预测精度和泛化能力，而不是先追求 CFMS 改善。

## 6.5 R4：两 Seed Pilot

建议三个对照组：

| 实验组 | 算法                                                   | 作用           |
| --- | ---------------------------------------------------- | ------------ |
| A   | Standard MAPPO                                       | 基准           |
| B   | Standard + Intervention Critic（只训练预测器，不给 Actor 风险梯度） | 验证预测器本身      |
| C   | IA-MAPPO                                             | 验证风险正则是否改善行为 |

这样可以把两个问题分开：

- Critic 是否预测准确？
- Critic 是否能够通过 Actor 优化产生实际收益？

建议先沿用 P4 的两个 Pilot Seed 和 40 个配对布局作为初筛，但正式统计不能只依赖两个 Seed。

### 我建议的 Pilot 门槛

以下是新的建议门槛，并非历史 V2 的既有 Gate：

| 指标                          | 建议要求             |
| --------------------------- | ---------------- |
| Material Total Modification | Pooled 至少下降 5 pp |
| Material Initial Guard      | 两个 Seed 均下降      |
| CFMS                        | 不低于 Standard     |
| Progress                    | 不低于 Standard     |
| Obstacle Collision          | 不增加              |
| Timeout                     | 不增加              |
| Persistent Brake            | 不增加              |
| Persistent Recovery         | 不增加              |
| Fixed Point                 | 不增加              |
| Pair/Boundary Collision     | 0                |

5 pp 是值得作为新 Pilot 目标的幅度，但正式预注册前还应结合 Material Modification 的基线发生率和测量误差校准。

注意：这里使用的是 Material Modification，而不是原来对任意浮点差异都计数的 Any Modification。

## 6.6 R5：正式多 Seed 实验

只有 Pilot 通过后才开始。

建议：

- 至少 5 个训练 Seed；
- 200 个配对评估布局；
- Standard 与 IA 使用相同布局；
- 不同实验组保持相同安全栈；
- 使用配对 Bootstrap；
- 同时报告 CFMS、Progress、Collision、Timeout、Material Modification。

最终需要区分两种论文结论：

结论 A：执行一致性改善

Material Modification 显著下降，任务表现非劣。

结论 B：整体算法性能改善

除了执行一致性改善，CFMS 也提高，Collision 和 Timeout 不恶化。

只有第二种才能支持 IA-MAPPO 优于 Standard MAPPO 的整体性能主张。

# 七、这条新路线与原 Projection-aware MAPPO 有什么区别？

| 对比项目                | V1       | V2 Mean   | 建议 IA-MAPPO         |
| ------------------- | -------- | --------- | ------------------- |
| 学习目标                | 模仿执行动作分布 | 模仿执行动作均值  | 预测并减少可避免的介入         |
| 核心机制                | NLL      | SmoothL1  | Intervention Critic |
| 是否学习执行动作            | 是        | 是         | 否                   |
| 是否有阶段级风险预测          | 否        | 否         | 是                   |
| 是否考虑联合安全上下文         | 不充分      | 不充分       | CTDE 全局上下文          |
| 是否保留 PPO 原始采样 ratio | 是        | 是         | 是                   |
| 是否保留确定性安全层          | 是        | 是         | 是                   |
| 是否增加部署推理负担          | 否        | 否         | 否，Critic 仅训练期使用     |
| 当前实验结论              | FAILED   | P4 FAILED | 尚未实验                |

我认为 IA-MAPPO 的创新不应该再定位为简单的动作投影模仿，而应该定位为：

安全执行介入风险感知的多智能体策略优化。

它有三个明确的研究贡献：

1. Stage-aware Intervention Modeling：对多阶段安全执行介入进行分解和预测。
2. Progress-preserving Intervention Optimization：减少可以避免的安全介入，同时保留任务推进能力。
3. Centralized Safety-aware Policy Training：利用团队级安全信息训练分布式 Actor，但不增加部署时的安全控制依赖。

# 八、我认为最合理的执行顺序

现在不应该马上修改 `trainer.py` 开始训练。

建议按照以下优先级推进：

## 下一阶段行动清单

建议立即开展

1. 冻结 FACS 和 Projection V1/V2 历史结果

   不覆盖原始 Gate，不恢复旧训练。
2. 完成 V2 P4 的 320 Episode 配对归因

   找出 16 个被挽救的 Timeout，以及 10 个由成功转为碰撞或 Timeout 的布局。
3. 修改执行介入遥测

   建立 per-agent、per-stage 的 Material Intervention 数据合同。
4. 开展 40-layout 可学习性审计

   比较局部观测和全局状态对安全介入的预测能力。
5. 决定是否训练 Intervention Critic

   只有标签稳定、预测能力足够才继续。
6. 进行 IA-MAPPO 双 Seed Pilot

   先证明介入下降能够转化为实际任务收益。
7. 正式多 Seed 验证

   通过后才形成新的算法性能结论。

## 九、什么时候应该再次停止这条研究路线？

我建议设置四个停止条件：

| 失败情况                                        | 决策               |
| ------------------------------------------- | ---------------- |
| Material Intervention 标签不可稳定预测              | 停止学习式介入优化        |
| Actor 局部观测无法表达关键安全信息，且观测改进仍无效               | 停止当前分布式 Actor 方案 |
| Intervention Critic 准确，但 Actor 优化后真实介入率没有下降 | 停止当前风险正则方案       |
| 介入率下降，但 CFMS 下降或 Collision/Timeout 增加       | 判定算法无效，停止主线      |

尤其不能再次出现：

动作一致性提高了，就认为算法成功了。

这次应该把真正的任务成功率和安全性放在第一位。

# 十、最终建议

我对当前 Projection-aware MAPPO 的评价是：

研究问题成立，V1 的损失函数设计失败，V2 修复了方差捷径，但仍然使用了不合适的学习目标。

因此，我不建议继续修改：

- NLL 的形式；
- SmoothL1 的参数；
- `lambda_exec` 的取值；
- 训练 Update 数量；
- PPO 的 Clip 参数。

这些调整没有触及 V2 的主要根因。

我建议把研究重心转到：

\\[ \boxed{ \text{Intervention Prediction} \rightarrow \text{Avoidable Intervention Optimization} \rightarrow \text{Safe MAPPO Execution} } \\]

其中最重要的第一步不是训练网络，而是证明安全层介入具有可学习性，并且 Actor 能够获得足够的信息来避免它。

我的最终决策建议：正式暂停 FACS/STAR，将后续研究独立命名为 IA-MAPPO。先做历史归因、阶段级数据修正和 40-layout 可学习性审计。只有这三项产生正面证据，才开始设计新的 Actor 优化目标。

这样可以避免重复 V1/V2 的问题，也能让下一轮研究真正回答一个更有价值的问题：

多智能体策略能否在不削弱安全保护、不牺牲任务成功率的前提下，学会主动产生更容易被安全执行系统接受的动作？
