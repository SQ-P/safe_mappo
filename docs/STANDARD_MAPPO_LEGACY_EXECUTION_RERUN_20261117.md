# Standard MAPPO 历史执行合同三种子重训说明（2026-11-17）

## 决策

重新训练 seed 23、41、67，统一预算 800 updates，不自动启动实验。

## 可执行合同

- 任务：3U+8S+2D；
- obstacle slots：6；
- 单体观测维度：52；
- 全局状态维度：156；
- 奖励、Oracle、MAPPO 网络和 PPO 超参数与历史运行保持一致；
- TTC：`causal_ttc_progress_projection_v2`；
- Recovery：`star_scoped_recovery_v4`；
- candidate feasibility：`guard_aligned_observable_v2`；
- checkpoint：100、200、400、600、800；
- 周期评估：每100 updates、10 episodes、seed base 111000；
- 输出写入新的 `standard_mappo_3u_8s2d_legacy_execution_v1` 目录，不覆盖旧三种子结果。

## 与字节级历史配置的唯一已知差异

当前 checkout 已删除并 fail-closed 禁用了历史高层 `facs_star_v0`
selector，无法仅凭保存的 manifest 可靠重建其候选生成逻辑。历史独立确认
记录显示该次评估没有 selector intervention，因此本次采用证据支持的最近
可执行复现：selector 保持禁用，将历史 Guard、progress-aware TTC、STAR
Recovery 和相关审计字段映射到当前共享安全执行接口。

因此，新运行必须称为“历史执行合同复现”，不能声称 task fingerprint 与
旧 checkpoint 完全相同。

## 训练配置

- `configs/experiments/standard_mappo_3u_8s2d_seed23.yaml`
- `configs/experiments/standard_mappo_3u_8s2d_seed41.yaml`
- `configs/experiments/standard_mappo_3u_8s2d_seed67.yaml`

待人工启动的命令：

```powershell
python scripts/train_ppo.py --config configs/experiments/standard_mappo_3u_8s2d_seed23.yaml
python scripts/train_ppo.py --config configs/experiments/standard_mappo_3u_8s2d_seed41.yaml
python scripts/train_ppo.py --config configs/experiments/standard_mappo_3u_8s2d_seed67.yaml
```

本次配置修改没有执行上述命令。

## 后续门控

新的 Standard 训练完成前：

- 不启动 Critic、CSA 或 MAPPO-Lagrangian 训练；
- 不访问 120000–120199 正式测试面板；
- 不预先假定 u800 优于 u400；
- 在查看下游结果前，冻结三个 seed 共同使用的父 checkpoint update；
- 新观测维度会改变 task/learner fingerprint，旧 Critic 不得复用。
