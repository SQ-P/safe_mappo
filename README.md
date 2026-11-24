# safe_mappo
一种帮助强化学习的安全层设计
# MAPPO Safety Alignment

这是面向论文扩展的精简研究底座。当前主线固定为：

`Standard recurrent MAPPO -> frozen staged safety execution -> counterfactual reference actions -> advantage-gated intervention critic -> zero-initialized safety-alignment adapter`

保留内容包括二维多 UAV 环境、Standard MAPPO/PPO、冻结安全执行器、反事实数据与优势 Critic、成对评估工具，以及可复核的 3U+8S+2D 基线证据。旧 FACS/STAR、HAPPO、Projection-aware 模仿和 LLM 自修复路线不再属于活动研究面。

主要入口：

```powershell
python scripts/train_ppo.py --config configs/active/standard_mappo_3u_8s2d.yaml
python scripts/evaluate_ppo.py --config configs/active/standard_mappo_3u_8s2d.yaml --checkpoint <checkpoint.pt>
python scripts/build_ia_mappo_r4_4_advantage_dataset.py --help
python scripts/train_ia_mappo_r4_4_advantage_critic.py --help
python scripts/certify_ia_mappo_r4_4_advantage_critic.py --help
python scripts/calibrate_csa_lambda.py --help
python scripts/freeze_lagrangian_budget.py --help
python scripts/compare_csa_formal.py --help
```

聚焦验证：

```powershell
pytest -q tests/test_mappo.py tests/test_csa_mappo.py tests/test_intervention_advantage.py tests/test_intervention_advantage_certification.py tests/ia_mappo_v1
```

研究边界见 [docs/PROJECT_SCOPE.md](docs/PROJECT_SCOPE.md)，正式执行顺序见 [docs/CSA_MAPPO_IMPLEMENTATION_AND_PROTOCOL.md](docs/CSA_MAPPO_IMPLEMENTATION_AND_PROTOCOL.md)。
