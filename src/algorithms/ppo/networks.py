"""中央联合循环 PPO 的 Actor 与 Critic。

当前版本已经完全移除通信消息分支：
Actor 只输出二维物理动作，不再输出 message/raw_message，也不再把消息概率
加入 PPO 的 log_prob。这样可以减少一个与当前基础导航任务无关的随机变量来源。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F


class TanhNormal:
    """经过 tanh 变换、动作范围位于 (-1, 1) 的对角高斯分布。"""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.base = Normal(mean, std)

    def sample(self, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.base.mean if deterministic else self.base.rsample()
        return torch.tanh(raw), raw

    def log_prob(self, raw: torch.Tensor) -> torch.Tensor:
        """计算包含 tanh 雅可比修正后的动作对数概率。"""
        log_two = torch.log(
            torch.tensor(2.0, dtype=raw.dtype, device=raw.device)
        )
        log_det_jacobian = 2.0 * (
            log_two - raw - F.softplus(-2.0 * raw)
        )
        return (
            self.base.log_prob(raw) - log_det_jacobian
        ).sum(dim=-1)

    def entropy(self, num_samples: int = 1) -> torch.Tensor:
        """用 -log pi(a) 蒙特卡洛估计 tanh 后真实动作分布的微分熵。

        旧版直接使用 tanh 前 Normal 的熵，可能出现“高斯方差变大，但 tanh 后
        动作反而更接近 ±1”的误导。这里的熵与环境实际接收的动作保持一致。
        """
        if int(num_samples) <= 0:
            raise ValueError("num_samples must be positive")
        raw = self.base.rsample((int(num_samples),))
        return -self.log_prob(raw).mean(dim=0)


@dataclass
class ActorStep:
    """Actor单步或序列评估结果。"""

    action: torch.Tensor
    raw_action: torch.Tensor
    log_prob: torch.Tensor
    action_entropy: torch.Tensor
    hidden: torch.Tensor

    @property
    def entropy(self) -> torch.Tensor:
        return self.action_entropy


class SafetyAlignmentAdapter(nn.Module):
    """Small zero-initialized residual on the pre-tanh action mean."""

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        *,
        rank: int = 16,
        scale: float = 0.15,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.rank = int(rank)
        self.scale = float(scale)
        if min(self.hidden_dim, self.action_dim, self.rank) <= 0:
            raise ValueError("adapter dimensions must be positive")
        if not 0.0 < self.scale <= 1.0:
            raise ValueError("adapter scale must be in (0,1]")
        self.normalization = nn.LayerNorm(self.hidden_dim)
        self.down = nn.Linear(self.hidden_dim, self.rank)
        self.up = nn.Linear(self.rank, self.action_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.down(self.normalization(features)))
        return self.scale * torch.tanh(self.up(hidden))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class RecurrentActor(nn.Module):
    """参数共享 GRU Actor。

    所有智能体共享同一套网络参数，但每个智能体维护独立 GRU hidden state。
    因此 ``num_agents`` 的改变不要求创建新的 Actor 类，只会改变 batch 行数。
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        action_dim: int = 2,
        *,
        initial_action_log_std: float = -1.2,
        log_std_min: float = -4.0,
        log_std_max: float = -0.3,
        safety_adapter_rank: int = 0,
        safety_adapter_scale: float = 0.15,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        if min(self.obs_dim, self.hidden_dim, self.action_dim) <= 0:
            raise ValueError("actor dimensions must be positive")
        if self.log_std_min >= self.log_std_max:
            raise ValueError("log_std_min must be smaller than log_std_max")

        self.encoder = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim)
        self.action_mean = nn.Linear(self.hidden_dim, self.action_dim)
        self.action_log_std = nn.Parameter(
            torch.full(
                (self.action_dim,),
                float(initial_action_log_std),
            )
        )
        self.safety_adapter = (
            SafetyAlignmentAdapter(
                self.hidden_dim,
                self.action_dim,
                rank=int(safety_adapter_rank),
                scale=float(safety_adapter_scale),
            )
            if int(safety_adapter_rank) > 0
            else None
        )

    def _action_mean(self, features: torch.Tensor) -> torch.Tensor:
        mean = self.action_mean(features)
        if self.safety_adapter is not None:
            mean = mean + self.safety_adapter(features)
        return mean

    def freeze_backbone_for_adapter(self) -> None:
        if self.safety_adapter is None:
            raise ValueError("cannot freeze for a missing safety adapter")
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("safety_adapter."))

    def adapter_parameter_count(self) -> int:
        return (
            0
            if self.safety_adapter is None
            else self.safety_adapter.parameter_count()
        )

    def initial_hidden(
        self,
        num_agents: int,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """为任意数量智能体创建独立隐藏状态。"""
        if int(num_agents) <= 0:
            raise ValueError("num_agents must be positive")
        return torch.zeros(
            1, int(num_agents), self.hidden_dim, device=device
        )

    def _effective_action_log_std(self) -> torch.Tensor:
        return self.action_log_std.clamp(
            self.log_std_min, self.log_std_max
        )

    def _distribution_and_state(
        self,
        local_obs: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[TanhNormal, torch.Tensor]:
        """编码局部观测并返回动作分布和下一隐藏状态。"""
        if local_obs.ndim != 2 or local_obs.shape[-1] != self.obs_dim:
            raise ValueError(
                f"local_obs must have shape [batch, {self.obs_dim}]"
            )
        if hidden.shape != (1, local_obs.shape[0], self.hidden_dim):
            raise ValueError("hidden must have shape [1, batch, hidden_dim]")

        encoded = self.encoder(local_obs.float()).unsqueeze(0)
        recurrent, next_hidden = self.gru(encoded, hidden.float())
        features = recurrent.squeeze(0)
        mean = self._action_mean(features)
        std = self._effective_action_log_std().exp().expand_as(mean)
        return TanhNormal(mean, std), next_hidden

    def distribution_stats(self) -> dict[str, float]:
        action_std = self._effective_action_log_std().detach().exp()
        return {
            "action_std_min": float(action_std.min().cpu()),
            "action_std_max": float(action_std.max().cpu()),
            "action_std_mean": float(action_std.mean().cpu()),
        }

    def forward_step(
        self,
        local_obs: torch.Tensor,
        hidden: torch.Tensor,
        deterministic: bool = False,
    ) -> ActorStep:
        """执行一次 Actor 推理。"""
        distribution, next_hidden = self._distribution_and_state(
            local_obs, hidden
        )
        action, raw_action = distribution.sample(deterministic)
        return ActorStep(
            action=action,
            raw_action=raw_action,
            log_prob=distribution.log_prob(raw_action),
            action_entropy=distribution.entropy(),
            hidden=next_hidden,
        )

    def evaluate_actions(
        self,
        local_obs: torch.Tensor,
        hidden: torch.Tensor,
        raw_actions: torch.Tensor,
    ) -> ActorStep:
        """评估一个 batch 中已采样动作的 log_prob 和 entropy。"""
        distribution, next_hidden = self._distribution_and_state(
            local_obs, hidden
        )
        if raw_actions.shape != (
            local_obs.shape[0],
            self.action_dim,
        ):
            raise ValueError(
                "raw_actions must align with actor batch and action width"
            )
        return ActorStep(
            action=torch.tanh(raw_actions),
            raw_action=raw_actions,
            log_prob=distribution.log_prob(raw_actions),
            action_entropy=distribution.entropy(),
            hidden=next_hidden,
        )

    def evaluate_sequence(
        self,
        local_obs: torch.Tensor,
        initial_hidden: torch.Tensor,
        raw_actions: torch.Tensor,
    ) -> ActorStep:
        """按时间序列评估 RNN 动作，用于 recurrent PPO 更新。"""
        if local_obs.ndim != 3 or local_obs.shape[-1] != self.obs_dim:
            raise ValueError(
                "sequence local_obs must have shape [time, batch, obs_dim]"
            )
        if initial_hidden.shape != (
            1,
            local_obs.shape[1],
            self.hidden_dim,
        ):
            raise ValueError(
                "initial_hidden must have shape [1, batch, hidden_dim]"
            )
        if raw_actions.shape != (
            *local_obs.shape[:2],
            self.action_dim,
        ):
            raise ValueError(
                "sequence raw_actions must align with time and batch"
            )

        encoded = self.encoder(local_obs.float())
        features, final_hidden = self.gru(
            encoded, initial_hidden.float()
        )
        mean = self._action_mean(features)
        distribution = TanhNormal(
            mean,
            self._effective_action_log_std().exp().expand_as(mean),
        )
        return ActorStep(
            action=torch.tanh(raw_actions),
            raw_action=raw_actions,
            log_prob=distribution.log_prob(raw_actions),
            action_entropy=distribution.entropy(),
            hidden=final_hidden,
        )

    def mean_action_sequence(
        self,
        local_obs: torch.Tensor,
        initial_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Return bounded policy means without consulting ``log_std``.

        Risk critics may use this path so the auxiliary objective cannot reduce
        its loss by inflating exploration variance. PPO sampling and likelihood
        evaluation remain unchanged.
        """
        if local_obs.ndim != 3 or local_obs.shape[-1] != self.obs_dim:
            raise ValueError(
                "sequence local_obs must have shape [time, batch, obs_dim]"
            )
        if initial_hidden.shape != (
            1,
            local_obs.shape[1],
            self.hidden_dim,
        ):
            raise ValueError(
                "initial_hidden must have shape [1, batch, hidden_dim]"
            )
        encoded = self.encoder(local_obs.float())
        features, _ = self.gru(encoded, initial_hidden.float())
        return torch.tanh(self._action_mean(features))


class CentralizedCritic(nn.Module):
    """集中式 Critic，输出每架智能体各自的状态价值。"""

    def __init__(
        self,
        global_state_dim: int,
        hidden_dim: int,
        num_agents: int,
    ):
        super().__init__()
        self.global_state_dim = int(global_state_dim)
        self.num_agents = int(num_agents)
        hidden_dim = int(hidden_dim)
        if (
            self.global_state_dim <= 0
            or hidden_dim <= 0
            or self.num_agents <= 0
        ):
            raise ValueError("critic dimensions must be positive")

        self.network = nn.Sequential(
            nn.Linear(self.global_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_agents),
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        if (
            global_state.ndim != 2
            or global_state.shape[-1] != self.global_state_dim
        ):
            raise ValueError(
                f"global_state must have shape [batch, {self.global_state_dim}]"
            )
        return self.network(global_state.float())
