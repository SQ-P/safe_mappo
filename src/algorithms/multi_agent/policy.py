"""Policy and optimizer containers shared by recurrent MAPPO and HAPPO."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn, optim

from src.algorithms.ppo.networks import (
    ActorStep,
    CentralizedCritic,
    RecurrentActor,
)


class MultiAgentPolicy(nn.Module):
    """One shared actor or one independent actor per homogeneous UAV."""

    def __init__(
        self,
        *,
        num_agents: int,
        obs_dim: int,
        hidden_dim: int,
        action_dim: int,
        parameter_sharing: bool,
        initial_action_log_std: float,
        log_std_min: float,
        log_std_max: float,
        safety_adapter_rank: int = 0,
        safety_adapter_scale: float = 0.15,
    ) -> None:
        super().__init__()
        self.num_agents = int(num_agents)
        self.obs_dim = int(obs_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.parameter_sharing = bool(parameter_sharing)
        if self.num_agents <= 0:
            raise ValueError("num_agents must be positive")

        kwargs = {
            "obs_dim": self.obs_dim,
            "hidden_dim": self.hidden_dim,
            "action_dim": self.action_dim,
            "initial_action_log_std": float(initial_action_log_std),
            "log_std_min": float(log_std_min),
            "log_std_max": float(log_std_max),
            "safety_adapter_rank": int(safety_adapter_rank),
            "safety_adapter_scale": float(safety_adapter_scale),
        }
        if self.parameter_sharing:
            self.shared_actor = RecurrentActor(**kwargs)
            self.actors = nn.ModuleList()
        else:
            self.shared_actor = None
            self.actors = nn.ModuleList(
                RecurrentActor(**kwargs) for _ in range(self.num_agents)
            )

    def actor_for(self, agent_id: int) -> RecurrentActor:
        agent_id = int(agent_id)
        if not 0 <= agent_id < self.num_agents:
            raise IndexError("agent_id is outside the policy")
        if self.parameter_sharing:
            assert self.shared_actor is not None
            return self.shared_actor
        return self.actors[agent_id]

    def actor_modules(self) -> tuple[RecurrentActor, ...]:
        if self.parameter_sharing:
            assert self.shared_actor is not None
            return (self.shared_actor,)
        return tuple(self.actors)

    def initial_hidden(
        self,
        num_agents: int,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        if int(num_agents) != self.num_agents:
            raise ValueError(
                "multi-agent policy requires one hidden row per environment agent"
            )
        return torch.zeros(
            1, self.num_agents, self.hidden_dim, device=device
        )

    @staticmethod
    def _concatenate_steps(steps: list[ActorStep]) -> ActorStep:
        return ActorStep(
            action=torch.cat([item.action for item in steps], dim=0),
            raw_action=torch.cat(
                [item.raw_action for item in steps], dim=0
            ),
            log_prob=torch.cat([item.log_prob for item in steps], dim=0),
            action_entropy=torch.cat(
                [item.action_entropy for item in steps], dim=0
            ),
            hidden=torch.cat([item.hidden for item in steps], dim=1),
        )

    def forward_step(
        self,
        local_obs: torch.Tensor,
        hidden: torch.Tensor,
        deterministic: bool = False,
    ) -> ActorStep:
        if local_obs.shape != (self.num_agents, self.obs_dim):
            raise ValueError(
                "decentralized actor input must contain one local observation per agent"
            )
        if hidden.shape != (1, self.num_agents, self.hidden_dim):
            raise ValueError("hidden state does not match the agent set")
        if self.parameter_sharing:
            assert self.shared_actor is not None
            return self.shared_actor.forward_step(
                local_obs, hidden, deterministic
            )
        return self._concatenate_steps(
            [
                actor.forward_step(
                    local_obs[agent_id : agent_id + 1],
                    hidden[:, agent_id : agent_id + 1],
                    deterministic,
                )
                for agent_id, actor in enumerate(self.actors)
            ]
        )

    def distribution_stats(self) -> dict[str, float]:
        values = torch.cat(
            [
                actor._effective_action_log_std().detach().exp().reshape(-1)
                for actor in self.actor_modules()
            ]
        )
        return {
            "action_std_min": float(values.min().cpu()),
            "action_std_max": float(values.max().cpu()),
            "action_std_mean": float(values.mean().cpu()),
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def adapter_parameter_count(self) -> int:
        return sum(actor.adapter_parameter_count() for actor in self.actor_modules())

    def freeze_backbone_for_adapter(self) -> None:
        for actor in self.actor_modules():
            actor.freeze_backbone_for_adapter()


class TeamCentralizedCritic(nn.Module):
    """One team value head, broadcast only at the rollout-buffer boundary."""

    def __init__(
        self,
        global_state_dim: int,
        hidden_dim: int,
        num_agents: int,
    ) -> None:
        super().__init__()
        self.num_agents = int(num_agents)
        self.value = CentralizedCritic(
            global_state_dim, hidden_dim, num_agents=1
        )

    def team_value(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.value(global_state).squeeze(-1)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        scalar = self.team_value(global_state)
        return scalar[:, None].expand(-1, self.num_agents)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class OptimizerBundle:
    """Checkpoint-compatible collection of one optimizer per actor."""

    STATE_VERSION = 1

    def __init__(self, optimizers: Iterable[optim.Optimizer]) -> None:
        self.optimizers = tuple(optimizers)
        if not self.optimizers:
            raise ValueError("optimizer bundle cannot be empty")

    @property
    def param_groups(self) -> list[dict]:
        return [
            group
            for optimizer in self.optimizers
            for group in optimizer.param_groups
        ]

    def state_dict(self) -> dict:
        return {
            "version": self.STATE_VERSION,
            "optimizers": [
                optimizer.state_dict() for optimizer in self.optimizers
            ],
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state.get("version", -1)) != self.STATE_VERSION:
            raise ValueError("actor optimizer bundle version is incompatible")
        saved = list(state.get("optimizers", []))
        if len(saved) != len(self.optimizers):
            raise ValueError("actor optimizer count is incompatible")
        for optimizer, payload in zip(self.optimizers, saved):
            optimizer.load_state_dict(payload)

