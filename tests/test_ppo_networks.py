import torch

from src.algorithms.ppo.networks import CentralizedCritic, RecurrentActor


def test_actor_supports_variable_agent_batch_size():
    actor = RecurrentActor(obs_dim=20, hidden_dim=16, action_dim=2)
    for num_agents in (3, 4, 6):
        obs = torch.zeros(num_agents, 20)
        hidden = actor.initial_hidden(num_agents)
        output = actor.forward_step(obs, hidden)
        assert output.action.shape == (num_agents, 2)
        assert output.raw_action.shape == (num_agents, 2)
        assert output.log_prob.shape == (num_agents,)
        assert output.action_entropy.shape == (num_agents,)
        assert output.hidden.shape == (1, num_agents, 16)
        assert torch.all(output.action.abs() < 1.0)


def test_actor_no_longer_has_communication_head():
    actor = RecurrentActor(obs_dim=8, hidden_dim=16)
    names = dict(actor.named_parameters())
    assert not any("message" in name for name in names)


def test_actor_sequence_evaluation_shapes():
    actor = RecurrentActor(obs_dim=6, hidden_dim=8, action_dim=2)
    observations = torch.randn(5, 3, 6)
    initial_hidden = torch.zeros(1, 3, 8)
    raw_actions = torch.randn(5, 3, 2)
    evaluated = actor.evaluate_sequence(observations, initial_hidden, raw_actions)
    assert evaluated.log_prob.shape == (5, 3)
    assert evaluated.action_entropy.shape == (5, 3)
    assert evaluated.hidden.shape == (1, 3, 8)


def test_critic_supports_variable_agent_counts():
    for num_agents in (3, 4, 6):
        critic = CentralizedCritic(42, 32, num_agents)
        values = critic(torch.randn(2, 42))
        assert values.shape == (2, num_agents)
