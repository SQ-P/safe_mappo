import numpy as np
import pytest

from src.algorithms.ppo.rollout_buffer import RecurrentRolloutBuffer


def _add_step(buffer, marker=0.0, reward=1.0, done=False):
    n = buffer.num_agents
    buffer.add(
        local_obs=np.full((n, 3), marker, dtype=np.float32),
        global_state=np.full(7, marker, dtype=np.float32),
        hidden=np.full((1, n, 6), marker, dtype=np.float32),
        actions=np.zeros((n, 2), dtype=np.float32),
        raw_actions=np.zeros((n, 2), dtype=np.float32),
        log_probs=np.zeros(n, dtype=np.float32),
        rewards=np.full(n, reward, dtype=np.float32),
        dones=np.full(n, done, dtype=bool),
        values=np.zeros(n, dtype=np.float32),
    )


def test_rollout_supports_variable_agent_counts():
    for n in (3, 4, 6):
        buffer = RecurrentRolloutBuffer(n)
        _add_step(buffer)
        _add_step(buffer, marker=1.0, done=True)
        buffer.compute_returns_and_advantages(
            np.zeros(n, dtype=np.float32), gamma=0.99, gae_lambda=0.95
        )
        assert buffer.rewards.shape == (2, n)
        assert buffer.advantages.shape == (2, n)
        assert buffer.returns.shape == (2, n)


def test_rollout_has_no_message_storage():
    buffer = RecurrentRolloutBuffer(4)
    assert not hasattr(buffer, "messages")
    assert not hasattr(buffer, "raw_messages")


def test_rollout_rejects_wrong_agent_rows():
    buffer = RecurrentRolloutBuffer(4)
    with pytest.raises(ValueError):
        buffer.add(
            local_obs=np.zeros((3, 2)),
            global_state=np.zeros(5),
            hidden=np.zeros((1, 4, 6)),
            actions=np.zeros((4, 2)),
            raw_actions=np.zeros((4, 2)),
            log_probs=np.zeros(4),
            rewards=np.zeros(4),
            dones=np.zeros(4, dtype=bool),
            values=np.zeros(4),
        )
