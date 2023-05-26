"""Tests for computing R2D2 loss and priorities."""

import gymnasium as gym
import torch

from porl.r2d2.r2d2 import compute_loss_and_priority
from porl.r2d2.utils import R2D2Config


def test_compute_loss_and_priorities():
    """Tests loss and priorities work as expected."""
    env_id = "CartPole-v1"
    config = R2D2Config(
        env_id=env_id,
        num_actors=1,
        num_envs_per_actor=1,
        seq_len=4,
        burnin_len=2,
    )
    env = config.make_env()
    act_space = env.action_space
    assert isinstance(act_space, gym.spaces.Discrete)

    T = config.seq_len
    B = config.num_envs_per_actor
    q_values = torch.randn(T + 1, B, act_space.n).float()
    actions = torch.randint(0, act_space.n, (T + 1, B)).long()
    rewards = torch.randn(T + 1, B).float()
    dones = torch.randint(0, 2, (T + 1, B)).long()

    loss, priorities = compute_loss_and_priority(
        config=config,
        q_values=q_values,
        actions=actions,
        rewards=rewards,
        dones=dones,
        target_q_values=q_values,
    )

    assert loss.shape == (B,)
    assert priorities.shape == (B,)
    assert (priorities >= 0).all()


def test_compute_loss_and_priorities_outputs_1step():
    """Tests loss and priorities calculations are correct."""
    env_id = "CartPole-v1"
    config = R2D2Config(
        env_id=env_id,
        num_actors=1,
        num_envs_per_actor=1,
        seq_len=2,
        burnin_len=2,
        n_steps=1,
        gamma=0.5,
        value_rescaling=False,
        value_rescaling_epsilon=0.1,
        priority_td_error_mix=0.5,
    )
    env = config.make_env()
    act_space = env.action_space
    assert isinstance(act_space, gym.spaces.Discrete)

    B = config.num_envs_per_actor
    q_values = torch.tensor(
        [
            [[0.0, 1.0]],
            [[0.0, 1.0]],
            [[0.0, 1.0]],
        ]
    ).float()
    actions = torch.tensor([[0], [0], [1]]).long()
    rewards = torch.tensor([[0.0], [1.0], [1.0]]).float()
    dones = torch.tensor([[0], [0], [0]]).long()

    loss, priorities = compute_loss_and_priority(
        config=config,
        q_values=q_values,
        actions=actions,
        rewards=rewards,
        dones=dones,
        target_q_values=q_values,
    )
    assert loss.shape == (B,)
    assert priorities.shape == (B,)
    assert (priorities >= 0).all()

    expected_actual_q_values = torch.tensor([[0.0], [1.0]]).float()
    expected_target_q_max = torch.tensor([[1.0], [1.0], [1.0]]).float()
    expected_bellman_target = rewards[1:] + config.gamma * expected_target_q_max[1:]
    expected_td_error = expected_bellman_target - expected_actual_q_values
    expected_loss = 0.5 * torch.sum(torch.square(expected_td_error), dim=0)

    eta = config.priority_td_error_mix
    abs_td_error = torch.abs(expected_td_error)
    priorities = eta * torch.max(abs_td_error, dim=0)[0] + (1 - eta) * torch.mean(
        abs_td_error, dim=0
    )
    # Clamp priorities to avoid NaNs
    priorities = torch.clamp(priorities, min=0.0001)

    print(loss, expected_loss)
    assert torch.allclose(loss, expected_loss)
    assert torch.allclose(priorities, priorities)


def test_rescaling():
    eps = 0.001

    def h(x):
        return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x

    def h_inv(y):
        return torch.sign(y) * (
            torch.square(
                (torch.sqrt(1 + 4 * eps * (torch.abs(y) + eps + 1)) - 1) / (2 * eps)
            )
            - 1
        )

    for _ in range(100):
        x = torch.randn(1)
        y = h(x)
        x_hat = h_inv(y)
        assert abs(x.item() - x_hat.item()) < eps, f"{x} != {x_hat}"


if __name__ == "__main__":
    test_compute_loss_and_priorities()
    test_compute_loss_and_priorities_outputs_1step()
    test_rescaling()
