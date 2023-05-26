"""Tests for R2D2 network."""

import gymnasium as gym
import numpy as np
import torch
from porl.r2d2.network import R2D2Network


def test_network_init():
    env = gym.make("CartPole-v1")
    obs_space = env.observation_space
    action_space = env.action_space
    trunk_sizes = [32, 32]
    lstm_size = 32
    head_sizes = [32, 32]

    network = R2D2Network(obs_space, action_space, trunk_sizes, lstm_size, head_sizes)
    print(network)


def test_network_single():
    env = gym.make("CartPole-v1")
    obs_space = env.observation_space
    action_space = env.action_space
    trunk_sizes = [32, 32]
    lstm_size = 32
    head_sizes = [32, 32]

    network = R2D2Network(obs_space, action_space, trunk_sizes, lstm_size, head_sizes)
    obs = env.reset()[0]
    obs = torch.from_numpy(obs).float()
    # Add batch dimension
    obs = obs.view(1, 1, -1)
    action = torch.tensor([[0]], dtype=torch.long)
    reward = torch.tensor([[0.0]], dtype=torch.float32)
    done = torch.tensor([[0]], dtype=torch.int8)
    lstm_state = (torch.zeros(1, 1, lstm_size), torch.zeros(1, 1, lstm_size))
    q_values, next_lstm_state = network(obs, action, reward, done, lstm_state)

    assert q_values.shape == (1, 1, action_space.n)
    assert len(next_lstm_state) == 2
    assert next_lstm_state[0].shape == (1, 1, lstm_size)
    assert next_lstm_state[1].shape == (1, 1, lstm_size)


def test_network_batch():
    batch_size = 4
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make("CartPole-v1") for _ in range(batch_size)]
    )
    obs_space = envs.single_observation_space
    assert isinstance(obs_space, gym.spaces.Box)
    action_space = envs.single_action_space
    trunk_sizes = [32, 32]
    lstm_size = 32
    head_sizes = [32, 32]

    network = R2D2Network(obs_space, action_space, trunk_sizes, lstm_size, head_sizes)
    obs = envs.reset()[0]
    assert obs.shape == (batch_size, *obs_space.shape)
    obs = torch.from_numpy(obs).float()
    # Add batch dimension
    obs = obs.view(1, batch_size, -1)
    action = torch.zeros((1, batch_size), dtype=torch.long)
    reward = torch.zeros((1, batch_size), dtype=torch.float32)
    done = torch.zeros((1, batch_size), dtype=torch.int8)
    lstm_state = (
        torch.zeros(1, batch_size, lstm_size),
        torch.zeros(1, batch_size, lstm_size),
    )
    q_values, next_lstm_state = network(obs, action, reward, done, lstm_state)

    assert q_values.shape == (1, batch_size, action_space.n)
    assert len(next_lstm_state) == 2
    assert next_lstm_state[0].shape == (1, batch_size, lstm_size)
    assert next_lstm_state[1].shape == (1, batch_size, lstm_size)


def test_network_single_sequence():
    seq_len = 4
    env = gym.make("CartPole-v1")
    obs_space = env.observation_space
    action_space = env.action_space
    trunk_sizes = [32, 32]
    lstm_size = 32
    head_sizes = [32, 32]

    network = R2D2Network(obs_space, action_space, trunk_sizes, lstm_size, head_sizes)
    obs = env.reset()[0]
    obs = torch.from_numpy(np.stack([obs] * seq_len)).float()
    obs = obs.view(seq_len, 1, -1)
    action = torch.zeros((seq_len, 1), dtype=torch.long)
    reward = torch.zeros((seq_len, 1), dtype=torch.float32)
    done = torch.zeros((seq_len, 1), dtype=torch.int8)
    lstm_state = (torch.zeros(1, 1, lstm_size), torch.zeros(1, 1, lstm_size))
    q_values, next_lstm_state = network(obs, action, reward, done, lstm_state)

    assert q_values.shape == (seq_len, 1, action_space.n)
    assert len(next_lstm_state) == 2
    assert next_lstm_state[0].shape == (1, 1, lstm_size)
    assert next_lstm_state[1].shape == (1, 1, lstm_size)


def test_network_batch_sequence():
    batch_size = 3
    seq_len = 4
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make("CartPole-v1") for _ in range(batch_size)]
    )
    obs_space = envs.single_observation_space
    assert isinstance(obs_space, gym.spaces.Box)
    action_space = envs.single_action_space
    trunk_sizes = [32, 32]
    lstm_size = 32
    head_sizes = [32, 32]

    network = R2D2Network(obs_space, action_space, trunk_sizes, lstm_size, head_sizes)
    obs = envs.reset()[0]
    assert obs.shape == (batch_size, *obs_space.shape)
    obs = torch.from_numpy(np.stack([obs] * seq_len)).float()
    obs = obs.view(seq_len, batch_size, -1)
    action = torch.zeros((seq_len, batch_size), dtype=torch.long)
    reward = torch.zeros((seq_len, batch_size), dtype=torch.float32)
    done = torch.zeros((seq_len, batch_size), dtype=torch.int8)
    lstm_state = (
        torch.zeros(1, batch_size, lstm_size),
        torch.zeros(1, batch_size, lstm_size),
    )
    q_values, next_lstm_state = network(obs, action, reward, done, lstm_state)

    assert q_values.shape == (seq_len, batch_size, action_space.n)
    assert len(next_lstm_state) == 2
    assert next_lstm_state[0].shape == (1, batch_size, lstm_size)
    assert next_lstm_state[1].shape == (1, batch_size, lstm_size)


def test_network_single_sequence_reset():
    seq_len = 4
    env = gym.make("CartPole-v1")
    obs_space = env.observation_space
    action_space = env.action_space
    trunk_sizes = [32, 32]
    lstm_size = 32
    head_sizes = [32, 32]

    network = R2D2Network(obs_space, action_space, trunk_sizes, lstm_size, head_sizes)

    obs = env.reset()[0]
    init_lstm_state = (torch.zeros(1, 1, lstm_size), torch.zeros(1, 1, lstm_size))

    # Get lstm state after a single step
    obs = torch.from_numpy(obs).float()
    obs = obs.view(1, 1, -1)
    action = torch.zeros((1, 1), dtype=torch.long)
    reward = torch.zeros((1, 1), dtype=torch.float32)
    done = torch.zeros((1, 1), dtype=torch.int8)
    one_step_q_values, one_step_lstm_state = network(
        obs, action, reward, done, init_lstm_state
    )

    # run for sequence with final step done (reseting LSTM state)
    # all obs, actions, and rewards are the same
    obs = torch.from_numpy(np.stack([obs] * seq_len)).float()
    obs = obs.view(seq_len, 1, -1)
    action = torch.zeros((seq_len, 1), dtype=torch.long)
    reward = torch.zeros((seq_len, 1), dtype=torch.float32)
    done = torch.zeros((seq_len, 1), dtype=torch.int8)
    done[-1] = 1
    n_step_q_values, n_step_next_lstm_state = network(
        obs, action, reward, done, init_lstm_state
    )

    assert n_step_q_values.shape == (seq_len, 1, action_space.n)
    assert len(n_step_next_lstm_state) == 2
    assert n_step_next_lstm_state[0].shape == (1, 1, lstm_size)
    assert n_step_next_lstm_state[1].shape == (1, 1, lstm_size)

    # one step and final step outputs should be the same
    assert torch.allclose(one_step_q_values, n_step_q_values[-1])
    assert torch.allclose(one_step_lstm_state[0], n_step_next_lstm_state[0])
    assert torch.allclose(one_step_lstm_state[1], n_step_next_lstm_state[1])


if __name__ == "__main__":
    # test_network_init()
    # test_network_single()
    # test_network_batch()
    # test_network_single_sequence()
    # test_network_batch_sequence()
    test_network_single_sequence_reset()
