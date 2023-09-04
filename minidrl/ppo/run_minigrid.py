"""Run PPO on minigrid environments."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pyrallis
import torch
from torch import nn
from torch.distributions.categorical import Categorical

from minidrl.ppo.ppo import PPOConfig, run_ppo


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0):
    """Initialize a linear layer."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class PPONetwork(nn.Module):
    """PPO neural network policy class.

    Has linear trunk with tanh activations followed by an LSTM layer. The output
    of the LSTM layer is split into two heads, one for the actor (policy) and one for
    the (critic) value function.

    """

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        act_space: gym.spaces.Discrete,
    ):
        super().__init__()
        prev_size = np.array(obs_space.shape).prod()

        self.trunk = nn.Sequential(
            layer_init(nn.Linear(prev_size, 64)),
            nn.Tanh(),
        )

        self.lstm = nn.LSTM(64, 64, batch_first=False)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        self.actor = nn.Sequential(
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_space.n), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1),
        )

    def get_states(self, x, lstm_state, done):
        """Get the next states from the LSTM.

        `batch_size` is typically the number of parallel environments contained in
        the current batch, or the number of chunked sequences in the batch.

        E.g. if each batch is collected from 8 parallel environments, and 128 steps
        are collected from each environment, and we chunk each sequence of 128 steps
        into 16 steps giving eight 16 step sequences per parallel environment, then
        `batch_size` is 8 * 8 = 64 and `seq_len` is 16.

        Arguments
        ---------
        x: The input to the network. Shape (seq_len * batch_size, input_size).
        lstm_state: The previous state of the LSTM. This is a tuple with two entries,
            each of which has shape=(lstm.num_layers, batch_size, lstm_size).
        done: Whether the episode is done, has shape=(seq_len * batch_size,).

        Returns
        -------
        new_hidden: The output of the LSTM layer for each input x, has
            shape=(seq_len * batch_size, lstm_size)
        lstm_state: The new state of the LSTM at the end of the sequence. This is a
            tuple with two entries, each of which has
            shape=(lstm.num_layers, batch_size, lstm_size).
        """
        hidden = self.trunk(x)

        # LSTM Logic
        # get hidden and lstm state for each timestep in the sequence
        # resets the hidden state when episode done
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]

        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        """Get the value from the critic."""
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        """Get action and value from the actor and critic.

        Arguments
        ---------
        x: The input to the network. Shape (batch_size, input_size).
        lstm_state: The previous hidden state of the LSTM.
            Shape (seq_len, batch_size, features).
        done: Whether the episode is done. Shape (batch_size,).
        action: The action taken. If None, sample from the actor. Shape (batch_size,).

        Returns
        -------
        action: The action. Shape (seq_len, batch_size,).
        log_prob: The log probability of the action. Shape (seq_len, batch_size,).
        entropy: The entropy of the action distribution. Shape (seq_len, batch_size,).
        value: The values from critic. Shape (seq_len, batch_size,).
        lstm_state: The new state of the LSTM. Shape (seq_len, batch_size, features).

        """
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic(hidden),
            lstm_state,
        )


def minigrid_model_loader(config: PPOConfig):
    """Load PPO policy model for minigrid envs."""
    env = config.env_creator_fn_getter(config, env_idx=0, worker_idx=None)()
    obs_space = env.observation_space
    act_space = env.action_space
    model = PPONetwork(obs_space, act_space)
    env.close()
    return model


def get_minigrid_env_creator_fn(
    config: PPOConfig, env_idx: int, worker_idx: int | None = None
):
    """Get environment creator function."""

    def thunk():
        env = gym.make(config.env_id, max_episode_steps=256)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if config.capture_video and worker_idx == 0 and env_idx == 0:
            env = gym.wrappers.RecordVideo(env, config.video_dir)

        # take image observations and normalise to [0, 1]
        obs_space = env.observation_space
        env = gym.wrappers.TransformObservation(env, lambda obs: obs["image"] / 255.0)
        # have to monkey patch the observation space
        # https://gymnasium.farama.org/api/wrappers/observation_wrappers/#gymnasium.wrappers.TransformObservation
        env.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=obs_space["image"].shape, dtype=np.float32
        )
        env = gym.wrappers.FlattenObservation(env)

        seed = config.seed + env_idx
        if worker_idx is not None:
            seed += worker_idx * PPOConfig.num_envs_per_worker
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def run_ppo_minigrid(config: PPOConfig):
    config.env_creator_fn_getter = get_minigrid_env_creator_fn
    config.model_loader = minigrid_model_loader
    run_ppo(config)


if __name__ == "__main__":
    cfg = pyrallis.parse(config_class=PPOConfig)
    run_ppo_minigrid(cfg)
