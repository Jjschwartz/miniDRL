"""Neural network for PPO."""
from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical

if TYPE_CHECKING:
    import gymnasium as gym


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0):
    """Initialize a linear layer."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class PPONetwork(nn.Module):
    """Main Neural Network class for PPO.

    Has linear trunk with tanh activations followed by an LSTM layer. The output
    of the LSTM layer is split into two heads, one for the actor (policy) and one for
    the (critic) value function.

    """

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        act_space: gym.spaces.Discrete,
        trunk_sizes: List[int],
        lstm_size: int,
        head_sizes: List[int],
    ):
        """Initialize."""
        super().__init__()
        prev_size = np.array(obs_space.shape).prod()

        trunk_layers = []
        for i in range(0, len(trunk_sizes)):
            trunk_layers.append(layer_init(nn.Linear(prev_size, trunk_sizes[i])))
            trunk_layers.append(nn.Tanh())
            prev_size = trunk_sizes[i]
        self.trunk = nn.Sequential(*trunk_layers)

        self.lstm = nn.LSTM(prev_size, lstm_size, batch_first=False)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        prev_size = lstm_size

        actor_head_layers = []
        critic_head_layers = []
        for i in range(0, len(head_sizes)):
            actor_head_layers.append(layer_init(nn.Linear(prev_size, head_sizes[i])))
            actor_head_layers.append(nn.Tanh())
            critic_head_layers.append(layer_init(nn.Linear(prev_size, head_sizes[i])))
            critic_head_layers.append(nn.Tanh())
            prev_size = head_sizes[i]
        actor_head_layers.append(
            layer_init(nn.Linear(prev_size, act_space.n), std=0.01)
        )
        critic_head_layers.append(layer_init(nn.Linear(prev_size, 1), std=1))

        self.actor = nn.Sequential(*actor_head_layers)
        self.critic = nn.Sequential(*critic_head_layers)

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
