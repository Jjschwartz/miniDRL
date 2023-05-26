"""Neural network for R2D2."""
from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    import gymnasium as gym


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0):
    """Initialize a linear layer."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class R2D2Network(nn.Module):
    """Main Neural Network class for R2D2.

    Has a Dueling DQN architecture with an LSTM layer. This includes a shared linear
    trunk with ReLU activations, followed by an LSTM layer. The output of the LSTM
    layer is split into two heads, one for the advantage function and one for the
    value function. The advantage and value functions are then combined to get the
    Q-values for each action.

    """

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Discrete,
        trunk_sizes: List[int],
        lstm_size: int,
        head_sizes: List[int],
    ):
        """Initialize."""
        super().__init__()
        self.action_space = action_space
        self.action_dim = action_space.n
        prev_size = np.array(obs_space.shape).prod()

        trunk_layers = []
        for i in range(0, len(trunk_sizes)):
            trunk_layers.append(layer_init(nn.Linear(prev_size, trunk_sizes[i])))
            trunk_layers.append(nn.ReLU())
            prev_size = trunk_sizes[i]
        self.trunk = nn.Sequential(*trunk_layers)

        # LSTM recieves trunk output plus the last action (one-hot encoded) and reward
        lstm_input_size = prev_size + self.action_dim + 1
        self.lstm = nn.LSTM(lstm_input_size, lstm_size, batch_first=False, num_layers=1)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        prev_size = lstm_size

        adv_head_layers = []
        value_head_layers = []
        for i in range(0, len(head_sizes)):
            adv_head_layers.append(layer_init(nn.Linear(prev_size, head_sizes[i])))
            adv_head_layers.append(nn.ReLU())
            value_head_layers.append(layer_init(nn.Linear(prev_size, head_sizes[i])))
            value_head_layers.append(nn.ReLU())
            prev_size = head_sizes[i]
        adv_head_layers.append(layer_init(nn.Linear(prev_size, self.action_dim)))
        value_head_layers.append(layer_init(nn.Linear(prev_size, 1)))
        self.adv_head = nn.Sequential(*adv_head_layers)
        self.value_head = nn.Sequential(*value_head_layers)

    def forward(self, o, a, r, done, lstm_state):
        """Get q-values for each action given inputs.

        T = seq_len
        B = batch_size (i.e. num parallel envs)

        Arguments
        ---------
        o: The time `t` observation: o_t. Shape=(T, B, *obs_shape).
        a: The previous (`t-1`) action: a_tm1. Shape=(T, B).
        r: The previous (`t-1`) reward: r_tm1. Shape=(T, B).
        done: Whether the episode is ended on last `t-1` step: d_tm1. Shape=(T, B).
        lstm_state: The previous state of the LSTM. This is a tuple with two entries,
            each of which has shape=(lstm.num_layers, B, lstm_size).

        Returns
        -------
        q: The q-values for each action for time `t`: q_t. Shape=(T, B, action_space.n).
        lstm_state: The new state of the LSTM, shape=(lstm.num_layers, B, lstm_size).

        """
        T, B, *_ = o.shape
        # merge batch and time dimensions, new shape=(T*B, *obs_shape)
        o = torch.flatten(o, 0, 1)

        hidden = self.trunk(o)
        # ensure trunk output has correct dims for LSTM layer
        hidden = hidden.view(T, B, -1)

        # ensure correct dims for actions and rewards
        a = a.view(T, B)
        r = r.view(T, B, 1)
        # get one-hot encoding of actions
        # one hot encoding will add extra dim so shape is (T, B, action_space.n)
        a_one_hot = F.one_hot(a, self.action_dim).float().to(device=o.device)

        # Append reward and one hot last action to trunk output
        # shape is (T, B, trunk_hidden + 1 + action_space.n)
        lstm_input = torch.cat([hidden, r, a_one_hot], dim=-1)

        # LSTM Logic
        # Unroll LSTM for T timesteps
        # resets the lstm state when episode done
        done = done.view(T, B, 1)
        new_hidden = []
        for h, d in zip(lstm_input, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        # concate all lstm outputs into (T*B, features)
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)

        # advantages (T*B, action_space.n)
        advantages = self.adv_head(new_hidden)
        # values (T*B, 1)
        values = self.value_head(new_hidden)

        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        # reshape to (T, B, action_space.n)
        q_values = q_values.view(T, B, self.action_dim)

        return q_values, lstm_state
