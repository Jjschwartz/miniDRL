"""Run R2D2 on atari environments."""
from __future__ import annotations

import argparse
import math
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from porl.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from porl.r2d2.r2d2 import run_r2d2
from porl.r2d2.utils import R2D2Config


def quadratic_episode_trigger(x: int) -> bool:
    """Quadratic episode trigger."""
    sqrt_x = math.sqrt(x)
    return x >= 1000 or int(sqrt_x) - sqrt_x == 0


def get_atari_env_creator_fn(
    config: R2D2Config, env_idx: int, actor_idx: int | None = None
):
    """Get atari environment creator function."""

    def thunk():
        capture_video = config.capture_video and actor_idx == 0 and env_idx == 0
        render_mode = "rgb_array" if capture_video else None
        env = gym.make(config.env_id, render_mode=render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            env = gym.wrappers.RecordVideo(
                env, config.video_dir, episode_trigger=quadratic_episode_trigger
            )

        # Atari specific wrappers
        # https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_lstm.py
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 1)

        seed = config.seed + env_idx
        if actor_idx is not None:
            seed += actor_idx * R2D2Config.num_envs_per_actor
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def layer_init(
    layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class R2D2AtariNetwork(nn.Module):
    """CNN based network for R2D2 on atari.

    Has a Dueling DQN architecture with an LSTM layer.

    CNN trunk -> LSTM -> Dueling DQN heads

    """

    def __init__(
        self,
        action_space: gym.spaces.Discrete,
    ):
        """Initialize."""
        super().__init__()
        self.action_space = action_space
        self.action_dim = action_space.n

        self.trunk = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )

        # LSTM recieves trunk output plus the last action (one-hot encoded) and reward
        lstm_input_size = 512 + self.action_dim + 1
        self.lstm = nn.LSTM(lstm_input_size, 512, batch_first=False, num_layers=1)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        # Dueling DQN heads
        self.adv_head = nn.Sequential(
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, self.action_dim)),
        )
        self.value_head = nn.Sequential(
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1)),
        )

    def forward(self, o, a, r, done, lstm_state):
        """Get q-values for each action given inputs.

        See porl.r2d2.r2d2.R2D2Network.forward for more details.
        """
        T, B, *_ = o.shape
        # merge batch and time dimensions, new shape=(T*B, *obs_shape)
        o = torch.flatten(o, 0, 1)
        o = o.float() / 255.0

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


def atari_model_loader(config: R2D2Config):
    """Generates model given config."""
    env = config.env_creator_fn_getter(config, env_idx=0, actor_idx=None)()
    model = R2D2AtariNetwork(env.action_space)
    env.close()
    return model


def parse_r2d2_atari_args() -> R2D2Config:
    """Parse command line arguments for R2D2 algorithm in Atari env."""
    # ruff: noqa: E501
    # fmt: off
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--exp-name", type=str, default="r2d2_atari",
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=0,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track-wandb", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project", type=str, default="porl",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    
    # Environment specific arguments
    parser.add_argument("--env-id", type=str, default="PongNoFrameskip-v4",
        help="the id of the environment")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Training configuration
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--num-actors", type=int, default=4,
        help="the number of rollout actors collecting trajectories in parallel")
    parser.add_argument("--num-envs-per-actor", type=int, default=8,
        help="the number of parallel game environments")
    parser.add_argument("--actor-update-interval", type=int, default=400,
        help="the number of environment steps between updating actor parameters")
    parser.add_argument("--save-interval", type=int, default=0,
        help="checkpoint saving interval, w.r.t. updates. If save-interval <= 0, no saving.")
    
    # Replay buffer configuration
    parser.add_argument("--seq-len", type=int, default=80,
        help="length of sequences stored and sampled from the replay buffer")
    parser.add_argument("--burnin-len", type=int, default=40,
        help="sequence burn-in length")
    parser.add_argument("--replay-buffer-size", type=int, default=100000,
        help="size of replay buffer (in terms of sequences)")
    parser.add_argument("--replay-priority-exponent", type=float, default=0.9,
        help="exponent for replay priority")
    parser.add_argument("--replay-priority-noise", type=float, default=1e-3,
        help="prioritized replay noise")
    parser.add_argument("--importance-sampling-exponent", type=float, default=0.6,
        help="exponent for importance sampling")
    
    # Training hyperparameters
    parser.add_argument("--gamma", type=float, default=0.997,
        help="the discount factor gamma")
    parser.add_argument("--batch-size", type=int, default=64,
        help="size of batches")
    parser.add_argument("--n-steps", type=int, default=5,
        help="number of steps for n-step returns")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--adam-eps", type=float, default=1e-3,
        help="adam epsilon")
    parser.add_argument("--target-network-update-interval", type=int, default=2500,
        help="target network update interval (in terms of number of updates)")
    parser.add_argument("--value-rescaling", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="use value function rescaling or not")
    parser.add_argument("--value-rescaling-epsilon", type=float, default=1e-3,
        help="epsilon for value function rescaling")
    parser.add_argument("--priority-td-error-mix", type=float, default=0.9,
        help="mean-max TD error mix proportion for priority calculation")
    parser.add_argument("--clip-grad-norm", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to clip the gradient norm")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="maximum gradient norm for clipping")
    
    # Model hyperparameters
    args = parser.parse_args()
    # fmt: on
    config = R2D2Config(**vars(args))
    config.env_creator_fn_getter = get_atari_env_creator_fn
    config.model_loader = atari_model_loader
    config.lstm_size = 512
    return config


if __name__ == "__main__":
    run_r2d2(parse_r2d2_atari_args())
