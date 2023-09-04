"""Run PPO on atari environments."""
from __future__ import annotations

import argparse
import math
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from minidrl.common.atari_wrappers import (
    ClipRewardRangeEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from minidrl.ppo.ppo import run_ppo
from minidrl.ppo.utils import PPOConfig


def quadratic_episode_trigger(x: int) -> bool:
    """Quadratic episode trigger."""
    sqrt_x = math.sqrt(x)
    return x >= 1000 or int(sqrt_x) - sqrt_x == 0


def get_atari_env_creator_fn(
    config: PPOConfig, env_idx: int, worker_idx: int | None = None
):
    """Get atari environment creator function."""

    def thunk():
        capture_video = config.capture_video and worker_idx == 0 and env_idx == 0
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
        # env = ClipRewardEnv(env)
        env = gym.wrappers.NormalizeReward(env)
        env = ClipRewardRangeEnv(env, -5, 5)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 1)

        seed = config.seed + env_idx
        if worker_idx is not None:
            seed += worker_idx * PPOConfig.num_envs_per_worker
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


class PPOAtariNetwork(nn.Module):
    """CNN based network for PPO on atari.

    https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_lstm.py
    """

    def __init__(self, obs_space: gym.spaces.Space, act_space: gym.spaces.Discrete):
        super().__init__()
        self.network = nn.Sequential(
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
        self.lstm = nn.LSTM(512, 128)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor = layer_init(nn.Linear(128, act_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1)

    def get_states(self, x, lstm_state, done):
        hidden = self.network(x / 255.0)

        # LSTM logic
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
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
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


def atari_model_loader(config: PPOConfig):
    """Generates model given config."""
    env = config.env_creator_fn_getter(config, env_idx=0, worker_idx=None)()
    model = PPOAtariNetwork(env.observation_space, env.action_space)
    env.close()
    return model


def parse_ppo_atari_args() -> PPOConfig:
    """Parse command line arguments for PPO algorithm."""
    # ruff: noqa: E501
    # fmt: off
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--exp-name", type=str, default="ppo_atari",
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=0,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track-wandb", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project", type=str, default="miniDRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    
    # Environment specific arguments
    parser.add_argument("--env-id", type=str, default="PongNoFrameskip-v4",
        help="the id of the environment")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Training arguments
    parser.add_argument("--total-timesteps", type=int, default=200000000,
        help="total timesteps of the experiments")
    parser.add_argument("--num-rollout-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--num-workers", type=int, default=4,
        help="the number of rollout workers collecting trajectories in parallel")
    parser.add_argument("--num-envs-per-worker", type=int, default=32,
        help="the number of parallel game environments, will be set automatically unless `--batch_size=-1`.")
    parser.add_argument("--batch-size", type=int, default=16384,
        help="the number of steps in each batch.")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches. Onle used if `--minibatch-size=-1`")
    parser.add_argument("--minibatch-size", type=int, default=2048,
        help="the number of mini-batches")
    parser.add_argument("--seq-len", type=int, default=8,
        help="the lengths of individual sequences used in training batches")
    
    # Loss and update hyperparameters
    parser.add_argument("--update-epochs", type=int, default=2,
        help="the K epochs to update the policy")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.999,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=5.0,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    
    # Evaluation specific arguments
    parser.add_argument("--eval-interval", type=int, default=100,
        help="evaluation interval w.r.t updates. If eval-interval <= 0, no evaluation.")
    parser.add_argument("--eval-num-steps", type=int, default=10000,
        help="minimum number of steps per evaluation (per eval environment = num-envs-per-worker)")
    
    # Other arguments
    parser.add_argument("--save-interval", type=int, default=0,
        help="checkpoint saving interval, w.r.t. updates. If save-interval <= 0, no saving.")
    

    args = parser.parse_args()
    # fmt: on
    config = PPOConfig(**vars(args))
    config.env_creator_fn_getter = get_atari_env_creator_fn
    config.model_loader = atari_model_loader
    return config


if __name__ == "__main__":
    run_ppo(parse_ppo_atari_args())
