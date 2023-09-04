"""Run PPO on atari environments."""
import math
from dataclasses import dataclass, field
from typing import Optional

import gymnasium as gym
import numpy as np
import pyrallis
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
from minidrl.ppo.ppo import PPOConfig, run_ppo


@dataclass
class AtariPPOConfig(PPOConfig):
    """Configuration for Distributed PPO in Atari."""

    # The name of this experiment
    exp_name: str = "ppo_atari"
    # The name of this run
    run_name: str = field(init=False, default="<exp_name>_<env_id>_<seed>_<time>")
    # Experiment seed
    seed: int = 0
    # Whether to set torch to deterministic mode
    # `torch.backends.cudnn.deterministic=False`
    torch_deterministic: bool = True
    # Whether to use CUDA
    cuda: bool = True
    # Number of updates between saving the policy model. If `save_interval > 0`, then
    # the policy model will be saved every `save_interval` updates as well as after the
    # final update.
    save_interval: int = 0

    # Wether to track the experiment with WandB
    track_wandb: bool = False
    # WandB project name
    wandb_project: str = "miniDRL"
    # WandB entity (team) name
    wandb_entity: Optional[str] = None

    # The ID of the atari environment
    env_id: str = "PongNoFrameskip-v4"
    # Whether to capture videos of the agent performances (check out `videos` folder)
    capture_video: bool = False

    # Total number of timesteps to train for
    total_timesteps: int = 200000000
    # Number of steps of the each vectorized environment per update
    num_rollout_steps: int = 128
    # The lengths of individual sequences (chunks) used in training batches. This
    # controls the length of Backpropagation Through Time (BPTT) in the LSTM. Should
    # be a factor of `num_rollout_steps`.
    seq_len: int = 16
    # Number of parallel rollout workers to use for collecting trajectories
    num_workers: int = 4
    # Number of parallel environments per worker.
    # Will be overwritten if `batch_size` is provided
    num_envs_per_worker: int = 32
    # Number of steps per update batch
    # If not provided (i.e. is set to -1), then
    # `batch_size = num_rollout_steps * num_envs_per_worker * num_workers`
    batch_size: int = 16384
    # Number of mini-batches per update
    # Will be overwritten if minibatch_size is provided
    num_minibatches: int = 8
    # Number of steps in each mini-batch.
    # If not provided (i.e. is set to -1), then
    # `minibatch_size = int(batch_size // num_minibatches)`
    minibatch_size: int = 2048

    # Number of epochs to train policy per update
    update_epochs: int = 2
    # Learning rate of the optimizer
    learning_rate: float = 2.5e-4
    # Whether to anneal the learning rate linearly to zero
    anneal_lr: bool = True
    # The discount factor
    gamma: float = 0.999
    # The GAE lambda parameter
    gae_lambda: float = 0.95
    # Whether to normalize advantages
    norm_adv: bool = True
    # Surrogate clip coefficient of PPO
    clip_coef: float = 0.2
    # Whether to use a clipped loss for the value function, as per the paper
    clip_vloss: bool = True
    # Coefficient of the value function loss
    vf_coef: float = 0.5
    # Coefficient of the entropy
    ent_coef: float = 0.01
    # The maximum norm for the gradient clipping
    max_grad_norm: float = 5.0
    # The target KL divergence threshold
    target_kl: Optional[float] = None

    # Function that returns an environment creator function
    # Callable[[PPOConfig, int, int | None], Callable[[], gym.Env]
    # Should be set after initialization
    env_creator_fn_getter: callable = field(init=False, default=None)
    # Function for loading model: Callable[[PPOConfig], nn.Module]
    # Should be set after initialization
    model_loader: callable = field(init=False, default=None)


def quadratic_episode_trigger(x: int) -> bool:
    """Quadratic episode trigger."""
    sqrt_x = math.sqrt(x)
    return x >= 1000 or int(sqrt_x) - sqrt_x == 0


def get_atari_env_creator_fn(
    config: PPOConfig, env_idx: int, worker_idx: Optional[int] = None
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

    Has a CNN trunk which feeds into a linear layer before going an LSTM layer. The
    output of the LSTM layer is split into two heads, one for the actor (policy) and
    one for the (critic) value function.

    """

    def __init__(self, act_space: gym.spaces.Discrete):
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
    model = PPOAtariNetwork(env.action_space)
    env.close()
    return model


if __name__ == "__main__":
    cfg = pyrallis.parse(config_class=AtariPPOConfig)
    cfg.env_creator_fn_getter = get_atari_env_creator_fn
    cfg.model_loader = atari_model_loader
    run_ppo(cfg)
