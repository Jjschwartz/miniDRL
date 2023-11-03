"""Run R2D2 on atari environments."""
import gymnasium as gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F

from minidrl.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from minidrl.r2d2.r2d2 import R2D2Config, R2D2Network, run_r2d2


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
            env = gym.wrappers.RecordVideo(env, config.video_dir)

        # Atari specific wrappers
        # https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_lstm.py
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        if not config.value_rescaling:
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


class R2D2AtariNetwork(R2D2Network):
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


if __name__ == "__main__":
    config = pyrallis.parse(config_class=R2D2Config)
    config.env_creator_fn_getter = get_atari_env_creator_fn
    config.model_loader = atari_model_loader
    config.lstm_size_ = 512
    run_r2d2(config)
