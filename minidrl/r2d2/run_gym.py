"""Run R2D2 on classic gym environments."""
import gymnasium as gym
import numpy as np
import pyrallis
import torch
import torch.nn.functional as F
from torch import nn

from minidrl.r2d2.r2d2 import R2D2Config, R2D2Network, run_r2d2


def get_gym_env_creator_fn(
    config: R2D2Config,
    env_idx: int,
    actor_idx: int | None = None,
) -> callable:
    """Get environment creator function."""

    def thunk():
        capture_video = config.capture_video and actor_idx == 0 and env_idx == 0
        render_mode = "rgb_array" if capture_video else None
        env = gym.make(config.env_id, render_mode=render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            env = gym.wrappers.RecordVideo(env, config.video_dir)
        seed = config.seed + env_idx
        if actor_idx is not None:
            seed += actor_idx * R2D2Config.num_envs_per_actor
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0):
    """Initialize a linear layer."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class R2D2GymNetwork(R2D2Network):
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
    ):
        """Initialize."""
        super().__init__()
        self.action_space = action_space
        self.action_dim = action_space.n
        input_size = np.array(obs_space.shape).prod()

        self.trunk = nn.Sequential(
            layer_init(nn.Linear(input_size, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 128)),
        )

        # LSTM recieves trunk output plus the last action (one-hot encoded) and reward
        self.lstm = nn.LSTM(
            128 + self.action_dim + 1, 128, batch_first=False, num_layers=1
        )
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        self.adv_head = nn.Sequential(
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, self.action_dim)),
        )
        self.value_head = nn.Sequential(
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1)),
        )

    def forward(self, o, a, r, done, lstm_state):
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


def gym_model_loader(config: R2D2Config):
    """Load PPO policy model for standard gymnasium envs."""
    env = config.env_creator_fn_getter(config, env_idx=0, actor_idx=None)()
    obs_space = env.observation_space
    act_space = env.action_space
    model = R2D2GymNetwork(obs_space, act_space)
    env.close()
    return model


class R2D2GymConfig(R2D2Config):
    """R2D2 gym specific configuration."""

    # The ID of the gymnasium environment
    env_id: str = "CartPole-v1"
    # Number of environment steps between updating actor parameters
    actor_update_interval: int = 100
    # Length of sequences stored and sampled from the replay buffer
    seq_len: int = 16
    # Length of burn-in sequence for each training sequence
    burnin_len: int = 0
    # Size of replay buffer (i.e. number of sequences)
    replay_buffer_size: int = 10000
    # Size of replay buffer before learning starts
    learning_starts: int = 1000
    # Target network update interval (in terms of number of updates)
    target_network_update_interval: int = 100


if __name__ == "__main__":
    config = pyrallis.parse(config_class=R2D2GymConfig)
    config.env_creator_fn_getter = get_gym_env_creator_fn
    config.model_loader = gym_model_loader
    config.lstm_size_ = 128
    run_r2d2(config)
