"""Calculates the size in memory of the replay buffer."""
import argparse

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from minidrl.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


def make_atari_env(env_id: str):
    """Make atari environment with all the wrappers."""
    env = gym.make(env_id)
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
    return env


def calculate_replay_size(
    obs_space: spaces.Box,
    buffer_size: int,
    seq_len: int,
    burnin_len: int,
    lstm_size: int,
):
    """Calculate the size of the replay buffer in bytes."""
    print(f"{obs_space=}")
    T, C = seq_len + burnin_len + 1, buffer_size

    obs_shape = obs_space.shape
    obs_dtype = torch.from_numpy(obs_space.sample()).dtype

    obs_element_size = torch.tensor((1, 1, *obs_shape), dtype=obs_dtype).element_size()
    obs_num_bytes = obs_element_size * T * C * np.prod(obs_shape)

    action_element_size = torch.tensor((1,), dtype=torch.long).element_size()
    action_num_bytes = action_element_size * T * C

    reward_element_size = torch.tensor((1,), dtype=torch.float32).element_size()
    reward_num_bytes = reward_element_size * T * C

    done_element_size = torch.tensor((1,), dtype=torch.int8).element_size()
    done_num_bytes = done_element_size * T * C

    lstm_element_size = torch.tensor((1,), dtype=torch.float32).element_size()
    lstm_num_bytes = lstm_element_size * 1 * C * lstm_size
    lstm_num_bytes *= 2  # for hidden and cell state

    total_num_bytes = (
        obs_num_bytes
        + action_num_bytes
        + reward_num_bytes
        + done_num_bytes
        + lstm_num_bytes
    )

    for k, v, n, s in (
        (
            "obs_num_bytes",
            obs_num_bytes,
            (T * C * np.prod(obs_shape)),
            obs_element_size,
        ),
        ("action_num_bytes", action_num_bytes, (T * C), action_element_size),
        ("reward_num_bytes", reward_num_bytes, (T * C), reward_element_size),
        ("done_num_bytes", done_num_bytes, (T * C), done_element_size),
        ("lstm_num_bytes", lstm_num_bytes, (2 * C * lstm_size), lstm_element_size),
        ("total_num_bytes", total_num_bytes, (), ()),
    ):
        print(
            f"{k}={v}\n"
            f"  {v / int(1e6):.2f} MB\n"
            f"  {v / int(1e9):.2f} GB\n"
            f"  num elements={n}"
            f"  element size={s} Bytes"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--env_id", type=str, default="CartPole-v1", help="...")
    parser.add_argument("--atari", action="store_true", default=False, help="...")
    parser.add_argument("--buffer_size", type=int, default=100000, help="...")
    parser.add_argument("--seq_len", type=int, default=80, help="...")
    parser.add_argument("--burnin_len", type=int, default=40, help="...")
    parser.add_argument("--lstm_size", type=int, default=512, help="...")
    args = parser.parse_args()

    if args.atari and args.env_id == "CartPole-v1":
        env = make_atari_env("PongNoFrameskip-v4")
    else:
        env = make_atari_env(args.env_id) if args.atari else gym.make(args.env_id)
    calculate_replay_size(
        env.observation_space,
        args.buffer_size,
        args.seq_len,
        args.burnin_len,
        args.lstm_size,
    )
