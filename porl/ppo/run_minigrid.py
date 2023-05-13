"""Run PPO on minigrid environments."""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from porl.ppo.ppo import run_ppo
from porl.ppo.utils import PPOConfig, parse_ppo_args


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


if __name__ == "__main__":
    config = parse_ppo_args()
    config.env_creator_fn_getter = get_minigrid_env_creator_fn
    run_ppo(config)
