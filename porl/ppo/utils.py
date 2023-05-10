"""Utils for PPO algorithm."""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from distutils.util import strtobool

import gymnasium as gym
import torch


def get_env_creator_fn(config: PPOConfig, env_idx: int, worker_idx: int | None = None):
    """Get environment creator function."""

    def thunk():
        env = gym.make(config.env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if config.capture_video and worker_idx == 0 and env_idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{config.run_name}")
        seed = config.seed + env_idx
        if worker_idx is not None:
            seed += worker_idx * PPOConfig.num_envs_per_worker
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


@dataclass
class PPOConfig:
    """Configuration for PPO algorithm."""

    # Experiment configuration
    # ------------------------
    # The name of this experiment
    exp_name: str = "ppo"
    # The name of this run
    # run_name = f"{exp_name}_{env_id}_{seed}_{time}"
    run_name: str = field(init=False)
    # Experiment seed
    seed: int = 0
    # Whether to set torch to deterministic mode
    # `torch.backends.cudnn.deterministic=False`
    torch_deterministic: bool = True
    # Whether to use CUDA
    cuda: bool = True
    # Device where model is run
    device: str = field(init=False)

    # WandB conbfiguration
    # --------------------
    track_wandb: bool = False
    wandb_project: str = "porl"
    wandb_entity: str | None = None

    # Environment configuration
    # -------------------------
    env_id: str = "CartPole-v1"
    # whether to capture videos of the agent performances (check out `videos` folder)
    capture_video: bool = False

    # Training configuration
    # ----------------------
    # total timesteps of the experiments
    total_timesteps: int = 10000000
    # number of parallel environments per worker
    num_envs_per_worker: int = 4
    # number of steps of the vectorized environment per update
    num_steps: int = 128
    # number of rollout workers
    num_workers: int = 4
    # number of mini-batches per update
    num_minibatches: int = 4
    # total number of environments
    # total_num_envs = num_envs_per_worker * num_workers
    total_num_envs: int = field(init=False)
    # number of steps per update batch
    # batch_size = num_steps * num_envs_per_worker * num_workers
    batch_size: int = field(init=False)
    # total number of updates
    # num_updates = total_timesteps // batch_size
    num_updates: int = field(init=False)
    # size of the mini-batches (in terms of timesteps)
    # minibatch_size = int(batch_size // num_minibatches)
    minibatch_size: int = field(init=False)
    # number of epochs to update the policy per update
    update_epochs: int = 4
    # learning rate of the optimizer
    learning_rate: float = 2.5e-4
    # whether to anneal the learning rate linearly to zero
    anneal_lr: bool = True
    # discount
    gamma: float = 0.99
    # GAE lambda parameter
    gae_lambda: float = 0.95
    # whether to normalize advantages
    norm_adv: bool = True
    # surrogate clip coefficient of PPO
    clip_coef: float = 0.2
    # whether to use a clipped loss for the value function, as per the paper
    clip_vloss: bool = True
    # coefficient of the value function loss
    vf_coef: float = 0.5
    # coefficient of the entropy
    ent_coef: float = 0.01
    # the maximum norm for the gradient clipping
    max_grad_norm: float = 0.5
    # the target KL divergence threshold
    target_kl: float | None = None
    # number of steps after which the model is saved
    save_interval: int = 100

    # Model configuration
    # -------------------
    # size of each layer of the MLP trunk
    trunk_sizes: list[int] = field(default_factory=lambda: [64])
    # size of the LSTM layer
    lstm_size: int = 64
    # size of each layer of the policy and value heads
    head_sizes: list[int] = field(default_factory=lambda: [64])

    def __post_init__(self):
        """Post initialization."""
        self.run_name = f"{self.exp_name}_{self.env_id}_{self.seed}_{int(time.time())}"
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.cuda else "cpu"
        )

        self.total_num_envs = self.num_envs_per_worker * self.num_workers
        self.batch_size = self.num_steps * self.total_num_envs
        self.num_updates = int(self.total_timesteps // self.batch_size)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        assert self.total_num_envs % self.num_minibatches == 0, (
            "The number of parallel environments (num_envs_per_worker * num_workers) "
            "must be a multiple of the number of minibatches."
        )


def parse_ppo_args() -> PPOConfig:
    """Parse command line arguments for PPO algorithm."""
    # ruff: noqa: E501
    # fmt: off
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--exp-name", type=str, default="ppo",
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
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
        help="the id of the environment")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-workers", type=int, default=4,
        help="the number of rollout workers collecting trajectories in parallel")
    parser.add_argument("--num-envs-per-worker", type=int, default=4,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
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
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    
    # Model specific arguments
    parser.add_argument("--trunk-sizes", nargs="+", type=int, default=[64],
        help="size of each layer of the MLP trunk")
    parser.add_argument("--lstm-size", type=int, default=64,
        help="size of LSTM layer")
    parser.add_argument("--head-sizes", nargs="+", type=int, default=[64],
        help="size of each layer of the MLP policy and value heads")
    args = parser.parse_args()
    # fmt: on
    return PPOConfig(**vars(args))
