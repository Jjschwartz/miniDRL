"""Utils for PPO algorithm."""
from __future__ import annotations

import argparse
import os
import time
import logging
from dataclasses import dataclass, field
from distutils.util import strtobool

import gymnasium as gym
import torch

from d2rl.config import BASE_RESULTS_DIR
from d2rl.ppo.network import PPONetwork


def get_env_creator_fn(
    config: PPOConfig,
    env_idx: int,
    worker_idx: int | None = None,
):
    """Get environment creator function."""

    def thunk():
        capture_video = config.capture_video and worker_idx == 0 and env_idx == 0
        render_mode = "rgb_array" if capture_video else None
        env = gym.make(config.env_id, render_mode=render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            env = gym.wrappers.RecordVideo(env, config.video_dir)
        seed = config.seed + env_idx
        if worker_idx is not None:
            seed += worker_idx * PPOConfig.num_envs_per_worker
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def default_model_loader(config: PPOConfig):
    """Get model creator function."""

    env = config.env_creator_fn_getter(config, env_idx=0, worker_idx=None)()
    obs_space = env.observation_space
    act_space = env.action_space
    model = PPONetwork(
        obs_space,
        act_space,
        trunk_sizes=config.trunk_sizes,
        lstm_size=config.lstm_size,
        head_sizes=config.head_sizes,
    )
    env.close()
    return model


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
    # Directory where the model and logs will be saved
    log_dir: str = field(init=False)
    # Directory where videos will be saved
    video_dir: str = field(init=False)
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
    wandb_project: str = "dtrl"
    wandb_entity: str | None = None

    # Environment configuration
    # -------------------------
    env_id: str = "CartPole-v1"
    # whether to capture videos of the agent performances (check out `videos` folder)
    capture_video: bool = False
    # function for getting the environment creator function
    # Callable[[PPOConfig, int, int | None], Callable[[], gym.Env]
    env_creator_fn_getter: callable = get_env_creator_fn

    # Training configuration
    # ----------------------
    # total timesteps of the experiments
    total_timesteps: int = 10000000
    # number of steps of the vectorized environment per update
    num_rollout_steps: int = 128
    # the lengths of individual sequences used in training batches
    seq_len: int = 16
    # the number of chunk sequences per rollout
    num_seqs_per_rollout: int = field(init=False)
    # number of rollout workers
    num_workers: int = 4
    # number of parallel environments per worker
    # is overwritten if batch_size is provided
    num_envs_per_worker: int = 4
    # total number of environments
    # total_num_envs = num_envs_per_worker * num_workers
    total_num_envs: int = field(init=False)
    # number of steps per update batch
    # if not provided (i.e. is set to -1), then
    # batch_size = num_rollout_steps * num_envs_per_worker * num_workers
    batch_size: int = -1
    # number of mini-batches per update
    # is overwritten if minibatch_size is provided
    num_minibatches: int = 4
    # size of the mini-batches (in terms of timesteps)
    # if not provided (i.e. is set to -1), then
    # minibatch_size = int(batch_size // num_minibatches)
    minibatch_size: int = -1
    # total number of updates
    # num_updates = total_timesteps // batch_size
    num_updates: int = field(init=False)

    # Loss and update hyperparameters
    # -------------------------------
    # number of epochs to update the policy per update
    update_epochs: int = 2
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

    # Model configuration
    # -------------------
    # function for loading model
    # Callable[[PPOConfig], nn.Module]
    model_loader: callable = default_model_loader
    # size of each layer of the MLP trunk
    trunk_sizes: list[int] = field(default_factory=lambda: [64])
    # size of the LSTM layer
    lstm_size: int = 64
    # size of each layer of the policy and value heads
    head_sizes: list[int] = field(default_factory=lambda: [64])

    # Evaluation configuration
    # ------------------------
    # number of updates between evaluations
    eval_interval: int = 0
    # number of steps per evaluation (per eval environment)
    eval_num_steps: int = 1500

    # Other configuration
    # -------------------
    # number of steps after which the model is saved
    save_interval: int = 0

    def __post_init__(self):
        """Post initialization."""
        self.run_name = self.exp_name
        if self.env_id not in self.exp_name:
            self.run_name += f"_{self.env_id}"
        self.run_name += f"_{self.seed}_{int(time.time())}"

        self.log_dir = os.path.join(BASE_RESULTS_DIR, self.run_name)
        self.video_dir = os.path.join(self.log_dir, "videos")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.cuda else "cpu"
        )

        assert self.num_rollout_steps % self.seq_len == 0
        self.num_seqs_per_rollout = self.num_rollout_steps // self.seq_len

        if self.batch_size == -1:
            self.total_num_envs = self.num_envs_per_worker * self.num_workers
            self.batch_size = self.num_rollout_steps * self.total_num_envs
        else:
            assert self.batch_size % self.num_rollout_steps == 0
            self.total_num_envs = self.batch_size // self.num_rollout_steps
            assert self.total_num_envs % self.num_workers == 0
            self.num_envs_per_worker = self.total_num_envs // self.num_workers

        self.num_updates = self.total_timesteps // self.batch_size

        if self.minibatch_size == -1:
            assert self.batch_size % self.num_minibatches == 0
            self.minibatch_size = self.batch_size // self.num_minibatches
        else:
            assert self.batch_size % self.minibatch_size == 0
            self.num_minibatches = self.batch_size // self.minibatch_size

        if self.total_num_envs % self.num_minibatches != 0:
            logging.warn(
                "The total number of parallel environments `%d` (num_envs_per_worker * "
                "num_workers) isn't a multiple of the number of minibatches `%d`. PPO "
                "will still run, but the trajectories from some environments may not "
                "be used for training. Consider using a different `minibatch_size`, "
                "`num_minibatches`, `num_workers`, or `num_envs_per_worker`. for "
                "maximum efficiency.",
                self.total_num_envs,
                self.num_minibatches,
            )

    def load_model(self):
        """Load the model."""
        return self.model_loader(self)


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
    parser.add_argument("--wandb-project", type=str, default="d2rl",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    
    # Environment specific arguments
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
        help="the id of the environment")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Training arguments
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--num-rollout-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--num-workers", type=int, default=4,
        help="the number of rollout workers collecting trajectories in parallel")
    parser.add_argument("--num-envs-per-worker", type=int, default=4,
        help="the number of parallel game environments, will be set automatically set unless `--batch_size=-1`.")
    parser.add_argument("--batch-size", type=int, default=4096,
        help="the number of steps in each batch. Automatically set if `--batch-size=-1`.")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches. Onle used if `--minibatch-size=-1`")
    parser.add_argument("--minibatch-size", type=int, default=-1,
        help="the number of mini-batches Automatically set if `--minibatch-size=-1`.")
    parser.add_argument("--seq-len", type=int, default=16,
        help="the lengths of individual sequences used in training batches")
    
    # Loss and update hyperparameters
    parser.add_argument("--update-epochs", type=int, default=2,
        help="the K epochs to update the policy")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
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
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--save-interval", type=int, default=0,
        help="checkpoint saving interval, w.r.t. updates. If save-interval <= 0, no saving.")
    
    # Model specific arguments
    parser.add_argument("--trunk-sizes", nargs="+", type=int, default=[64],
        help="size of each layer of the MLP trunk")
    parser.add_argument("--lstm-size", type=int, default=64,
        help="size of LSTM layer")
    parser.add_argument("--head-sizes", nargs="+", type=int, default=[64],
        help="size of each layer of the MLP policy and value heads")

    # Evaluation specific arguments
    parser.add_argument("--eval-interval", type=int, default=100,
        help="evaluation interval w.r.t updates. If eval-interval <= 0, no evaluation.")
    parser.add_argument("--eval-num-steps", type=int, default=1500,
        help="minimum number of steps per evaluation (per eval environment = num-envs-per-worker)")

    args = parser.parse_args()
    # fmt: on
    return PPOConfig(**vars(args))
