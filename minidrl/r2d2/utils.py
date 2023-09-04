"""Utils for R2D2 algorithm."""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from distutils.util import strtobool

import gymnasium as gym
import torch

from minidrl.config import BASE_RESULTS_DIR
from minidrl.r2d2.network import R2D2Network


def get_env_creator_fn(config: R2D2Config, env_idx: int, actor_idx: int | None = None):
    """Get environment creator function."""

    def thunk():
        env = gym.make(config.env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if config.capture_video and actor_idx == 0 and env_idx == 0:
            env = gym.wrappers.RecordVideo(env, config.video_dir)
        seed = config.seed + env_idx
        if actor_idx is not None:
            seed += actor_idx * R2D2Config.num_envs_per_actor
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def default_model_loader(config: R2D2Config):
    """Get model creator function."""

    env = config.env_creator_fn_getter(config, env_idx=0, actor_idx=None)()
    obs_space = env.observation_space
    act_space = env.action_space
    model = R2D2Network(
        obs_space,
        act_space,
        trunk_sizes=config.trunk_sizes,
        lstm_size=config.lstm_size,
        head_sizes=config.head_sizes,
    )
    env.close()
    return model


@dataclass
class R2D2Config:
    """Configuration for R2D2 algorithm."""

    # Experiment configuration
    # ------------------------
    # The name of this experiment
    exp_name: str = "r2d2"
    # The name of this run
    # run_name = f"{exp_name}_{env_id}_{seed}_{time}"
    run_name: str = field(init=False)
    # Directory where the model and logs will be saved
    # (default=<base_results_dir>/<run_name>)
    log_dir: str = field(init=False)
    # Directory where videos will be saved (default=<log_dir>/videos)
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
    wandb_project: str = "miniDRL"
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
    # number of trajectory collection actors
    num_actors: int = 4
    # number of parallel environments per actor
    num_envs_per_actor: int = 4
    # number of environment steps between updating actor parameters
    actor_update_interval: int = 400
    # base epsilon for actor epsilon-greedy exploration
    actor_base_epsilon: float = 0.05
    # device used by actor models
    actor_device: torch.device = torch.device("cpu")
    # total number of environments
    # total_num_envs = num_envs_per_actor * num_actors
    total_num_envs: int = field(init=False)
    # number of steps after which the model is saved
    save_interval: int = 0

    # Replay buffer configuration
    # ---------------------------
    # length of sequences stored and sampled from the replay buffer
    seq_len: int = 80
    # sequence burn-in length
    burnin_len: int = 40
    # size of replay buffer (in terms of sequences)
    replay_buffer_size: int = 100000
    # exponent for replay priority
    replay_priority_exponent: float = 0.9
    # prioritized replay noise
    replay_priority_noise: float = 1e-3
    # exponent for importance sampling
    importance_sampling_exponent: float = 0.6

    # Training hyperparameters
    # ------------------------
    # discount
    gamma: float = 0.997
    # size of each batch
    batch_size: int = 32
    # number of steps for n-step return
    n_steps: int = 5
    # learning rate of the optimizer
    learning_rate: float = 1e-4
    # adam epsilon
    adam_eps: float = 1e-3
    # size of replay buffer before learning starts
    learning_starts: int = 500
    # target network update interval (in terms of number of updates)
    target_network_update_interval: int = 2500
    # use value function rescaling or not
    value_rescaling: bool = True
    # epsilon used for value function rescaling
    value_rescaling_epsilon: float = 0.001
    # mean-max TD error mix proportion for priority calculation
    priority_td_error_mix: float = 0.9
    # whether to clip the gradient norm
    clip_grad_norm: bool = True
    # maximum gradient norm
    max_grad_norm: float = 0.5

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

    # Debugging configuration
    # -----------------------
    # number of steps to run actor for, can be set so run_actor can be tested
    debug_actor_steps: int | None = None
    # disable logging
    debug_disable_tensorboard: bool = False

    def __post_init__(self):
        """Post initialization."""
        self.run_name = f"{self.exp_name}_{self.env_id}_{self.seed}_{int(time.time())}"
        self.log_dir = os.path.join(BASE_RESULTS_DIR, self.run_name)
        self.video_dir = os.path.join(self.log_dir, "videos")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.cuda else "cpu"
        )

        self.actor_device = torch.device(
            self.actor_device if torch.cuda.is_available() and self.cuda else "cpu"
        )

        assert self.seq_len > 0, "Sequence length must be greater than 0."
        assert (
            self.seq_len == 1 or self.seq_len % 2 == 0
        ), "Sequence length must be 1 or even."
        assert self.replay_buffer_size >= self.learning_starts >= self.batch_size

        self.total_num_envs = self.num_envs_per_actor * self.num_actors

    def load_model(self):
        """Load the model."""
        return self.model_loader(self)

    def get_actor_epsilon(self, actor_idx: int) -> float:
        """Get the epsilon for the actor with index `actor_idx`.

        Similar to the Ape-X paper each actor is assigned a different epsilon value
        for exploration. Specificall actor i in [0, num_actors) has an epsilon of:

            epsilon_i = base_epsilon * (1 - i / num_actors)

        This is different from the original paper where the epsilon is:

            epsilon_i = base_epsilon ** (1 + i / num_actors * alpha)

        The original paper's epsilon schedule doesn't work as well when the number of
        actors is low (it assigns very low epsilon valus to all actors).

        """
        return self.actor_base_epsilon * (1 - actor_idx / self.num_actors)

    def make_env(self) -> gym.Env:
        """Get the environment."""
        return self.env_creator_fn_getter(self, 0, None)()


def parse_r2d2_args() -> R2D2Config:
    """Parse command line arguments for R2D2 algorithm."""
    # ruff: noqa: E501
    # fmt: off
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--exp-name", type=str, default="r2d2",
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

    # Training configuration
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--num-actors", type=int, default=4,
        help="the number of rollout actors collecting trajectories in parallel")
    parser.add_argument("--num-envs-per-actor", type=int, default=8,
        help="the number of parallel game environments")
    parser.add_argument("--actor-update-interval", type=int, default=400,
        help="the number of environment steps between updating actor parameters")
    parser.add_argument("--actor-base-epsilon", type=float, default=0.05,
        help="base epsilon for actor epsilon-greedy exploration")
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
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--batch-size", type=int, default=32,
        help="size of batches")
    parser.add_argument("--n-steps", type=int, default=5,
        help="number of steps for n-step returns")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
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
    parser.add_argument("--trunk-sizes", nargs="+", type=int, default=[64],
        help="size of each layer of the MLP trunk")
    parser.add_argument("--lstm-size", type=int, default=64,
        help="size of LSTM layer")
    parser.add_argument("--head-sizes", nargs="+", type=int, default=[64],
        help="size of each layer of the MLP policy and value heads")
    args = parser.parse_args()
    # fmt: on
    return R2D2Config(**vars(args))
