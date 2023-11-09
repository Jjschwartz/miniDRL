"""The R2D2 algorithm.

From the paper "Recurrent Experience Replay in Distributed Reinforcement Learning"
https://openreview.net/pdf?id=r1lyTjAqYX.

Also with details from the Ape-X paper "Distributed Prioritized Experience Replay"
https://arxiv.org/abs/1803.00933

Reference implementations:
https://github.com/michaelnny/deep_rl_zoo/tree/main

"""
import math
import os
import random
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from multiprocessing.queues import Empty, Full
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from minidrl.config import BASE_RESULTS_DIR
from minidrl.r2d2.replay import run_replay_process


class R2D2Network(nn.Module):
    """Abstract Neural Network class for R2D2.

    All R2D2 networks should inherit from this class or at least implement it's
    interface (i.e. the `forward` method).
    """

    def forward(
        self,
        o: torch.tensor,
        a: torch.tensor,
        r: torch.tensor,
        done: torch.tensor,
        lstm_state: tuple[torch.tensor, : torch.tensor],
    ):
        """Get q-values for each action given inputs.

        T = seq_len
        B = batch_size (i.e. num parallel envs)

        Arguments
        ---------
        o
            The time `t` observation: o_t. Shape=(T, B, *obs_shape).
        a
            The previous (`t-1`) action: a_tm1. Shape=(T, B).
        r
            The previous (`t-1`) reward: r_tm1. Shape=(T, B).
        done
            Whether the episode is ended on last `t-1` step: d_tm1. Shape=(T, B).
        lstm_state
            The previous state of the LSTM. This is a tuple with two entries, each of
            which has shape=(lstm.num_layers, B, lstm_size).

        Returns
        -------
        q
            Q-values for each action for time `t`: q_t. Shape=(T, B, action_space.n).
        lstm_state
            The new state of the LSTM, shape=(lstm.num_layers, B, lstm_size).
        """
        raise NotImplementedError


@dataclass(kw_only=True)
class R2D2Config:
    """Configuration for R2D2 algorithm.

    Note, the `env_creator_fn_getter` and `model_loader` attributes must be set after
    initialization (this is to be compatible with pyrallis).):

    ```python
    config = R2D2Config()
    config.env_creator_fn_getter = my_env_creator_fn_getter
    config.model_loader = my_model_loader
    ```

    For examples see the `minidrl/r2d2/run_atari.py` and `minidrl/r2d2/run_gym.py`
    files.

    """

    # The name of this experiment
    exp_name: str = "r2d2"
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

    # The ID of the gymnasium environment
    env_id: str = "PongNoFrameskip-v4"
    # Whether to capture videos of the agent performances (check out `videos` folder)
    capture_video: bool = False

    # Total number of timesteps to train for
    # This is the total number of steps/transitions/frames used to update the model,
    # not the number of steps taken in the environment by the actors.
    # Note each update will use `batch_size * seq_len` steps.
    total_timesteps: int = 2000000
    # Number of parallel actors to use for collecting trajectories
    num_actors: int = 4
    # Device used by actor models
    actor_device: torch.device = torch.device("cpu")
    # Number of parallel environments per actor
    num_envs_per_actor: int = 16
    # Number of environment steps between updating actor parameters
    actor_update_interval: int = 400
    # Base epsilon for actor epsilon-greedy exploration
    # Each actor has a different epsilon value for exploration. See `get_actor_epsilon`
    # for details.
    actor_base_epsilon: float = 0.4
    # Actor exploration alpha hyperparameter. See `get_actor_epsilon` for details.
    actor_alpha: float = 7.0

    # Length of sequences used for learning that are stored and sampled from the replay
    # buffer
    seq_len: int = 80
    # Length of burn-in sequence for each training sequence
    burnin_len: int = 40
    # Size of replay buffer in terms of number of sequences
    replay_buffer_size: int = 100000
    # Exponent for replay priority
    replay_priority_exponent: float = 0.9
    # Prioritized replay noise
    replay_priority_noise: float = 1e-3
    # Exponent for replay importance sampling
    importance_sampling_exponent: float = 0.6
    # Mean-max TD error mix proportion for priority calculation
    priority_td_error_mix: float = 0.9
    # The Discount factor
    gamma: float = 0.997
    # Number of batches to sample from replay buffer at a time
    # Increasing this will increase the amount of memory used, but should also increase
    # the speed of training, by reducing the overall time spent sampling from replay.
    num_prefetch_batches: int = 8
    # Size (i.e. number of sequence) of each learner update batch
    batch_size: int = 64
    # Number of steps for n-step return
    n_steps: int = 5
    # Learning rate of the optimizer
    learning_rate: float = 1e-4
    # Adam optimizer epsilon
    adam_eps: float = 1e-3
    # Size of replay buffer before learning starts
    learning_starts: int = 1000
    # Target network update interval (in terms of number of learner updates)
    target_network_update_interval: int = 2500
    # Whether to use value function rescaling or not
    value_rescaling: bool = True
    # Epsilon used for value function rescaling
    value_rescaling_epsilon: float = 0.001
    # Whether to clip the gradient norm
    clip_grad_norm: bool = True
    # Maximum gradient norm
    max_grad_norm: float = 40.0

    # Size of LSTM hidden state
    # Should be set based on model being used or will attempt to be inferred from
    # the model, assuming it has an LSTM layer.
    lstm_size_: Optional[int] = None

    # Function that returns an environment creator function
    # Callable[[R2D2Config, int, int | None], Callable[[], gym.Env]
    # Should be set after initialization
    env_creator_fn_getter: callable = field(init=False, default=None)
    # Function for loading model: Callable[[R2D2Config], nn.Module]
    # Should be set after initialization
    model_loader: callable = field(init=False, default=None)

    def __post_init__(self):
        """Post initialization."""
        self.run_name = self.exp_name
        if self.env_id not in self.exp_name:
            self.run_name += f"_{self.env_id}"
        self.run_name += f"_{self.seed}_{int(time.time())}"

        self.actor_device = torch.device(
            self.actor_device if torch.cuda.is_available() and self.cuda else "cpu"
        )

        assert self.seq_len > 0, "Sequence length must be greater than 0."
        assert (
            self.seq_len == 1 or self.seq_len % 2 == 0
        ), "Sequence length must be 1 or even."
        assert self.replay_buffer_size >= self.learning_starts >= self.batch_size

    @property
    def log_dir(self) -> str:
        """Directory where the model and logs will be saved."""
        return os.path.join(BASE_RESULTS_DIR, self.run_name)

    @property
    def video_dir(self) -> str:
        """Directory where videos will be saved."""
        return os.path.join(self.log_dir, "videos")

    @property
    def device(self) -> torch.device:
        """Device where learner model is run."""
        return torch.device(
            "cuda" if torch.cuda.is_available() and self.cuda else "cpu"
        )

    @property
    def num_updates(self) -> int:
        """Number of updates to perform."""
        return math.ceil(self.total_timesteps / (self.batch_size * self.seq_len))

    @property
    def lstm_size(self) -> int:
        """Size of LSTM hidden state."""
        if self.lstm_size_ is None:
            # Attempt to infer lstm size from model
            model = self.load_model()
            for m in model.modules():
                if isinstance(m, torch.nn.LSTM):
                    self.lstm_size_ = m.hidden_size

            assert self.lstm_size_ is not None, (
                "Could not infer LSTM size from model. Please ensure that the model "
                "has an torch.nn.LSTM layer or that the `lstm_size_` attribute is set "
                "in the config."
            )

        return self.lstm_size_

    def load_model(self) -> R2D2Network:
        """Load the model."""
        return self.model_loader(self)

    def get_actor_epsilon(self, actor_idx: int) -> float:
        """Get the epsilon for the actor with index `actor_idx`.

        As per the Ape-X paper, each actor is assigned a different epsilon value
        for exploration. Specifically, actor i in [0, num_actors-1] has an epsilon of:

            epsilon_i = base_epsilon ** (1 + i / (num_actors-1) * alpha)

        Where `base_epsilon` and `alpha` are hyperparameters, set to  0.4 and 7 by
        default (same as Ape-X and R2D2 paper).

        """
        if self.num_actors < 4:
            # If using less than 4 actors, the default function leads to quite high and low
            # epsilons for the low number of actors.
            # So we pretend we have 16 actors, and select the epsilons for the actors
            # from intervals in the middle of the 16 actor epsilons.
            epsilons = np.power(
                self.actor_base_epsilon, 1 + np.arange(16) / (16 - 1) * self.actor_alpha
            )
            return epsilons[(actor_idx + 1) * 16 // (self.num_actors + 3)]
        return self.actor_base_epsilon ** (
            1 + (actor_idx / (self.num_actors - 1)) * self.actor_alpha
        )

    def is_reporting_actor(self, actor_idx: int) -> bool:
        """Whether the actor with index `actor_idx` should report results.

        The actor with exploration epsilon closest to 0.05 will report results. This is
        so that the results are not biased by the exploration epsilon. When comparing
        results across different runs with varying number of actors.
        """
        epsilons = [self.get_actor_epsilon(i) for i in range(self.num_actors)]
        closest_idx = np.argmin(np.abs(np.array(epsilons) - 0.05))
        return closest_idx == actor_idx


def compute_loss_and_priority(
    config: R2D2Config,
    q_values: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    target_q_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the loss and priority for a batch of transitions.

    T = sequence length (i.e. unrolled sequence length, excluding burnin)
    B = batch size
    A = action space size

    Use T+1 since target for time t is calculated using time t+n_steps+1

    Arguments
    ---------
    config
        Configuration for R2D2.
    q_values
        Predicted Q values for step `t`: q_t. Shape (T+1, B, A)
    actions
        Actual actions taken at step `t-1`: a_tm1 Shape (T+1, B)
    rewards
        Rewards received on step `t-1`: r_tm1. Shape (T+1, B)
    dones
        Whether the episode terminated on step `t-1`: d_tm1. Shape (T+1, B)
    target_q_values
        Target Q values for step `t`: q_target_t. Shape (T+1, B, A)

    Returns
    -------
    loss
        The loss for each transition sequence. Shape (B,)
    priorities
        The priority for each transition sequence. Shape (B,)

    """
    # Get the Q value for the action taken at each step. Shape (T, B)
    # we get values for all steps except the last one, and must shift the actions
    # forward by one step since they are for the previous step
    q_actual = torch.gather(
        q_values[:-1], dim=-1, index=actions[1:].unsqueeze(-1)
    ).squeeze(-1)

    # Get the target Q value for the action taken at each step. Shape (T+1, B)
    best_actions = torch.argmax(q_values, dim=-1)
    target_q_max = torch.gather(
        target_q_values, dim=-1, index=best_actions.unsqueeze(-1)
    ).squeeze(-1)

    # Apply signed parabolic rescaling to TD targets
    # This is the inverse to hyperbolic rescaling h(x) = y <=> x = h^(-1)(y)
    # h^(-1)(x) = sign(x)[([sqrt(1 + 4*eps*[eps + 1 + |x|]) - 1] / (2*eps))^2 - 1]
    if config.value_rescaling:
        eps = config.value_rescaling_epsilon
        target_q_max = torch.sign(target_q_max) * (
            torch.square(
                (torch.sqrt(1 + 4 * eps * (eps + 1 + torch.abs(target_q_max))) - 1)
                / (2 * eps)
            )
            - 1
        )

    # Calculate n-step TD target. Shape (T, B)
    # = [sum_{k=0}^{n-1} gamma^k * r_{t+k}] + [gamma^n * Q_target(o_{t+n}, a*)]
    # where a* = argmax_a Q(o_{t+n}, a)

    # Get rewards, dones, target Q values into shape (T + n, B)
    # Append n steps of zeros for rewards and dones
    # Note we skip first step since these are for time t-1
    n_steps, gamma = config.n_steps, config.gamma
    dones = torch.cat(
        [dones[1:]] + [torch.zeros_like(dones[0:1])] * n_steps, dim=0
    ).float()
    rewards = torch.cat(
        [rewards[1:]] + [torch.zeros_like(rewards[0:1])] * n_steps, dim=0
    )
    # For target Q values, we append n-1 steps of final Q_target
    # Final n-1 steps are divided by gamma**k to correct for fact they are "fake"
    # Note first step will be ignored since we use targets for time t+1
    bellman_target = torch.cat(
        [target_q_max] + [target_q_max[-1:] / gamma**k for k in range(1, n_steps)],
        dim=0,
    )
    # iteratively calculate n-step TD targets
    # Outputs shape=(T, B)
    for _ in range(n_steps):
        rewards = rewards[:-1]
        dones = dones[:-1]
        bellman_target = rewards + gamma * (1 - dones) * bellman_target[1:]

    # signed hyperbolic rescaling
    # h(x) = sign(x)(sqrt(|x| + 1) - 1) + eps*x
    if config.value_rescaling:
        eps = config.value_rescaling_epsilon
        bellman_target = (
            torch.sign(bellman_target) * (torch.sqrt(torch.abs(bellman_target) + 1) - 1)
            + eps * bellman_target
        )

    # Calculate the TD Error
    td_error = bellman_target - q_actual

    # Calculate priorities
    # p = eta * max(|td_error|) + (1 - eta) * mean(|td_error|)
    # Shape (B,)
    with torch.no_grad():
        eta = config.priority_td_error_mix
        abs_td_error = torch.abs(td_error)
        priorities = eta * torch.max(abs_td_error, dim=0)[0] + (1 - eta) * torch.mean(
            abs_td_error, dim=0
        )
        # Clamp priorities to avoid NaNs
        priorities = torch.clamp(priorities, min=0.0001)

    # Calculate the loss. Shape (B,)
    # Summing over time dimension
    loss = 0.5 * torch.sum(torch.square(td_error), dim=0)

    return loss, priorities


def run_actor(
    actor_idx: int,
    config: R2D2Config,
    replay_queue: mp.JoinableQueue,
    learner_send_queue: mp.JoinableQueue,
    learner_recv_queue: mp.JoinableQueue,
    terminate_event: mp.Event,
):
    """Run an R2D2 actor process.

    Each R2D2 process collects trajectories and sends them to replay while periodically
    requesting the latest model parameters from the learner.

    Arguments
    ---------
    actor_idx
        The index of the actor.
    config
        Configuration for R2D2.
    replay_queue
        Queue for sending trajectories to replay.
    learner_send_queue
        Queue for sending results and parameter requests to learner.
    learner_recv_queue
        Queue for receiving parameters from learner.
    terminate_event
        Event for signaling termination of run.
    """
    print(f"actor={actor_idx}: Actor started.")

    # Limit each actor to using a single CPU thread.
    # This prevents each actor from using all available cores, which can lead to each
    # actor being slower due to contention.
    torch.set_num_threads(1)

    # seeding
    np_rng = np.random.RandomState(actor_idx)

    epsilon = config.get_actor_epsilon(actor_idx)
    is_reporting_actor = config.is_reporting_actor(actor_idx)
    device = config.actor_device

    print(f"actor={actor_idx}: {device=} {epsilon=:.4f}.")

    # env setup
    # Note: SyncVectorEnv runs multiple-env instances serially.
    envs = gym.vector.SyncVectorEnv(
        [
            config.env_creator_fn_getter(config, env_idx=i, actor_idx=actor_idx)
            for i in range(config.num_envs_per_actor)
        ]
    )
    obs_space = envs.single_observation_space
    action_space = envs.single_action_space
    action_dim = action_space.n
    assert isinstance(obs_space, gym.spaces.Box)
    num_envs = config.num_envs_per_actor

    # model setup
    actor_model = config.load_model()
    actor_model.to(device)
    # disable autograd for actor model
    for param in actor_model.parameters():
        param.requires_grad = False

    # Sequence buffer setup
    total_seq_len = config.seq_len + config.burnin_len
    # timesteps to drop from the start of the sequence after sequence added to replay
    seq_drop_len = max(1, config.seq_len // 2)
    # sequence buffers: o_t, a_tm1, r_tm1, d_tm1, q_t, h_0
    # stores burnin_len + seq_len + 1 timesteps
    obs_seq_buffer = torch.zeros(
        (total_seq_len + 1, num_envs, *obs_space.shape), dtype=torch.float32
    )
    action_seq_buffer = torch.zeros((total_seq_len + 1, num_envs), dtype=torch.long)
    reward_seq_buffer = torch.zeros((total_seq_len + 1, num_envs), dtype=torch.float32)
    done_buffer = torch.zeros((total_seq_len + 1, num_envs), dtype=torch.long)
    q_vals_seq_buffer = torch.zeros(
        (total_seq_len + 1, num_envs, action_dim), dtype=torch.float32
    )
    lstm_h_seq_buffer = torch.zeros(
        (1, num_envs, config.lstm_size), dtype=torch.float32
    )
    lstm_c_seq_buffer = torch.zeros(
        (1, num_envs, config.lstm_size), dtype=torch.float32
    )

    # Stores init lstm state for the next sequence
    lstm_state_queue = []

    # Step variables
    # R2D2 network expects inputs of shape (seq_len, batch_size, ...)
    # Hence why we add timestep dimension
    obs = torch.Tensor(envs.reset()[0]).to(device).unsqueeze(0)
    prev_action = torch.zeros((1, num_envs), dtype=torch.long).to(device)
    prev_reward = torch.zeros((1, num_envs), dtype=torch.float32).to(device)
    prev_done = torch.zeros((1, num_envs), dtype=torch.long).to(device)
    q_vals = torch.zeros((num_envs, action_dim), dtype=torch.float32).to(device)
    lstm_state = (
        torch.zeros((1, num_envs, config.lstm_size), dtype=torch.float32).to(device),
        torch.zeros((1, num_envs, config.lstm_size), dtype=torch.float32).to(device),
    )

    step = 0
    current_seq_len, num_episodes_completed = 0, 0
    # buffer for storing last 100 episodes returns
    episode_returns = np.zeros(100, dtype=np.float32)
    episode_lengths = np.zeros(100, dtype=np.int64)
    start_time = time.time()

    # get reference to learner model weights
    learner_send_queue.put(("get_model_params", actor_idx))
    learner_model_weights = None
    while not terminate_event.is_set():
        try:
            result = learner_recv_queue.get(timeout=1)
            assert result[0] == "params"
            learner_model_weights = result[1]
            actor_model.load_state_dict(result[1])
            break
        except Empty:
            pass

    # main loop
    while not terminate_event.is_set():
        if step % config.actor_update_interval == 0:
            actor_model.load_state_dict(learner_model_weights)

        # q_t = Q(o_t, a_tm1, r_tm1, d_tm1)
        q_vals, lstm_state = actor_model.forward(
            obs, prev_action, prev_reward, prev_done, lstm_state
        )
        # remove timestep dimension shape=(num_envs, action_dim)
        q_vals = q_vals.squeeze(0)

        if np_rng.rand() < epsilon:
            action = torch.from_numpy(
                np.random.randint(0, action_dim, size=(num_envs,))
            ).to(device)
        else:
            action = torch.argmax(q_vals, dim=-1)

        # o_tp1, r_t, d_t, i_t = env.step(a_t)
        next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())

        # add to sequence buffers: o_t, a_tm1, r_tm1, d_tm1, q_t
        obs_seq_buffer[current_seq_len] = obs
        action_seq_buffer[current_seq_len] = prev_action
        reward_seq_buffer[current_seq_len] = prev_reward
        done_buffer[current_seq_len] = prev_done
        q_vals_seq_buffer[current_seq_len] = q_vals

        # update step variables (adding timestep dimension)
        obs = torch.Tensor(next_obs).to(device).unsqueeze(0)
        prev_action = action.unsqueeze(0)
        prev_reward = torch.from_numpy(reward).float().to(device).unsqueeze(0)
        prev_done = (
            torch.from_numpy(terminated | truncated).long().to(device).unsqueeze(0)
        )

        current_seq_len += 1
        if current_seq_len == total_seq_len + 1:
            # sequence is full, add to replay buffer

            # calculate priority, skipping the burnin steps
            # and using the same q values as target q values
            _, priority = compute_loss_and_priority(
                config=config,
                q_values=q_vals_seq_buffer[config.burnin_len :],
                actions=action_seq_buffer[config.burnin_len :],
                rewards=reward_seq_buffer[config.burnin_len :],
                dones=done_buffer[config.burnin_len :],
                target_q_values=q_vals_seq_buffer[config.burnin_len :],
            )

            while not terminate_event.is_set():
                try:
                    replay_queue.put(
                        (
                            obs_seq_buffer.clone(),
                            action_seq_buffer.clone(),
                            reward_seq_buffer.clone(),
                            done_buffer.clone(),
                            lstm_h_seq_buffer.clone(),
                            lstm_c_seq_buffer.clone(),
                            priority.clone(),
                        ),
                        block=True,  # block until space available
                        timeout=1,  # timeout after 1 second, to check for terminate
                    )
                    break
                except Full:
                    pass

            # reset sequence buffers, keeping the last burnin + seq_len / 2 steps
            # from current buffer
            current_seq_len = config.burnin_len + config.seq_len // 2 + 1

            # drop first seq_len // 2 steps from the sequence buffers
            obs_seq_buffer = obs_seq_buffer.roll(-seq_drop_len, dims=0)
            action_seq_buffer = action_seq_buffer.roll(-seq_drop_len, dims=0)
            reward_seq_buffer = reward_seq_buffer.roll(-seq_drop_len, dims=0)
            done_buffer = done_buffer.roll(-seq_drop_len, dims=0)

            # reset sequences init lstm state from the queue
            buffer_lstm_state = lstm_state_queue.pop(0)
            lstm_h_seq_buffer[0:] = buffer_lstm_state[0]
            lstm_c_seq_buffer[0:] = buffer_lstm_state[1]

        if current_seq_len > 0 and current_seq_len % seq_drop_len == 0:
            # add lstm state to queue for next sequence
            lstm_state_queue.append(lstm_state)

        step += 1

        if is_reporting_actor:
            # accumulate and send results to learner
            # only send from first actor since it will have the same exploration epsilon
            # irrespective of the number of actors
            for item in [
                i
                for i in info.get("final_info", [])
                if i is not None and "episode" in i
            ]:
                episode_returns[num_episodes_completed % 100] = item["episode"]["r"][0]
                episode_lengths[num_episodes_completed % 100] = item["episode"]["l"][0]
                num_episodes_completed += 1

            if step > 0 and step % 1000 == 0:
                # print(f"actor={actor_idx} - t={step}: Sending results to learner.")
                sps = int(step * config.num_envs_per_actor) / (time.time() - start_time)
                ep_returns = episode_returns[:num_episodes_completed]
                ep_lengths = episode_lengths[:num_episodes_completed]
                results = {
                    "actor_steps": step * config.num_envs_per_actor,
                    "actor_sps": sps,
                    "estimated_total_actor_sps": sps * config.num_actors,
                    "mean_episode_returns": ep_returns.mean().item(),
                    "max_episode_returns": ep_returns.max().item(),
                    "min_episode_returns": ep_returns.min().item(),
                    "mean_episode_lengths": ep_lengths.mean().item(),
                    "actor_episodes_completed": num_episodes_completed,
                }
                learner_send_queue.put(("send_results", results))

    print(f"actor={actor_idx} - t={step} terminate signal recieved.")
    envs.close()

    # delete references to shared memory, and signal that it's been done
    del learner_model_weights
    learner_recv_queue.task_done()

    print(f"actor={actor_idx} - t={step}: Waiting for shared resources to be released.")
    replay_queue.join()
    learner_send_queue.join()

    print(f"actor={actor_idx} - t={step}: Actor Finished.")


def run_learner(
    config: R2D2Config,
    replay_send_queue: mp.JoinableQueue,
    replay_recv_queue: mp.JoinableQueue,
    actor_recv_queue: mp.JoinableQueue,
    actor_send_queues: list[mp.JoinableQueue],
    termination_event: mp.Event,
):
    """Run the R2D2 learner process.

    The learner process samples trajectories from replay and updates the model.

    Arguments
    ---------
    config
        Configuration for R2D2.
    replay_send_queue
        Queue for sending requests to the replay.
    replay_rec_queue
        Queue for recieving trajectories from the replay.
    actor_recv_queue
        Queue for receiving results and requests for parameters from actors.
    actor_send_queues
        Queues (one for each actor) for sending parameters to each actor.
    termination_event
        Event for signaling termination of run.
    """

    # Limit learner to using a single CPU thread.
    # This prevents learner from using all available cores, which can lead to contention
    # with actor and replay processes.
    torch.set_num_threads(1)

    # Logging setup
    # Do this after workers are spawned to avoid log duplication
    if config.track_wandb:
        import wandb

        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            sync_tensorboard=True,
            config=vars(config),
            name=config.run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(config.log_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(config).items()])),
    )
    uploaded_video_files = set()

    # model setup
    device = config.device
    model = config.load_model()
    model.to(device)

    target_model = config.load_model()
    target_model.to(device)
    target_model.load_state_dict(model.state_dict())

    # Optimizer setup
    optimizer = optim.Adam(
        model.parameters(), lr=config.learning_rate, eps=config.adam_eps
    )

    # Global training step
    # Careful: shared beween threads on learner process so needs to be set up before
    # defining thread function
    global_step = 0
    update = 1

    # Setup background thread
    end_training_event = threading.Event()

    def run_background_thread():
        """Runs a thread that handles various background tasks."""
        print("learner: param_sync_thread - started")
        last_replay_log_time = time.time()
        last_video_upload_time = time.time()
        replay_stats_requested = False
        while not end_training_event.is_set() and not termination_event.is_set():
            try:
                # set timeout so that we can check if training is done
                request = actor_recv_queue.get(timeout=1)
                if request[0] == "get_model_params":
                    actor_recv_queue.task_done()
                    actor_send_queues[request[1]].put(("params", model.state_dict()))
                elif request[0] == "send_results":
                    # log actor results
                    for key, value in request[1].items():
                        writer.add_scalar(f"actor/{key}", value, global_step)

                    result_output = [f"\nlearner: actor results (step={global_step})"]
                    for key, value in request[1].items():
                        # log to stdout
                        if isinstance(value, float):
                            result_output.append(f"{key}={value:0.4f}")
                        else:
                            result_output.append(f"{key}={value}")
                    print("\n  ".join(result_output))
                    actor_recv_queue.task_done()
                else:
                    actor_recv_queue.task_done()
                    raise ValueError(f"Unknown request {request}")
            except Empty:
                pass

            if time.time() - last_replay_log_time >= 10:
                if not replay_stats_requested:
                    replay_send_queue.put(("get_replay_stats", None))
                    replay_stats_requested = True
                try:
                    replay_stats = replay_recv_queue.get(timeout=1)
                    replay_recv_queue.task_done()

                    for key, value in replay_stats.items():
                        writer.add_scalar(f"replay/{key}", value, global_step)

                    result_output = [
                        f"\nlearner: replay stats (step={global_step} update={update})"
                    ]
                    for key, value in replay_stats.items():
                        # log to stdout
                        if isinstance(value, float):
                            result_output.append(f"{key}={value:0.4f}")
                        else:
                            result_output.append(f"{key}={value}")
                    print("\n  ".join(result_output))
                    replay_stats_requested = False
                    last_replay_log_time = time.time()
                except Empty:
                    pass

            if (
                config.capture_video
                and config.track_wandb
                and time.time() - last_video_upload_time >= 5
            ):
                video_filenames = [
                    fname
                    for fname in os.listdir(config.video_dir)
                    if fname.endswith(".mp4") and fname not in uploaded_video_files
                ]
                if video_filenames:
                    print(f"learner: uploading {len(video_filenames)} videos")
                    video_filenames.sort()
                    for filename in video_filenames:
                        wandb.log(  # type:ignore
                            {
                                "video": wandb.Video(  # type:ignore
                                    os.path.join(config.video_dir, filename)
                                )
                            }
                        )
                        uploaded_video_files.add(filename)
                last_video_upload_time = time.time()

        print("learner: background_thread - done")

    background_thread = threading.Thread(target=run_background_thread)
    background_thread.start()

    # Wait for actors to start and fill replay buffer
    print("learner: Waiting for actors to start and fill replay buffer...")
    replay_send_queue.put(("get_buffer_size", None))
    while not termination_event.is_set():
        try:
            buffer_size = replay_recv_queue.get(timeout=1)
            replay_recv_queue.task_done()
            assert buffer_size >= config.learning_starts
            break
        except Empty:
            pass

    # Training loop
    print("learner: Starting training loop...")
    train_start_time, last_report_time = time.time(), time.time()
    while update < config.num_updates + 1 and not termination_event.is_set():
        update_start_time = time.time()
        # samples: (T, B, ...), indices: (B,), weights: (B,)
        replay_send_queue.put(
            ("sample", config.batch_size * config.num_prefetch_batches)
        )
        batch = None
        while not termination_event.is_set():
            try:
                batch = replay_recv_queue.get(timeout=1)
                break
            except Empty:
                pass
        if termination_event.is_set():
            if batch is not None:
                del batch
                replay_recv_queue.task_done()
            break

        sample_time = time.time() - update_start_time

        (obs, actions, rewards, dones, lstm_h, lstm_c), indices, weights = batch

        burnin_time, learning_time = 0.0, 0.0
        updated_priorities = []
        q_values = 0.0
        for batch_idx in range(config.num_prefetch_batches):
            start_idx = batch_idx * config.batch_size
            end_idx = start_idx + config.batch_size

            b_obs = obs[:, start_idx:end_idx]
            b_actions = actions[:, start_idx:end_idx]
            b_rewards = rewards[:, start_idx:end_idx]
            b_dones = dones[:, start_idx:end_idx]
            b_lstm_h = lstm_h[:, start_idx:end_idx]
            b_lstm_c = lstm_c[:, start_idx:end_idx]
            b_weights = weights[start_idx:end_idx]

            b_target_lstm_h, b_target_lstm_c = b_lstm_h.clone(), b_lstm_c.clone()

            burnin_time_start = time.time()
            # burn in lstm state
            if config.burnin_len > 0:
                with torch.no_grad():
                    _, (b_lstm_h, b_lstm_c) = model.forward(
                        b_obs[: config.burnin_len],
                        b_actions[: config.burnin_len],
                        b_rewards[: config.burnin_len],
                        b_dones[: config.burnin_len],
                        (b_lstm_h, b_lstm_c),
                    )
                    _, (b_target_lstm_h, b_target_lstm_c) = model.forward(
                        b_obs[: config.burnin_len],
                        b_actions[: config.burnin_len],
                        b_rewards[: config.burnin_len],
                        b_dones[: config.burnin_len],
                        (b_target_lstm_h, b_target_lstm_c),
                    )
            burnin_time = time.time() - burnin_time_start

            # compute loss and gradients and update parameters
            learning_start_time = time.time()

            q_values, _ = model.forward(
                b_obs[config.burnin_len :],
                b_actions[config.burnin_len :],
                b_rewards[config.burnin_len :],
                b_dones[config.burnin_len :],
                (b_lstm_h, b_lstm_c),
            )

            with torch.no_grad():
                target_q_values, _ = target_model.forward(
                    b_obs[config.burnin_len :],
                    b_actions[config.burnin_len :],
                    b_rewards[config.burnin_len :],
                    b_dones[config.burnin_len :],
                    (b_target_lstm_h, b_target_lstm_c),
                )

            loss, priorities = compute_loss_and_priority(
                config=config,
                q_values=q_values,
                actions=b_actions[config.burnin_len :],
                rewards=b_rewards[config.burnin_len :],
                dones=b_dones[config.burnin_len :],
                target_q_values=target_q_values,
            )
            loss = torch.mean(loss * b_weights.detach())

            updated_priorities.extend(priorities.tolist())

            optimizer.zero_grad()
            loss.backward()
            if config.clip_grad_norm:
                unclipped_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.max_grad_norm
                )
            else:
                unclipped_grad_norm = None
            optimizer.step()

            learning_time = time.time() - learning_start_time

            if update > 1 and update % config.target_network_update_interval == 0:
                target_model.load_state_dict(model.state_dict())

            # log metrics
            update += 1
            global_step += config.batch_size * config.seq_len
            q_max = torch.mean(torch.max(q_values, dim=-1)[0])

            writer.add_scalar(
                "losses/learning_rate", optimizer.param_groups[0]["lr"], global_step
            )
            writer.add_scalar("losses/value_loss", loss.item(), global_step)
            writer.add_scalar("losses/q_max", q_max.item(), global_step)
            writer.add_scalar(
                "losses/mean_priorities", priorities.mean().item(), global_step
            )
            writer.add_scalar(
                "losses/max_priorities", priorities.max().item(), global_step
            )
            writer.add_scalar(
                "losses/min_priorities", priorities.min().item(), global_step
            )
            if unclipped_grad_norm is not None:
                writer.add_scalar(
                    "losses/unclipped_grad_norm",
                    unclipped_grad_norm.item(),
                    global_step,
                )
            writer.add_scalar("charts/update", update, global_step)
            writer.add_scalar("charts/burnin_time", burnin_time, global_step)
            writer.add_scalar("charts/learning_time", learning_time, global_step)

            if config.save_interval > 0 and update % config.save_interval == 0:
                print("Saving model")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "global_step": global_step,
                        "update": update,
                        "config": asdict(config),
                    },
                    os.path.join(config.log_dir, f"checkpoint_{update}.pt"),
                )

            if batch_idx == 4:
                del b_obs, b_actions, b_rewards, b_dones, b_lstm_h, b_lstm_c

        replay_send_queue.put(("update_priorities", indices, updated_priorities))

        # release shared memory references
        del obs, actions, rewards, dones, lstm_h, lstm_c, indices, weights, batch
        replay_recv_queue.task_done()

        ups = update / (time.time() - train_start_time)
        sps = global_step / (time.time() - train_start_time)
        update_time = time.time() - update_start_time
        writer.add_scalar("charts/learner_SPS", sps, global_step)
        writer.add_scalar("charts/learner_UPS", ups, global_step)
        writer.add_scalar("charts/sample_time", sample_time, global_step)
        writer.add_scalar("charts/update_time", update_time, global_step)
        writer.add_scalar(
            "charts/sample_time_per_batch",
            sample_time / config.num_prefetch_batches,
            global_step,
        )
        writer.add_scalar(
            "charts/update_time_per_batch",
            update_time / config.num_prefetch_batches,
            global_step,
        )

        if time.time() - last_report_time > 5:
            # periodically log to stdout
            training_time = timedelta(seconds=int(time.time() - train_start_time))
            output = "\n".join(
                [
                    f"\nlearner: time={training_time} {update=} {global_step=}",
                    f"  UPS={ups:.1f}",
                    f"  SPS={sps:.1f}",
                ]
            )
            print(output)
            last_report_time = time.time()

    if termination_event.is_set():
        print("learner: Training terminated early due to error.")
    else:
        # set termination event so replay and actor processes know to stop
        # and to signal to main that learner finished as expected
        termination_event.set()
        print("learner: Training complete.")

    print("learner: Shutting down background thread.")
    end_training_event.set()
    background_thread.join()

    writer.close()

    print("learner: waiting for shared resources to be released")
    replay_send_queue.join()
    for q in actor_send_queues:
        q.join()

    print("learner: All done.")


def run_r2d2(config: R2D2Config):
    """Run R2D2.

    This function spawns the replay, actor, and learner processes, and sets up
    communication between them. It also handles cleanup after training is finished
    or in the event an error occurs on any of the processes.

    Arguments
    ---------
    config
        Configuration for R2D2.

    """
    # seeding
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic

    print("Running R2D2:")
    print(f"Env ID: {config.env_id}")

    print("main: spawning replay, actor, and learner processes.")
    # `fork` not supported by CUDA
    # https://pytorch.org/docs/main/notes/multiprocessing.html#cuda-in-multiprocessing
    # must use context to set start method
    mp_ctxt = mp.get_context("spawn")

    # event for signaling end of training
    terminate_event = mp_ctxt.Event()

    # create queues for communication between learner, actors, and replay
    actor_replay_queue = mp_ctxt.JoinableQueue(maxsize=100)
    actor_recv_queue = mp_ctxt.JoinableQueue()
    actor_send_queues = [mp_ctxt.JoinableQueue() for _ in range(config.num_actors)]
    replay_send_queue = mp_ctxt.JoinableQueue()
    replay_recv_queue = mp_ctxt.JoinableQueue()

    print("main: Spawning replay process.")
    replay_process = mp_ctxt.Process(
        target=run_replay_process,
        args=(
            config,
            actor_replay_queue,
            replay_send_queue,
            replay_recv_queue,
            terminate_event,
        ),
    )
    replay_process.start()

    print("main: Spawning actor processes.")
    actors = []
    for actor_idx in range(config.num_actors):
        actor = mp_ctxt.Process(
            target=run_actor,
            args=(
                actor_idx,
                config,
                actor_replay_queue,
                actor_recv_queue,
                actor_send_queues[actor_idx],
                terminate_event,
            ),
        )
        actor.start()
        actors.append(actor)

    print("main: Spawning learner process.")
    learner = mp_ctxt.Process(
        target=run_learner,
        args=(
            config,
            replay_send_queue,
            replay_recv_queue,
            actor_recv_queue,
            actor_send_queues,
            terminate_event,
        ),
    )
    learner.start()

    print("main: Waiting for replay, actor, and learner processes to finish.")
    while (
        not terminate_event.is_set()
        and learner.is_alive()
        and replay_process.is_alive()
        and all(a.is_alive() for a in actors)
    ):
        time.sleep(1)

    print("main: Training ended, shutting down.")
    if terminate_event.is_set():
        print("main: Learner process finished training as expected.")
    elif not learner.is_alive():
        print("main: Learner process crashed.")
    elif not replay_process.is_alive():
        print("main: Replay process crashed.")
    else:
        for i, actor in enumerate(actors):
            if not actor.is_alive():
                print(f"main: Actor {i} process crashed.")

    if not terminate_event.is_set():
        terminate_event.set()

    # give time for processes to close and stop adding to queues
    time.sleep(2)

    print("main: Draining and closing replay and actor queues.")
    for q in [
        actor_replay_queue,
        replay_send_queue,
        replay_recv_queue,
        actor_recv_queue,
    ] + actor_send_queues:
        while not q.empty():
            q.get()
            q.task_done()
        q.close()

    print("main: Joining replay process.")
    replay_process.join()

    print("main: Joining actors.")
    for i in range(config.num_actors):
        actors[i].join()

    print("main: Joining learner.")
    learner.join()

    print("main: Replay, actor and learner processes successfully joined.")
    print("main: All done")
