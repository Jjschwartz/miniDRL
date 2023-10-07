"""The R2D2 algorithm.

From the paper "Recurrent Experience Replay in Distributed Reinforcement Learning"
https://openreview.net/pdf?id=r1lyTjAqYX.

Also with details from the Ape-X paper "Distributed Prioritized Experience Replay"
https://arxiv.org/abs/1803.00933

https://github.com/michaelnny/deep_rl_zoo/tree/main


TODO:
- [ ] Test for any issues with multiprocessing race-conditions
- [ ] Evaluation worker
- [ ] Zero out prev action and reward for first step in episode?
"""
import os
import random
import threading
import time
from dataclasses import asdict, dataclass, field
from multiprocessing.queues import Empty
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from minidrl.config import BASE_RESULTS_DIR
from minidrl.r2d2.replay import run_replay_process


@dataclass
class R2D2Config:
    """Configuration for R2D2 algorithm.

    Note, the `env_creator_fn_getter` and `model_loader` attributes must be set after
    initialization:

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
    env_id: str = "CartPole-v1"
    # Whether to capture videos of the agent performances (check out `videos` folder)
    capture_video: bool = False

    # Total number of timesteps to train for
    total_timesteps: int = 10000000
    # Number of parallel actors to use for collecting trajectories
    num_actors: int = 4
    # Device used by actor models
    actor_device: torch.device = torch.device("cpu")
    # Number of parallel environments per actor
    num_envs_per_actor: int = 16
    # Number of environment steps between updating actor parameters
    actor_update_interval: int = 400
    # base epsilon for actor epsilon-greedy exploration
    actor_base_epsilon: float = 0.05

    # Length of sequences stored and sampled from the replay buffer
    seq_len: int = 80
    # Length of burn-in sequence for each training sequence
    burnin_len: int = 40
    # Size of replay buffer (i.e. number of sequences)
    replay_buffer_size: int = 100000
    # Exponent for replay priority
    replay_priority_exponent: float = 0.9
    # Prioritized replay noise
    replay_priority_noise: float = 1e-3
    # Exponent for importance sampling
    importance_sampling_exponent: float = 0.6
    # Mean-max TD error mix proportion for priority calculation
    priority_td_error_mix: float = 0.9
    # The Discount factor
    gamma: float = 0.997
    # Size (i.e. number of sequence) of each learner update batch
    batch_size: int = 64
    # Number of steps for n-step return
    n_steps: int = 5
    # Learning rate of the optimizer
    learning_rate: float = 1e-4
    # Adam optimizer epsilon
    adam_eps: float = 1e-3
    # Size of replay buffer before learning starts
    learning_starts: int = 500
    # Target network update interval (in terms of number of updates)
    target_network_update_interval: int = 2500
    # Whether to use value function rescaling or not
    value_rescaling: bool = True
    # Epsilon used for value function rescaling
    value_rescaling_epsilon: float = 0.001
    # Whether to clip the gradient norm
    clip_grad_norm: bool = True
    # Maximum gradient norm
    max_grad_norm: float = 0.5

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
    def total_num_envs(self) -> int:
        """Total number of environments."""
        return self.num_envs_per_actor * self.num_actors

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

    def load_model(self):
        """Load the model."""
        return self.model_loader(self)

    def get_actor_epsilon(self, actor_idx: int) -> float:
        """Get the epsilon for the actor with index `actor_idx`.

        Similar to the Ape-X paper each actor is assigned a different epsilon value
        for exploration. Specifically, actor i in [0, num_actors) has an epsilon of:

            epsilon_i = base_epsilon * (1 - i / num_actors)

        This is different from the original paper where the epsilon is:

            epsilon_i = base_epsilon ** (1 + i / num_actors * alpha)

        The original paper's epsilon schedule doesn't work as well when the number of
        actors is low (it assigns very low epsilon valus to all actors).

        """
        return self.actor_base_epsilon * (1 - actor_idx / self.num_actors)


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
    config : Configuration for R2D2.
    q_values : Predicted Q values for step `t`: q_t. Shape (T+1, B, A)
    actions : Actual actions taken at step `t-1`: a_tm1 Shape (T+1, B)
    rewards : Rewards received on step `t-1`: r_tm1. Shape (T+1, B)
    dones : Whether the episode terminated on step `t-1`: d_tm1. Shape (T+1, B)
    target_q_values : Target Q values for step `t`: q_target_t. Shape (T+1, B, A)

    Returns
    -------
    loss : The loss for each transition sequence. Shape (B,)
    priorities : The priority for each transition sequence. Shape (B,)

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
    # [sum_{k=0}^{n-1} gamma^k * r_{t+k}] + [gamma^n * Q_target(o_{t+n}, a*)]
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
    replay_queue: mp.Queue,
    learner_send_queue: mp.Queue,
    learner_recv_queue: mp.Queue,
):
    """Run an R2D2 actor process that collects trajectories and sends them to replay.

    Arguments
    ---------
    actor_idx : The index of the actor.
    config : Configuration for R2D2.
    replay_queue : Queue for sending trajectories to replay.
    learner_send_queue : Queue for sending results and parameter requests to learner.
    learner_recv_queue : Queue for receiving parameters and other signals from learner.
    """
    print(f"actor={actor_idx}: Actor started.")

    # Limit each actor to using a single CPU thread.
    # This prevents each actor from using all available cores, which can lead to each
    # actor being slower due to contention.
    torch.set_num_threads(1)

    # seeding
    np_rng = np.random.RandomState(actor_idx)

    epsilon = config.get_actor_epsilon(actor_idx)
    device = config.actor_device

    print(f"actor={actor_idx}: {device=} {epsilon=:.4f}.")
    print(f"actor={actor_idx}")

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
    current_seq_len = 0
    episode_returns = np.zeros(100, dtype=np.float32)
    episode_lengths = np.zeros(100, dtype=np.int64)
    num_episodes_completed = 0
    start_time = time.time()
    # main loop - runs until learner sends end signal
    while True:
        # check if learner has finished training
        try:
            learner_recv_queue.get_nowait()
            print(f"actor={actor_idx} - t={step}: End training signal recieved.")
            break
        except Empty:
            pass

        if step % config.actor_update_interval == 0:
            # request latest model parameters from learner
            # print(f"actor={actor_idx} - t={step}: Updating actor model.")
            learner_send_queue.put(("get_latest_params", actor_idx))
            result = learner_recv_queue.get()
            # need to handle case where exit signal has been sent in the meantime
            if result[0] == "params":
                actor_model.load_state_dict(result[1])
                del result
            else:
                assert result[0] == "terminate"
                print(f"actor={actor_idx} - t={step}: End training signal recieved.")
                break

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

            replay_queue.put(
                (
                    obs_seq_buffer,
                    action_seq_buffer,
                    reward_seq_buffer,
                    done_buffer,
                    lstm_h_seq_buffer,
                    lstm_c_seq_buffer,
                    priority,
                )
            )

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

        if actor_idx == config.num_actors - 1:
            # accumulate and send results to learner
            # only send from last actor since it has smallest exploration epsilon
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
                    "actor_steps": step,
                    "actor_sps": sps,
                    "mean_episode_returns": ep_returns.mean().item(),
                    "max_episode_returns": ep_returns.max().item(),
                    "min_episode_returns": ep_returns.min().item(),
                    "mean_episode_lengths": ep_lengths.mean().item(),
                    "actor_episodes_completed": num_episodes_completed,
                }
                learner_send_queue.put(("send_results", results))

        step += 1

    envs.close()

    # wait for queue to empty
    print(f"actor={actor_idx} - t={step}: Waiting for queues to empty.")
    while not replay_queue.empty():
        time.sleep(1)

    print(f"actor={actor_idx} - t={step}: Actor Finished.")


def run_r2d2(config: R2D2Config):
    """Run R2D2."""
    # seeding
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic

    # env setup
    # created here for generating model
    env = config.env_creator_fn_getter(config, env_idx=0, actor_idx=None)()
    obs_space = env.observation_space
    act_space = env.action_space

    print("Running R2D2:")
    print(f"Env-id: {config.env_id}")
    print(f"Observation space: {obs_space}")
    print(f"Action space: {act_space}")

    # model setup
    device = config.device
    model = config.load_model()
    model.to(device)

    target_model = config.load_model()
    target_model.to(device)
    target_model.load_state_dict(model.state_dict())

    # Actor Setup
    # `fork` not supported by CUDA
    # https://pytorch.org/docs/main/notes/multiprocessing.html#cuda-in-multiprocessing
    # must use context to set start method
    mp_ctxt = mp.get_context("spawn")

    # create queues for communication between learner, actors, and replay
    actor_replay_queue = mp_ctxt.Queue(maxsize=100)
    actor_recv_queue = mp_ctxt.Queue()
    actor_send_queues = [mp_ctxt.Queue() for _ in range(config.num_actors)]
    replay_send_queue = mp_ctxt.Queue()
    replay_recv_queue = mp_ctxt.Queue()

    # spawn replay process
    replay_process = mp_ctxt.Process(
        target=run_replay_process,
        args=(
            config,
            actor_replay_queue,
            replay_send_queue,
            replay_recv_queue,
        ),
    )
    replay_process.start()

    # spawn actor processes
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
            ),
        )
        actor.start()
        actors.append(actor)

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

    # Optimizer setup
    optimizer = optim.Adam(
        model.parameters(), lr=config.learning_rate, eps=config.adam_eps
    )

    # Global training step
    # Careful: shared beween threads on learner process so needs to be set up before
    # defining thread function
    global_step = 0

    # Setup parameter synchronization thread
    parameter_lock = threading.Lock()
    end_training_event = threading.Event()

    def run_param_sync_thread():
        """Runs a thread that synchronizes parameters between learner and actors."""
        print("param_sync_thread: started")
        while not end_training_event.is_set():
            try:
                # set timeout so that we can check if training is done
                request = actor_recv_queue.get(timeout=1)
                if request[0] == "get_latest_params":
                    with parameter_lock:
                        model_state = {
                            k: v.cpu() for k, v in model.state_dict().items()
                        }
                    actor_send_queues[request[1]].put(("params", model_state))
                elif request[0] == "send_results":
                    # log actor results
                    for key, value in request[1].items():
                        writer.add_scalar(f"charts/{key}", value, global_step)

                    result_output = [f"\nlearner: actor results (step={global_step})"]
                    for key, value in request[1].items():
                        # log to stdout
                        if isinstance(value, float):
                            result_output.append(f"{key}={value:0.4f}")
                        else:
                            result_output.append(f"learner: {key}={value}")
                    print("\n  ".join(result_output))

                else:
                    raise ValueError(f"Unknown request {request}")
            except Empty:
                pass

        print("param_sync_thread: done")

    param_sync_thread = threading.Thread(target=run_param_sync_thread)
    param_sync_thread.start()

    # Wait for actors to start and fill replay buffer
    replay_send_queue.put(("get_buffer_size", None))
    buffer_size = replay_recv_queue.get()
    assert buffer_size >= config.learning_starts

    # Training loop
    print("learner: Starting training loop...")
    start_time = time.time()
    # placeholders for samples, so we can release them later
    samples, indices, weights = None, None, None
    obs, actions, rewards, dones, lstm_h, lstm_c = None, None, None, None, None, None
    while global_step < config.total_timesteps:
        sample_start_time = time.time()
        # samples: (T, B, ...), indices: (B,), weights: (B,)
        replay_send_queue.put(("sample", config.batch_size))
        samples, indices, weights = replay_recv_queue.get()
        sample_time = time.time() - sample_start_time

        # unpack samples: o_t, a_t, r_t, d_t, lsmt_h_0, lstm_c_0
        obs, actions, rewards, dones, lstm_h, lstm_c = samples
        target_lstm_c, target_lstm_h = lstm_c.clone(), lstm_h.clone()

        burnin_time_start = time.time()
        # burn in lstm state
        if config.burnin_len > 0:
            with torch.no_grad():
                _, (lstm_h, lstm_c) = model.forward(
                    obs[: config.burnin_len],
                    actions[: config.burnin_len],
                    rewards[: config.burnin_len],
                    dones[: config.burnin_len],
                    (lstm_h, lstm_c),
                )
                _, (target_lstm_h, target_lstm_c) = model.forward(
                    obs[: config.burnin_len],
                    actions[: config.burnin_len],
                    rewards[: config.burnin_len],
                    dones[: config.burnin_len],
                    (target_lstm_c, target_lstm_h),
                )
        burnin_time = time.time() - burnin_time_start

        # compute loss and gradients and update parameters
        learning_start_time = time.time()
        optimizer.zero_grad()

        q_values, _ = model.forward(
            obs[config.burnin_len :],
            actions[config.burnin_len :],
            rewards[config.burnin_len :],
            dones[config.burnin_len :],
            (lstm_h, lstm_c),
        )

        with torch.no_grad():
            target_q_values, _ = target_model.forward(
                obs[config.burnin_len :],
                actions[config.burnin_len :],
                rewards[config.burnin_len :],
                dones[config.burnin_len :],
                (target_lstm_h, target_lstm_c),
            )

        loss, priorities = compute_loss_and_priority(
            config=config,
            q_values=q_values,
            actions=actions[config.burnin_len :],
            rewards=rewards[config.burnin_len :],
            dones=dones[config.burnin_len :],
            target_q_values=target_q_values,
        )
        loss = torch.mean(loss * weights.detach())

        with parameter_lock:
            loss.backward()
            if config.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

        learning_time = time.time() - learning_start_time

        # update priorities
        replay_send_queue.put(("update_priorities", indices, priorities.tolist()))

        if global_step > 1 and global_step % config.target_network_update_interval == 0:
            print(f"\nlearner: {global_step=} - updating target network.")
            target_model.load_state_dict(model.state_dict())

        # log metrics
        ups = int(global_step / (time.time() - start_time))
        sps = int(global_step * config.batch_size * config.seq_len) / (
            time.time() - start_time
        )
        q_max = torch.mean(torch.max(q_values, dim=-1)[0])

        if global_step > 0 and global_step % config.actor_update_interval == 0:
            # periodically log to stdout
            print(f"\nlearner: {global_step=}")
            print(f"learner: loss={loss.item():0.6f}")
            print(f"learner: q_max={q_max.item():0.6f}")
            print(f"learner: {ups=:d}")

        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", loss.item(), global_step)
        writer.add_scalar("losses/q_max", q_max.item(), global_step)
        writer.add_scalar(
            "losses/mean_priorities", priorities.mean().item(), global_step
        )
        writer.add_scalar("losses/max_priorities", priorities.max().item(), global_step)
        writer.add_scalar("losses/min_priorities", priorities.min().item(), global_step)
        writer.add_scalar("times/learner_SPS", sps, global_step)
        writer.add_scalar("times/learner_UPS", ups, global_step)
        writer.add_scalar("times/sample_time", sample_time, global_step)
        writer.add_scalar("times/burnin_time", burnin_time, global_step)
        writer.add_scalar("times/learning_time", learning_time, global_step)

        if (
            config.save_interval > 0
            and global_step > 0
            and global_step % config.save_interval == 0
        ):
            print("Saving model")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "global_step": global_step,
                    "config": asdict(config),
                },
                os.path.join(config.log_dir, f"checkpoint_{global_step}.pt"),
            )

        global_step += 1

    print("Training complete. Shutting down...")

    # cleanup
    env.close()

    print("Shutting down parameter synchronization thread.")
    end_training_event.set()
    param_sync_thread.join()

    writer.close()

    print("Sending end signals to replay and actor processes.")
    del samples, indices, weights
    del obs, actions, rewards, dones, lstm_h, lstm_c
    replay_send_queue.put(("terminate", None))
    for i in range(config.num_actors):
        actor_send_queues[i].put("terminate")

    print("Joining replay process.")
    replay_process.join()

    print("Draining replay and actor queues.")
    while not actor_replay_queue.empty():
        actor_replay_queue.get()
    while not replay_recv_queue.empty():
        replay_recv_queue.get()
    while not actor_recv_queue.empty():
        actor_recv_queue.get()

    print("Joining actors.")
    for i in range(config.num_actors):
        actors[i].join()

    print("Replay and actor processes successfully joined.")
    print("Cleaning up communication queues.")
    actor_replay_queue.close()
    replay_send_queue.close()
    replay_recv_queue.close()
    actor_recv_queue.close()
    for i in range(config.num_actors):
        actor_send_queues[i].close()

    print("All done")
