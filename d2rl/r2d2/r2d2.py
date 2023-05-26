"""The R2D2 algorithm.

From the paper "Recurrent Experience Replay in Distributed Reinforcement Learning"
https://openreview.net/pdf?id=r1lyTjAqYX.

Also with details from the Ape-X paper "Distributed Prioritized Experience Replay"
https://arxiv.org/abs/1803.00933

https://github.com/michaelnny/deep_rl_zoo/tree/main


TODO:
- [ ] Test for any issues with multiprocessing race-conditions
- [ ] Evaluation worker
- [ ] Add +1 to seq len when computing loss and priority
- [ ] Update code to account for o_t, a_tm1, r_tm1, d_tm1, q_t in buffer
- [ ] Zero out prev action and reward for first step in episode?
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import asdict
from multiprocessing.queues import Empty
from typing import TYPE_CHECKING, Dict, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from d2rl.r2d2.replay import R2D2ActorReplayBuffer, R2D2PrioritizedReplay

if TYPE_CHECKING:
    from d2rl.r2d2.network import R2D2Network
    from d2rl.r2d2.utils import R2D2Config


def compute_loss_and_priority(
    config: R2D2Config,
    q_values: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    target_q_values: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    learner_model: R2D2Network,
    input_queue: mp.Queue,
    actor_storage: Dict[str, torch.Tensor],
    actor_lock: mp.Lock,
    output_queue: mp.Queue | None = None,
):
    """Run an R2D2 actor process that collects trajectories."""
    print(f"actor={actor_idx}: Actor started.")
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

    # replay buffer setup
    replay = R2D2ActorReplayBuffer(
        actor_idx=actor_idx, config=config, actor_lock=actor_lock, **actor_storage
    )

    total_seq_len = config.seq_len + config.burnin_len
    # how many timesteps to drop from the start of the sequence after each sequence
    # is added to the replay buffer
    seq_drop_len = max(1, config.seq_len // 2)

    # sequence buffers: o_t, a_tm1, r_tm1, d_tm1, q_t, h_0
    # stores burnin_len + seq_len + 1timesteps
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
    # or if in debug mode, until debug_actor_steps is reached
    while config.debug_actor_steps is None or step < config.debug_actor_steps:
        # check if learner has finished training
        try:
            input_queue.get_nowait()
            print(f"actor={actor_idx} - t={step}: End training signal recieved.")
            break
        except Empty:
            pass

        if step % config.actor_update_interval == 0:
            # print(f"i={actor_idx} - t={step}: Updating actor model.")
            actor_model.load_state_dict(learner_model.state_dict())

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

            replay.add(
                obs_seq_buffer,
                action_seq_buffer,
                reward_seq_buffer,
                done_buffer,
                lstm_h_seq_buffer,
                lstm_c_seq_buffer,
                priority,
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

        for item in [
            i for i in info.get("final_info", []) if i is not None and "episode" in i
        ]:
            episode_returns[num_episodes_completed % 100] = item["episode"]["r"][0]
            episode_lengths[num_episodes_completed % 100] = item["episode"]["l"][0]
            num_episodes_completed += 1

        if step > 0 and step % 1000 == 0 and output_queue is not None:
            # send results to learner
            print(f"actor={actor_idx} - t={step}: Sending results to learner.")
            sps = int(step * config.num_envs_per_actor) / (time.time() - start_time)
            ep_returns = episode_returns[:num_episodes_completed]
            ep_lengths = episode_lengths[:num_episodes_completed]
            output_queue.put(
                {
                    "actor_steps": step,
                    "actor_sps": sps,
                    "mean_episode_returns": ep_returns.mean().item(),
                    "max_episode_returns": ep_returns.max().item(),
                    "min_episode_returns": ep_returns.min().item(),
                    "mean_episode_lengths": ep_lengths.mean().item(),
                    "actor_episodes_completed": num_episodes_completed,
                }
            )

        step += 1

    envs.close()
    print(f"i={actor_idx} - t={step}: Actor Finished.")


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
    model.share_memory()

    target_model = config.load_model()
    target_model.to(device)
    target_model.load_state_dict(model.state_dict())

    # replay buffer setup
    replay = R2D2PrioritizedReplay(obs_space, config)

    # Actor Setup
    # `fork` not supported by CUDA
    # https://pytorch.org/docs/main/notes/multiprocessing.html#cuda-in-multiprocessing
    # must use context to set start method
    mp_ctxt = mp.get_context("spawn")

    # create queues for communication
    input_queues = []
    output_queue = mp_ctxt.Queue()
    actors = []
    for actor_idx in range(config.num_actors):
        input_queues.append(mp_ctxt.Queue())
        actor_storage, actor_lock = replay.get_actor_storage(actor_idx)
        actor = mp_ctxt.Process(
            target=run_actor,
            args=(
                actor_idx,
                config,
                model,
                input_queues[actor_idx],
                actor_storage,
                actor_lock,
                # get episode results from last actor since it has smallest exploration
                # epsilon
                output_queue if actor_idx == config.num_actors - 1 else None,
            ),
        )
        actor.start()
        actors.append(actor)

    # Logging setup
    # Do this after workers are spawned to avoid log duplication
    if not config.debug_no_logging and config.track_wandb:
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
    if not config.debug_no_logging:
        writer = SummaryWriter(config.log_dir)
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in vars(config).items()])),
        )
    else:
        writer = None

    # Optimizer setup
    optimizer = optim.Adam(
        model.parameters(), lr=config.learning_rate, eps=config.adam_eps
    )

    # Wait for actors to start and fill replay buffer
    print("learner: Waiting for actors to start and replay buffer to fill...")
    while replay.size < config.learning_starts:
        time.sleep(0.1)

    # Training loop
    print("learner: Starting training loop...")
    start_time = time.time()
    for step in range(config.total_timesteps):
        sample_start_time = time.time()
        # samples: (T, B, ...), indices: (B,), weights: (B,)
        samples, indices, weights = replay.sample(config.batch_size, device=device)
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

        learning_start_time = time.time()
        # compute loss and gradients
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
        loss.backward()

        if config.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        optimizer.step()
        learning_time = time.time() - learning_start_time

        # update priorities
        priority_update_start_time = time.time()
        replay.update_priorities(indices, priorities.tolist())
        priority_update_time = time.time() - priority_update_start_time

        # update target network
        if step > 1 and step % config.target_network_update_interval == 0:
            print(f"\nlearner: {step=} - updating target network.")
            target_model.load_state_dict(model.state_dict())

        # log metrics
        sps = int(step / (time.time() - start_time))
        q_max = torch.mean(torch.max(q_values, dim=-1)[0])

        try:
            actor_results = output_queue.get_nowait()
        except Empty:
            actor_results = {}

        if actor_results:
            # periodically log to stdout
            print(f"\nlearner: {step=}")
            print(f"learner: loss={loss.item():0.6f}")
            print(f"learner: q_max={q_max.item():0.6f}")
            print(f"learner: {sps=:d}")
            for key, value in actor_results.items():
                if isinstance(value, float):
                    print(f"learner: {key}={value:0.4f}")
                else:
                    print(f"learner: {key}={value}")

        if writer is not None:
            writer.add_scalar(
                "charts/learning_rate", optimizer.param_groups[0]["lr"], step
            )
            for key, value in actor_results.items():
                writer.add_scalar(f"charts/{key}", value, step)
            writer.add_scalar("losses/value_loss", loss.item(), step)
            writer.add_scalar("losses/q_max", q_max.item(), step)
            writer.add_scalar("losses/mean_priorities", priorities.mean().item(), step)
            writer.add_scalar("losses/max_priorities", priorities.max().item(), step)
            writer.add_scalar("losses/min_priorities", priorities.min().item(), step)
            writer.add_scalar("times/learner_SPS", sps, step)
            writer.add_scalar("times/sample_time", sample_time, step)
            writer.add_scalar("times/burnin_time", burnin_time, step)
            writer.add_scalar("times/learning_time", learning_time, step)
            writer.add_scalar("times/priority_update_time", priority_update_time, step)

        if config.save_interval > 0 and step > 0 and step % config.save_interval == 0:
            print("Saving model")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "global_step": step,
                    "config": asdict(config),
                },
                os.path.join(config.log_dir, f"checkpoint_{step}.pt"),
            )

    # cleanup
    env.close()

    if writer is not None:
        writer.close()

    print("Training complete")
    print("Sending stop signal to actors.")
    for i in range(config.num_actors):
        input_queues[i].put("terminate")

    print("Stop signal sent, joining actors.")
    for i in range(config.num_actors):
        actors[i].join()

    print("Actors successfully joined. Cleaning up communication queues.")
    for i in range(config.num_actors):
        input_queues[i].close()

    print("All done")
