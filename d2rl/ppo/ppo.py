"""PPO with multiple rollout workers collecting trajectories in parallel."""
import os
import random
import time
from dataclasses import asdict
from datetime import timedelta

import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from d2rl.ppo.utils import PPOConfig


def run_rollout_worker(
    worker_id: int,
    config: PPOConfig,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
):
    """Rollout worker function for collecting trajectories.

    The rollout worker uses CPU for sampling actions, and then stores trajectories in
    shared buffers (which my be on GPU).

    """
    # Limit each rollout worker to using a single CPU thread.
    # This prevents each rollout worker from using all available cores, which can
    # lead to each rollout worker being slower due to contention.
    torch.set_num_threads(1)

    # env setup
    # Note: SyncVectorEnv runs multiple-env instances serially.
    envs = gym.vector.SyncVectorEnv(
        [
            config.env_creator_fn_getter(config, env_idx=i, worker_idx=worker_id)
            for i in range(config.num_envs_per_worker)
        ]
    )
    obs_space = envs.single_observation_space
    act_space = envs.single_action_space

    # model setup
    worker_model = config.load_model()
    worker_model.cpu()

    # worker buffers are stored on CPU
    buf_shape = (config.num_rollout_steps, config.num_envs_per_worker)
    obs = torch.zeros(buf_shape + obs_space.shape)
    actions = torch.zeros(buf_shape + act_space.shape)
    logprobs = torch.zeros(buf_shape)
    rewards = torch.zeros(buf_shape)
    # +1 for bootstrapped value
    dones = torch.zeros((buf_shape[0] + 1, buf_shape[1]))
    values = torch.zeros((buf_shape[0] + 1, buf_shape[1]))
    lstm_state_shape = (
        worker_model.lstm.num_layers,
        config.num_envs_per_worker,
        worker_model.lstm.hidden_size,
    )
    initial_lstm_states = (
        torch.zeros((config.num_seqs_per_rollout,) + lstm_state_shape),
        torch.zeros((config.num_seqs_per_rollout,) + lstm_state_shape),
    )
    ep_return_stats = torch.zeros(3)
    ep_len_stats = torch.zeros(3)

    # setup variables for tracking current step outputs
    next_obs = torch.Tensor(envs.reset()[0])
    next_done = torch.zeros(config.num_envs_per_worker)
    next_lstm_state = (
        torch.zeros(lstm_state_shape),
        torch.zeros(lstm_state_shape),
    )

    while True:
        # wait for learner to signal ready for next batch
        learner_model = input_queue.get()
        if learner_model == 0:
            # learner has finished training, so end work
            break

        # sync weights
        worker_model.load_state_dict(learner_model.state_dict())

        # collect batch of experience
        episode_returns = []
        episode_lengths = []
        num_episodes = 0
        for step in range(0, config.num_rollout_steps):
            obs[step] = next_obs
            dones[step] = next_done

            if step % config.seq_len == 0:
                seq_num = step // config.seq_len
                # store lstm for start of each seq chunk in buffer for use
                # by learner during update
                initial_lstm_states[0][seq_num][:] = next_lstm_state[0].clone()
                initial_lstm_states[1][seq_num][:] = next_lstm_state[1].clone()

            # sample next action
            with torch.no_grad():
                (
                    action,
                    logprob,
                    _,
                    value,
                    next_lstm_state,
                ) = worker_model.get_action_and_value(
                    next_obs, next_lstm_state, next_done
                )
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # execute step and log data.
            next_obs, reward, terminated, truncated, info = envs.step(
                action.cpu().numpy()
            )

            done = terminated | truncated
            rewards[step] = torch.tensor(reward).view(-1)
            next_obs = torch.Tensor(next_obs)
            next_done = torch.Tensor(done)

            for item in [
                i
                for i in info.get("final_info", [])
                if i is not None and "episode" in i
            ]:
                num_episodes += 1
                episode_returns.append(item["episode"]["r"][0])
                episode_lengths.append(item["episode"]["l"][0])

        # log episode stats
        if num_episodes > 0:
            episode_returns = torch.tensor(episode_returns, dtype=torch.float32)
            episode_lengths = torch.tensor(episode_lengths, dtype=torch.float32)
            ep_return_stats[0] = torch.mean(episode_returns)
            ep_return_stats[1] = torch.min(episode_returns)
            ep_return_stats[2] = torch.max(episode_returns)
            ep_len_stats[0] = torch.mean(episode_lengths)
            ep_len_stats[1] = torch.min(episode_lengths)
            ep_len_stats[2] = torch.max(episode_lengths)
        else:
            ep_return_stats[:] = 0
            ep_len_stats[:] = 0

        # bootstrap value for final entry of batch if not done
        with torch.no_grad():
            next_value = worker_model.get_value(
                next_obs,
                next_lstm_state,
                next_done,
            ).reshape(1, -1)
            dones[-1] = next_done
            values[-1] = next_value

        # calculate advantages and monte-carlo returns
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(config.num_rollout_steps)):
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
            delta = rewards[t] + config.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = (
                delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
            )
        returns = advantages + values[:-1]

        # reshape batch to be (seq_len, num_envs_per_worker * num_seqs_per_rollout)
        b_obs = torch.concatenate(torch.split(obs, config.seq_len), dim=1)
        b_actions = torch.concatenate(torch.split(actions, config.seq_len), dim=1)
        b_logprobs = torch.concatenate(torch.split(logprobs, config.seq_len), dim=1)
        b_advantages = torch.concatenate(torch.split(advantages, config.seq_len), dim=1)
        b_returns = torch.concatenate(torch.split(returns, config.seq_len), dim=1)
        # -1 to remove the last step, which is only used for calculating final
        # advantage and returns
        b_dones = torch.concatenate(torch.split(dones[:-1], config.seq_len), dim=1)
        b_values = torch.concatenate(torch.split(values[:-1], config.seq_len), dim=1)

        # send batch of data to learner
        output_queue.put(
            {
                "obs": b_obs,
                "actions": b_actions,
                "logprobs": b_logprobs,
                "advantages": b_advantages,
                "returns": b_returns,
                "dones": b_dones,
                "values": b_values,
                "initial_lstm_states": (
                    initial_lstm_states[0].view(
                        worker_model.lstm.num_layers,
                        config.num_envs_per_worker * config.num_seqs_per_rollout,
                        worker_model.lstm.hidden_size,
                    ),
                    initial_lstm_states[1].view(
                        worker_model.lstm.num_layers,
                        config.num_envs_per_worker * config.num_seqs_per_rollout,
                        worker_model.lstm.hidden_size,
                    ),
                ),
                "ep_returns": ep_return_stats,
                "ep_lens": ep_len_stats,
            }
        )
    envs.close()


def run_evaluation_worker(
    config: PPOConfig,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
):
    """Worker function for running evaluations.

    The evaluation worker uses the current learner model to run a number of evaluation
    episodes, then reports the results back to the main process.

    In addition, will record videos of the evaluation episodes if requested.

    """
    torch.set_num_threads(1)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            config.env_creator_fn_getter(config, env_idx=i, worker_idx=None)
            for i in range(config.num_envs_per_worker)
        ]
    )

    # model setup
    device = config.device
    eval_model = config.load_model()
    eval_model.to(device)

    while True:
        # wait for main process to signal ready for evaluation
        learner_model = input_queue.get()
        if learner_model == 0:
            # main process has finished training, so end work
            break

        start_time = time.time()
        # sync weights
        eval_model.load_state_dict(learner_model.state_dict())

        # run evaluation episodes
        # reseting envs and all variables
        next_obs = torch.Tensor(envs.reset()[0]).to(device)
        next_done = torch.zeros(config.num_envs_per_worker).to(device)
        next_lstm_state = (
            torch.zeros(
                eval_model.lstm.num_layers,
                config.num_envs_per_worker,
                eval_model.lstm.hidden_size,
            ).to(device),
            torch.zeros(
                eval_model.lstm.num_layers,
                config.num_envs_per_worker,
                eval_model.lstm.hidden_size,
            ).to(device),
        )
        start_time = time.time()
        num_episodes = 0
        episode_returns = []
        episode_lengths = []

        steps = 0
        while num_episodes == 0 or steps < config.eval_num_steps:
            # sample next action
            with torch.no_grad():
                action, _, _, _, next_lstm_state = eval_model.get_action_and_value(
                    next_obs, next_lstm_state, next_done
                )

            # execute step and log data.
            next_obs, _, terminated, truncated, info = envs.step(action.cpu().numpy())

            done = terminated | truncated
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(done).to(device)
            steps += 1

            for item in [
                i
                for i in info.get("final_info", [])
                if i is not None and "episode" in i
            ]:
                num_episodes += 1
                episode_returns.append(item["episode"]["r"][0])
                episode_lengths.append(item["episode"]["l"][0])

        eval_time = time.time() - start_time
        total_steps = steps * config.num_envs_per_worker
        output_queue.put(
            {
                "total_steps": total_steps,
                "parallel_steps": steps,
                "SPS": int(total_steps / eval_time),
                "num_episodes": num_episodes,
                "episode_return_mean": np.mean(episode_returns),
                "episode_return_min": np.min(episode_returns),
                "episode_return_max": np.max(episode_returns),
                "episode_length_mean": np.mean(episode_lengths),
                "episode_length_min": np.min(episode_lengths),
                "episode_length_max": np.max(episode_lengths),
                "time": eval_time,
            }
        )

    envs.close()


def run_ppo(config: PPOConfig):
    # seeding
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic

    # env setup
    # created here for generating model
    env = config.env_creator_fn_getter(config, env_idx=0, worker_idx=None)()
    obs_space = env.observation_space
    act_space = env.action_space

    print("Running PPO:")
    print(f"Env-id: {config.env_id}")
    print(f"Observation space: {obs_space}")
    print(f"Action space: {act_space}")

    # model setup
    device = config.device
    model = config.load_model()
    model.to(device)

    # Experience buffer setup
    seq_len, num_seqs_per_rollout = config.seq_len, config.num_seqs_per_rollout
    seqs_per_batch = config.total_num_envs * num_seqs_per_rollout

    print(f"Rollout Length per worker: {config.num_rollout_steps}")
    print(f"Sequence Length: {seq_len}")
    print(f"Number of Sequences per Rollout: {num_seqs_per_rollout}")
    print(f"Num Envs per Worker: {config.num_envs_per_worker}")
    print(f"Num Workers: {config.num_workers}")
    print(f"Num sequences per batch: {seqs_per_batch}")
    print(f"Batch size: {config.batch_size}")

    buf_shape = (seq_len, seqs_per_batch)
    obs = torch.zeros(buf_shape + obs_space.shape).to(device)
    actions = torch.zeros(buf_shape + act_space.shape).to(device)
    logprobs = torch.zeros(buf_shape).to(device)
    advantages = torch.zeros(buf_shape).to(device)
    returns = torch.zeros(buf_shape).to(device)
    # +1 for bootstrapped value
    dones = torch.zeros(buf_shape).to(device)
    values = torch.zeros(buf_shape).to(device)
    # buffer for storing lstm state for each worker-env at start of each update
    initial_lstm_state = (
        torch.zeros(model.lstm.num_layers, seqs_per_batch, model.lstm.hidden_size).to(
            device
        ),
        torch.zeros(model.lstm.num_layers, seqs_per_batch, model.lstm.hidden_size).to(
            device
        ),
    )

    # buffers for tracking episode stats (mean, min, max)
    ep_returns = torch.zeros((config.num_workers, 3)).cpu()
    ep_lens = torch.zeros((config.num_workers, 3)).cpu()

    # load model and buffers into shared memory
    model.share_memory()

    # Spawn workers
    # `fork` not supported by CUDA
    # https://pytorch.org/docs/main/notes/multiprocessing.html#cuda-in-multiprocessing
    # must use context to set start method
    mp_ctxt = mp.get_context("spawn")

    # create queues for communication
    input_queues = []
    output_queues = []
    workers = []
    # placeholder for worker batch, so we can release it later
    worker_batch = {}
    for worker_id in range(config.num_workers):
        input_queues.append(mp_ctxt.Queue())
        output_queues.append(mp_ctxt.Queue())
        worker = mp_ctxt.Process(
            target=run_rollout_worker,
            args=(
                worker_id,
                config,
                input_queues[worker_id],
                output_queues[worker_id],
            ),
        )
        worker.start()
        workers.append(worker)

    # create eval worker
    if config.eval_interval > 0:
        eval_input_queue = mp_ctxt.Queue()
        eval_output_queue = mp_ctxt.Queue()
        eval_worker = mp_ctxt.Process(
            target=run_evaluation_worker,
            args=(
                config,
                eval_input_queue,
                eval_output_queue,
            ),
        )
        eval_worker.start()
    else:
        eval_input_queue = None
        eval_output_queue = None
        eval_worker = None

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

    # Optimizer setup
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)

    # Training loop
    print("Starting training loop...")
    global_step = 0
    sps_start_time = time.time()
    for update in range(1, config.num_updates + 1):
        if config.anneal_lr:
            frac = 1.0 - (update - 1.0) / config.num_updates
            lrnow = frac * config.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # run evaluation
        if config.eval_interval > 0 and update % config.eval_interval == 0:
            assert eval_input_queue is not None
            assert eval_output_queue is not None
            print("Running evaluation...")
            eval_start_time = time.time()
            eval_input_queue.put(model)
            eval_results = eval_output_queue.get()

            print(
                f"global_step={global_step}, "
                f"evaluation/episode_return={eval_results['episode_return_mean']:.2f}, "
                f"evaluation/episode_length={eval_results['episode_length_mean']:.2f}, "
                f"evaluation/SPS={eval_results['SPS']:.2f}, "
                f"evaluation/time={eval_results['time']:.2f}"
            )
            for key, value in eval_results.items():
                writer.add_scalar(f"evaluation/{key}", value, global_step)
            # remove evaluation time from sps timer
            # otherwise results will be skewed by evaluation time
            sps_start_time += time.time() - eval_start_time

        experience_collection_start_time = time.time()
        # signal workers to collect next batch of experience
        for i in range(config.num_workers):
            input_queues[i].put(model)

        # wait for workers to finish collecting experience
        for i in range(config.num_workers):
            worker_batch = output_queues[i].get()
            buf_idx_start = i * (
                config.num_envs_per_worker * config.num_seqs_per_rollout
            )
            buf_idx_end = buf_idx_start + (
                config.num_envs_per_worker * config.num_seqs_per_rollout
            )

            obs[:, buf_idx_start:buf_idx_end] = worker_batch["obs"]
            actions[:, buf_idx_start:buf_idx_end] = worker_batch["actions"]
            logprobs[:, buf_idx_start:buf_idx_end] = worker_batch["logprobs"]
            advantages[:, buf_idx_start:buf_idx_end] = worker_batch["advantages"]
            returns[:, buf_idx_start:buf_idx_end] = worker_batch["returns"]
            dones[:, buf_idx_start:buf_idx_end] = worker_batch["dones"]
            values[:, buf_idx_start:buf_idx_end] = worker_batch["values"]
            initial_lstm_state[0][:, buf_idx_start:buf_idx_end] = worker_batch[
                "initial_lstm_states"
            ][0]
            initial_lstm_state[1][:, buf_idx_start:buf_idx_end] = worker_batch[
                "initial_lstm_states"
            ][1]
            ep_returns[i] = worker_batch["ep_returns"]
            ep_lens[i] = worker_batch["ep_lens"]

        experience_collection_time = time.time() - experience_collection_start_time

        # log episode stats
        global_step += config.batch_size
        episode_len_mean = torch.mean(ep_lens[:, 0]).item()
        if episode_len_mean > 0:
            # only log if there were episodes completed
            # using results from workers with completed episodes
            worker_idxs = torch.nonzero(ep_lens[:, 0]).squeeze()
            episode_return_mean = torch.mean(ep_returns[worker_idxs, 0]).item()
            print(
                f"{timedelta(seconds=int(time.time()-sps_start_time))} "
                f"global_step={global_step}, "
                f"episode_return={episode_return_mean:.2f}, "
                f"episodic_length={episode_len_mean:.2f}"
            )
            writer.add_scalar(
                "charts/episode_return_mean", episode_return_mean, global_step
            )
            writer.add_scalar(
                "charts/episode_return_min",
                torch.min(ep_returns[worker_idxs, 1]).item(),
                global_step,
            )
            writer.add_scalar(
                "charts/episode_return_max",
                torch.max(ep_returns[worker_idxs, 2]).item(),
                global_step,
            )
            writer.add_scalar(
                "charts/episode_length_mean", episode_len_mean, global_step
            )
            writer.add_scalar(
                "charts/episode_length_min",
                torch.min(ep_lens[worker_idxs, 1]).item(),
                global_step,
            )
            writer.add_scalar(
                "charts/episode_length_max",
                torch.max(ep_lens[worker_idxs, 2]).item(),
                global_step,
            )

        learning_start_time = time.time()

        # flatten the batch
        b_obs = obs.reshape((-1,) + obs_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + act_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_dones = dones.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        seqs_per_minibatch = seqs_per_batch // config.num_minibatches
        seq_indxs = np.arange(seqs_per_batch)
        flat_indxs = np.arange(config.batch_size).reshape(
            config.seq_len, seqs_per_batch
        )
        clipfracs = []
        approx_kl, old_approx_kl, unclipped_grad_norm = 0, 0, 0
        entropy_loss, pg_loss, v_loss, loss = 0, 0, 0, 0
        for epoch in range(config.update_epochs):
            np.random.shuffle(seq_indxs)

            # do minibatch update
            # each minibatch uses data from randomized subset of envs
            for start in range(0, seqs_per_batch, seqs_per_minibatch):
                end = start + seqs_per_minibatch
                mb_seq_indxs = seq_indxs[start:end]
                # be really careful about the index
                mb_indxs = flat_indxs[:, mb_seq_indxs].ravel()

                _, newlogprob, entropy, newvalue, _ = model.get_action_and_value(
                    b_obs[mb_indxs],
                    (
                        initial_lstm_state[0][:, mb_seq_indxs],
                        initial_lstm_state[1][:, mb_seq_indxs],
                    ),
                    b_dones[mb_indxs],
                    b_actions.long()[mb_indxs],
                )
                logratio = newlogprob - b_logprobs[mb_indxs]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > config.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_indxs]
                if config.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - config.clip_coef, 1 + config.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if config.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_indxs]) ** 2
                    v_clipped = b_values[mb_indxs] + torch.clamp(
                        newvalue - b_values[mb_indxs],
                        -config.clip_coef,
                        config.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_indxs]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_indxs]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = (
                    pg_loss - config.ent_coef * entropy_loss + config.vf_coef * v_loss
                )

                optimizer.zero_grad()
                loss.backward()
                unclipped_grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), config.max_grad_norm
                )
                optimizer.step()

            if config.target_kl is not None and approx_kl > config.target_kl:
                break

        learning_time = time.time() - learning_start_time

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # record learning statistics
        writer.add_scalar("charts/update", update, global_step)
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/loss", loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar(
            "losses/unclipped_grad_norm", unclipped_grad_norm.item(), global_step
        )

        # record timing stats
        sps = int(global_step / (time.time() - sps_start_time))
        print("SPS:", sps)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar(
            "charts/collection_time", experience_collection_time, global_step
        )
        writer.add_scalar("charts/learning_time", learning_time, global_step)

        if config.save_interval > 0 and (
            update % config.save_interval == 0 or update == config.num_updates
        ):
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

        if config.capture_video and config.track_wandb:
            video_filenames = [
                fname
                for fname in os.listdir(config.video_dir)
                if fname.endswith(".mp4")
            ]
            video_filenames.sort()
            for filename in video_filenames:
                if filename not in uploaded_video_files:
                    print("Uploading video:", filename)
                    wandb.log(  # type:ignore
                        {
                            "video": wandb.Video(  # type:ignore
                                os.path.join(config.video_dir, filename)
                            )
                        }
                    )
                    uploaded_video_files.add(filename)
                    break

    env.close()
    writer.close()

    print("Training complete")
    print("Sending stop signal to workers.")
    for i in range(config.num_workers):
        input_queues[i].put(0)

    if eval_input_queue is not None:
        eval_input_queue.put(0)

    print("Closing communication queues.")
    for i in range(config.num_workers):
        input_queues[i].close()
        output_queues[i].close()

    if eval_input_queue is not None:
        eval_input_queue.close()
    if eval_output_queue is not None:
        eval_output_queue.close()

    print("Stop signal sent, joining workers.")
    for i in range(config.num_workers):
        workers[i].join()

    if eval_worker is not None:
        eval_worker.join()

    print("Workers successfully joined.")
    print("All done")
