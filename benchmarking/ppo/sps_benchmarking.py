"""Script for testing SPS of different PPO configurations."""
import os
import random
import time
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from d2rl.ppo.network import PPONetwork
from d2rl.ppo.ppo import run_rollout_worker
from d2rl.ppo.utils import PPOConfig

DEFAULT_CONFIG = {
    "exp_name": "ppo_benchmarking",
    "seed": 0,
    "torch_deterministic": True,
    "cuda": True,
    "track_wandb": False,
    "env_id": "CartPole-v1",
    "num_minibatches": 1,
}

NETWORKS = {
    "small": {
        "trunk_sizes": [64],
        "lstm_size": 64,
        "head_sizes": [64],
    },
    "medium": {
        "trunk_sizes": [128, 128],
        "lstm_size": 128,
        "head_sizes": [128, 128],
    },
    "large": {
        "trunk_sizes": [256, 256, 256],
        "lstm_size": 256,
        "head_sizes": [256, 256, 256],
    },
}

NUM_WORKERS = [1, 2]
NUM_ENVS_PER_WORKER = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256]
NUM_STEPS = [32, 64, 128, 256]

# parameters to play with
# - num_workers
# - num_envs_per_worker
# - num_steps

# parameters to set
# - total_timesteps - should be chosen so that num_updates == 100
#     100 = total_timesteps / (num_steps * num_envs_per_worker * num_workers)
#     total_timesteps = 100 * (num_steps * num_envs_per_worker * num_workers)


def run_benchmarking_ppo(config: PPOConfig):
    """Run PPO only recording timing statistics."""
    setup_start_time = time.time()

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

    # model setup
    device = config.device
    model = PPONetwork(
        obs_space,
        act_space,
        trunk_sizes=config.trunk_sizes,
        lstm_size=config.lstm_size,
        head_sizes=config.head_sizes,
    ).to(device)

    # Experience buffer setup
    total_num_envs, num_steps = config.total_num_envs, config.num_steps
    obs = torch.zeros((num_steps, total_num_envs) + obs_space.shape).to(device)
    actions = torch.zeros((num_steps, total_num_envs) + act_space.shape).to(device)
    logprobs = torch.zeros((num_steps, total_num_envs)).to(device)
    rewards = torch.zeros((num_steps, total_num_envs)).to(device)
    # +1 for bootstrapped value
    dones = torch.zeros((num_steps + 1, total_num_envs)).to(device)
    values = torch.zeros((num_steps + 1, total_num_envs)).to(device)
    # buffer for storing lstm state for each worker-env at start of each update
    initial_lstm_state = (
        torch.zeros(model.lstm.num_layers, total_num_envs, model.lstm.hidden_size).to(
            device
        ),
        torch.zeros(model.lstm.num_layers, total_num_envs, model.lstm.hidden_size).to(
            device
        ),
    )

    # buffers for tracking episode stats
    ep_returns = torch.zeros((config.num_workers, 1)).cpu()
    ep_lens = torch.zeros((config.num_workers, 1)).cpu()

    # load model and buffers into shared memory
    model.share_memory()
    obs.share_memory_()
    actions.share_memory_()
    logprobs.share_memory_()
    rewards.share_memory_()
    dones.share_memory_()
    values.share_memory_()
    initial_lstm_state[0].share_memory_()
    initial_lstm_state[1].share_memory_()
    ep_returns.share_memory_()
    ep_lens.share_memory_()

    # Spawn workers
    # `fork` not supported by CUDA
    # https://pytorch.org/docs/main/notes/multiprocessing.html#cuda-in-multiprocessing
    # must use context to set start method
    mp_ctxt = mp.get_context("spawn")

    # create queues for communication
    input_queues = []
    output_queues = []
    workers = []
    for worker_id in range(config.num_workers):
        input_queues.append(mp_ctxt.Queue())
        output_queues.append(mp_ctxt.Queue())
        buf_idx_start = worker_id * config.num_envs_per_worker
        buf_idx_end = buf_idx_start + config.num_envs_per_worker
        worker = mp_ctxt.Process(
            target=run_rollout_worker,
            args=(
                worker_id,
                config,
                model,
                input_queues[worker_id],
                output_queues[worker_id],
                [
                    obs[:, buf_idx_start:buf_idx_end],
                    actions[:, buf_idx_start:buf_idx_end],
                    logprobs[:, buf_idx_start:buf_idx_end],
                    rewards[:, buf_idx_start:buf_idx_end],
                    dones[:, buf_idx_start:buf_idx_end],
                    values[:, buf_idx_start:buf_idx_end],
                    initial_lstm_state[0][:, buf_idx_start:buf_idx_end],
                    initial_lstm_state[1][:, buf_idx_start:buf_idx_end],
                    ep_returns[:, buf_idx_start:buf_idx_end],
                    ep_lens[:, buf_idx_start:buf_idx_end],
                ],
            ),
        )
        worker.start()
        workers.append(worker)

    # Optimizer setup
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)

    setup_time = time.time() - setup_start_time

    # Training loop
    global_step = 0
    start_time = time.time()
    experience_collection_time = 0
    learning_time = 0
    for update in range(1, config.num_updates + 1):
        if config.anneal_lr:
            frac = 1.0 - (update - 1.0) / config.num_updates
            lrnow = frac * config.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        experience_collection_start_time = time.time()
        # signal workers to collect next batch of experience
        for i in range(config.num_workers):
            input_queues[i].put(1)

        # wait for workers to finish collecting experience
        for i in range(config.num_workers):
            output_queues[i].get()

        experience_collection_time += time.time() - experience_collection_start_time

        # log episode stats
        global_step += config.batch_size

        learning_start_time = time.time()
        # calculate advantages and monte-carlo returns
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(config.num_steps)):
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
            delta = rewards[t] + config.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = (
                delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
            )
        returns = advantages + values[:-1]

        # flatten the batch
        b_obs = obs.reshape((-1,) + obs_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + act_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        # -1 to remove the last step, which is only used for calculating final
        # advantage and returns
        b_dones = dones[:-1].reshape(-1)
        b_values = values[:-1].reshape(-1)

        # Optimizing the policy and value network
        envsperbatch = total_num_envs // config.num_minibatches
        envinds = np.arange(total_num_envs)
        flatinds = np.arange(config.batch_size).reshape(
            config.num_steps, total_num_envs
        )
        clipfracs = []
        for epoch in range(config.update_epochs):
            np.random.shuffle(envinds)

            # do minibatch update
            # each minibatch uses data from randomized subset of envs
            for start in range(0, total_num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[
                    :, mbenvinds
                ].ravel()  # be really careful about the index

                _, newlogprob, entropy, newvalue, _ = model.get_action_and_value(
                    b_obs[mb_inds],
                    (
                        initial_lstm_state[0][:, mbenvinds],
                        initial_lstm_state[1][:, mbenvinds],
                    ),
                    b_dones[mb_inds],
                    b_actions.long()[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > config.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
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
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -config.clip_coef,
                        config.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = (
                    pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

            if config.target_kl is not None and approx_kl > config.target_kl:
                break

        learning_time += time.time() - learning_start_time

    total_time = time.time() - start_time
    env.close()

    for i in range(config.num_workers):
        input_queues[i].put(0)

    for i in range(config.num_workers):
        workers[i].join()

    for i in range(config.num_workers):
        input_queues[i].close()
        output_queues[i].close()

    return {
        "total_steps": global_step,
        "total_time": total_time,
        "sps": int(global_step / total_time),
        "setup_time": setup_time,
        "experience_collection_time": experience_collection_time,
        "learning_time": learning_time,
    }


def run(append_results: bool = False):
    save_file = os.path.join(os.path.dirname(__file__), "benchmarking_results.csv")

    if not append_results:
        with open(save_file, "w") as f:
            f.write(
                "exp_num,network,num_workers,num_envs_per_worker,num_steps,"
                "total_steps,total_time,sps,setup_time,"
                "experience_collection_time,learning_time\n"
            )
        exp_num = 0
    else:
        with open(save_file, "r") as f:
            exp_num = len(f.readlines()) - 1

    for net_name, nw, ne, ns in product(
        NETWORKS, NUM_WORKERS, NUM_ENVS_PER_WORKER, NUM_STEPS
    ):
        # at least 1 update or 8192 steps (whichever is larger)
        # at most 65536 steps or 10 updates (whichever is smaller)
        # except if update size is bigger than 65536, then use update size
        total_timesteps = max(max(8192, nw * ne * ns), min(65536, 10 * nw * ne * ns))
        print(
            f"Running exp_num={exp_num}, network={net_name}, num_workers={nw}, "
            f"num_envs_per_worker={ne}, num_steps={ns} for {total_timesteps} steps"
        )
        network = NETWORKS[net_name]
        config = PPOConfig(
            total_timesteps=total_timesteps,
            num_envs_per_worker=ne,
            num_workers=nw,
            num_steps=ns,
            trunk_sizes=network["trunk_sizes"],
            lstm_size=network["lstm_size"],
            head_sizes=network["head_sizes"],
            **DEFAULT_CONFIG,
        )
        result = run_benchmarking_ppo(config)
        print(f"SPS: {result['sps']}")

        # result = {
        #     "total_steps": 1000,
        #     "total_time": 100.1,
        #     "sps": 12,
        #     "setup_time": 1.01,
        #     "experience_collection_time": 6.5,
        #     "learning_time": 0.9,
        # }

        with open(save_file, "a") as f:
            f.write(
                f"{exp_num},{net_name},{nw},{ne},{ns},"
                + ",".join(map(str, result.values()))
                + "\n"
            )
        exp_num += 1


def plot():
    save_file = os.path.join(os.path.dirname(__file__), "benchmarking_results.csv")
    df = pd.read_csv(save_file)
    print(df.columns)

    sns.set_theme()
    sns.relplot(
        data=df,
        kind="line",
        x="num_steps",
        y="sps",
        hue="num_envs_per_worker",  # z
        col="num_workers",  # separate plots along columns
        row="network",  # seperate plots along rows
        style="num_envs_per_worker",
        # size="total_steps",
    )
    # plt.yscale("log")

    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("action", type=str, choices=["run", "plot"])
    parser.add_argument("--append-results", action="store_true")
    args = parser.parse_args()
    if args.action == "run":
        run(args.append_results)
    elif args.action == "plot":
        plot()
    else:
        raise ValueError("Invalid action")
