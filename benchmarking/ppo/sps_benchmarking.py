"""Script for testing SPS of different PPO configurations.

Specifically, we measure the steps-per-second (SPS) executed by PPO during training
for different batch sizes and number of workers. 

Along with the total SPS, we also measure the SPS for experience collection and
learning steps separately. This is useful for determining how our algorithm scales
in each of these aspects as we increase the number of workers for different batch sizes.

Note, for each experiment we keep the minibatch size constant, and similarly for the 
number of rollout steps. Furthermore, the number of environments per worker is
calculated based on the batch size and number of workers.

The results of each experiment are saved to a csv file. The results can then be plotted
using the `plot` action of this script.

The script supports running benchmarking for both the simple gym environments (it uses
"CartPole-v1" by default), as well as atari environments ("Pong" is used by default).

Example usage, run benchmarking for CartPole-v1:

    python benchmarking.py run

Example usage, run benchmarking for Pong:

    python benchmarking.py run --atari

Example usage, plot results for CartPole-v1:

    python benchmarking.py plot

Example usage, plot results for Pong (using default save file):

    python benchmarking.py plot --atari

    
Note each of the above will save and load results to a default location. To specify a 
custom save location, use the `--save-file` argument.

For all options, run:

    python benchmarking.py --help

"""
import gc
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

from minidrl.ppo.ppo import PPOConfig, run_rollout_worker
from minidrl.ppo.run_atari import atari_model_loader, get_atari_env_creator_fn
from minidrl.ppo.run_gym import get_gym_env_creator_fn, gym_model_loader

CARTPOLE_CONFIG = {
    "config_kwargs": {
        "exp_name": "ppo_benchmarking",
        "seed": 0,
        "torch_deterministic": True,
        "cuda": True,
        "track_wandb": False,
        "env_id": "CartPole-v1",
        "num_rollout_steps": 128,
        "minibatch_size": 2048,
    },
    "env_creator_fn_getter": get_gym_env_creator_fn,
    "model_loader": gym_model_loader,
}

ATARI_CONFIG = {
    "config_kwargs": {
        "exp_name": "ppo_benchmarking_atari",
        "seed": 0,
        "torch_deterministic": True,
        "cuda": True,
        "track_wandb": False,
        "env_id": "PongNoFrameskip-v4",
        "num_rollout_steps": 128,
        "minibatch_size": 2048,
    },
    "env_creator_fn_getter": get_atari_env_creator_fn,
    "model_loader": atari_model_loader,
}


# number of updates to run per experiment
NUM_UPDATES = 20

# full exp
NUM_WORKERS = [1, 2, 4, 8, 16, 32]
NUM_ENVS_SINGLE_WORKER = [1, 2, 4, 8, 16, 32, 64, 128, 256]
NUM_ENVS_PER_WORKER = [1, 2, 4, 8]
CARTPOLE_BATCH_SIZES = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
ATARI_BATCH_SIZES = [2048, 4096, 8192, 16384, 32768]

# smaller exp
# NUM_UPDATES = 2
# NUM_WORKERS = [1, 2]
# NUM_ENVS_SINGLE_WORKER = [1, 2, 4, 8, 16]
# NUM_ENVS_PER_WORKER = [1, 2, 4]
# CARTPOLE_BATCH_SIZES = [2048, 4096]
# ATARI_BATCH_SIZES = [2048, 4096]


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
    model = config.load_model()
    model.to(device)

    # Experience buffer setup
    buf_shape = (config.seq_len, config.num_seqs_per_batch)
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
        torch.zeros(
            model.lstm.num_layers, config.num_seqs_per_batch, model.lstm.hidden_size
        ).to(device),
        torch.zeros(
            model.lstm.num_layers, config.num_seqs_per_batch, model.lstm.hidden_size
        ).to(device),
    )

    # buffers for tracking episode stats (mean, min, max)
    ep_returns = torch.zeros((config.num_workers, 3)).cpu()
    ep_lens = torch.zeros((config.num_workers, 3)).cpu()

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
        input_queues.append(mp_ctxt.SimpleQueue())
        output_queues.append(mp_ctxt.SimpleQueue())
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

    # Optimizer setup
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)

    # Initialize rollout workers by doing a single batch
    # This first batch is always much slower due to start up overhead
    for i in range(config.num_workers):
        input_queues[i].put({k: v.cpu() for k, v in model.state_dict().items()})
    for i in range(config.num_workers):
        output_queues[i].get()

    setup_time = time.time() - setup_start_time

    # Training loop
    global_step = 0
    experience_sps_per_update = []
    learning_sps_per_update = []
    start_time = time.time()
    experience_collection_time = 0
    learning_time = 0
    for update in range(1, config.num_updates + 1):
        # current model state to share with workers
        model_state = {k: v.cpu() for k, v in model.state_dict().items()}

        experience_collection_start_time = time.time()
        # signal workers to collect next batch of experience
        for i in range(config.num_workers):
            input_queues[i].put(model_state)

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

        # log episode stats
        experience_collection_time += time.time() - experience_collection_start_time
        experience_sps_per_update.append(
            config.batch_size / (time.time() - experience_collection_start_time)
        )
        global_step += config.batch_size

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
        seq_indxs = np.arange(config.num_seqs_per_batch)
        flat_indxs = np.arange(config.batch_size).reshape(
            config.seq_len, config.num_seqs_per_batch
        )
        for epoch in range(config.update_epochs):
            np.random.shuffle(seq_indxs)

            # do minibatch update
            # each minibatch uses data from randomized subset of envs
            for start in range(
                0,
                config.num_seqs_per_batch,
                config.num_seqs_per_batch // config.num_minibatches,
            ):
                end = start + (config.num_seqs_per_batch // config.num_minibatches)
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
                    pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

        learning_time += time.time() - learning_start_time
        learning_sps_per_update.append(
            config.batch_size / (time.time() - learning_start_time)
        )

    total_time = time.time() - start_time
    env.close()

    for i in range(config.num_workers):
        input_queues[i].put(0)

    for i in range(config.num_workers):
        input_queues[i].close()
        output_queues[i].close()

    for i in range(config.num_workers):
        workers[i].join()

    return {
        "total_steps": global_step,
        "total_time": total_time,
        "sps": int(global_step / total_time),
        "experience_sps": int(np.mean(experience_sps_per_update)),
        "experience_sps_std": int(np.std(experience_sps_per_update)),
        "learning_sps": int(np.mean(learning_sps_per_update)),
        "learning_sps_std": int(np.std(learning_sps_per_update)),
        "setup_time": setup_time,
        "experience_collection_time": experience_collection_time,
        "learning_time": learning_time,
    }


def run_fixed_batch_size(
    num_workers: list[int],
    batch_sizes: list[int],
    config_params: dict,
    save_file: str,
    append_results: bool = False,
):
    """Benchmark SPS vs num workers for a fixed batch size."""
    print("Running benchmarking fixed batch size experiments")
    print(f"Saving results to: {save_file}")
    num_exps = len(num_workers) * len(batch_sizes)
    print(f"Running a total of {num_exps} experiments")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    header_written, exp_num = False, 0
    if append_results:
        header_written = True
        with open(save_file, "r") as f:
            exp_num = len(f.readlines()) - 1

    seq_len = config_params["config_kwargs"]["num_rollout_steps"]
    for nw, batch_size in product(num_workers, batch_sizes):
        # Num envs per worker
        ne = max(1, int(np.ceil(batch_size / (nw * seq_len))))

        total_timesteps = NUM_UPDATES * batch_size
        print(
            f"Running exp_num={exp_num}, batch_size={batch_size}, "
            f"num_workers={nw}, num_envs_per_worker={ne}, "
            f"num_rollout_steps={seq_len} for {total_timesteps} steps"
        )
        config = PPOConfig(
            total_timesteps=total_timesteps,
            num_envs_per_worker=ne,
            num_workers=nw,
            **config_params["config_kwargs"],
        )
        config.env_creator_fn_getter = config_params["env_creator_fn_getter"]
        config.model_loader = config_params["model_loader"]

        result = run_benchmarking_ppo(config)
        print(
            f"SPS: {result['sps']} | "
            f"EXP-SPS: {result['experience_sps']} +/- "
            f"{result['experience_sps_std']:.2f} | "
            f"LEARN-SPS: {result['learning_sps']} +/- "
            f"{result['learning_sps_std']:.2f}"
        )

        if not header_written:
            with open(save_file, "w") as f:
                f.write(
                    "exp_num,num_workers,batch_size,num_envs_per_worker,"
                    + "num_rollout_steps,minibatch_size,num_minibatches,"
                    + ",".join(result.keys())
                    + "\n"
                )
            header_written = True

        with open(save_file, "a") as f:
            f.write(
                f"{exp_num},{nw},{batch_size},{ne},"
                + f"{seq_len},{config.minibatch_size},{config.num_minibatches},"
                + ",".join(map(str, result.values()))
                + "\n"
            )
        exp_num += 1

        # give time for CUDA cleanup so we don't get OOM errors
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
        gc.collect()
        time.sleep(2)

    print("Benchmarking experiments finished")


def run_fixed_num_envs(
    num_workers: list[int],
    num_envs_per_worker: list[int],
    config_params: dict,
    save_file: str,
    append_results: bool = False,
):
    """Benchmark SPS vs num_envs_per_worker for a single worker."""
    print("Running fixed num_envs_per_worker experiments")
    print(f"Saving results to: {save_file}")
    print(
        f"Running a total of {len(num_envs_per_worker) * len(num_workers)} experiments"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    header_written, exp_num = False, 0
    if append_results:
        header_written = True
        with open(save_file, "r") as f:
            exp_num = len(f.readlines()) - 1

    config_params["config_kwargs"].pop("minibatch_size")
    seq_len = config_params["config_kwargs"]["num_rollout_steps"]
    for nw, ne in product(num_workers, num_envs_per_worker):
        batch_size = ne * seq_len * nw
        total_timesteps = NUM_UPDATES * batch_size

        print(
            f"Running exp_num={exp_num}, batch_size={batch_size}, "
            f"num_workers={nw}, num_envs_per_worker={ne}, "
            f"num_rollout_steps={seq_len} for {total_timesteps} steps"
        )
        config = PPOConfig(
            total_timesteps=total_timesteps,
            num_envs_per_worker=ne,
            num_workers=nw,
            minibatch_size=batch_size // 8,  # same number of minibatches
            **config_params["config_kwargs"],
        )
        config.env_creator_fn_getter = config_params["env_creator_fn_getter"]
        config.model_loader = config_params["model_loader"]

        result = run_benchmarking_ppo(config)
        print(
            f"SPS: {result['sps']} | "
            f"EXP-SPS: {result['experience_sps']} +/- "
            f"{result['experience_sps_std']:.2f} | "
            f"LEARN-SPS: {result['learning_sps']} +/- "
            f"{result['learning_sps_std']:.2f}"
        )

        if not header_written:
            with open(save_file, "w") as f:
                f.write(
                    "exp_num,num_workers,batch_size,num_envs_per_worker,"
                    + "num_rollout_steps,minibatch_size,num_minibatches,"
                    + ",".join(result.keys())
                    + "\n"
                )
            header_written = True

        with open(save_file, "a") as f:
            f.write(
                f"{exp_num},{nw},{batch_size},{ne},"
                + f"{seq_len},{config.minibatch_size},{config.num_minibatches},"
                + ",".join(map(str, result.values()))
                + "\n"
            )
        exp_num += 1

        # give time for CUDA cleanup so we don't get OOM errors
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
        gc.collect()
        time.sleep(2)

    print("Benchmarking experiments finished")


def plot_fixed_batch_size(save_file: str):
    """Plot the results of fixed batch size vs num workers experiments."""
    df = pd.read_csv(save_file)

    sns.set_theme()
    num_workers = df["num_workers"].unique().tolist()
    num_workers.sort()
    batch_sizes = df["batch_size"].unique().tolist()
    batch_sizes.sort()

    y_keys = [
        ("experience_sps", "experience_sps_std"),
        ("learning_sps", "learning_sps_std"),
        ("sps", None),
        ("num_envs_per_worker", None),
    ]

    fig, axs = plt.subplots(
        nrows=len(y_keys),
        ncols=1,
        squeeze=True,
        sharex=True,
        sharey=False,
    )

    # Plot the results for each batch size by num workers
    df.sort_values(by=["num_workers"], inplace=True, ascending=True)
    for row, (y, yerr) in enumerate(y_keys):
        ax = axs[row]
        for batch_size in batch_sizes:
            df_subset = df[(df["batch_size"] == batch_size)]
            if yerr:
                ax.errorbar(
                    x=df_subset["num_workers"],
                    y=df_subset[y],
                    yerr=df_subset[yerr],
                    label=f"{batch_size}",
                )
            else:
                ax.plot(
                    df_subset["num_workers"],
                    df_subset[y],
                    label=f"{batch_size}",
                )

        # ax.set_yscale("log", base=2)
        ax.set_ylabel(f"{y}")
        if row == len(y_keys) - 1:
            ax.set_xlabel("Num Workers")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center right",
        ncol=1,
        title="Batch Size",
    )
    fig.tight_layout(rect=(0, 0, 0.8, 1))
    plt.show()


def plot_single_worker_fixed_num_envs(save_file: str):
    """Plot the results of single worker, fixed num envs experiments."""
    df = pd.read_csv(save_file)

    sns.set_theme()
    df = df[(df["num_workers"] == 1)]

    y_keys = [
        ("experience_sps", "experience_sps_std"),
        ("learning_sps", "learning_sps_std"),
        ("sps", None),
        ("batch_size", None),
    ]

    fig, axs = plt.subplots(
        nrows=len(y_keys),
        ncols=1,
        squeeze=True,
        sharex=True,
        sharey=False,
    )

    df.sort_values(by=["num_envs_per_worker"], inplace=True, ascending=True)
    for row, (y, yerr) in enumerate(y_keys):
        ax = axs[row]

        if yerr:
            ax.errorbar(
                x=df["num_envs_per_worker"],
                y=df[y],
                yerr=df[yerr],
            )
        else:
            ax.plot(
                df["num_envs_per_worker"],
                df[y],
            )

        # ax.set_yscale("log", base=2)
        ax.set_ylabel(f"{y}")
        ax.set_xscale("log", base=2)
        if row == len(y_keys) - 1:
            ax.set_xlabel("Num Envs Per Worker")

    fig.tight_layout()
    plt.show()


def plot_fixed_num_envs(save_file: str):
    """Plot the results of fixed num envs vs num workers experiments."""
    df = pd.read_csv(save_file)

    sns.set_theme()
    num_workers = df["num_workers"].unique().tolist()
    num_workers.sort()
    num_envs_per_worker = df["num_envs_per_worker"].unique().tolist()
    num_envs_per_worker.sort()

    y_keys = [
        ("experience_sps", "experience_sps_std"),
        ("learning_sps", "learning_sps_std"),
        ("sps", None),
        ("batch_size", None),
    ]

    fig, axs = plt.subplots(
        nrows=len(y_keys),
        ncols=1,
        squeeze=True,
        sharex=True,
        sharey=False,
    )

    # Plot the results for each batch size by num workers
    df.sort_values(by=["num_workers"], inplace=True, ascending=True)
    for row, (y, yerr) in enumerate(y_keys):
        ax = axs[row]
        for num_envs in num_envs_per_worker:
            df_subset = df[(df["num_envs_per_worker"] == num_envs)]
            if yerr:
                ax.errorbar(
                    x=df_subset["num_workers"],
                    y=df_subset[y],
                    yerr=df_subset[yerr],
                    label=f"{num_envs}",
                )
            else:
                ax.plot(
                    df_subset["num_workers"],
                    df_subset[y],
                    label=f"{num_envs}",
                )

        # ax.set_yscale("log", base=2)
        ax.set_ylabel(f"{y}")
        if row == len(y_keys) - 1:
            ax.set_xlabel("Num Workers")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center right",
        ncol=1,
        title="Num Envs",
    )
    fig.tight_layout(rect=(0, 0, 0.8, 1))
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "action",
        type=str,
        choices=["run_fbs", "run_swne", "run_fne", "plot_fbs", "plot_swne", "plot_fne"],
        help=(
            "Action to perform. "
            "`run_fbs` - fixed batch size vs num workers. "
            "`run_swne` - fixed number of envs per worker with single worker. "
            "`run_fne` - fixed number of envs per worker vs num workers. "
            "`plot_fbs` - plot fixed batch size vs num workers results. "
            "`plot_swne` - fixed number of envs per worker with single worker. "
            "`plot_fne` - plot fixed number of envs per worker vs num workers."
        ),
    )
    parser.add_argument(
        "--atari", action="store_true", help="Run benchmarking on Atari environment."
    )
    parser.add_argument("--append-results", action="store_true")
    parser.add_argument(
        "--save-file",
        type=str,
        default=None,
        help=(
            "Path to csv file to save results to. If not specified, results are "
            "saved to a default location."
        ),
    )
    args = parser.parse_args()

    save_file_arg = args.save_file
    if not save_file_arg:
        save_file_name = "benchmarking_results"
        if args.atari:
            save_file_name += "_atari"
        save_file_name += "_" + args.action.split("_")[1]
        save_file_name += ".csv"
        save_file_arg = os.path.join(os.path.dirname(__file__), save_file_name)

    if args.action == "run_fbs":
        run_fixed_batch_size(
            num_workers=NUM_WORKERS,
            batch_sizes=ATARI_BATCH_SIZES if args.atari else CARTPOLE_BATCH_SIZES,
            config_params=ATARI_CONFIG if args.atari else CARTPOLE_CONFIG,
            save_file=save_file_arg,
            append_results=args.append_results,
        )
    elif args.action == "run_swne":
        run_fixed_num_envs(
            num_workers=[1],
            num_envs_per_worker=NUM_ENVS_SINGLE_WORKER,
            config_params=ATARI_CONFIG if args.atari else CARTPOLE_CONFIG,
            save_file=save_file_arg,
            append_results=args.append_results,
        )
    elif args.action == "run_fne":
        run_fixed_num_envs(
            num_workers=NUM_WORKERS,
            num_envs_per_worker=NUM_ENVS_PER_WORKER,
            config_params=ATARI_CONFIG if args.atari else CARTPOLE_CONFIG,
            save_file=save_file_arg,
            append_results=args.append_results,
        )
    elif args.action == "plot_fbs":
        plot_fixed_batch_size(save_file_arg)
    elif args.action == "plot_swne":
        plot_single_worker_fixed_num_envs(save_file_arg)
    elif args.action == "plot_fne":
        plot_fixed_num_envs(save_file_arg)
    else:
        raise ValueError("Invalid action")
