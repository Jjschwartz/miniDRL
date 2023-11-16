"""Runs SPS benchmarking for R2D2 replay buffer."""
import os
import random
import time
from multiprocessing.queues import Empty, Full

import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.multiprocessing as mp
from minidrl.r2d2.r2d2 import R2D2Config
from minidrl.r2d2.replay import run_replay_process
from minidrl.r2d2.run_atari import atari_model_loader, get_atari_env_creator_fn
from minidrl.r2d2.run_gym import get_gym_env_creator_fn, gym_model_loader


CARTPOLE_CONFIG = {
    "config_kwargs": {
        "exp_name": "r2d2_replay_benchmarking_gym",
        "seed": 0,
        "torch_deterministic": True,
        "cuda": True,
        "track_wandb": False,
        "env_id": "CartPole-v1",
        "seq_len": 10,
        "burnin_len": 0,
        "num_prefetch_batches": 8,
        "batch_size": 64,
        "replay_size": 5000,
        "learning_starts": 8 * 64,
    },
    "env_creator_fn_getter": get_gym_env_creator_fn,
    "model_loader": gym_model_loader,
}

ATARI_CONFIG = {
    "config_kwargs": {
        "exp_name": "r2d2_replay_benchmarking_atari",
        "seed": 0,
        "torch_deterministic": True,
        "cuda": True,
        "track_wandb": False,
        "env_id": "PongNoFrameskip-v4",
        "seq_len": 80,
        "burnin_len": 40,
        "num_prefetch_batches": 8,
        "batch_size": 64,
        "replay_size": 2500,
        "learning_starts": 8 * 64,
    },
    "env_creator_fn_getter": get_atari_env_creator_fn,
    "model_loader": atari_model_loader,
}


NUM_ENVS_PER_ACTOR = [1, 2, 4, 8, 16, 32, 64, 128, 256]
BATCH_SIZES = [8, 16, 32, 64, 128, 256, 512]


def run_add_experiment(
    config: R2D2Config,
    num_batches: int,
    obs_space: gym.spaces.Box,
    num_actions: int,
) -> dict:
    """Runs SPS benchmarking for adding batches of sequences to R2D2 replay buffer."""
    mp_ctxt = mp.get_context("spawn")
    terminate_event = mp_ctxt.Event()
    actor_to_replay_queue = mp_ctxt.JoinableQueue(maxsize=config.num_actors * 2)
    learner_to_replay_queue = mp_ctxt.JoinableQueue()
    replay_to_learner_queue = mp_ctxt.JoinableQueue()
    log_queue = mp_ctxt.JoinableQueue()

    print("main: Spawning replay process.")
    replay_process = mp_ctxt.Process(
        target=run_replay_process,
        args=(
            config,
            actor_to_replay_queue,
            learner_to_replay_queue,
            replay_to_learner_queue,
            log_queue,
            terminate_event,
        ),
    )
    replay_process.start()

    total_seq_len = config.seq_len + config.burnin_len
    num_envs = config.num_envs_per_actor

    obs_seq_buffer = torch.randn(
        (total_seq_len + 1, num_envs, *obs_space.shape), dtype=torch.float32
    )
    action_seq_buffer = torch.randint(
        0, num_actions, (total_seq_len + 1, num_envs), dtype=torch.long
    )
    reward_seq_buffer = torch.randn((total_seq_len + 1, num_envs), dtype=torch.float32)
    done_buffer = torch.randint(0, 2, (total_seq_len + 1, num_envs), dtype=torch.long)
    lstm_h_seq_buffer = torch.randn(
        (1, num_envs, config.lstm_size), dtype=torch.float32
    )
    lstm_c_seq_buffer = torch.randn(
        (1, num_envs, config.lstm_size), dtype=torch.float32
    )
    priorities = torch.rand((100, num_envs))

    batch, max_q_size = 0, 0
    add_start_time = time.time()
    while batch < num_batches and replay_process.is_alive():
        # Send a batch of replay data to the replay process.
        priority = priorities[random.randint(0, 99)]
        while replay_process.is_alive():
            try:
                actor_to_replay_queue.put(
                    (
                        obs_seq_buffer,
                        action_seq_buffer,
                        reward_seq_buffer,
                        done_buffer,
                        lstm_h_seq_buffer,
                        lstm_c_seq_buffer,
                        priority,
                    ),
                    block=True,  # block until space available
                    timeout=1,  # timeout after 1 second, to check for terminate
                )
                break
            except Full:
                pass

        max_q_size = max(max_q_size, actor_to_replay_queue.qsize())

        batch += 1
        if batch % max(1, num_batches // 10) == 0:
            print(f"main: batch={batch}/{num_batches}")

    if not replay_process.is_alive():
        print("main: replay process terminated early.")
        terminate_event.set()
        replay_process.join()
        raise AssertionError("Replay process terminated early.")

    # wait for all replay data to be processed
    actor_to_replay_queue.join()
    add_time_taken = time.time() - add_start_time

    replay_stats = {
        "add_time_taken": add_time_taken,
        "num_batches": num_batches,
        "seqs_added": num_batches * num_envs,
        "added_seq_per_sec": (num_batches * num_envs) / add_time_taken,
        "added_batch_per_sec": num_batches / add_time_taken,
        "max_q_size": max_q_size,
    }

    print("main: shutting down.")
    terminate_event.set()

    actor_to_replay_queue.close()
    learner_to_replay_queue.close()
    replay_to_learner_queue.close()

    while not log_queue.empty():
        log_queue.get()
        log_queue.task_done()
    log_queue.close()

    replay_process.join()
    print("main: All done")
    return replay_stats


def run_sample_experiment(
    config: R2D2Config,
    num_batches: int,
    batch_sizes: list[int],
    obs_space: gym.spaces.Box,
    num_actions: int,
) -> dict:
    """Runs SPS benchmarking for adding batches of sequences to R2D2 replay buffer."""
    mp_ctxt = mp.get_context("spawn")
    terminate_event = mp_ctxt.Event()
    actor_to_replay_queue = mp_ctxt.JoinableQueue(maxsize=config.num_actors * 2)
    learner_to_replay_queue = mp_ctxt.JoinableQueue()
    replay_to_learner_queue = mp_ctxt.JoinableQueue()
    log_queue = mp_ctxt.JoinableQueue()

    print("main: Spawning replay process.")
    replay_process = mp_ctxt.Process(
        target=run_replay_process,
        args=(
            config,
            actor_to_replay_queue,
            learner_to_replay_queue,
            replay_to_learner_queue,
            log_queue,
            terminate_event,
        ),
    )
    replay_process.start()

    total_seq_len = config.seq_len + config.burnin_len
    num_envs = config.num_envs_per_actor

    obs_seq_buffer = torch.randn(
        (total_seq_len + 1, num_envs, *obs_space.shape), dtype=torch.float32
    )
    action_seq_buffer = torch.randint(
        0, num_actions, (total_seq_len + 1, num_envs), dtype=torch.long
    )
    reward_seq_buffer = torch.randn((total_seq_len + 1, num_envs), dtype=torch.float32)
    done_buffer = torch.randint(0, 2, (total_seq_len + 1, num_envs), dtype=torch.long)
    lstm_h_seq_buffer = torch.randn(
        (1, num_envs, config.lstm_size), dtype=torch.float32
    )
    lstm_c_seq_buffer = torch.randn(
        (1, num_envs, config.lstm_size), dtype=torch.float32
    )
    priorities = torch.rand((100, num_envs))

    print("main: filling replay buffer.")
    step, last_report_step = 0, 0
    while step < config.replay_size and replay_process.is_alive():
        # Send a batch of replay data to the replay process.
        priority = priorities[random.randint(0, priorities.shape[0] - 1)]
        while replay_process.is_alive():
            try:
                actor_to_replay_queue.put(
                    (
                        obs_seq_buffer,
                        action_seq_buffer,
                        reward_seq_buffer,
                        done_buffer,
                        lstm_h_seq_buffer,
                        lstm_c_seq_buffer,
                        priority,
                    ),
                    block=True,  # block until space available
                    timeout=1,  # timeout after 1 second, to check for terminate
                )
                break
            except Full:
                pass
        step += num_envs
        if step - last_report_step > 1000:
            print(f"main: replay buffer size={step}/{config.replay_size}")
            last_report_step = step

    if not replay_process.is_alive():
        print("main: replay process terminated early.")
        terminate_event.set()
        replay_process.join()
        raise AssertionError("Replay process terminated early.")

    print("main: waiting for replay buffer to process all added batches.")
    actor_to_replay_queue.join()

    print("main: running benchmarking sampling.")
    replay_stats = []
    for exp_num, batch_size in enumerate(batch_sizes):
        print(f"\nmain: running {exp_num=} {batch_size=}.")

        batch = 0
        start_time = time.time()
        while batch < num_batches and replay_process.is_alive():
            learner_to_replay_queue.put(("sample", batch_size))
            while replay_process.is_alive():
                try:
                    replay_to_learner_queue.get(timeout=1)
                    replay_to_learner_queue.task_done()
                    break
                except Empty:
                    pass
            batch += 1
            if batch % max(1, num_batches // 10) == 0:
                print(f"main: sample_batch={batch}/{num_batches}")

        time_taken = time.time() - start_time

        batch_replay_stats = {
            "exp_num": exp_num,
            "batch_size": batch_size,
            "time_taken": time_taken,
            "seqs_sampled": num_batches * batch_size,
            "sampled_seq_per_sec": (num_batches * batch_size) / time_taken,
            "sampled_batch_per_sec": num_batches / time_taken,
        }
        replay_stats.append(batch_replay_stats)

    print("main: shutting down.")
    terminate_event.set()

    actor_to_replay_queue.close()
    learner_to_replay_queue.join()
    learner_to_replay_queue.close()
    replay_to_learner_queue.join()
    replay_to_learner_queue.close()

    while not log_queue.empty():
        log_queue.get()
        log_queue.task_done()
    log_queue.close()

    replay_process.join()
    print("main: All done")

    return replay_stats


def run_replay_add_benchmarking(
    num_batches: int,
    num_envs_per_actor: list[int],
    config_params: dict,
    save_file: str,
):
    """Run R2D2 Benchmarking."""
    print("Running R2D2 replay add benchmarking experiments")
    num_exps = len(num_envs_per_actor)
    print(f"Running a total of {num_exps} experiments")

    learning_starts = config_params["config_kwargs"]["learning_starts"]
    assert num_batches * min(num_envs_per_actor) >= learning_starts, (
        "Not enough batches to fill replay buffer. Need at least "
        f"{learning_starts / min(num_envs_per_actor)} batches."
    )

    header_written = False
    obs_space, num_actions = None, None
    for exp_num, ne in enumerate(num_envs_per_actor):
        print(
            f"Running exp_num={exp_num}, "
            f"num_envs_per_actor={ne}, "
            f"for {num_batches} batches"
        )
        config = R2D2Config(
            num_envs_per_actor=ne,
            **config_params["config_kwargs"],
        )
        config.env_creator_fn_getter = config_params["env_creator_fn_getter"]
        config.model_loader = config_params["model_loader"]

        if obs_space is None:
            env = config.env_creator_fn_getter(config, env_idx=0)()
            obs_space = env.observation_space
            num_actions = env.action_space.n

        exp_start_time = time.time()
        exp_results = run_add_experiment(config, num_batches, obs_space, num_actions)

        if not header_written:
            with open(save_file, "w") as f:
                f.write(
                    "exp_num,num_envs_per_actor," + ",".join(exp_results.keys()) + "\n"
                )
            header_written = True

        with open(save_file, "a") as f:
            f.write(
                f"{exp_num},{ne}," + ",".join(map(str, exp_results.values())) + "\n"
            )

        exp_time = time.time() - exp_start_time
        print(f"{exp_num=} complete. Time taken = {exp_time:.3f} s")

    print("Benchmarking experiments finished")


def run_replay_sample_benchmarking(
    num_batches: int,
    batch_sizes: list[int],
    config_params: dict,
    save_file: str,
):
    """Run R2D2 Sample Benchmarking."""
    print("Running R2D2 replay sample benchmarking experiments")
    num_exps = len(batch_sizes)
    print(f"Running a total of {num_exps} experiments")

    config = R2D2Config(
        num_envs_per_actor=128,
        **config_params["config_kwargs"],
    )
    config.env_creator_fn_getter = config_params["env_creator_fn_getter"]
    config.model_loader = config_params["model_loader"]
    print(f"Running on device: {config.device}")

    env = config.env_creator_fn_getter(config, env_idx=0)()
    obs_space = env.observation_space
    num_actions = env.action_space.n

    results = run_sample_experiment(
        config, num_batches, batch_sizes, obs_space, num_actions
    )

    with open(save_file, "w") as f:
        f.write(",".join(results[0].keys()) + "\n")

    for exp_results in results:
        with open(save_file, "a") as f:
            f.write(",".join(map(str, exp_results.values())) + "\n")

    print("Sample Benchmarking experiments finished")


def plot_add(save_file: str):
    """Plot results from R2D2 replay benchmarking experiments."""
    df = pd.read_csv(save_file)

    sns.set_theme()

    y_keys = [
        "seqs_added",
        "added_seq_per_sec",
        "added_batch_per_sec",
        "max_q_size",
    ]

    fig, axs = plt.subplots(
        nrows=min(len(y_keys), 4),
        ncols=len(y_keys) // 4 + int(len(y_keys) % 4 != 0),
        squeeze=False,
        sharex=True,
        sharey=False,
    )

    df.sort_values(by=["num_envs_per_actor"], inplace=True, ascending=True)
    for i, y in enumerate(y_keys):
        row, col = i % 4, i // 4
        ax = axs[row][col]

        ax.plot(
            df["num_envs_per_actor"],
            df[y],
        )

        # ax.set_yscale("log", base=2)
        # ax.set_xscale("log", base=2)
        ax.set_title(f"{y}")
        ax.set_xticks(df["num_envs_per_actor"])
        if row == len(y_keys) - 1:
            ax.set_xlabel("Num Envs Per Actor")

    fig.tight_layout()
    plt.show()


def plot_sample(save_file: str):
    """Plot results from R2D2 replay benchmarking experiments."""
    df = pd.read_csv(save_file)

    sns.set_theme()

    y_keys = [
        "seqs_sampled",
        "sampled_seq_per_sec",
        "sampled_batch_per_sec",
    ]

    fig, axs = plt.subplots(
        nrows=min(len(y_keys), 4),
        ncols=len(y_keys) // 4 + int(len(y_keys) % 4 != 0),
        squeeze=False,
        sharex=True,
        sharey=False,
    )

    df.sort_values(by=["batch_size"], inplace=True, ascending=True)
    for i, y in enumerate(y_keys):
        row, col = i % 4, i // 4
        ax = axs[row][col]

        ax.plot(
            df["batch_size"],
            df[y],
        )

        # ax.set_yscale("log", base=2)
        # ax.set_xscale("log", base=2)
        ax.set_title(f"{y}")
        ax.set_xticks(df["batch_size"])
        if row == len(y_keys) - 1:
            ax.set_xlabel("Batch Size")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "action",
        type=str,
        choices=["run_add", "run_sample", "plot_add", "plot_sample"],
        help=(
            "Action to perform. "
            "`run_add` - run add experiments. "
            "`run_sample` - run sample experiments. "
            "`plot_add` - plot add_results. "
            "`plot_sample` - plot sample_results. "
        ),
    )
    parser.add_argument(
        "--atari", action="store_true", help="Run benchmarking on Atari environment."
    )
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help=(
            "Path to csv file to save results to. If not specified, results are "
            "saved to a default location."
        ),
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=int(1e4),
        help="Number of batches to run each experiment for.",
    )
    args = parser.parse_args()

    save_file_arg = args.save_file
    if not save_file_arg:
        save_file_name = "replay_benchmarking_results"
        if args.atari:
            save_file_name += "_atari"
        save_file_name += "_" + args.action.split("_")[1]
        save_file_name += ".csv"
        save_file_arg = os.path.join(os.path.dirname(__file__), save_file_name)

    if args.action == "run_add":
        run_replay_add_benchmarking(
            num_batches=args.num_batches,
            num_envs_per_actor=NUM_ENVS_PER_ACTOR,
            config_params=ATARI_CONFIG if args.atari else CARTPOLE_CONFIG,
            save_file=save_file_arg,
        )
    elif args.action == "run_sample":
        run_replay_sample_benchmarking(
            num_batches=args.num_batches,
            batch_sizes=BATCH_SIZES,
            config_params=ATARI_CONFIG if args.atari else CARTPOLE_CONFIG,
            save_file=save_file_arg,
        )
    elif args.action == "plot_add":
        plot_add(save_file_arg)
    elif args.action == "plot_sample":
        plot_sample(save_file_arg)
    else:
        raise ValueError("Invalid action")
