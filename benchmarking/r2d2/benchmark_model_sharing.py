"""Runs SPS benchmarking for R2D2 replay buffer."""
import os
import time
from itertools import product
from multiprocessing.queues import Empty

import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.multiprocessing as mp
import torch.optim as optim
from minidrl.r2d2.r2d2 import R2D2Config, compute_loss_and_priority
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
        "batch_size": 64,
        "replay_size": 5000,
        "learning_starts": 100,
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
        "batch_size": 64,
        "replay_size": 5000,
        "learning_starts": 100,
    },
    "env_creator_fn_getter": get_atari_env_creator_fn,
    "model_loader": atari_model_loader,
}


def run_pretend_actor(
    actor_idx: int,
    request_interval: float,
    config: R2D2Config,
    learner_recv_queue: mp.JoinableQueue,
    terminate_event: mp.Event,
):
    print(f"actor={actor_idx}: Actor started.")
    torch.set_num_threads(1)

    actor_model = config.load_model()
    actor_model.to(config.actor_device)

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

    while not terminate_event.is_set():
        time.sleep(request_interval)
        actor_model.load_state_dict(learner_model_weights)

    del learner_model_weights
    learner_recv_queue.task_done()

    print(f"actor={actor_idx} - Actor Finished.")


def run_share_weights_experiment(
    config: R2D2Config,
    request_interval: float,
    num_updates: int,
    obs_space: gym.spaces.Box,
    num_actions: int,
) -> dict:
    """Runs SPS benchmarking for adding batches of sequences to R2D2 replay buffer."""
    torch.set_num_threads(1)
    mp_ctxt = mp.get_context("spawn")
    actor_recv_queue = mp_ctxt.JoinableQueue()
    terminate_event = mp_ctxt.Event()

    print("main: Spawning actor processes.")
    actors = []
    for actor_idx in range(config.num_actors):
        actor = mp_ctxt.Process(
            target=run_pretend_actor,
            args=(
                actor_idx,
                request_interval,
                config,
                actor_recv_queue,
                terminate_event,
            ),
        )
        actor.start()
        actors.append(actor)

    model = config.load_model()
    model.share_memory()
    model.to(config.device)

    optimizer = optim.Adam(
        model.parameters(), lr=config.learning_rate, eps=config.adam_eps
    )

    # Send shared memory model weights reference to actors
    for _ in range(config.num_actors):
        actor_recv_queue.put(("params", model.state_dict()))

    total_seq_len = config.seq_len

    b_obs = torch.randn(
        (total_seq_len + 1, config.batch_size, *obs_space.shape), dtype=torch.float32
    ).to(config.device)
    b_actions = torch.randint(
        0, num_actions, (total_seq_len + 1, config.batch_size), dtype=torch.long
    ).to(config.device)
    b_rewards = torch.randn(
        (total_seq_len + 1, config.batch_size), dtype=torch.float32
    ).to(config.device)
    b_dones = torch.randint(
        0, 2, (total_seq_len + 1, config.batch_size), dtype=torch.long
    ).to(config.device)
    b_lstm_h = torch.randn(
        (1, config.batch_size, config.lstm_size), dtype=torch.float32
    ).to(config.device)
    b_lstm_c = torch.randn(
        (1, config.batch_size, config.lstm_size), dtype=torch.float32
    ).to(config.device)

    update = 1
    train_start_time = time.time()
    while update < num_updates + 1 and not terminate_event.is_set():
        q_values, _ = model.forward(
            b_obs, b_actions, b_rewards, b_dones, (b_lstm_h, b_lstm_c)
        )

        loss, _ = compute_loss_and_priority(
            config=config,
            q_values=q_values,
            actions=b_actions,
            rewards=b_rewards,
            dones=b_dones,
            target_q_values=q_values,
        )

        loss = torch.mean(loss)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()

        update += 1

    train_time = time.time() - train_start_time

    print("main: shutting down.")
    terminate_event.set()
    actor_recv_queue.join()

    for actor in actors:
        actor.join()
    print("main: All done")

    return {
        "num_actors": config.num_actors,
        "request_interval": request_interval,
        "RPS": config.num_actors * (1.0 / request_interval),
        "UPS": update / train_time,
    }


def run_share_weights_benchmarking(
    request_intervals: list[float],
    num_updates: int,
    num_actors: list[int],
    config_params: dict,
    save_file: str,
):
    """Run R2D2 Benchmarking."""
    print("Running R2D2 share weights benchmarking experiments")
    num_exps = len(request_intervals) * len(num_actors)
    print(f"Running a total of {num_exps} experiments")

    header_written = False
    obs_space, num_actions = None, None
    for exp_num, (ri, na) in enumerate(product(request_intervals, num_actors)):
        print(
            f"Running exp_num={exp_num}, "
            f"request_interval={ri}, num_actors={na}, "
            f"for {num_updates} updates"
        )
        config = R2D2Config(
            num_actors=na,
            **config_params["config_kwargs"],
        )
        config.env_creator_fn_getter = config_params["env_creator_fn_getter"]
        config.model_loader = config_params["model_loader"]

        if obs_space is None:
            env = config.env_creator_fn_getter(config, env_idx=0)()
            obs_space = env.observation_space
            num_actions = env.action_space.n

        exp_start_time = time.time()
        exp_results = run_share_weights_experiment(
            config, ri, num_updates, obs_space, num_actions
        )

        if not header_written:
            with open(save_file, "w") as f:
                f.write("exp_num," + ",".join(exp_results.keys()) + "\n")
            header_written = True

        with open(save_file, "a") as f:
            f.write(f"{exp_num}," + ",".join(map(str, exp_results.values())) + "\n")

        exp_time = time.time() - exp_start_time
        print(f"{exp_num=} complete. Time taken = {exp_time:.3f} s")

    print("Benchmarking experiments finished")


def plot_share_weights(save_file: str):
    """Plot results from R2D2 share weights benchmarking experiments."""
    df = pd.read_csv(save_file)

    sns.set_theme()

    y_keys = ["UPS"]

    fig, axs = plt.subplots(
        nrows=min(len(y_keys), 4),
        ncols=len(y_keys) // 4 + int(len(y_keys) % 4 != 0),
        squeeze=False,
        sharex=True,
        sharey=False,
    )

    df.sort_values(by=["request_interval"], inplace=True, ascending=True)

    x_ticks = df["request_interval"].unique().tolist()
    x_ticks.sort()

    num_actors = df["num_actors"].unique().tolist()
    num_actors.sort()
    for i, y in enumerate(y_keys):
        row, col = i % 4, i // 4
        ax = axs[row][col]

        for na in num_actors:
            df_na = df[df["num_actors"] == na]

            ax.plot(
                df_na["request_interval"],
                df_na[y],
                label=f"{na}",
            )

        # ax.set_yscale("log", base=2)
        ax.set_xscale("log", base=10)
        ax.set_title(f"{y}")
        ax.set_xticks(x_ticks)
        ax.set_ylabel(y)
        if row == len(y_keys) - 1:
            ax.set_xlabel("Request Interval")

        ax.legend(title="Num Actors")

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
        choices=["run", "plot"],
        help=(
            "Action to perform. " "`run` - run experiments. " "`plot` - plot results. "
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
        "--num_updates",
        type=int,
        default=int(1e5),
        help="Number of updates to run each experiment for.",
    )
    args = parser.parse_args()

    save_file_arg = args.save_file
    if not save_file_arg:
        save_file_name = "model_sharing_benchmarking_results"
        if args.atari:
            save_file_name += "_atari"
        # save_file_name += "_" + args.action.split("_")[1]
        save_file_name += ".csv"
        save_file_arg = os.path.join(os.path.dirname(__file__), save_file_name)

    if args.action == "run":
        run_share_weights_benchmarking(
            request_intervals=[0.01, 0.05, 0.1, 0.5, 1.0],
            num_updates=args.num_updates,
            num_actors=[1, 2, 4],
            config_params=ATARI_CONFIG if args.atari else CARTPOLE_CONFIG,
            save_file=save_file_arg,
        )
    elif args.action == "plot":
        plot_share_weights(save_file_arg)
    else:
        raise ValueError("Invalid action")
