"""Script for testing SPS of different R2D2 configurations.

Specifically, we measure the metrics related to steps-per-second (SPS) for the R2D2 
learner, actors and replay during training for different and number of actors. 

For the learner we are interested in updates-per-second (UPS) and steps-per-second (SPS)
(1 update = batch_size * seq_len), and also more fine-grained metrics like sample time,
burnin time, and learning time.

For the actors we are interested in steps-per-second (SPS) collected in the environment.

Lastly, for the replay we are interested in how many new steps and sequences are add to
the replay buffer per second.

For the experiments we mostly use the default hyperparameters, but adjust the `seq_len`
and `burnin_len` depending on the environment. We also reduce the replay buffer size
to avoid any memory issues and since we don't care about performance, just speed.

The results of each experiment are saved to wandb.

The script supports running benchmarking for both the simple gym environments (it uses
"CartPole-v1" by default), as well as atari environments ("Pong" is used by default).

Example usage, run benchmarking for CartPole-v1:

    python benchmarking.py

Example usage, run benchmarking for Pong:

    python benchmarking.py --atari

    
For all options, run:

    python benchmarking.py --help

"""
import argparse
import gc
import math
import time
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from minidrl.r2d2.r2d2 import R2D2Config, run_r2d2
from minidrl.r2d2.run_atari import atari_model_loader, get_atari_env_creator_fn
from minidrl.r2d2.run_gym import get_gym_env_creator_fn, gym_model_loader

CARTPOLE_CONFIG = {
    "config_kwargs": {
        "exp_name": "r2d2_benchmarking_gym",
        "seed": 0,
        "torch_deterministic": True,
        "cuda": True,
        # "track_wandb": True,
        "track_wandb": False,
        "env_id": "CartPole-v1",
        "seq_len": 10,
        "burnin_len": 0,
        "batch_size": 64,
        "replay_buffer_size": 5000,
        "learning_starts": 100,
    },
    "env_creator_fn_getter": get_gym_env_creator_fn,
    "model_loader": gym_model_loader,
}

ATARI_CONFIG = {
    "config_kwargs": {
        "exp_name": "r2d2_benchmarking_atari",
        "seed": 0,
        "torch_deterministic": True,
        "cuda": True,
        # "track_wandb": True,
        "track_wandb": False,
        "env_id": "PongNoFrameskip-v4",
        "seq_len": 80,
        "burnin_len": 40,
        "batch_size": 64,
        "replay_buffer_size": 5000,
        "learning_starts": 100,
    },
    "env_creator_fn_getter": get_atari_env_creator_fn,
    "model_loader": atari_model_loader,
}


# full exp
NUM_ACTORS = [1, 2, 4, 8, 16, 32]
NUM_ENVS_PER_ACTOR = [1, 2, 4, 8, 16, 32, 64, 128, 256]

# smaller exp
SMALL_NUM_ACTORS = [1, 2]
SMALL_NUM_ENVS_PER_ACTOR = [1, 2, 4]


def run_r2d2_benchmarking(
    num_updates: int,
    num_actors: list[int],
    num_envs_per_actor: list[int],
    config_params: dict,
):
    """Run R2D2 Benchmarking."""
    print("Running R2D2 benchmarking experiments")
    num_exps = len(num_actors) * len(num_envs_per_actor)
    print(f"Running a total of {num_exps} experiments")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    batch_size = config_params["config_kwargs"]["batch_size"]
    seq_len = config_params["config_kwargs"]["seq_len"]
    for exp_num, (na, ne) in enumerate(product(num_actors, num_envs_per_actor)):
        exp_start_time = time.time()
        total_timesteps = num_updates * batch_size * seq_len
        print(
            f"Running exp_num={exp_num}, "
            f"num_actors={na}, "
            f"num_envs_per_actor={na}, "
            f"for {total_timesteps} steps"
        )
        config = R2D2Config(
            total_timesteps=total_timesteps,
            num_actors=na,
            num_envs_per_actor=ne,
            **config_params["config_kwargs"],
        )
        config.env_creator_fn_getter = config_params["env_creator_fn_getter"]
        config.model_loader = config_params["model_loader"]

        run_r2d2(config)
        exp_time = time.time() - exp_start_time
        print(f"{exp_num=} complete. Time taken = {exp_time:.3f} s")

        # give time for CUDA cleanup so we don't get OOM errors
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
        gc.collect()
        time.sleep(2)

    print("Benchmarking experiments finished")


def plot_results(results_file: str):
    """Plot the results."""
    df = pd.read_csv(results_file)

    columns = df.columns.to_list()
    sorted_columns = sorted(columns)
    for c in sorted_columns:
        print(c)

    num_crashed = df["State"].value_counts()["crashed"]
    if num_crashed > 0:
        print(
            f"WARNING: {num_crashed} experiments crashed. These will be filtered out."
        )
        df = df[df["State"] != "crashed"]

    sns.set_theme()

    # first plot results for single actor with different number of envs per worker
    df_single_actor = df[(df["num_actors"] == 1)]
    df_single_actor.sort_values(by=["num_envs_per_actor"], inplace=True, ascending=True)

    y_keys = [
        "actor/actor_sps",
        "replay/seq_per_sec",
        "replay/q_size",
        "replay/size",
        "replay/seqs_added",
        "times/learner_UPS",
        "times/burnin_time",
        "times/learning_time",
        "times/sample_time",
        "times/update_time",
    ]
    if len(y_keys) > 3:
        nrows, ncols = math.ceil(len(y_keys) / 3), 3
    else:
        nrows, ncols = 1, len(y_keys)

    print(f"Plotting {nrows} x {ncols} plots")

    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        squeeze=False,
        sharex=False,
        sharey=False,
        figsize=(6, 6),
    )

    print(df_single_actor["num_envs_per_actor"])
    for i, y in enumerate(y_keys):
        if len(y_keys) > 3:
            row, col = i // 3, i % 3
        else:
            row, col = 0, i
        ax = axs[row][col]

        print(f"Plotting {y}")
        print(df_single_actor[y])
        ax.plot(
            df_single_actor["num_envs_per_actor"],
            df_single_actor[y],
        )

        # ax.set_yscale("log", base=2)
        ax.set_ylabel(f"{y}")
        # ax.set_xscale("log", base=2)
        ax.set_xlabel("Num Envs Per Actor")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "action",
        type=str,
        choices=["run", "plot"],
        help=(
            "Action to perform. "
            "`run` - run benchmark experiments. "
            "`plot` - plot results. "
        ),
    )
    parser.add_argument(
        "--atari", action="store_true", help="Run benchmarking on Atari environment."
    )
    parser.add_argument(
        "--num_updates",
        type=int,
        default=500,
        help="Number of learner updates to run each benchmark experiment for.",
    )
    parser.add_argument(
        "--small_run",
        action="store_true",
        help="Run smaller benchmark experiments.",
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default=None,
        help="Path to csv results file (required if plotting).",
    )
    args = parser.parse_args()

    if args.action == "run":
        run_r2d2_benchmarking(
            num_updates=args.num_updates,
            num_actors=SMALL_NUM_ACTORS if args.small_run else NUM_ACTORS,
            num_envs_per_actor=(
                SMALL_NUM_ENVS_PER_ACTOR if args.small_run else NUM_ENVS_PER_ACTOR
            ),
            config_params=ATARI_CONFIG if args.atari else CARTPOLE_CONFIG,
        )
    else:
        assert args.results_file is not None, "results_file is required for plotting"
        plot_results(args.results_file)
