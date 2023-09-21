"""Script for generating SPS plots for PPO docs."""
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_fixed_batch_size(
    df: pd.DataFrame, batch_size: int, y_keys: list, output_dir: str
):
    """Plot the results for a fixed batch size."""
    df = df[(df["batch_size"] == batch_size)]
    df.sort_values(by=["num_workers"], inplace=True, ascending=True)
    for y, yerr in y_keys:
        fig, ax = plt.subplots(nrows=1, ncols=1)

        if yerr:
            ax.errorbar(
                x=df["num_workers"],
                y=df[y],
                yerr=df[yerr],
            )
        else:
            ax.plot(
                df["num_workers"],
                df[y],
            )

        # ax.set_yscale("log", base=2)
        ax.set_ylabel(f"{y}")
        ax.set_xlabel("Num Workers")

        # fig.tight_layout(rect=(0, 0, 0.8, 1))
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, f"fbs_{batch_size}_{y}.png"),
            bbox_inches="tight",
        )


def plot_by_num_envs(df: pd.DataFrame, y_keys: list, output_dir: str):
    """Plot results by number of environments for single worker."""
    df = df[(df["num_workers"] == 1)]
    df.sort_values(by=["num_envs_per_worker"], inplace=True, ascending=True)
    num_envs_per_worker = df["num_envs_per_worker"].unique().tolist()
    num_envs_per_worker.sort()

    for y, yerr in y_keys:
        fig, ax = plt.subplots(nrows=1, ncols=1)
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
        ax.set_xlabel("Num Envs")

        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, f"bne_{y}.png"),
            bbox_inches="tight",
        )


def plot_fixed_num_envs_per_worker(df: pd.DataFrame, y_keys: list, output_dir: str):
    """Plot the results for a fixed number of envs per worker."""
    df = df[(df["num_envs_per_worker"] == 8)]
    df.sort_values(by=["num_workers"], inplace=True, ascending=True)
    for y, yerr in y_keys:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        if yerr:
            ax.errorbar(
                x=df["num_workers"],
                y=df[y],
                yerr=df[yerr],
            )
        else:
            ax.plot(
                df["num_workers"],
                df[y],
            )

        # ax.set_yscale("log", base=2)
        ax.set_ylabel(f"{y}")
        ax.set_xlabel("Num Workers")

        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, f"fne_{y}.png"),
            bbox_inches="tight",
        )


def plot(save_file: str):
    """Plot the results of experiments."""
    df = pd.read_csv(save_file)
    print(str(df.columns))

    output_dir = os.path.dirname(save_file)

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

    plot_fixed_batch_size(df, 16384, y_keys, output_dir)
    y_keys[-1] = ("batch_size", None)
    plot_by_num_envs(df, y_keys, output_dir)
    plot_fixed_num_envs_per_worker(df, y_keys, output_dir)

    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
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
        save_file_name = "benchmarking_results_atari.csv"
        save_file_arg = os.path.join(os.path.dirname(__file__), save_file_name)

    plot(save_file_arg)
