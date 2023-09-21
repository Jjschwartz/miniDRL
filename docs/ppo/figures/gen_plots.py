"""Script for generating SPS plots for PPO docs."""
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_fixed_batch_size(output_dir: str):
    """Plot the results for a fixed batch size."""
    save_file = os.path.join(
        os.path.dirname(__file__), "benchmarking_results_atari_fbs.csv"
    )
    df = pd.read_csv(save_file)

    y_keys = [
        ("experience_sps", "experience_sps_std"),
        ("learning_sps", "learning_sps_std"),
        ("sps", None),
        ("num_envs_per_worker", None),
    ]

    df.sort_values(by=["num_workers"], inplace=True, ascending=True)
    num_workers = df["num_workers"].unique().tolist()
    num_workers.sort()
    batch_sizes = df["batch_size"].unique().tolist()
    batch_sizes.sort()

    for y, yerr in y_keys:
        fig, ax = plt.subplots(nrows=1, ncols=1)

        for batch_size in batch_sizes:
            df_i = df[(df["batch_size"] == batch_size)]
            if yerr:
                ax.errorbar(
                    x=df_i["num_workers"],
                    y=df_i[y],
                    yerr=df_i[yerr],
                    label=f"{batch_size}",
                )
            else:
                ax.plot(
                    df_i["num_workers"],
                    df_i[y],
                    label=f"{batch_size}",
                )

        # ax.set_yscale("log", base=2)
        ax.set_ylabel(f"{y}")
        ax.set_xlabel("Num Workers")
        ax.set_xticks(num_workers)
        ax.legend(title="Batch Size")

        fig.savefig(
            os.path.join(output_dir, f"fbs_{y}.svg"),
            bbox_inches="tight",
        )


def plot_by_num_envs(output_dir: str):
    """Plot results by number of environments for single worker."""
    save_file = os.path.join(
        os.path.dirname(__file__), "benchmarking_results_atari_swne.csv"
    )
    df = pd.read_csv(save_file)

    y_keys = [
        ("experience_sps", "experience_sps_std"),
        ("learning_sps", "learning_sps_std"),
        ("sps", None),
        ("batch_size", None),
    ]

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
        ax.set_xticks(num_envs_per_worker)

        fig.savefig(
            os.path.join(output_dir, f"swne_{y}.svg"),
            bbox_inches="tight",
        )


def plot_fixed_num_envs_per_worker(output_dir: str):
    """Plot the results for a fixed number of envs per worker."""
    save_file = os.path.join(
        os.path.dirname(__file__), "benchmarking_results_atari_fne.csv"
    )
    df = pd.read_csv(save_file)

    y_keys = [
        ("experience_sps", "experience_sps_std"),
        ("learning_sps", "learning_sps_std"),
        ("sps", None),
        ("batch_size", None),
    ]

    df.sort_values(by=["num_workers"], inplace=True, ascending=True)
    num_workers = df["num_workers"].unique().tolist()
    num_workers.sort()
    num_envs_per_worker = df["num_envs_per_worker"].unique().tolist()
    num_envs_per_worker.sort()

    for y, yerr in y_keys:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        for ne in num_envs_per_worker:
            df_i = df[(df["num_envs_per_worker"] == ne)]
            if yerr:
                ax.errorbar(
                    x=df_i["num_workers"],
                    y=df_i[y],
                    yerr=df_i[yerr],
                    label=f"{ne}",
                )
            else:
                ax.plot(
                    df_i["num_workers"],
                    df_i[y],
                    label=f"{ne}",
                )

        # ax.set_yscale("log", base=2)
        ax.set_ylabel(f"{y}")
        ax.set_xlabel("Num Workers")
        ax.set_xticks(num_workers)
        ax.legend(title="Num Envs")

        fig.savefig(
            os.path.join(output_dir, f"fne_{y}.svg"),
            bbox_inches="tight",
        )


def plot():
    """Plot the results of experiments."""
    output_dir = os.path.dirname(__file__)

    sns.set_theme()

    plot_fixed_batch_size(output_dir)
    plot_by_num_envs(output_dir)
    plot_fixed_num_envs_per_worker(output_dir)

    # plt.show()


if __name__ == "__main__":
    plot()
