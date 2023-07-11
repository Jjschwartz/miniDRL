"""Benchmarks FLOPS for numpy Matrix Multiplication.

This is important since it is the most common operation used by the rollout workers
which use the CPU for policy inference.

"""
import os
import time

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1


import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402


NUM_ROUNDS = 1000
SIZES = [2**i for i in range(1, 12)]


def run(save_file: str):
    with open(save_file, "w") as f:
        f.write("size,time,time_std,flops,flops_std\n")

    for size in SIZES:
        print(f"Benchmarking size {size}")
        times = []
        for _ in range(NUM_ROUNDS):
            a = np.random.randn(size, size)
            b = np.random.randn(size, size)
            start = time.time()
            np.matmul(a, b)
            times.append(time.time() - start)

        times = np.array(times)
        op_gflops = 2 * size**3 / 1e9
        gflops = op_gflops / times
        with open(save_file, "a") as f:
            f.write(
                "{},{},{},{},{}\n".format(
                    size,
                    np.mean(times),
                    np.std(times),
                    np.mean(gflops),
                    np.std(gflops),
                )
            )
        print(f"Mean time: {np.mean(times)} | Mean gflops: {np.mean(gflops)}")


def plot(save_file: str):
    df = pd.read_csv(save_file)
    print("Columns:", str(df.columns.tolist()))

    sns.set_theme()
    df.sort_values(by=["size"], inplace=True)

    fig, axs = plt.subplots(nrows=1, ncols=2)

    axs[0].plot(df["size"], df["gflops"], label="GFLOPS")
    axs[0].fill_between(
        df["size"],
        df["gflops"] - df["gflops_std"],
        df["gflops"] + df["gflops_std"],
        alpha=0.5,
    )
    axs[0].set_xscale("log", base=2)
    axs[0].get_xaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)))
    )
    # axs[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axs[0].set_yscale("log", base=10)
    axs[0].set_xlabel("Size")
    axs[0].set_ylabel("GFLOPS")
    axs[0].set_title("GFLOPS vs Size")

    axs[1].plot(df["size"], df["time"], label="Time")
    axs[1].fill_between(
        df["size"], df["time"] - df["time_std"], df["time"] + df["time_std"], alpha=0.5
    )
    axs[1].set_xscale("log", base=2)
    axs[1].get_xaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)))
    )
    axs[1].set_yscale("log", base=10)
    axs[1].set_xlabel("Size")
    axs[1].set_ylabel("Time (s)")
    axs[1].set_title("Time vs Size")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("action", type=str, choices=["run", "plot"])
    parser.add_argument("--save-file", type=str, default="np_gemm.csv")
    args = parser.parse_args()
    if args.action == "run":
        run(args.save_file)
    elif args.action == "plot":
        plot(args.save_file)
    else:
        raise ValueError("Invalid action")
