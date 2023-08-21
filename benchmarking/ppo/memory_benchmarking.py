"""Script for plotting memory usage statistics for PPO."""
from io import StringIO

import pandas as pd
import matplotlib.pyplot as plt

TOTAL_CPU_MEM = 15744
TOTAL_GPU_MEM = 8000

RESULTS = """
    n,mode,learner_gpu,learner_cpu,worker_gpu,worker_cpu,sps,
    1,CUDA,2304,3126,1440,1402,603,
    1,CPU,2304,3565,110,2270,590,
    2,CUDA,2304,3124,778,1288,1252,
    2,CPU,2640,3275,110,1640,1237,
    4,CUDA,2304,3123,448,1229,2298,
    4,CPU,2304,3237,110,1447,2310,
"""

df = pd.read_csv(StringIO(RESULTS.strip()), sep=",")
df["total_gpu"] = df["learner_gpu"] + df["n"] * df["worker_gpu"]
df["total_cpu"] = df["learner_cpu"] + df["n"] * df["worker_cpu"]
df["total_mem"] = df["total_gpu"] + df["total_cpu"]

# Plot memory usage.
fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(16, 6))
for process, ax in zip(["learner", "worker", "total"], axs):
    for mode in ["CUDA", "CPU"]:
        for mem_type in ["mem", "cpu", "gpu"] if process == "total" else ["cpu", "gpu"]:
            ax.plot(
                df[df["mode"] == mode]["n"],
                df[df["mode"] == mode][f"{process}_{mem_type}"],
                label=f"{mode} - {mem_type}",
            )
    ax.set_xlabel("Number of workers")
    ax.set_ylabel("Memory usage (MiB)")
    if process == "worker":
        ax.set_title("PPO per worker memory usage")
    else:
        ax.set_title(f"PPO {process} memory usage")
    ax.set_xscale("log", base=2)
    ax.set_xticks(df["n"])
    ax.set_xticklabels(df["n"])
    ax.legend()


# Plot speed.
ax = axs[-1]
for mode in ["CUDA", "CPU"]:
    ax.plot(df[df["mode"] == mode]["n"], df[df["mode"] == mode]["sps"], label=mode)
ax.set_xlabel("Number of workers")
ax.set_ylabel("Steps per second")
ax.set_title("PPO speed")
ax.set_xscale("log", base=2)
ax.set_xticks(df["n"])
ax.set_xticklabels(df["n"])
ax.legend()

fig.tight_layout()
plt.show()
