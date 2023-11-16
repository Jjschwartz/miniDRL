"""Compares between different schemes for setting actor exploration epsilon in R2D2."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def apex_actor_epsilons(
    num_actors: int, base_epsilon: float = 0.4, alpha: float = 7
) -> np.ndarray:
    """Get actor exploration epsilons used in Ape-X (and R2D2) papers."""
    return np.power(
        base_epsilon, 1 + alpha * np.arange(num_actors) / max(1, num_actors - 1)
    )


def tweaked_apex_actor_epsilons(
    num_actors: int, base_epsilon: float = 0.4, alpha: float = 7
) -> np.ndarray:
    """Get actor exploration epsilons used in Ape-X (and R2D2) papers."""
    if num_actors < 4:
        epsilons = np.power(base_epsilon, 1 + alpha * np.arange(16) / (16 - 1))
        selected_epsilons = [
            epsilons[(i + 1) * 16 // (num_actors + 3)] for i in range(num_actors)
        ]
        return np.array(selected_epsilons)
    return np.power(base_epsilon, 1 + alpha * np.arange(num_actors) / (num_actors - 1))


def linear_actor_epsilons(
    num_actors: int, base_epsilon: float = 0.4, final_epsilon: float = 0.01
) -> np.ndarray:
    """Get actor exploration epsilons used in R2D2 paper."""
    return np.linspace(base_epsilon, final_epsilon, num_actors)


def main():
    """Plot actor exploration epsilons."""
    num_actors_list = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256]
    base_epsilon = 0.4
    final_epsilon = 5e-4
    alpha = 7

    sns.set_theme()
    fig, axs = plt.subplots(
        nrows=1,
        ncols=len(num_actors_list),
        sharey=True,
        figsize=(3 * len(num_actors_list), 4),
    )
    for num_actors, ax in zip(num_actors_list, axs):
        apex_epsilons = apex_actor_epsilons(num_actors, base_epsilon, alpha)
        tweaked_apex_epsilons = tweaked_apex_actor_epsilons(
            num_actors, base_epsilon, alpha
        )
        linear_epsilons = linear_actor_epsilons(num_actors, base_epsilon, final_epsilon)

        if num_actors == 1:
            ax.bar(
                [0, 1, 2],
                [
                    linear_epsilons[0],
                    tweaked_apex_epsilons[0],
                    apex_epsilons[0],
                ],
                tick_label=[
                    "Lin",
                    "AX*",
                    "AX",
                ],
            )
        else:
            ax.plot(linear_epsilons, label="Linear")
            ax.plot(tweaked_apex_epsilons, label="Tweaked Ape-X")
            ax.plot(apex_epsilons, label="Ape-X", linestyle="--")

        ax.set_xlabel("Actor Index")
        ax.set_title(f"N={num_actors}")

    axs[0].set_ylabel("Actor Epsilon")
    lines, labels = axs[-1].get_legend_handles_labels()
    fig.legend(lines, labels, ncols=3, loc="lower center")
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    main()
