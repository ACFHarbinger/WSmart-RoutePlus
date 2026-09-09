"""Visualization utilities for training logs and statistics.

Separated from log_utils to isolate heavy plotting dependencies (matplotlib).

Attributes:
    plot_training_logs: Plots logged training history (mean and std dev per metric type).
    log_plot: Execution function for saving static plots.

Example:
    >>> from logic.src.utils.plotting.log_visualization import log_plot
    >>> log_plot(fig=my_fig, fig_filename="plot.png")
"""

import os
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

import logic.src.constants as udef


def plot_training_logs(
    loss_keys: List[str], xname: str, x_values: List[int], swapped_df: pd.DataFrame, output_dir: str, wandb_mode: str
) -> List[str]:
    """Plots logged training history (mean and std dev per metric type).

    Args:
        loss_keys: List of loss column names.
        xname: Name of the x-axis.
        x_values: Values of the x-axis.
        swapped_df: DataFrame containing training stats.
        output_dir: Directory to save the plots.
        wandb_mode: WandB mode (disabled, online, offline).

    Returns:
        List[str]: List of paths to the generated plots.
    """
    fig_paths: List[str] = []
    for l_id, l_key in enumerate(loss_keys):
        mean_loss: np.ndarray = swapped_df[f"mean_{l_key}"].to_numpy()  # pyrefly: ignore [bad-assignment]
        std_loss: np.ndarray = swapped_df[f"std_{l_key}"].to_numpy()  # pyrefly: ignore [bad-assignment]
        if np.all(mean_loss == std_loss):
            continue
        max_loss: np.ndarray = swapped_df[f"max_{l_key}"].to_numpy()  # pyrefly: ignore [bad-assignment]
        min_loss: np.ndarray = swapped_df[f"min_{l_key}"].to_numpy()  # pyrefly: ignore [bad-assignment]
        lower_bound: np.ndarray = np.maximum(mean_loss - std_loss, min_loss)
        upper_bound: np.ndarray = np.minimum(mean_loss + std_loss, max_loss)

        fig: Any = plt.figure(
            l_id,
            figsize=(20, 10),
            facecolor="white",
            edgecolor="black",
            layout="constrained",
        )
        label: str = l_key if l_key in udef.LOSS_KEYS else f"{l_key}_cost"
        ax: Any = fig.add_subplot(1, 1, 1)
        ax.plot(x_values, mean_loss, label=f"{label} μ", linewidth=2)
        ax.fill_between(x_values, lower_bound, upper_bound, alpha=0.3, label=f"{label} clip(μ ± σ)")
        ax.scatter(x_values, mean_loss, color="red", s=50, zorder=5)
        ax.set_xlabel(xname)
        ax.set_ylabel(label)
        ax.set_title(f"{label} per {xname}")
        ax.legend()
        ax.grid(True, linestyle="-", alpha=0.9)
        fig_path: str = os.path.join(output_dir, f"{label}.png")
        fig.savefig(fig_path)
        plt.close(fig)
        if wandb_mode != "disabled":
            wandb.log({label: wandb.Image(fig_path)})
        fig_paths.append(fig_path)
    return fig_paths


def log_plot(visualize: bool = False, **kwargs: Any) -> None:
    """Execution function for saving static plots.

    Args:
        visualize: Whether to show the plot. Defaults to False.
        kwargs: Must contain 'fig' (matplotlib Figure) and 'fig_filename' (str).
    """
    kwargs["fig"].savefig(kwargs["fig_filename"], bbox_inches="tight")
    if visualize:
        plt.show()
    plt.close(kwargs["fig"])
