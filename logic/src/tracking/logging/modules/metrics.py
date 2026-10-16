"""Training metrics logging and synchronization.

This module provides functions for recording real-time training progress across
multiple backends, including standard console output, TensorBoard, and
Weights & Biases (WandB). It handles loss aggregation, gradient norm
tracking, and epoch-level statistics generation.

Attributes:
    log_values: Per-step metric logging to all active backends.
    log_epoch: Summarizes loss metrics at the end of an epoch.
    get_loss_stats: Computes descriptive statistics for a loss distribution.

Example:
    >>> from logic.src.tracking.logging.modules import metrics
    >>> metrics.log_values(cost, norms, epoch, batch_id, step, l_dict, tb, cfg)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


from typing import Any, Dict, List, Optional, Tuple, Union

from omegaconf import DictConfig

import logic.src.constants as udef
from logic.src.configs import Config


def log_values(
    cost: torch.Tensor,
    grad_norms: Tuple[torch.Tensor, ...],
    epoch: int,
    batch_id: int,
    step: int,
    l_dict: Dict[str, torch.Tensor],
    tb_logger: Any,
    cfg: Union[Config, DictConfig],
) -> None:
    """Logs training metrics to console, TensorBoard, and WandB.

    Args:
        cost: The batch cost tensor.
        grad_norms: Tuple of (raw_norms, clipped_norms).
        epoch: Current training epoch index.
        batch_id: Index of the current batch within the epoch.
        step: Global training step index.
        l_dict: Dictionary of auxiliary losses (e.g. nll, reinforce).
        tb_logger: TensorBoard SummaryWriter instance.
        cfg: Root Hydra configuration.
    """
    import wandb

    avg_cost: float = cost.mean().item()
    norms, norms_clipped = grad_norms
    train = cfg.train
    rl = cfg.rl

    print(
        "{}: {}, train_batch_id: {}, avg_cost: {}".format(
            "day" if train.train_time else "epoch", epoch, batch_id, avg_cost
        )
    )
    print("grad_norm: {}, clipped: {}".format(norms[0], norms_clipped[0]))

    no_tensorboard: bool = getattr(rl, "no_tensorboard", True)
    wandb_mode: str = getattr(rl, "wandb_mode", "disabled")
    baseline: Optional[str] = getattr(rl, "baseline", None)

    if not no_tensorboard:
        tb_logger.add_scalar("avg_cost", avg_cost, step)
        tb_logger.add_scalar("actor_loss", l_dict["reinforce_loss"].mean().item(), step)
        tb_logger.add_scalar("nll", l_dict["nll"].mean().item(), step)
        tb_logger.add_scalar("grad_norm", norms[0].item(), step)
        tb_logger.add_scalar("grad_norm_clipped", norms_clipped[0], step)
        if "imitation_loss" in l_dict:
            tb_logger.add_scalar("imitation_loss", l_dict["imitation_loss"].item(), step)
        if baseline == "critic":
            tb_logger.add_scalar("critic_loss", l_dict["baseline_loss"].item(), step)
            tb_logger.add_scalar("critic_grad_norm", norms[1].item(), step)
            tb_logger.add_scalar("critic_grad_norm_clipped", norms_clipped[1].item(), step)

    if wandb_mode != "disabled":
        wandb_data = {
            "avg_cost": avg_cost,
            "actor_loss": l_dict["reinforce_loss"].mean().item(),
            "nll": l_dict["nll"].mean().item(),
            "grad_norm": norms[0].item(),
            "grad_norm_clipped": norms_clipped[0],
        }
        if "imitation_loss" in l_dict:
            wandb_data["imitation_loss"] = l_dict["imitation_loss"].item()
        wandb.log(wandb_data)
        if baseline == "critic":
            wandb.log(
                {
                    "critic_loss": l_dict["baseline_loss"].item(),
                    "critic_grad_norm": norms[1].item(),
                    "critic_grad_norm_clipped": norms_clipped[1].item(),
                }
            )

    if "imitation_loss" in l_dict and l_dict["imitation_loss"].item() != 0:
        print(f"imitation_loss: {l_dict['imitation_loss'].item():.6f}")


def log_epoch(
    x_tup: Tuple[str, int],
    loss_keys: List[str],
    epoch_loss: Dict[str, List[torch.Tensor]],
    cfg: Union[Config, DictConfig],
) -> None:
    """Logs summary statistics for a completed epoch.

    Args:
        x_tup: Tuple of (unit_name, index) e.g., ("epoch", 10).
        loss_keys: List of keys to summarize from the epoch_loss dict.
        epoch_loss: Map of keys to lists of per-step loss tensors.
        cfg: Root Hydra configuration.
    """
    import torch
    import wandb

    wandb_mode: str = getattr(cfg.rl, "wandb_mode", "disabled")

    log_str: str = f"Finished {x_tup[0]} {x_tup[1]} log:"
    for _id, key in enumerate(loss_keys):
        if not epoch_loss.get(key):
            continue
        lname: str = key if key in udef.LOSS_KEYS else f"{key}_cost"
        try:
            lmean: float = torch.cat(epoch_loss[key]).float().mean().item()
        except Exception:
            lmean = 0.0

        log_str += f" {lname}: {lmean:.4f}"
        if wandb_mode != "disabled":
            wandb.log({x_tup[0]: x_tup[1], lname: lmean}, commit=(key == loss_keys[-1]))
    print(log_str)


def get_loss_stats(epoch_loss: Dict[str, List[torch.Tensor]]) -> List[float]:
    """Computes mean, std, min, and max for each loss key.

    Args:
        epoch_loss: Map of keys to lists of per-step loss tensors.

    Returns:
        List[float]: Flattened list of [mean, std, min, max] for each key.
    """
    import torch

    loss_stats: List[float] = []
    for key in epoch_loss:
        loss_tensor: torch.Tensor = torch.cat(epoch_loss[key]).float()
        loss_tmp: List[float] = [
            torch.mean(loss_tensor).item(),
            torch.std(loss_tensor).item(),
            torch.min(loss_tensor).item(),
            torch.max(loss_tensor).item(),
        ]
        loss_stats.extend(loss_tmp)
    return loss_stats
