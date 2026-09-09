"""Loss landscape visualization tools for routing models.

This module provides tools for computing and visualizing the loss surface
of neural routing models. It supports both Imitation Learning landscapes
(negative log-likelihood) and Reinforcement Learning landscapes (greedy cost).
It uses random plane perturbations to generate 2D and 3D surface plots.

Attributes:
    plot_loss_landscape: Primary entry point for landscape generation.
    ImitationMetric: Adapter for imitation loss computation.
    RLMetric: Adapter for reinforcement learning cost computation.

Example:
    >>> from logic.src.utils.plotting import plot_loss_landscape
    >>> plot_loss_landscape(model, cfg, "output/landscape/")
"""

import os
from typing import Any, Union

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import loss_landscapes
import matplotlib.pyplot as plt
from loss_landscapes.metrics import Metric
from omegaconf import DictConfig

from logic.src.configs import Config

from .helpers import MyModelWrapper, get_batch


def imitation_loss_fn(m, x_batch, pi_target, cost_weights=None):
    """Computes imitation loss (log likelihood of target) for loss landscape.

    Args:
        m: The model or model wrapper instance.
        x_batch: Input batch of graph data.
        pi_target: Ground truth action distribution or sequence.
        cost_weights: Optional scaling factors for loss. Defaults to None.

    Returns:
        float: The scalar imitation loss value.
    """
    model_to_call = m.modules[0] if hasattr(m, "modules") else m
    if hasattr(model_to_call, "model"):
        model_to_call = model_to_call.model
    model_to_call.eval()

    # Ensure cost_weights are on the same device as the model
    dev = next(model_to_call.parameters()).device
    if cost_weights is not None:
        cost_weights = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in cost_weights.items()}
    elif hasattr(model_to_call, "cost_weights"):
        cost_weights = {
            k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in model_to_call.cost_weights.items()
        }

    with torch.no_grad():
        res = model_to_call(x_batch, cost_weights=cost_weights, return_pi=False, expert_pi=pi_target)
        log_likelihood = res[1]
    return -log_likelihood.mean().item()


def rl_loss_fn(m, x_batch, cost_weights=None):
    """Computes RL loss (greedy cost) for loss landscape.

    Args:
        m: The model or model wrapper instance.
        x_batch: Input batch of graph data.
        cost_weights: Optional scaling factors for cost components. Defaults to None.

    Returns:
        float: The mean greedy cost across the batch.
    """
    model_to_call = m.modules[0] if hasattr(m, "modules") else m
    if hasattr(model_to_call, "model"):
        model_to_call = model_to_call.model
    model_to_call.eval()

    # Ensure cost_weights are on the same device as the model
    dev = next(model_to_call.parameters()).device
    if cost_weights is not None:
        cost_weights = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in cost_weights.items()}
    elif hasattr(model_to_call, "cost_weights"):
        cost_weights = {
            k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in model_to_call.cost_weights.items()
        }

    with torch.no_grad():
        model_to_call.set_strategy("greedy")
        cost, _, _, _, _ = model_to_call(x_batch, cost_weights=cost_weights, return_pi=False)
    return cost.float().mean().item()


class ImitationMetric(Metric):
    """Metric class for imitation loss.

    Attributes:
        x_batch: The stable batch used for perturbations.
        pi_target: The stable target sequence.
        cost_weights: Optional cost scaling configuration.
    """

    def __init__(self, x_batch, pi_target, cost_weights=None):
        """Initializes the imitation metric.

        Args:
            x_batch: The input batch data for the perturbation grid.
            pi_target: The target policy sequence for the imitation loss.
            cost_weights: Optional scaling factors for cost components. Defaults to None.
        """
        super().__init__()
        self.x_batch = x_batch
        self.pi_target = pi_target
        self.cost_weights = cost_weights

    def __call__(self, model_wrapper):
        """Computes imitation loss for the current model state.

        Args:
            model_wrapper: The wrapped model to evaluate.

        Returns:
            float: The computed imitation loss.
        """
        return imitation_loss_fn(model_wrapper, self.x_batch, self.pi_target, cost_weights=self.cost_weights)


class RLMetric(Metric):
    """Metric class for RL cost.

    Attributes:
        x_batch: The stable batch used for perturbations.
        cost_weights: Optional cost scaling configuration.
    """

    def __init__(self, x_batch, cost_weights=None):
        """Initializes the RL metric.

        Args:
            x_batch: The input batch data for the perturbation grid.
            cost_weights: Optional scaling factors for cost components. Defaults to None.
        """
        super().__init__()
        self.x_batch = x_batch
        self.cost_weights = cost_weights

    def __call__(self, model_wrapper):
        """Computes RL cost for the current model state.

        Args:
            model_wrapper: The wrapped model to evaluate.

        Returns:
            float: The computed RL cost.
        """
        return rl_loss_fn(model_wrapper, self.x_batch, cost_weights=self.cost_weights)


def plot_loss_landscape(
    model: Any,
    cfg: Union[Config, DictConfig],
    output_dir: str,
    epoch: int = 0,
    size: int = 50,
    batch_size: int = 16,
    resolution: int = 10,
    span: float = 1.0,
) -> None:
    """Computes and plots 2D and 3D loss landscapes for Imitation Loss and RL Cost.

    Args:
        model: The neural model to perturb.
        cfg: Root Hydra configuration.
        output_dir: Directory where the generated images will be saved.
        epoch: Current training epoch index. Defaults to 0.
        size: Number of nodes in the graph. Defaults to 50.
        batch_size: Number of instances per perturbation. Defaults to 16.
        resolution: Grid resolution of the landscape. Defaults to 10.
        span: Range of perturbation along the random directions. Defaults to 1.0.

    Returns:
        None
    """
    from logic.src.policies.vector.local_search import vectorized_two_opt

    print("Computing Loss Landscape...")
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")

    temporal_horizon = 0
    if not isinstance(cfg, dict):
        train = getattr(cfg, "train", None)
        if train is not None:
            policy = getattr(train, "policy", None)
            if policy is not None:
                model_cfg = getattr(policy, "model", None)
                if model_cfg is not None:
                    temporal_horizon = getattr(model_cfg, "temporal_horizon", 0)
        if temporal_horizon == 0:
            model_cfg = getattr(cfg, "model", None)
            if model_cfg is not None:
                temporal_horizon = getattr(model_cfg, "temporal_horizon", 0)
    else:
        train = cfg.get("train", {})
        if isinstance(train, dict):
            temporal_horizon = (
                train.get("policy", {})
                .get("model", {})
                .get("temporal_horizon", cfg.get("model", {}).get("temporal_horizon", 0))
            )
        else:
            # Check if getattr is needed for OmegaConf/object types in dict-like wrapper
            policy = getattr(train, "policy", None)
            if policy is not None:
                model_cfg = getattr(policy, "model", None)
                temporal_horizon = getattr(model_cfg, "temporal_horizon", 0) if model_cfg is not None else 0
            else:
                temporal_horizon = cfg.get("model", {}).get("temporal_horizon", 0)

    # Generate random batch for landscape
    x_batch = get_batch(
        device,
        size=size,
        batch_size=batch_size,
        temporal_horizon=temporal_horizon,
    )

    print("Generating expert targets for landscape...")
    model.set_strategy("greedy")
    with torch.no_grad():
        _, _, _, pi, _ = model(x_batch, return_pi=True)
        x_dist = x_batch["dist"]
        if x_dist.dim() == 2:
            x_dist = x_dist.unsqueeze(0)
        if x_dist.size(0) == 1:
            x_dist = x_dist.expand(pi.size(0), -1, -1)
        pi_with_depot = torch.cat([torch.zeros((pi.size(0), 1), dtype=torch.long, device=device), pi], dim=1)

        pi_opt = vectorized_two_opt(pi_with_depot, x_dist, max_iterations=100)
        pi_target = pi_opt[:, 1:]

    wrapped_model = MyModelWrapper(model)

    model_device = next(model.parameters()).device

    def move_dict_to_device(d, dev):
        """Recursively move dictionary values to the specified device."""
        return {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in d.items()}

    x_batch = move_dict_to_device(x_batch, model_device)
    pi_target = pi_target.to(model_device)

    # Imitation
    print(f"Computing Imitation Landscape on {model_device}...")

    orig_cost_weights = getattr(model, "cost_weights", None)

    # Move data to CPU for landscape computation consistency if needed by wrapper
    m_dev = torch.device("cpu")
    x_m = move_dict_to_device(x_batch, m_dev)
    pi_m = pi_target.to(m_dev)

    imitation_metric = ImitationMetric(x_m, pi_m, cost_weights=orig_cost_weights)
    try:
        data = loss_landscapes.random_plane(
            wrapped_model,
            imitation_metric,
            distance=span,
            steps=resolution,
            deepcopy_model=True,
        )

        plt.figure()
        plt.contour(data, levels=50)
        plt.title(f"Imitation Loss Landscape (Epoch {epoch})")
        plt.savefig(os.path.join(output_dir, f"landscape_imitation_ep{epoch}.png"))
        plt.close()

        plt.close()

        plt.figure()
        ax = plt.axes(projection="3d")
        X, Y = np.meshgrid(np.arange(resolution), np.arange(resolution))
        ax.plot_surface(X, Y, np.array(data), cmap="viridis")  # type: ignore[attr-defined]
        plt.title(f"Imitation Loss Surface (Epoch {epoch})")
        plt.savefig(os.path.join(output_dir, f"surface_imitation_ep{epoch}.png"))
        plt.close()
    except Exception as e:
        print(f"Error computing imitation landscape: {e}")

    # RL
    print(f"Computing RL Cost Landscape on {model_device}...")

    rl_metric = RLMetric(x_m, cost_weights=orig_cost_weights)
    try:
        data = loss_landscapes.random_plane(
            wrapped_model,
            rl_metric,
            distance=span,
            steps=resolution,
            deepcopy_model=True,
        )

        plt.figure()
        plt.contour(data, levels=50)
        plt.title(f"RL Cost Landscape (Epoch {epoch})")
        plt.savefig(os.path.join(output_dir, f"landscape_rl_ep{epoch}.png"))
        plt.close()

        plt.close()

        plt.figure()
        ax = plt.axes(projection="3d")
        X, Y = np.meshgrid(np.arange(resolution), np.arange(resolution))
        ax.plot_surface(X, Y, np.array(data), cmap="magma")  # type: ignore[attr-defined]
        plt.title(f"RL Cost Surface (Epoch {epoch})")
        plt.savefig(os.path.join(output_dir, f"surface_rl_ep{epoch}.png"))
        plt.close()
    except Exception as e:
        print(f"Error computing RL landscape: {e}")
