"""Weight monitoring and analysis hooks for PyTorch models.

This module provides utilities to track parameter evolution during training.
It includes hooks to monitor weight drift, distribution statistics, update
magnitudes from the optimizer, and hard constraints on weight norms. These
tools are vital for detecting mode collapse and initialization pathology in
routing models.

Attributes:
    add_weight_change_monitor_hook: Tracks parameter drift from start state.
    add_weight_distribution_monitor: Computes moment statistics of weights.
    add_weight_update_monitor_hook: Intercepts optimizer to track step sizes.
    add_weight_norm_constraint_hook: Enforces hard limits on weight norms.

Example:
    >>> from logic.src.tracking.hooks.weight_hooks import add_weight_change_monitor_hook
    >>> hook_data = add_weight_change_monitor_hook(model)
    >>> # After training loop:
    >>> drift = compute_weight_changes(model, hook_data)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import torch
from torch import nn


def add_weight_change_monitor_hook(
    model: nn.Module,
    layer_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Monitors weight changes during training to detect learning issues.

    Stores a snapshot of the initial state and enables future calculation of
    absolute and relative weight drift.

    Args:
        model: The target PyTorch model.
        layer_names: Optional specific parameter names to monitor. If None,
            monitors all model parameters. Defaults to None.

    Returns:
        Dict[str, Any]: Mapping containing 'initial_weights' and metadata.
    """
    initial_weights = {}

    for name, param in model.named_parameters():
        if layer_names is None or name in layer_names:
            initial_weights[name] = param.data.clone().detach()

    return {
        "initial_weights": initial_weights,
        "weight_changes": {},
    }


def compute_weight_changes(
    model: nn.Module,
    hook_data: Dict[str, Any],
    metric: str = "norm",
) -> Dict[str, float]:
    """Computes weight changes from the initial state captured in hook_data.

    Args:
        model: The current model state.
        hook_data: Data mapping created by add_weight_change_monitor_hook.
        metric: Magnitude metric ('norm', 'mean_abs', 'max_abs', 'relative').
            Defaults to "norm".

    Returns:
        Dict[str, float]: Mapping of parameter names to change magnitudes.

    Raises:
        ValueError: If an unsupported metric is specified.
    """
    initial_weights = hook_data["initial_weights"]
    changes = {}

    for name, param in model.named_parameters():
        if name in initial_weights:
            initial = initial_weights[name]
            current = param.data
            diff = current - initial

            if metric == "norm":
                changes[name] = diff.norm().item()
            elif metric == "mean_abs":
                changes[name] = diff.abs().mean().item()
            elif metric == "max_abs":
                changes[name] = diff.abs().max().item()
            elif metric == "relative":
                initial_norm = initial.norm().item()
                if initial_norm > 0:
                    changes[name] = (diff.norm() / initial_norm).item()
                else:
                    changes[name] = 0.0
            else:
                raise ValueError(f"Unknown metric: {metric}")

    return changes


def add_weight_distribution_monitor(
    model: nn.Module,
    layer_types: Tuple[type, ...] = (nn.Linear, nn.Conv2d),
) -> Dict[str, Any]:
    """Monitors weight distribution statistics (mean, std, min, max, sparsity).

    Useful for identifying layers with initialization pathology or that
    become saturated during training.

    Args:
        model: The target model.
        layer_types: Tuple of layer classes to analyze. Defaults to
            (nn.Linear, nn.Conv2d).

    Returns:
        Dict[str, Any]: Mapping containing current 'statistics' per layer.
    """
    statistics = {}

    for name, module in model.named_modules():
        if isinstance(module, layer_types) and hasattr(module, "weight"):
            weight = module.weight.data
            stats = {
                "mean": weight.mean().item(),
                "std": weight.std().item(),
                "min": weight.min().item(),
                "max": weight.max().item(),
                "norm": weight.norm().item(),
                "sparsity": (weight.abs() < 1e-6).sum().item() / weight.numel(),
            }
            statistics[name] = stats

    return {"statistics": statistics}


def add_weight_update_monitor_hook(
    optimizer: torch.optim.Optimizer,
    log_interval: int = 10,
) -> Dict[str, Any]:
    """Monitors raw parameter update magnitudes by wrapping the optimizer step.

    This tracks the delta applied to weights by the optimizer (including
    momentum and weight decay effects).

    Args:
        optimizer: The PyTorch optimizer to instrument.
        log_interval: Frequency of update logging (to minimize overhead).
            Defaults to 10.

    Returns:
        Dict[str, Any]: Mapping with history buffer and metadata.
    """
    update_history: Dict[str, List[float]] = defaultdict(list)
    step_count = [0]
    original_step = optimizer.step

    # Pre-map parameters to names once outside the loop
    param_names = {}
    for i, group in enumerate(optimizer.param_groups):
        for j, param in enumerate(group["params"]):
            param_names[id(param)] = f"group{i}_param{j}"

    def monitored_step(*args: Any, **kwargs: Any) -> Any:
        """Wrapped optimizer step that calculates weight deltas.

        Args:
            *args: Positional args for optimizer.step.
            **kwargs: Keyword args for optimizer.step.

        Returns:
            Any: The result of the original optimizer.step call.
        """
        step_count[0] += 1
        pre_weights: Dict[int, torch.Tensor] = {}

        # Log updates only on intervals to save memory/compute
        should_log = step_count[0] % log_interval == 0

        if should_log:
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        pre_weights[id(param)] = param.data.clone()

        result = original_step(*args, **kwargs)

        if should_log:
            for group in optimizer.param_groups:
                for param in group["params"]:
                    p_id = id(param)
                    if p_id in pre_weights:
                        # Normalize update magnitude by weight magnitude
                        update_norm = (param.data - pre_weights[p_id]).norm().item()
                        update_history[param_names[p_id]].append(update_norm)

        return result

    optimizer.step = monitored_step  # type: ignore[method-assign]

    return {
        "update_history": update_history,
        "step_count": step_count,
        "original_step": original_step,
    }


def restore_optimizer_step(optimizer: torch.optim.Optimizer, hook_data: Dict[str, Any]) -> None:
    """Restores the original optimizer.step() implementation.

    Args:
        optimizer: The PyTorch optimizer to restore.
        hook_data: The mapping returned by add_weight_update_monitor_hook.
    """
    if "original_step" in hook_data:
        optimizer.step = hook_data["original_step"]  # type: ignore[method-assign]


def add_weight_norm_constraint_hook(
    model: nn.Module,
    max_norm: float = 10.0,
    layer_names: Optional[List[str]] = None,
) -> List[Any]:
    """Registers hooks to enforce maximum weight norms during training.

    Directly scales parameter data if its norm exceeds the specified threshold
    during the backward pass.

    Args:
        model: The target model.
        max_norm: Maximum allowed L2 weight norm per layer. Defaults to 10.0.
        layer_names: Optional layers to restrict. If None, restricts all trainable.
            Defaults to None.

    Returns:
        List[Any]: List of hook handles.
    """
    handles: List[Any] = []

    def norm_constraint_hook(target_param: torch.nn.Parameter, norm_threshold: float) -> Callable[[torch.Tensor], None]:
        """Creates a hook that enforces the norm constraint on a parameter.

        Args:
            target_param: The parameter to monitor.
            norm_threshold: The maximum allowed L2 norm.

        Returns:
            Callable: The registered constraint hook.
        """

        def hook(_grad: torch.Tensor) -> None:
            """Gradient hook that applies weight norm constraint in-place.

            Args:
                _grad: The incoming gradient (ignored).
            """
            with torch.no_grad():
                norm = target_param.data.norm()
                if norm > norm_threshold:
                    target_param.data *= norm_threshold / norm

        return hook

    for name, param in model.named_parameters():
        if param.requires_grad and (layer_names is None or name in layer_names):
            handle = param.register_hook(norm_constraint_hook(param, max_norm))
            handles.append(handle)

    return handles


def detect_weight_symmetry_breaking(
    model: nn.Module,
    threshold: float = 1e-4,
) -> Dict[str, bool]:
    """Verifies that weights have broken symmetry from their initialization.

    Args:
        model: The target model.
        threshold: Minimum standard deviation required to consider symmetry broken.
            Defaults to 1e-4.

    Returns:
        Dict[str, bool]: Mapping of layer names to their symmetry broken status.
    """
    symmetry_status = {}

    for name, module in model.named_modules():
        if hasattr(module, "weight"):
            weight = module.weight.data

            # Check if weights have sufficient variance
            std = cast(Any, weight).std().item()
            symmetry_status[name] = std > threshold

    return symmetry_status


def print_weight_summary(
    weight_changes: Dict[str, float],
    weight_stats: Dict[str, Dict[str, float]],
    top_k: int = 10,
) -> None:
    """Renders a diagnostic summary of weight evolution and statistics.

    Args:
        weight_changes: Drifts computed via compute_weight_changes.
        weight_stats: Moments recorded from add_weight_distribution_monitor.
        top_k: Number of layers with largest drift to display. Defaults to 10.
    """
    print(f"\n{'=' * 100}")
    print(f"{'Layer Name':<40} {'Change':>10} {'Mean':>10} {'Std':>10} {'Sparsity':>10}")
    print(f"{'=' * 100}")

    # Sort by change magnitude
    sorted_layers = sorted(weight_changes.keys(), key=lambda x: weight_changes[x], reverse=True)

    for name in sorted_layers[:top_k]:
        change = weight_changes.get(name, 0.0)
        stats = weight_stats.get(name, {})

        mean = stats.get("mean", 0.0)
        std = stats.get("std", 0.0)
        sparsity = stats.get("sparsity", 0.0)

        print(f"{name:<40} {change:>10.6f} {mean:>10.4f} {std:>10.4f} {sparsity:>9.2%}")

    print(f"{'=' * 100}\n")


def analyze_weight_updates(
    update_history: Dict[str, List[float]],
    window_size: int = 10,
) -> Dict[str, Dict[str, float]]:
    """Analyzes raw update magnitudes to detect training stagnation or explosion.

    Args:
        update_history: Buffer of update magnitudes from the step hook.
        window_size: Rolling window size for trend analysis. Defaults to 10.

    Returns:
        Dict[str, Dict[str, float]]: Mapping of parameter names to diagnostic flags.
    """
    analysis = {}

    for name, updates in update_history.items():
        if len(updates) < window_size:
            continue

        recent = updates[-window_size:]
        mean_update = sum(recent) / len(recent)
        variance = sum((x - mean_update) ** 2 for x in recent) / len(recent)

        # Detect potential issues
        is_stuck = mean_update < 1e-8  # Weights not changing
        is_exploding = mean_update > 1.0  # Large updates
        is_oscillating = variance > mean_update**2 * 10  # High variance

        analysis[name] = {
            "mean_update": mean_update,
            "variance": variance,
            "is_stuck": float(is_stuck),
            "is_exploding": float(is_exploding),
            "is_oscillating": float(is_oscillating),
        }

    return analysis
