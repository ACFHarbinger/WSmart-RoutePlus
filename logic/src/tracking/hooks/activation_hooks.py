"""Activation capturing and monitoring hooks for PyTorch models.

This module provides utilities to intercept and record layer activations during
the forward pass. These hooks are essential for tracking representation drift,
detecting dead neurons, and analyzing information flow through neural routing
architectures.

Attributes:
    add_activation_capture_hooks: Utility to record raw layer outputs.
    add_dead_neuron_detector_hook: Utility to count zero-activation units.
    add_activation_statistics_hook: Utility to compute running stats (mean/std).
    add_activation_sparsity_hook: Utility to measure activation sparse rates.

Example:
    >>> from logic.src.tracking.hooks.activation_hooks import add_activation_statistics_hook
    >>> hook_data = add_activation_statistics_hook(model)
    >>> # After forward pass:
    >>> stats = compute_activation_statistics(hook_data['statistics'])
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn


def add_activation_capture_hooks(
    model: nn.Module,
    layer_types: Optional[Tuple[type, ...]] = None,
    layer_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Registers forward hooks to capture raw activations from specified layers.

    Args:
        model: The PyTorch model to instrument.
        layer_types: Optional tuple of layer classes to monitor (e.g., nn.Linear).
            Defaults to None.
        layer_names: Optional list of explicit layer names to monitor.
            Defaults to None.

    Returns:
        Dict[str, Any]: Mapping containing 'activations' (results) and 'handles'.
    """
    activations: Dict[str, Optional[torch.Tensor]] = {}
    handles: List[Any] = []

    def capture_hook(name: str) -> Callable:
        """Creates a closure-based hook for capturing specific layer output.

        Args:
            name: Human-readable name for the layer.

        Returns:
            Callable: The registered forward hook.
        """

        def hook(module: nn.Module, input_tensors: Tuple[torch.Tensor, ...], output: Any) -> None:
            """Internal forward hook that detaches and records activations.

            Args:
                module: The layer generating the activation.
                input_tensors: Tensors entering the layer.
                output: Resulting activation tensor.
            """
            # Store activation (detached from graph to save memory)
            if isinstance(output, torch.Tensor):
                activations[name] = output.detach()
            elif isinstance(output, (tuple, list)):
                # For layers returning multiple outputs, store first one
                activations[name] = output[0].detach() if len(output) > 0 else None

        return hook

    # Register hooks
    for name, module in model.named_modules():
        should_hook = False

        # Check if this layer should be hooked
        if (
            layer_names is not None
            and name in layer_names
            or layer_types is not None
            and isinstance(module, layer_types)
            or layer_types is None
            and layer_names is None
        ):
            should_hook = True

        if should_hook and name:  # Skip empty name (root module)
            handle = module.register_forward_hook(capture_hook(name))
            handles.append(handle)

    return {"activations": activations, "handles": handles}


def add_dead_neuron_detector_hook(
    model: nn.Module,
    threshold: float = 1e-6,
    layer_types: Tuple[type, ...] = (nn.Linear, nn.Conv2d),
) -> Dict[str, Any]:
    """Detects neurons that fail to activate (remain below threshold).

    Args:
        model: The PyTorch model to instrument.
        threshold: Absolute activation value below which a neuron is 'dead'.
            Defaults to 1e-6.
        layer_types: Target layer types for monitoring. Defaults to (nn.Linear, nn.Conv2d).

    Returns:
        Dict[str, Any]: Mapping containing 'dead_neurons' counts and 'handles'.
    """
    dead_neuron_counts: Dict[str, int] = defaultdict(int)
    activation_counts: Dict[str, int] = defaultdict(int)
    handles: List[Any] = []

    def dead_neuron_hook(name: str, threshold_val: float) -> Callable:
        """Creates a hook that identifies and counts dead units.

        Args:
            name: Unique identifier for the layer.
            threshold_val: The 'deadness' threshold.

        Returns:
            Callable: The registered check hook.
        """

        def hook(module: nn.Module, input_tensors: Tuple[torch.Tensor, ...], output: Any) -> None:
            """Hook that analyzes the batch max activation per neuron.

            Args:
                module: The layer generating the activation.
                input_tensors: Tensors entering the layer.
                output: Resulting activation tensor.
            """
            if isinstance(output, torch.Tensor):
                # Count neurons with max activation below threshold
                max_activation = output.abs().max(dim=0)[0]  # Max across batch
                if max_activation.dim() > 0:  # Handle different output shapes
                    dead_count = (max_activation < threshold_val).sum().item()
                    dead_neuron_counts[name] += int(dead_count)
                    activation_counts[name] += 1

        return hook

    for name, module in model.named_modules():
        if isinstance(module, layer_types) and name:
            handle = module.register_forward_hook(dead_neuron_hook(name, threshold))
            handles.append(handle)

    return {
        "dead_neurons": dict(dead_neuron_counts),
        "activation_counts": dict(activation_counts),
        "handles": handles,
    }


def add_activation_statistics_hook(
    model: nn.Module,
    layer_types: Tuple[type, ...] = (nn.Linear, nn.Conv2d, nn.MultiheadAttention),
) -> Dict[str, Any]:
    """Computes running statistics of activations across batches.

    Args:
        model: The PyTorch model to instrument.
        layer_types: Target layer types for monitoring. Defaults to
            (nn.Linear, nn.Conv2d, nn.MultiheadAttention).

    Returns:
        Dict[str, Any]: Mapping containing raw 'statistics' and 'handles'.
    """
    statistics: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"sum": 0.0, "sum_sq": 0.0, "min": float("inf"), "max": float("-inf"), "count": 0}
    )
    handles: List[Any] = []

    def stats_hook(name: str) -> Callable:
        """Creates a hook to accumulate statistical moments.

        Args:
            name: Unique identifier for the layer.

        Returns:
            Callable: The registered stats hook.
        """

        def hook(module: nn.Module, input_tensors: Tuple[torch.Tensor, ...], output: Any) -> None:
            """Hook that updates running sum, sum-of-squares, min, and max.

            Args:
                module: The layer generating the activation.
                input_tensors: Tensors entering the layer.
                output: Resulting activation tensor.
            """
            if isinstance(output, torch.Tensor):
                stats = statistics[name]
                stats["sum"] += output.sum().item()
                stats["sum_sq"] += (output**2).sum().item()
                stats["min"] = min(stats["min"], output.min().item())
                stats["max"] = max(stats["max"], output.max().item())
                stats["count"] += output.numel()

        return hook

    for name, module in model.named_modules():
        if isinstance(module, layer_types) and name:
            handle = module.register_forward_hook(stats_hook(name))
            handles.append(handle)

    return {"statistics": dict(statistics), "handles": handles}


def compute_activation_statistics(statistics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Calculates final mean and standard deviation from raw accumulated moments.

    Args:
        statistics: Raw stats mapping returned by add_activation_statistics_hook.

    Returns:
        Dict[str, Dict[str, float]]: Mapping of layer names to computed stats.
    """
    final_stats = {}
    for name, stats in statistics.items():
        count = int(stats["count"])
        if count > 0:
            mean = stats["sum"] / count
            variance = (stats["sum_sq"] / count) - (mean**2)
            std = variance**0.5 if variance > 0 else 0.0

            final_stats[name] = {
                "mean": mean,
                "std": std,
                "min": stats["min"],
                "max": stats["max"],
                "count": float(count),
            }
    return final_stats


def add_activation_sparsity_hook(
    model: nn.Module,
    threshold: float = 0.01,
    layer_types: Tuple[type, ...] = (nn.Linear, nn.Conv2d),
) -> Dict[str, Any]:
    """Measures activation sparsity by counting near-zero values.

    Args:
        model: The PyTorch model to instrument.
        threshold: Absolute value below which an activation is considered zero.
            Defaults to 0.01.
        layer_types: Target layer types for monitoring. Defaults to
            (nn.Linear, nn.Conv2d).

    Returns:
        Dict[str, Any]: Mapping containing 'sparsity' raw counts and 'handles'.
    """
    sparsity_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"zeros": 0, "total": 0})
    handles: List[Any] = []

    def sparsity_hook(name: str, threshold_val: float) -> Callable:
        """Creates a hook that counts batch-wise sparse activations.

        Args:
            name: Unique identifier for the layer.
            threshold_val: The sparsity cutoff.

        Returns:
            Callable: The registered sparsity hook.
        """

        def hook(module: nn.Module, input_tensors: Tuple[torch.Tensor, ...], output: Any) -> None:
            """Hook that updates cumulative zero and total element counts.

            Args:
                module: The layer generating the activation.
                input_tensors: Tensors entering the layer.
                output: Resulting activation tensor.
            """
            if isinstance(output, torch.Tensor):
                stats = sparsity_stats[name]
                stats["zeros"] += int((output.abs() < threshold_val).sum().item())
                stats["total"] += output.numel()

        return hook

    for name, module in model.named_modules():
        if isinstance(module, layer_types) and name:
            handle = module.register_forward_hook(sparsity_hook(name, threshold))
            handles.append(handle)

    return {"sparsity": dict(sparsity_stats), "handles": handles}


def compute_sparsity_percentages(sparsity_stats: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    """Computes final sparsity percentages from raw accumulated counts.

    Args:
        sparsity_stats: Raw sparsity counts from the sparsity hook.

    Returns:
        Dict[str, float]: Mapping of layer names to sparsity percentage (0.0-1.0).
    """
    return {
        name: (stats["zeros"] / stats["total"]) if stats["total"] > 0 else 0.0 for name, stats in sparsity_stats.items()
    }


def print_activation_summary(
    statistics: Dict[str, Dict[str, float]],
    sparsity: Optional[Dict[str, float]] = None,
    dead_neurons: Optional[Dict[str, int]] = None,
) -> None:
    """Renders a comprehensive diagnostic table of activation dynamics.

    Args:
        statistics: Processed stats from compute_activation_statistics.
        sparsity: Optional sparsity mapping from compute_sparsity_percentages.
            Defaults to None.
        dead_neurons: Optional mapping of dead unit counts. Defaults to None.
    """
    print(f"\n{'=' * 100}")
    print(f"{'Layer Name':<40} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Sparsity':>10}")
    print(f"{'=' * 100}")

    for name in sorted(statistics.keys()):
        stats = statistics[name]
        sparsity_val = sparsity.get(name, 0.0) if sparsity else 0.0

        print(
            f"{name:<40} "
            f"{stats['mean']:>10.4f} "
            f"{stats['std']:>10.4f} "
            f"{stats['min']:>10.4f} "
            f"{stats['max']:>10.4f} "
            f"{sparsity_val:>9.2%}"
        )

    if dead_neurons:
        print(f"\n{'=' * 100}")
        print("Dead Neurons (per layer):")
        print(f"{'=' * 100}")
        for name, count in sorted(dead_neurons.items()):
            print(f"  {name:<40}: {count} dead neurons")

    print(f"{'=' * 100}\n")


def remove_all_hooks(hook_data: Dict[str, Any]) -> None:
    """Safely detaches all registered PyTorch hooks to avoid memory leaks.

    Args:
        hook_data: Mapping returned by any registration function.
    """
    if "handles" in hook_data:
        for handle in hook_data["handles"]:
            handle.remove()
        hook_data["handles"].clear()
