"""
Configuration loading utilities for model arguments.

Attributes:
    _load_normalization_args: Loads normalization arguments from a JSON file.
    _load_activation_args: Loads activation arguments from a JSON file.
    load_args: Loads argument configuration from a JSON file.

Example:
    >>> from logic.src.utils.model.config_utils import load_args
    >>> args = load_args("path/to/args.json")
    >>> isinstance(args, dict)
    True
"""

from __future__ import annotations

import json
from typing import Any, Dict


def _load_normalization_args(subnet_args: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Loads normalization arguments from a JSON file.

    Args:
        subnet_args: The loaded arguments for the subnet (encoder or decoder).
        args: The loaded arguments.

    Returns:
        The loaded arguments.
    """
    # Normalization parameters
    norm = subnet_args.get("normalization", {})
    if isinstance(norm, dict):
        if "norm_type" in norm:
            args["normalization"] = norm["norm_type"]
        if "epsilon" in norm:
            args["epsilon_alpha"] = norm["epsilon"]
        if "momentum" in norm:
            args["momentum_beta"] = norm["momentum"]
        if "learn_affine" in norm:
            args["learn_affine"] = norm["learn_affine"]
        if "track_stats" in norm:
            args["track_stats"] = norm["track_stats"]
        if "k_lrnorm" in norm:
            args["lrnorm_k"] = norm["k_lrnorm"]
        if "n_groups" in norm:
            args["gnorm_groups"] = norm["n_groups"]
    return args


def _load_activation_args(subnet_args: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Loads activation arguments from a JSON file.

    Args:
        subnet_args: The loaded arguments for the subnet (encoder or decoder).
        args: The loaded arguments.

    Returns:
        The loaded arguments.
    """
    act = subnet_args.get("activation", {})
    if isinstance(act, dict):
        if "name" in act:
            args["activation"] = act["name"]
        if "param" in act:
            args["af_param"] = act["param"]
        if "threshold" in act:
            args["af_threshold"] = act["threshold"]
        if "replacement_value" in act:
            args["af_replacement"] = act["replacement_value"]
        if "n_params" in act:
            args["af_nparams"] = act["n_params"]
        if "range" in act:
            args["af_urange"] = act["range"]
    return args


def load_args(filename: str) -> Dict[str, Any]:
    """
    Loads argument configuration from a JSON file.
    Handles deprecated keys for backward compatibility.

    Args:
        filename: Path to args.json.

    Returns:
        The loaded arguments.
    """
    with open(filename, "r") as f:
        args = json.load(f)

    # 1. Flatten nested kwargs if present (legacy model format)
    if "kwargs" in args and isinstance(args["kwargs"], dict):
        kwargs = args.pop("kwargs")
        for k, v in kwargs.items():
            if k not in args:
                args[k] = v

    # 2. Extract architecture parameters from the nested 'decoder' block if it exists
    # These override top-level values as they represent the specific model config
    if "decoder" in args and isinstance(args["decoder"], dict):
        dec = args["decoder"]

        # Normalization parameters
        args = _load_normalization_args(dec, args)

        # Activation parameters
        args = _load_activation_args(dec, args)

        # Other decoder parameters (including n_heads from decoder)
        if "dropout" in dec:
            args["dropout"] = dec["dropout"]
        if "tanh_clipping" in dec:
            args["tanh_clipping"] = dec["tanh_clipping"]
        if "n_heads" in dec:
            args["n_heads"] = dec["n_heads"]

    # 3. Handle backward compatibility for other keys
    if "n_encode_layers" not in args and "num_encoder_layers" in args:
        args["n_encode_layers"] = args["num_encoder_layers"]

    # Final Backwards compatibility for problem/distribution
    if "data_distribution" not in args:
        args["data_distribution"] = None
        if "problem" in args and args["problem"] is not None:
            probl, *dist = args["problem"].split("_")
            if "vrpp" in probl:
                args["problem"] = probl
                args["data_distribution"] = dist[0] if dist else None
        else:
            args["problem"] = "vrpp"  # Default fallback

    return args
