"""
Configuration and data sanitization utilities.

Attributes:
    deep_sanitize: Recursively convert NumPy types, Tensors, and OmegaConf containers to Python primitives.
    get_pol_name: Extract a concise policy name from a structured or string config.

Example:
    deep_sanitize(val)
    get_pol_name(pol_obj)
"""

import ast
import dataclasses
import sys
from typing import Any

import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf

from logic.src.configs.learning.envs.graph import GraphConfig


def deep_sanitize(val: Any) -> Any:
    """
    Recursively convert NumPy types, Tensors, and OmegaConf containers to Python primitives.

    Args:
        val: The value to sanitize.

    Returns:
        The sanitized value (dict, list, int, float, str, etc.).
    """
    if val is None:
        return None

    # Handle OmegaConf containers
    if isinstance(val, (DictConfig, ListConfig)):
        val = OmegaConf.to_container(val, resolve=True)

    # Handle NumPy scalars
    if isinstance(val, np.generic):
        return val.item()

    # Handle NumPy arrays
    if isinstance(val, np.ndarray):
        return val.tolist()

    # Handle PyTorch Tensors
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(val, torch.Tensor):
        if val.numel() == 1:
            return val.item()
        return val.detach().cpu().numpy().tolist()

    # Handle standard containers
    if isinstance(val, dict):
        return {str(k): deep_sanitize(v) for k, v in val.items()}

    if isinstance(val, (list, tuple)):
        return [deep_sanitize(item) for item in val]

    return val


def get_pol_name(pol_obj: Any) -> str:
    """
    Extract a concise policy name from a structured or string config.
    Handles Hydra/OmegaConf objects, dicts, and strings.

    Args:
        pol_obj: The policy object to extract the name from.

    Returns:
        The policy name.
    """
    try:
        # Sanitize to plain Python types (handles DictConfig, Tensor, etc.)
        sanitized = deep_sanitize(pol_obj)

        if isinstance(sanitized, dict):
            if len(sanitized) == 1:
                # Return the top-level key (e.g., 'rl_alns')
                return str(list(sanitized.keys())[0])
            return "complex_policy"

        if isinstance(sanitized, list):
            return f"policy_list_{len(sanitized)}"

        # If it's a string, or became one, ensure it's not a serialized dict/block
        name = str(sanitized)
        if "{" in name or "[" in name:
            try:
                # Attempt to parse a string repr of a dict/list
                parsed = ast.literal_eval(name)
                return get_pol_name(parsed)
            except (ValueError, SyntaxError):
                return "unnamed_policy"

        if len(name) > 256:
            return "unnamed_policy"

        return name
    except Exception:
        return "unknown_policy"


def get_graph_config(sim: Any) -> Any:
    """Extract and validate GraphConfig from a simulation configuration object.

    Supports both GraphConfig dataclass instances and dict/DictConfig containers.
    """
    graph = getattr(sim, "graph", None)
    if graph is None:
        return GraphConfig()

    if isinstance(graph, GraphConfig):
        return graph

    sanitized = deep_sanitize(graph)
    if not isinstance(sanitized, dict):
        return GraphConfig()

    valid_fields = {f.name for f in dataclasses.fields(GraphConfig)}
    init_args = {k: v for k, v in sanitized.items() if k in valid_fields}

    # Handle the mapping of legacy keys
    if "days" in sanitized and "n_days" not in init_args:
        init_args["n_days"] = sanitized["days"]
    if "size" in sanitized and "num_loc" not in init_args:
        init_args["num_loc"] = sanitized["size"]

    return GraphConfig(**init_args)
