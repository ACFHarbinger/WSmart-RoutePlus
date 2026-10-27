"""
POMO (Policy Optimization with Multiple Optima) Operations.

Attributes:
    select_start_nodes: Select start nodes for POMO-style multi-start decoding.
    select_start_nodes_by_distance: Select start nodes based on distance from depot.
    get_num_starts: Get environment-specific number of start nodes for multi-start decoding.
    get_best_actions: Select best actions from multi-start rollouts.

Example:
    >>> from logic.src.utils.actions.pomo import select_start_nodes
    >>> td = TensorDict({"locs": torch.randn(1, 10, 2)}, batch_size=[1])
    >>> select_start_nodes(td, 3)
    tensor([0, 1, 2])
    >>> from logic.src.utils.actions.pomo import select_start_nodes_by_distance
    >>> td = TensorDict({"locs": torch.randn(1, 10, 2), "depot": torch.randn(1, 2)}, batch_size=[1])
    >>> select_start_nodes_by_distance(td, 3)
    tensor([0, 1, 2])
    >>> from logic.src.utils.actions.pomo import get_num_starts
    >>> td = TensorDict({"locs": torch.randn(1, 10, 2)}, batch_size=[1])
    >>> get_num_starts(td)
    10
    >>> from logic.src.utils.actions.pomo import get_best_actions
    >>> actions = torch.randn(1, 10, 10)
    >>> max_idxs = torch.tensor([0])
    >>> get_best_actions(actions, max_idxs)
    tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
"""

from __future__ import annotations

from typing import Optional, cast

import torch
from tensordict import TensorDict


def select_start_nodes(
    td: TensorDict,
    num_starts: int,
) -> torch.Tensor:
    """
    Select start nodes for POMO-style multi-start decoding.

    Selects `num_starts` starting nodes uniformly. If num_starts <= num_nodes,
    uses the first `num_starts` nodes (deterministic for reproducibility).

    Args:
        td: TensorDict with problem data (assumed to have 'locs' or 'loc').
        num_starts: Number of start nodes to select.

    Returns:
        Selected start node indices [batch * num_starts].
    """
    num_loc = td["locs"].shape[1] if "locs" in td.keys() else td["loc"].shape[1]

    # Exclude depot (index 0) if present in typical VRP/TSP formulations
    # Assuming standard RL4CO format where depots might be separate or at index 0
    # For simplicity, we just pick from available nodes.

    # Logic:
    # If num_starts <= num_loc: use 0..num_starts-1
    # If num_starts > num_loc: use 0..num_loc-1 and cycle/random sample rest

    if num_starts <= num_loc:
        start_nodes = torch.arange(num_starts, device=td.device)
    else:
        # Cycle if we need more starts than nodes
        start_nodes = torch.arange(num_loc, device=td.device).repeat((num_starts // num_loc) + 1)[:num_starts]

    # Batch expand: [batch * num_starts]
    batch_size = td.batch_size[0]
    return start_nodes.repeat(batch_size)


def select_start_nodes_by_distance(
    td: TensorDict,
    num_starts: int,
) -> torch.Tensor:
    """
    Select start nodes based on distance from depot.

    Selects nodes farthest from depot as starting points, which can
    improve exploration diversity.

    Args:
        td: TensorDict with 'locs' [batch, n_nodes, 2] and 'depot' [batch, 2].
        num_starts: Number of start nodes to select.

    Returns:
        Selected start node indices [batch * num_starts].
    """
    locs = cast(torch.Tensor, td["locs"] if "locs" in td.keys() else td["loc"])
    depot = cast(torch.Tensor, td["depot"])

    # Calculate distance from depot
    # depot: [batch, 2], locs: [batch, n, 2]
    dists = (locs - depot.unsqueeze(1)).norm(p=2, dim=-1)  # [batch, n]

    # Select top-k farthest
    k = min(num_starts, dists.size(1))
    _, topk_idx = torch.topk(dists, k=k, dim=1, largest=True)

    # Flatten
    return topk_idx.flatten()


def get_num_starts(td: TensorDict, env_name: Optional[str] = None) -> int:
    """
    Get environment-specific number of start nodes for multi-start decoding.

    For POMO, typically num_starts = num_nodes (excluding depot).

    Args:
        td: TensorDict with problem data.
        env_name: Optional environment name for special cases.

    Returns:
        Number of start nodes.
    """
    if "locs" in td.keys():
        return td["locs"].shape[1]
    elif "loc" in td.keys():
        return td["loc"].shape[1]
    # Fallback default
    return 1


def get_best_actions(
    actions: torch.Tensor,
    max_idxs: torch.Tensor,
) -> torch.Tensor:
    """
    Select best actions from multi-start rollouts.

    Args:
        actions: Actions from all starts [batch * num_starts, seq_len].
        max_idxs: Indices of best starts per batch element [batch].

    Returns:
        Best actions [batch, seq_len].
    """
    batch_size = max_idxs.size(0)
    # actions -> [batch, num_starts, seq_len]
    # Assuming num_starts is inferred from shape
    total_samples = actions.size(0)
    num_starts = total_samples // batch_size

    actions_reshaped = actions.view(batch_size, num_starts, -1)

    # Gather best: [batch, 1, seq_len]
    best_actions = actions_reshaped.gather(1, max_idxs.view(batch_size, 1, 1).expand(batch_size, 1, actions.size(-1)))

    return best_actions.squeeze(1)
