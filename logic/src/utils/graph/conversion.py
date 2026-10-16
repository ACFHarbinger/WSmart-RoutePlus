"""
Graph format conversion utilities.

Attributes:
    adj_to_idx: Converts dimensionality-2 adjacency matrix to a [2, num_edges] index array.
    idx_to_adj: Converts edge index array back to an adjacency matrix.
    tour_to_adj: Converts a sequence of nodes (tour) into an adjacency matrix.
    sort_by_pairs: Sorts edge indices by their linear index (row * size + col).

Example:
    adj_to_idx(adj_matrix)
    idx_to_adj(edge_idx)
    tour_to_adj(tour_nodes)
    sort_by_pairs(graph_size, edge_idx)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


from typing import Union

import numpy as np


def adj_to_idx(adj_matrix: np.ndarray, negative: bool = True) -> np.ndarray:
    """
    Converts dimensionality-2 adjacency matrix to a [2, num_edges] index array.

    Args:
        adj_matrix: The adjacency matrix.
        negative: If True, 0 represents an edge. Defaults to True.

    Returns:
        np.ndarray: Matrix of edge indices.
    """
    filter_val = 0 if negative else 1
    src, dst = np.where(adj_matrix == filter_val)
    return np.vstack((src, dst))


def idx_to_adj(edge_idx: Union[torch.Tensor, np.ndarray], negative: bool = False) -> np.ndarray:
    """
    Converts edge index array back to an adjacency matrix.

    Args:
        edge_idx: Edge indices.
        negative: If True, 0 represents an edge. Defaults to False.

    Returns:
        np.ndarray: The adjacency matrix.
    """
    fill_values = (1, 0) if negative else (0, 1)
    num_nodes = int(edge_idx.max().item() + 1)
    adj_matrix = np.full((num_nodes, num_nodes), fill_values[0])
    # Handle both tensor and numpy array indexing
    if not isinstance(edge_idx, np.ndarray):
        src, dst = edge_idx[0].cpu().numpy(), edge_idx[1].cpu().numpy()
    else:
        src, dst = edge_idx[0], edge_idx[1]
    adj_matrix[src, dst] = fill_values[1]
    return adj_matrix


def tour_to_adj(tour_nodes: list) -> np.ndarray:
    """
    Converts a sequence of nodes (tour) into an adjacency matrix.

    Args:
        tour_nodes: List of node indices in visit order.

    Returns:
        np.ndarray: Adjacency matrix representation of the tour.
    """
    num_nodes = len(tour_nodes)
    tour_edges = np.zeros((num_nodes, num_nodes))
    for idx in range(len(tour_nodes) - 1):
        i = tour_nodes[idx]
        j = tour_nodes[idx + 1]
        tour_edges[i][j] = 1
        tour_edges[j][i] = 1

    # Add final connection
    if num_nodes > 0:
        j = tour_nodes[-1]
        tour_edges[j][tour_nodes[0]] = 1
        tour_edges[tour_nodes[0]][j] = 1
    return tour_edges


def sort_by_pairs(graph_size: int, edge_idx: torch.Tensor) -> torch.Tensor:
    """
    Sorts edge indices by their linear index (row * size + col).

    Args:
        graph_size: Size of the graph (number of nodes).
        edge_idx: Edge indices.

    Returns:
        Tensor: Sorted edge indices.
    """
    import torch

    assert len(edge_idx.size()) == 2, (
        f"edge_idx must be a 2D tensor, got shape {edge_idx.size()} with {len(edge_idx.size())} dimensions"
    )
    assert edge_idx.size(dim=0) == 2 or edge_idx.size(dim=-1) == 2, (
        f"edge_idx must have shape (2, num_edges) or (num_edges, 2), got {edge_idx.size()}"
    )

    # Transpose the tensor if it has size (2, num_edges)
    is_transpose = edge_idx.size(dim=-1) != 2
    if is_transpose:
        edge_idx = torch.transpose(edge_idx, 0, 1)

    tmp = edge_idx.select(1, 0) * graph_size + edge_idx.select(1, 1)
    ind = tmp.sort().indices
    sorted_idx = edge_idx.index_select(0, ind)
    if is_transpose:
        sorted_idx = torch.transpose(sorted_idx, 0, 1)
    return sorted_idx
