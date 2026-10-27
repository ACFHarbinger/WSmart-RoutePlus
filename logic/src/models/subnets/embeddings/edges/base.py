"""Edge embedding modules for graph-based encoders.

This module provides the EdgeEmbedding base class, which transforms raw pairwise
features into latent representations used by Graph Neural Networks (GatedGCN, GAT).

Attributes:
    EdgeEmbedding: Base module for constructing graph-structured data from distances.

Example:
    >>> from logic.src.models.subnets.embeddings.edges import EdgeEmbedding
    >>> embedder = EdgeEmbedding(embed_dim=128, k_sparse=20)
    >>> pyg_batch = embedder(td, init_embeddings)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, List, Optional, Union, cast

import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import BaseData

from logic.src.utils.actions import (
    get_distance_matrix,
    get_full_graph_edge_index,
    sparsify_graph,
)


class EdgeEmbedding(nn.Module):
    """Base class for edge embeddings in graph reinforcement learning.

    Subclasses define how raw edge features are extracted from problem data
    and projected into the latent space. Includes support for k-NN graph
    sparsification.

    Attributes:
        node_dim (int): Dimensionality of raw input edge features.
        sparsify (bool): Whether to create a sparse k-NN representation.
        edge_embed (nn.Linear): Linear projection for edge features.
        _get_k_sparse (Callable): Logic for calculating number of neighbors.
    """

    node_dim: int = 1

    def __init__(
        self,
        embed_dim: int,
        linear_bias: bool = True,
        sparsify: bool = True,
        k_sparse: Optional[Union[int, Callable[[int], int]]] = None,
    ) -> None:
        """Initializes EdgeEmbedding.

        Args:
            embed_dim: Output embedding dimension for edge features.
            linear_bias: Whether to include bias in the edge projection.
            sparsify: Whether to create a sparse k-NN graph.
            k_sparse: Number of neighbors per node, or a callable mapping
                instance size to neighbor count. Defaults to max(n // 5, 10).
        """
        assert Batch is not None, (
            "torch_geometric is required for EdgeEmbedding. Install via: pip install torch_geometric"
        )
        super().__init__()

        self._get_k_sparse: Callable[[int], int]
        if k_sparse is None:
            self._get_k_sparse = lambda n: max(n // 5, 10)
        elif isinstance(k_sparse, int):
            self._get_k_sparse = lambda n: k_sparse
        elif callable(k_sparse):
            self._get_k_sparse = k_sparse
        else:
            raise ValueError("k_sparse must be an int or a callable")

        self.sparsify = sparsify
        self.edge_embed = nn.Linear(self.node_dim, embed_dim, linear_bias)

    def forward(self, td: Any, init_embeddings: torch.Tensor) -> Batch:
        """Builds a batched PyG graph with embedded edge features.

        Args:
            td: Current problem instance data containing 'locs'.
            init_embeddings: Static node embeddings of shape (batch, nodes, dim).

        Returns:
            Batch: PyG Batch object containing x, edge_index, and edge_attr.
        """
        cost_matrix = get_distance_matrix(td["locs"])
        batch = self._cost_matrix_to_graph(cost_matrix, init_embeddings)
        return batch

    def _cost_matrix_to_graph(
        self,
        batch_cost_matrix: torch.Tensor,
        init_embeddings: torch.Tensor,
    ) -> Batch:
        """Converts batched cost matrices to a single batched PyG graph.

        Args:
            batch_cost_matrix: (batch, n, n) cost/distance matrices.
            init_embeddings: (batch, n, embed_dim) node feature embeddings.

        Returns:
            Batch: Unified PyG Batch object.
        """
        k_sparse = self._get_k_sparse(batch_cost_matrix.shape[-1])
        graph_data: List[BaseData] = []

        for idx, cost_matrix in enumerate(batch_cost_matrix):
            if self.sparsify:
                edge_index, edge_attr = sparsify_graph(cost_matrix, k_sparse, self_loop=False)
            else:
                edge_index = get_full_graph_edge_index(cost_matrix.shape[0], self_loop=False).to(cost_matrix.device)
                edge_attr = cost_matrix[edge_index[0], edge_index[1]].unsqueeze(-1)

            graph = Data(x=init_embeddings[idx], edge_index=edge_index, edge_attr=edge_attr)
            graph_data.append(graph)

        batch = Batch.from_data_list(graph_data)
        graph_batch = cast(Any, batch)
        graph_batch.edge_attr = self.edge_embed(graph_batch.edge_attr)
        return batch
