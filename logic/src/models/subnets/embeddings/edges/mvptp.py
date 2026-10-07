"""MVPTP Edge embedding module.

This module provides the MVPTPEdgeEmbedding layer, which constructs graph
representations for Capacitated VRPs, preserving critical depot connectivity.

Attributes:
    MVPTPEdgeEmbedding: Edge encoder with guaranteed depot-to-customer links.

Example:
    >>> from logic.src.models.subnets.embeddings.edges.mvptp import MVPTPEdgeEmbedding
    >>> embedder = MVPTPEdgeEmbedding(embed_dim=128)
    >>> pyg_batch = embedder(td, init_embeddings)
"""

from __future__ import annotations

from typing import List

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import BaseData

from logic.src.utils.actions import get_full_graph_edge_index, sparsify_graph

from .base import EdgeEmbedding


class MVPTPEdgeEmbedding(EdgeEmbedding):
    """Edge embedding for capacitated VRP problems.

    Extends the base EdgeEmbedding logic to ensure all customer nodes maintain
    direct edges to and from the depot (node 0), even when graph sparsification
    is applied. Depot connectivity is critical for routing feasibility.

    Attributes:
        node_dim (int): Dimensionality of raw edge features.
    """

    def _cost_matrix_to_graph(
        self,
        batch_cost_matrix: torch.Tensor,
        init_embeddings: torch.Tensor,
    ) -> Batch:
        """Converts batched cost matrices to a graph with forced depot connections.

        Args:
            batch_cost_matrix: (batch, n, n) distance/cost matrices.
            init_embeddings: (batch, n, embed_dim) node feature embeddings.

        Returns:
            Batch: PyG Batch object containing the augmented sparse graph.
        """
        k_sparse = self._get_k_sparse(batch_cost_matrix.shape[-1])
        graph_data: List[BaseData] = []

        for idx, cost_matrix in enumerate(batch_cost_matrix):
            n = cost_matrix.shape[0]

            if self.sparsify:
                # Sparsify customer-to-customer edges (exclude depot at index 0)
                edge_index, edge_attr = sparsify_graph(cost_matrix[1:, 1:], k_sparse, self_loop=False)
                edge_index = edge_index + 1  # Shift indices to account for removed depot

                # Add depot-to-all and all-to-depot edges
                customer_indices = torch.arange(1, n, device=cost_matrix.device)
                depot_zeros = torch.zeros(n - 1, dtype=torch.long, device=cost_matrix.device)

                depot_edges = torch.cat(
                    [
                        edge_index,
                        torch.stack([customer_indices, depot_zeros]),  # customer -> depot
                        torch.stack([depot_zeros, customer_indices]),  # depot -> customer
                    ],
                    dim=1,
                )
                depot_attrs = torch.cat(
                    [
                        edge_attr,
                        cost_matrix[1:, 0:1],  # customer -> depot distances
                        cost_matrix[0:1, 1:].t(),  # depot -> customer distances
                    ],
                    dim=0,
                )
                edge_index = depot_edges
                edge_attr = depot_attrs
            else:
                edge_index = get_full_graph_edge_index(n, self_loop=False).to(cost_matrix.device)
                edge_attr = cost_matrix[edge_index[0], edge_index[1]].unsqueeze(-1)

            graph = Data(x=init_embeddings[idx], edge_index=edge_index, edge_attr=edge_attr)
            graph_data.append(graph)

        batch = Batch.from_data_list(graph_data)
        batch.edge_attr = self.edge_embed(batch.edge_attr)
        return batch
