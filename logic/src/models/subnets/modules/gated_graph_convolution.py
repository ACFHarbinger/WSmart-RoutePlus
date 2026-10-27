"""Gated Graph Convolution (GatedGCN) with explicit edge updates.

This module provides the GatedGraphConvolution layer, which explicitly models
edge features and uses a gating mechanism to control information flow between
neighboring nodes in a graph.

Attributes:
    GatedGraphConvolution: Gated graph convolution layer for node and edge updates.

Example:
    >>> import torch
    >>> from logic.src.models.subnets.modules.gated_graph_convolution import GatedGraphConvolution
    >>> layer = GatedGraphConvolution(hidden_dim=128)
    >>> h = torch.randn(1, 10, 128)
    >>> e = torch.randn(1, 10, 10, 128)
    >>> mask = torch.zeros(1, 10, 10, dtype=torch.bool)
    >>> h_out, e_out = layer(h, e, mask)
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn

from .activation_function import ActivationFunction
from .normalization import Normalization


class GatedGraphConvolution(nn.Module):
    """Gated Graph Convolution Layer (GatedGCN).

    Explicitly models edge features and uses a gating mechanism to control
    information flow between neighbors. Updates both node and edge features.

    Implements the Gated Graph ConvNet layer as described in Dwivedi et al. (2020).

    Attributes:
        hidden_dim (int): Dimensionality of hidden features.
        aggregation (str): Strategy for neighborhood pooling ('sum', 'mean', 'max').
        norm (str): Type of normalization strategy ('batch', 'layer', 'instance').
        learn_affine (bool): Whether normalization has learnable parameters.
        gated (bool): Whether the gating mechanism is active.
        U (nn.Linear): Transformation for source nodes in node update.
        V (nn.Linear): Transformation for neighbor nodes in node update.
        A (nn.Linear): Transformation for source nodes in edge update.
        B (nn.Linear): Transformation for target nodes in edge update.
        C (nn.Linear): Transformation for edge features in edge update.
        norm_h (Normalization): Normalization layer for node features.
        norm_e (Normalization): Normalization layer for edge features.
        activation (ActivationFunction): Non-linear activation module.

    References:
        Dwivedi, V. P., et al. (2020). Benchmarking graph neural networks.
        arXiv preprint arXiv:2003.00982.
    """

    def __init__(
        self,
        hidden_dim: int,
        aggregation: str = "sum",
        norm: str = "batch",
        activation: str = "relu",
        learn_affine: bool = True,
        gated: bool = True,
        bias: bool = True,
    ) -> None:
        """Initializes GatedGraphConvolution.

        Args:
            hidden_dim: Dimensionality of inputs and internal representations.
            aggregation: Strategy for neighborhood pooling ('sum', 'mean', 'max').
            norm: Normalization type ('batch', 'layer', 'instance').
            activation: Activation function name (e.g., 'relu', 'swish').
            learn_affine: Whether to learn gain and bias in normalization layers.
            gated: Whether to use gated message passing. Must be True for this layer.
            bias: Whether to use bias terms in linear transformations.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation
        self.norm = norm
        self.learn_affine = learn_affine
        self.gated = gated
        assert self.gated, "GatedGraphConvolution requires gated=True."

        self.U = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.A = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.B = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.C = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        self.norm_h = Normalization(hidden_dim, self.norm, learn_affine)
        self.norm_e = Normalization(hidden_dim, self.norm, learn_affine)
        self.activation = ActivationFunction(activation)

    def reset_parameters(self) -> None:
        """Initializes the linear layer weights using a uniform distribution."""
        for param in self.parameters():
            if param.dim() > 0:
                stdv = 1.0 / math.sqrt(param.size(-1))
                param.data.uniform_(-stdv, stdv)

    def forward(self, h: torch.Tensor, e: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Updates node and edge features using GatedGCN logic.

        Args:
            h: Input node features of shape (batch, nodes, hidden_dim).
            e: Input edge features of shape (batch, nodes, nodes, hidden_dim).
            mask: Adjacency/mask tensor of shape (batch, nodes, nodes).
                True indicates a masked edge.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - Updated node features of shape (batch, nodes, hidden_dim).
                - Updated edge features of shape (batch, nodes, nodes, hidden_dim).
        """
        _, num_nodes, _ = h.shape

        # Linear transformations for node update
        Uh = self.U(h)  # B x V x H
        Vh = self.V(h).unsqueeze(1).expand(-1, num_nodes, -1, -1)  # B x V x V x H

        # Linear transformations for edge update and gating
        Ah = self.A(h)  # B x V x H
        Bh = self.B(h)  # B x V x H
        Ce = self.C(e)  # B x V x V x H

        # Update edge features and compute edge gates
        e = Ah.unsqueeze(1) + Bh.unsqueeze(2) + Ce  # B x V x V x H
        gates = torch.sigmoid(e)  # B x V x V x H

        # Update node features
        h = Uh + self.aggregate(Vh, mask, gates)  # B x V x H

        # Normalize node features
        h = self.norm_h(h) if self.norm_h is not None else h

        # Normalize edge features
        batch_size, nodes_v, _, hidden_dim = e.shape
        e = (
            self.norm_e(e.view(batch_size, -1, hidden_dim)).view(batch_size, nodes_v, nodes_v, hidden_dim)
            if self.norm_e is not None
            else e
        )

        # Apply non-linearity
        h = self.activation(h)
        e = self.activation(e)

        return h, e

    def aggregate(self, Vh: torch.Tensor, mask: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        """Aggregates neighborhood information using gating and graph structure.

        Args:
            Vh: Neighborhood features of shape (batch, nodes, nodes, hidden_dim).
            mask: Adjacency mask of shape (batch, nodes, nodes).
            gates: Edge gates of shape (batch, nodes, nodes, hidden_dim).

        Returns:
            torch.Tensor: Aggregated feature representation of shape (batch, nodes, H).
        """
        # Perform feature-wise gating mechanism
        Vh = gates * Vh  # B x V x V x H

        # Enforce graph structure through masking
        Vh[mask.unsqueeze(-1).expand_as(Vh)] = 0

        if self.aggregation == "mean":
            return torch.sum(Vh, dim=2) / torch.sum(1 - mask, dim=2).unsqueeze(-1).type_as(Vh)
        elif self.aggregation == "max":
            return torch.max(Vh, dim=2)[0]
        else:
            return torch.sum(Vh, dim=2)
