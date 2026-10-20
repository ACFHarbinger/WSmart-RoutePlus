"""Standard Graph Convolutional Network (GCN) layer.

This module provides the GraphConvolution layer, which performs neighborhood
aggregation on node features based on a graph adjacency matrix.

Attributes:
    GraphConvolution: Standard graph convolution layer for node feature aggregation.

Example:
    >>> import torch
    >>> from logic.src.models.subnets.modules.graph_convolution import GraphConvolution
    >>> model = GraphConvolution(in_channels=128, out_channels=128)
    >>> x = torch.randn(1, 10, 128)
    >>> adj = torch.eye(10)
    >>> out = model(x, adj)
"""

from __future__ import annotations

import math

import torch
from torch import nn

try:
    from torch_geometric.utils import scatter

    PYG_AVAILABLE = True
except (ImportError, OSError):
    PYG_AVAILABLE = False
    scatter = None


class GraphConvolution(nn.Module):
    """Standard Graph Convolution layer.

    Performs message passing by aggregating features from neighbors according to:
    h_i' = W * h_i + Agg({h_j})

    Attributes:
        in_channels (int): Dimensionality of input node features.
        out_channels (int): Dimensionality of output node features.
        aggregation (str): Pooling strategy ('sum', 'mean', 'max').
        linear (nn.Linear): Transformation layer for input features.
        bias (Optional[nn.Parameter]): Learnable bias vector.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggregation: str = "sum",
        bias: bool = True,
    ) -> None:
        """Initializes GraphConvolution.

        Args:
            in_channels: Input feature dimensionality.
            out_channels: Output feature dimensionality.
            aggregation: Strategy for neighborhood pooling ('sum', 'mean', 'max').
            bias: Whether to add a learnable bias term.

        Raises:
            ImportError: If 'max' aggregation is requested but PyTorch Geometric
                is not installed.
        """
        if not PYG_AVAILABLE and aggregation == "max":
            raise ImportError(
                "torch_geometric is required for 'max' aggregation in GCN. Check your dependencies and CUDA version."
            )
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregation = aggregation

        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)
        self.init_parameters()

    def init_parameters(self) -> None:
        """Initializes the linear layer weights using a uniform distribution."""
        for param in self.parameters():
            if param.dim() > 0:
                stdv = 1.0 / math.sqrt(param.size(-1))
                param.data.uniform_(-stdv, stdv)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Applies graph convolution to a batch of node features.

        Args:
            h: Input features of shape (batch, nodes, in_channels).
            mask: Adjacency/mask matrix of shape (nodes, nodes) or (batch, nodes, nodes).

        Returns:
            torch.Tensor: Updated features of shape (batch, nodes, out_channels).
        """
        batch_size, num_nodes = h.size(0), h.size(1)
        support = self.linear(h)
        if self.aggregation == "max":
            source_idx, target_idx = mask.squeeze(0).nonzero(as_tuple=True)
            num_edges = source_idx.size(0)
            batch_indices = torch.arange(batch_size, device=h.device)

            expanded_source = source_idx.repeat(batch_size)
            expanded_target = target_idx.repeat(batch_size)
            batch_id = batch_indices.repeat_interleave(num_edges)

            # Get messages from source nodes for all batches
            messages = support[batch_id, expanded_source]
            batch_offset = batch_id * num_nodes
            unique_targets = batch_offset + expanded_target

            # Use scatter to aggregate messages to targets
            flat_output = scatter(
                messages,
                unique_targets,
                dim=0,
                dim_size=batch_size * num_nodes,
                reduce="max",
            )
            out = flat_output.view(batch_size, num_nodes, self.out_channels)
        elif self.aggregation == "mean":
            # Check if mask is batched (B, V, V) or shared (1, V, V) / (V, V)
            if mask.dim() == 3:
                degrees = mask.sum(dim=-1, keepdim=True).clamp(min=1)
                # degrees is (B, V, 1)
                if degrees.size(0) == 1:  # Shared graph but 3D
                    degrees = degrees.expand(batch_size, -1, -1)
            else:  # (V, V)
                degrees = mask.sum(dim=-1).view(1, num_nodes, 1).expand(batch_size, -1, -1).clamp(min=1)

            if mask.dim() == 2:
                mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
            elif mask.size(0) == 1:
                mask = mask.expand(batch_size, -1, -1)

            out = torch.bmm(mask.to(support.dtype), support)
            out = out / degrees
        else:
            out = torch.bmm(mask.expand(batch_size, -1, -1).to(support.dtype), support)

        if self.bias is not None:
            out += self.bias
        return out
