"""Optimized Graph Convolution implementation with multiple aggregators.

This module provides the EfficientGraphConvolution (EGC) layer, which combines
multiple neighborhood aggregation strategies (mean, max, std, etc.) to learn
richer node representations.

Attributes:
    EfficientGraphConvolution: Multi-aggregator GCN layer with basis weighting.

Example:
    >>> import torch
    >>> from logic.src.models.subnets.modules.efficient_graph_convolution import EfficientGraphConvolution
    >>> layer = EfficientGraphConvolution(in_channels=128, out_channels=128)
    >>> x = torch.randn(1, 10, 128)
    >>> edge_index = torch.tensor([[0, 1], [1, 0]])
    >>> out = layer(x=x, edge_index=edge_index)
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple, cast

import torch
from torch import Tensor
from torch.nn import Linear, Parameter

try:
    import torch_sparse
    from torch_geometric.nn import MessagePassing
    from torch_geometric.nn.conv.gcn_conv import gcn_norm
    from torch_geometric.nn.inits import glorot, zeros
    from torch_geometric.typing import Adj, OptTensor, SparseTensor
    from torch_geometric.utils import add_remaining_self_loops, scatter

    PYG_AVAILABLE = True
except (ImportError, OSError):
    # Handle both missing package and DLL load failures (common with CUDA mismatches)
    PYG_AVAILABLE = False
    MessagePassing = object  # type: ignore[assignment, misc]
    gcn_norm = None
    glorot = None
    zeros = None
    add_remaining_self_loops = None
    scatter = None

    # Define minimal types for structural compatibility
    Adj = Any
    OptTensor = Optional[Tensor]
    SparseTensor = Any


MessagePassing = cast(Any, MessagePassing)  # optional torch_geometric base
_gcn_norm: Any = gcn_norm  # optional torch_geometric helper
_glorot: Any = glorot  # optional torch_geometric helper
_zeros: Any = zeros  # optional torch_geometric helper
_add_remaining_self_loops: Any = add_remaining_self_loops  # optional torch_geometric helper
_scatter: Any = scatter  # optional torch_geometric helper


# Adapted from https://github.com/shyam196/egc
class EfficientGraphConvolution(MessagePassing):
    """Efficient Graph Convolution (EGC) with multiple aggregators.

    Computes node updates using a linear combination of neighborhood pooling
    results (mean, max, sum, var, std, symnorm) and self-features, using basis
    functions to maintain high efficiency.

    Attributes:
        in_channels (int): Dimensionality of input node samples.
        out_channels (int): Dimensionality of output node samples.
        n_heads (int): Number of attention-like heads.
        num_bases (int): Number of basis functions for weight consolidation.
        cached (bool): Whether to cache graph normalization weights.
        add_self_loops (bool): Whether to include self-loops in the graph.
        aggregators (Tuple[str, ...]): Active neighborhood pooling methods.
        sigmoid (bool): Whether to apply sigmoid activation to basis weighting.
        bases_weight (Parameter): Consolidated basis transformation weights.
        comb_weight (Linear): Layer to compute basis mixing coefficients.
        bias (Optional[Parameter]): Learnable output bias vector.
        _cached_adj_t (Optional[SparseTensor]): Cached sparse graph structure.
        _cached_edge_index (Optional[Tuple[Tensor, OptTensor]]): Cached edge weights.
    """

    _cached_edge_index: Optional[Tuple[Tensor, OptTensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggrs: Iterable[str] = ("symnorm",),
        n_heads: int = 8,
        num_bases: int = 4,
        cached: bool = False,
        add_self_loops: bool = True,
        bias: bool = True,
        sigmoid: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initializes the EGC layer.

        Args:
            in_channels: Input feature dimensionality.
            out_channels: Output feature dimensionality.
            aggrs: Collection of aggregation methods (e.g., 'mean', 'max').
            n_heads: Number of attention-style heads.
            num_bases: Count of basis functions for weight efficiency.
            cached: If True, caches graph normalization for static graphs.
            add_self_loops: Whether to include self-loops in graph structure.
            bias: Whether to add a learnable bias term.
            sigmoid: If True, compresses basis weights using sigmoid.
            kwargs: Additional arguments passed to the parent `MessagePassing`.

        Raises:
            ImportError: If PyTorch Geometric dependencies are missing.
            ValueError: If `out_channels` is not divisible by `n_heads` or
                unsupported aggregators are provided.
        """
        if not PYG_AVAILABLE:
            raise ImportError(
                "torch_geometric and torch_sparse are required for EGC. Check for CUDA/PyTorch version mismatches."
            )

        super().__init__(node_dim=1, **kwargs)
        if out_channels % n_heads != 0:
            raise ValueError("out_channels must be divisible by the number of heads")

        for a in aggrs:
            if a not in {"sum", "mean", "symnorm", "min", "max", "var", "std"}:
                raise ValueError("Unsupported aggregator: {}".format(a))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.num_bases = num_bases
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.aggregators = tuple(aggrs)
        self.sigmoid = sigmoid

        self.bases_weight = Parameter(torch.Tensor(in_channels, (out_channels // n_heads) * num_bases))
        self.comb_weight = Linear(in_channels, n_heads * num_bases * len(self.aggregators))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets the learnable weights using Glorot/zeros initialization."""
        _glorot(self.bases_weight)
        self.comb_weight.reset_parameters()
        if self.bias is not None:
            _zeros(self.bias)
        self._cached_adj_t = None
        self._cached_edge_index = None

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        """Computes the Efficient Graph Convolutional update.

        Args:
            args: Positional node features and structure (x, edge_index).
            kwargs: Named arguments 'x' and 'edge_index'.

        Returns:
            Tensor: Updated node features of shape (batch, nodes, out_channels).

        Raises:
            ValueError: If required graph inputs (x, edge_index) are missing.
        """
        x = kwargs.get("x", args[0] if len(args) > 0 else None)
        edge_index = kwargs.get("edge_index", args[1] if len(args) > 1 else None)

        if x is None or edge_index is None:
            raise ValueError("Forward requires x and edge_index")

        symnorm_weight: OptTensor = None
        if "symnorm" in self.aggregators or self.add_self_loops:
            edge_index, symnorm_weight = self._norm_and_cache(x, edge_index)

        batch_size = x.size(0)
        num_nodes = x.size(1)

        # [num_nodes, (out_channels // n_heads) * num_bases]
        bases = torch.matmul(x, self.bases_weight)
        # [num_nodes, n_heads * num_bases * num_aggrs]
        weightings = self.comb_weight(x)
        if self.sigmoid:
            weightings = torch.sigmoid_(weightings)

        if symnorm_weight is not None:
            symnorm_weight = symnorm_weight.view(-1, 1)

        # propagate_type: (x: Tensor, symnorm_weight: OptTensor)
        aggregated = self.propagate(edge_index, x=bases, symnorm_weight=symnorm_weight, size=None)

        weightings = weightings.view(
            batch_size,
            num_nodes,
            self.n_heads,
            self.num_bases * len(self.aggregators),
        )
        aggregated = aggregated.view(
            batch_size,
            num_nodes,
            len(self.aggregators) * self.num_bases,
            self.out_channels // self.n_heads,
        )

        out = torch.matmul(weightings, aggregated)
        out = out.view(batch_size, num_nodes, self.out_channels)
        if self.bias is not None:
            out += self.bias

        return out

    def _norm_and_cache(self, x: Tensor, edge_index: Adj) -> Tuple[Adj, OptTensor]:
        """Handles graph normalization and persistence for static structures.

        Args:
            x: Current node features.
            edge_index: Adjacency or sparse graph structure.

        Returns:
            Tuple[Adj, OptTensor]: Stabilized edge indices and optional normalization weights.
        """
        num_nodes = x.size(self.node_dim)

        if "symnorm" in self.aggregators:
            if isinstance(edge_index, Tensor):
                if self._cached_edge_index is None:
                    edge_index, sw = _gcn_norm(
                        edge_index,
                        num_nodes=num_nodes,
                        add_self_loops=self.add_self_loops,
                    )
                    if self.cached:
                        self._cached_edge_index = (edge_index, sw)
                    return edge_index, sw
                assert self._cached_edge_index is not None
                return self._cached_edge_index

            if self._cached_adj_t is None:
                st_edge_index = cast(torch_sparse.SparseTensor, edge_index)
                adj_t = _gcn_norm(st_edge_index, num_nodes=num_nodes, add_self_loops=self.add_self_loops)
                if self.cached:
                    self._cached_adj_t = cast(SparseTensor, adj_t)
                return cast(Adj, adj_t), None
            assert self._cached_adj_t is not None
            return cast(Adj, self._cached_adj_t), None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                if self._cached_edge_index is None:
                    edge_index, _ = _add_remaining_self_loops(edge_index)
                    if self.cached:
                        self._cached_edge_index = (edge_index, None)
                    return edge_index, None
                assert self._cached_edge_index is not None
                return self._cached_edge_index[0], None

            if self._cached_adj_t is None:
                st_edge_index = cast(torch_sparse.SparseTensor, edge_index)
                adj_t = torch_sparse.fill_diag(st_edge_index, 1.0)
                if self.cached:
                    self._cached_adj_t = cast(SparseTensor, adj_t)
                return cast(Adj, adj_t), None
            assert self._cached_adj_t is not None
            return cast(Adj, self._cached_adj_t), None

        return edge_index, None

    def message(self, x_j: Tensor) -> Tensor:  # type: ignore[override]
        """Transfers source node features along the graph edges.

        Args:
            x_j: Source node features.

        Returns:
            Tensor: Raw messages.
        """
        return x_j

    def aggregate(  # type: ignore[override]
        self,
        inputs: Tensor,
        index: Tensor,
        dim_size: Optional[int] = None,
        symnorm_weight: OptTensor = None,
    ) -> Tensor:
        """Pools neighborhood messages using the configured subset of aggregators.

        Args:
            inputs: Raw incoming messages.
            index: Mapping indices from edges to nodes.
            dim_size: Total number of destination nodes.
            symnorm_weight: Optional pre-computed normalization weights.

        Returns:
            Tensor: Stacked aggregation results.
        """
        aggregated = []
        inputs = inputs.permute(1, 0, 2)
        for aggregator in self.aggregators:
            out = self._run_aggregator(aggregator, inputs, index, dim_size, symnorm_weight)
            aggregated.append(out)
        return torch.stack(aggregated, dim=1)

    def _run_aggregator(
        self,
        aggregator: str,
        inputs: Tensor,
        index: Tensor,
        dim_size: Optional[int],
        symnorm_weight: OptTensor,
    ) -> Tensor:
        """Invokes a specific aggregation function (e.g., mean, std).

        Args:
            aggregator: Strategy name ('sum', 'mean', 'max', 'min', 'symnorm', 'var', 'std').
            inputs: Edge messages to process.
            index: Node assignment indices.
            dim_size: Target dimension size.
            symnorm_weight: coefficients for symnorm pooling.

        Returns:
            Tensor: Single pooled representation.

        Raises:
            ValueError: If an unknown aggregator name is provided.
        """
        if aggregator == "sum":
            return _scatter(inputs, index, 0, dim_size, reduce="sum")
        if aggregator == "symnorm":
            assert symnorm_weight is not None
            return _scatter(inputs * symnorm_weight.unsqueeze(-1), index, 0, dim_size, reduce="sum")
        if aggregator == "mean":
            return _scatter(inputs, index, 0, dim_size, reduce="mean")
        if aggregator == "min":
            return _scatter(inputs, index, 0, dim_size, reduce="min")
        if aggregator == "max":
            return _scatter(inputs, index, 0, dim_size, reduce="max")
        if aggregator in {"var", "std"}:
            mean = _scatter(inputs, index, 0, dim_size, reduce="mean")
            mean_squares = _scatter(inputs * inputs, index, 0, dim_size, reduce="mean")
            out = mean_squares - mean * mean
            if aggregator == "std":
                out = torch.sqrt(torch.relu(out) + 1e-5)
            return out
        raise ValueError(f'Unknown aggregator "{aggregator}".')

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:  # type: ignore[override]
        """Optimized single-step message passing for sparse adjacency matrices.

        Args:
            adj_t: Multi-head or normalized sparse adjacency representation.
            x: Node features to propagate.

        Returns:
            Tensor: Combined neighborhood representations.
        """
        aggregated = []
        if len(self.aggregators) > 1 and "symnorm" in self.aggregators:
            st_adj_t = cast(torch_sparse.SparseTensor, adj_t)
            adj_t_nonorm = st_adj_t.set_value(None, layout=None)
        else:
            adj_t_nonorm = adj_t

        for aggregator in self.aggregators:
            out = self._run_sparse_aggregator(aggregator, adj_t, adj_t_nonorm, x)
            aggregated.append(out)

        return torch.stack(aggregated, dim=1)

    def _run_sparse_aggregator(
        self, aggregator: str, adj_t: SparseTensor, adj_t_nonorm: SparseTensor, x: Tensor
    ) -> Tensor:
        """Executes a pooling operation on a sparse matrix structure.

        Args:
            aggregator: Strategy name.
            adj_t: Normalized sparse tensor.
            adj_t_nonorm: Unnormalized/binary sparse tensor.
            x: Input feature matrix.

        Returns:
            Tensor: Sparse aggregation result.
        """
        if aggregator == "symnorm":
            out = torch_sparse.matmul(adj_t, x, reduce="sum")
            return cast(Tensor, out)

        if aggregator in ["var", "std"]:
            mean = torch_sparse.matmul(adj_t_nonorm, x, reduce="mean")
            mean_sq = torch_sparse.matmul(adj_t_nonorm, x * x, reduce="mean")
            assert isinstance(mean, Tensor)
            assert isinstance(mean_sq, Tensor)
            out = mean_sq - mean * mean
            if aggregator == "std":
                out = torch.sqrt(torch.relu(out) + 1e-5)
            return out

        out = torch_sparse.matmul(adj_t_nonorm, x, reduce=aggregator)
        return cast(Tensor, out)

    def __repr__(self) -> str:
        """Provides a formatted description of the EGC layer.

        Returns:
            str: Layer name, input/output dimensions, and active aggregators.
        """
        return "{}({}, {}, {})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.aggregators,
        )
