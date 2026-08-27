"""Backward-compatibility shim for CriticNetwork.

This module provides the legacy implementation of the critic network and
re-exports the canonical `CriticNetwork` from the policy module. It serves
as a bridge for older problem-based pipelines.

Attributes:
    LegacyCriticNetwork: The legacy critic network implementation.

Example:
    >>> critic = LegacyCriticNetwork(problem, factory, embed_dim=128, hidden_dim=256)
"""

from __future__ import annotations

import warnings
from typing import Any, Dict

import torch
from torch import nn

# Re-export the canonical CriticNetwork and factory
from logic.src.models.common.critic_network.policy import (  # noqa: F401
    CriticNetwork,
    create_critic_from_actor,
)
from logic.src.models.subnets.embeddings import (
    VRPPContextEmbedder,
    WCVRPContextEmbedder,
)
from logic.src.models.subnets.modules import ActivationFunction


class LegacyCriticNetwork(nn.Module):
    """Legacy Critic Network using problem objects and component factories.

    This is the original implementation kept for backward compatibility with
    the legacy training pipeline. It evaluates graph states by aggregating
    node embeddings and passing them through a value head.

    Attributes:
        hidden_dim: Dimensionality of the value head hidden layers.
        embed_dim: Dimensionality of the node embeddings.
        aggregation_graph: Graph aggregation mode ('avg', 'sum', 'max').
        is_wc: Whether the problem is waste-collection based.
        is_vrpp: Whether the problem is VRPP-based.
        context_embedder: Problem-specific node feature encoder.
        encoder: The graph neural network used for feature extraction.
        value_head: Linear layers to predict state value.
    """

    def __init__(
        self,
        problem: Any,
        component_factory: Any,
        embed_dim: int,
        hidden_dim: int,
        n_layers: int,
        n_sublayers: int,
        encoder_normalization: str = "batch",
        activation: str = "gelu",
        n_heads: int = 8,
        aggregation_graph: str = "avg",
        dropout_rate: float = 0.0,
        temporal_horizon: int = 0,
    ) -> None:
        """Initialize the LegacyCriticNetwork.

        Args:
            problem: Problem instance defining the environment type.
            component_factory: Factory used to instantiate the encoder.
            embed_dim: Input dimensionality of node features.
            hidden_dim: Hidden dimensionality for the value head.
            n_layers: Number of layers in the encoder.
            n_sublayers: Number of sub-layers per encoder block.
            encoder_normalization: Normalization type for the encoder.
            activation: Nonlinearity to use (e.g., 'relu', 'gelu').
            n_heads: Number of attention heads in the encoder.
            aggregation_graph: Message aggregation strategy.
            dropout_rate: Dropout probability.
            temporal_horizon: Look-ahead horizon for dynamic features.
        """
        super().__init__()
        warnings.warn(
            "LegacyCriticNetwork is deprecated. Use logic.src.models.critic.CriticNetwork instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.aggregation_graph = aggregation_graph

        self.is_wc = problem.NAME in ("wcvrp", "cwcvrp", "sdwcvrp")
        self.is_vrpp = problem.NAME in ("vrpp", "cvrpp", "ttop")

        assert self.is_wc or self.is_vrpp, f"Unsupported problem: {problem.NAME}"

        node_dim = 3

        if self.is_wc:
            self.context_embedder = WCVRPContextEmbedder(
                embed_dim, node_dim=node_dim, temporal_horizon=temporal_horizon
            )
        else:
            self.context_embedder = VRPPContextEmbedder(embed_dim, node_dim=node_dim, temporal_horizon=temporal_horizon)  # type: ignore[assignment]

        self.encoder = component_factory.create_encoder(
            n_heads=n_heads,
            embed_dim=embed_dim,
            n_layers=n_layers,
            n_sublayers=n_sublayers,
            normalization=encoder_normalization,
            activation=activation,
            dropout_rate=dropout_rate,
            feed_forward_hidden=self.hidden_dim,
        )

        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            ActivationFunction(activation),
            nn.Linear(hidden_dim, 1),
        )

    def _init_embed(self, inputs: Dict[str, Any] | torch.Tensor) -> torch.Tensor:
        """Initialize node embeddings from raw node features.

        Args:
            inputs: Raw node feature tensor or input dictionary.

        Returns:
            torch.Tensor: Initial latent representations.
        """
        if isinstance(inputs, torch.Tensor):
            # Wrap tensor in a dict for the context embedder
            return self.context_embedder.init_node_embeddings({"nodes": inputs})
        return self.context_embedder.init_node_embeddings(inputs)

    def forward(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """Estimate the state value for the given input instance.

        Args:
            inputs: Dictionary containing 'nodes' (or 'loc', 'depot', etc.) and optional 'edges'.

        Returns:
            torch.Tensor: Scalar value prediction of shape [batch, 1].
        """
        edges = inputs.get("edges")
        # Try to use 'nodes' key if present, otherwise pass the whole dict
        input_data = inputs.get("nodes", inputs)
        embeddings = self.encoder(self._init_embed(input_data), edges)
        if self.aggregation_graph == "avg":
            graph_embeddings = embeddings.mean(1)
        elif self.aggregation_graph == "sum":
            graph_embeddings = embeddings.sum(1)
        else:
            assert self.aggregation_graph == "max", f"Unsupported aggregation: {self.aggregation_graph}"
            graph_embeddings = embeddings.max(1)[0]
        return self.value_head(graph_embeddings)


__all__ = [
    "CriticNetwork",
    "LegacyCriticNetwork",
    "create_critic_from_actor",
]
