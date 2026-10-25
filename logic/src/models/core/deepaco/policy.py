"""DeepACO Policy for non-autoregressive construction.

This module provides the `DeepACOPolicy`, which utilizes a Graph Neural Network
to predict an edge heatmap (representation of pheromones/heuristics) and an
Ant Colony Optimization algorithm to sample solutions from this heatmap.

Attributes:
    DeepACOPolicy: Hybrid policy combining GNN heatmaps with ACO search.

Example:
    >>> from logic.src.models.core.deepaco.policy import DeepACOPolicy
    >>> policy = DeepACOPolicy(embed_dim=128, n_ants=20)
    >>> out = policy(td, env)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, cast

from tensordict import TensorDict

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.common.non_autoregressive.encoder import (
    NonAutoregressiveEncoder,
)
from logic.src.models.common.non_autoregressive.policy import (
    NonAutoregressivePolicy,
)
from logic.src.models.subnets.decoders.deepaco import ACODecoder
from logic.src.models.subnets.encoders.deepaco.encoder import DeepACOEncoder


class DeepACOPolicy(NonAutoregressivePolicy):
    """DeepACO inference policy.

    Orchestrates the two-stage execution:
    1. Edge prediction via GNN to generate a probability heatmap.
    2. Solution construction via parallel ACO ants using the predicted heatmap.

    Attributes:
        encoder (DeepACOEncoder): GNN that outputs edge weights.
        decoder (ACODecoder): ACO implementation for path construction and refinement.
    """

    def __init__(
        self,
        encoder: Optional[NonAutoregressiveEncoder] = None,
        decoder: Optional[ACODecoder] = None,
        embed_dim: int = 128,
        num_encoder_layers: int = 3,
        num_heads: int = 8,
        n_ants: int = 20,
        n_iterations: int = 1,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.1,
        use_local_search: bool = True,
        env_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the DeepACOPolicy.

        Args:
            encoder: Optional custom non-autoregressive encoder.
            decoder: Optional custom ACO decoder.
            embed_dim: Dimensionality of node features.
            num_encoder_layers: Number of Transformer/GNN layers.
            num_heads: Number of attention heads.
            n_ants: Number of ants per problem instance.
            n_iterations: Number of ACO pheromone update cycles.
            alpha: Pheromone importance exponent.
            beta: Heuristic visibility importance exponent.
            rho: Pheromone evaporation rate.
            use_local_search: Whether to apply refinement after construction.
            env_name: Name of the environment identifier.
            kwargs: Additional keyword arguments.
        """
        if encoder is None:
            encoder = DeepACOEncoder(
                embed_dim=embed_dim,
                num_layers=num_encoder_layers,
                num_heads=num_heads,
                **kwargs,
            )
        if decoder is None:
            decoder = ACODecoder(
                n_ants=n_ants,
                n_iterations=n_iterations,
                alpha=alpha,
                beta=beta,
                rho=rho,
                use_local_search=use_local_search,
            )
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            env_name=env_name,
            embed_dim=embed_dim,
            **kwargs,
        )

    def forward(
        self,
        td: TensorDict,
        env: RL4COEnvBase,
        num_starts: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Executes heatmap prediction followed by ACO solution search.

        Args:
            td: TensorDict containing problem instance data.
            env: Environment managing problem physics.
            num_starts: Number of parallel construction starts.
            kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: Output dictionary containing:
                - reward (torch.Tensor): best reward found by ants.
                - actions (torch.Tensor): best tour found.
                - log_likelihood (torch.Tensor): log-prob of construction.
                - heatmap (torch.Tensor): the predicted edge weights.
        """
        # stage 1: GNN Edge weight (heatmap) prediction
        encoder = cast(NonAutoregressiveEncoder, self.encoder)
        heatmap = encoder(td, **kwargs)

        # stage 2: solution sampling via ACO
        decoder = cast(ACODecoder, self.decoder)
        out = decoder.construct(td, heatmap, env, num_starts=num_starts, **kwargs)

        out["heatmap"] = heatmap
        return out
