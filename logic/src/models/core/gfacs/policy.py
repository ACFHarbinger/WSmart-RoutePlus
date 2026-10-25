"""GFACS Policy: GFlowNet Ant Colony System.

This module implements the `GFACSPolicy`, which integrates a GFlowNet-style
learnable partition function (`logZ`) into the DeepACO framework. This enabling
Trajectory Balance loss training while maintaining the construction benefits
of Ant Colony System.

Attributes:
    GFACSPolicy: Neural GFlowNet policy with ACO path construction.

Example:
    >>> from logic.src.models.core.gfacs.policy import GFACSPolicy
    >>> policy = GFACSPolicy(env_name="tsp")
    >>> out = policy(td, my_env)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.core.deepaco.policy import DeepACOPolicy
from logic.src.models.subnets.decoders.deepaco import ACODecoder
from logic.src.models.subnets.encoders.gfacs.encoder import GFACSEncoder


class GFACSPolicy(DeepACOPolicy):
    """GFACS (GFlowNet Ant Colony System) Policy.

    Extends `DeepACOPolicy` by injecting a learnable scalar `logZ` representing
    the log partition function of the flow network. This parameter is essential
    for satisfying the Trajectory Balance condition during training.

    Attributes:
        logZ (nn.Parameter): Learnable scalar for the global flow normalization.
        encoder (GFACSEncoder): GNN-based problem encoder.
        decoder (ACODecoder): Probabilistic path construction module.
    """

    def __init__(
        self,
        encoder: Optional[GFACSEncoder] = None,
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
        """Initializes the GFACSPolicy.

        Args:
            encoder: Optional GFlowNet problem encoder.
            decoder: Optional ACO path decoder.
            embed_dim: Dimensionality of latent embeddings.
            num_encoder_layers: Number of encoder processing layers.
            num_heads: Number of attention heads.
            n_ants: Number of concurrent agents for solution discovery.
            n_iterations: Number of sequential ACO updates per forward pass.
            alpha: Pheromone importance factor.
            beta: Heuristic importance factor.
            rho: Pheromone evaporation rate.
            use_local_search: Whether to refine solutions via local search.
            env_name: Name of the environment identifier.
            kwargs: Additional keyword arguments.
        """
        if encoder is None:
            encoder = GFACSEncoder(
                embed_dim=embed_dim,
                num_layers=num_encoder_layers,
                num_heads=num_heads,
                **kwargs,
            )

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            n_ants=n_ants,
            n_iterations=n_iterations,
            alpha=alpha,
            beta=beta,
            rho=rho,
            use_local_search=use_local_search,
            env_name=env_name,
            **kwargs,
        )

        # GFlowNet partition scale parameter
        self.logZ = nn.Parameter(torch.tensor(0.0))

    def forward(  # type: ignore[override]
        self,
        td: TensorDict,
        env: RL4COEnvBase,
        num_starts: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Performs constructive sampling with flow normalization.

        Args:
            td: TensorDict containing problem instance data.
            env: Environment managing problem physics.
            num_starts: Number of parallel construction starts.
            kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: map with 'actions', 'reward', 'log_likelihood', and 'logZ'.
        """
        # Ensure all ants are processed for TB loss integrity
        if "return_all" not in kwargs:
            kwargs["return_all"] = True

        out = super().forward(td, env, num_starts=num_starts, **kwargs)

        # Inject global flow constant into result dictionary
        out["logZ"] = self.logZ
        if "ls_actions" in out:
            out["ls_logZ"] = self.logZ
        return out
