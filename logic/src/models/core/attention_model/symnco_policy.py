"""SymNCO Policy with projection head.

This module implements the policy for SymNCO (Kim et al. 2022), which extends
the standard Attention Model with a projection head to support symbolic
invariance and augmentation-based training objectives.

Attributes:
    SymNCOPolicy: Specialized policy with embedding projection support.

Example:
    >>> from logic.src.models.core.attention_model.symnco_policy import SymNCOPolicy
    >>> policy = SymNCOPolicy(embed_dim=128, use_projection_head=True)
    >>> out = policy(td, env)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.models.core.attention_model.policy import AttentionModelPolicy


class SymNCOPolicy(AttentionModelPolicy):
    """SymNCO Policy variant for contrastive/invariant RL.

    Enhances the Attention Model with a projection MLP that maps initial
    embeddings into a space suitable for invariance losses, while maintaining
    standard autoregressive construction capabilities.

    Attributes:
        use_projection_head (bool): Flag to enable the invariance projection.
        projection_head (nn.Module): 3-layer MLP for embedding transformation.
    """

    def __init__(self, embed_dim: int = 128, use_projection_head: bool = True, **kwargs: Any) -> None:
        """Initializes the SymNCOPolicy.

        Args:
            embed_dim: Dimensionality of node features and latent projections.
            use_projection_head: Whether to include the invariance projection MLP.
            kwargs: Additional keyword arguments for AttentionModelPolicy.
        """
        super().__init__(embed_dim=embed_dim, **kwargs)
        self.use_projection_head = use_projection_head

        if self.use_projection_head:
            # RL4CO standard 3-layer MLP architecture
            self.projection_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )

    def forward(  # type: ignore[override]  # multi-start signature is wider than ConstructivePolicy.forward
        self,
        td: TensorDict,
        env: Any = None,
        strategy: str = "sampling",
        num_starts: int = 1,
        actions: Optional[torch.Tensor] = None,
        start_nodes: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Executes decoding and computes projection embeddings.

        Args:
            td: TensorDict containing problem instance data.
            env: Environment managing problem physics.
            strategy: Decoding strategy identifier (e.g., "greedy").
            num_starts: Number of parallel construction starts.
            actions: Optional pre-selected actions.
            kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: Policy outputs extended with `proj_embeddings`.
        """
        # Ensure initial embeddings are returned for the projection head
        kwargs["return_init_embeds"] = True
        out = super().forward(
            td,
            env,
            strategy=strategy,
            num_starts=num_starts,
            actions=actions,
            start_nodes=start_nodes,
            **kwargs,
        )

        if self.use_projection_head:
            # Project the raw features [batch, nodes, dim]
            init_embeds = out["init_embeds"]
            out["proj_embeddings"] = self.projection_head(init_embeds)

        return out
