"""Mixture of Experts Attention Model.

This module provides `MoEAttentionModel`, which inherits from the base
`AttentionModel` but injects a `MoEComponentFactory` to instantiate
sparse-routed expert layers within the transformer architectures.

Attributes:
    MoEAttentionModel: Capacity-boosted attention model with expert routing.

Example:
    >>> model = MoEAttentionModel(embed_dim=128, hidden_dim=512, problem=env.problem)
    >>> out = model(td)
"""

from __future__ import annotations

from typing import Any, Optional, cast

from logic.src.models.core.attention_model import AttentionModel
from logic.src.models.subnets.factories import MoEComponentFactory


class MoEAttentionModel(AttentionModel):
    """Attention Model with sparse Mixture of Experts (MoE).

    Increases model capacity while maintaining sparse activation patterns. This
    leads to better task-specific specialization in heterogeneous data
    distributions.

    Attributes:
        total_experts (int): Accessor for the active expert population count.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        problem: Any,
        n_encode_layers: int = 2,
        n_encode_sublayers: Optional[int] = None,
        n_decode_layers: Optional[int] = None,
        dropout_rate: float = 0.1,
        normalization: str = "batch",
        n_heads: int = 8,
        num_experts: int = 4,
        k: int = 2,
        noisy_gating: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initializes the MoE Attention Model.

        Args:
            embed_dim: Dimensionality of latent embeddings.
            hidden_dim: Dimensionality of hidden layers.
            problem: Environment or problem logic wrapper.
            n_encode_layers: Number of transformer encoder layers.
            n_encode_sublayers: Optional internal encoder depth.
            n_decode_layers: Optional decoder layers.
            dropout_rate: Dropout probability for regularization.
            normalization: Type of layer normalization.
            n_heads: Number of attention heads.
            num_experts: Total number of experts in the MoE layer.
            k: Number of experts to activate per token (top-k).
            noisy_gating: Whether to use noise in the gating network.
            kwargs: Additional keyword arguments.
        """
        # Create the MoE Factory to build expert-enabled subnets
        component_factory = MoEComponentFactory(num_experts=num_experts, k=k, noisy_gating=noisy_gating)

        super().__init__(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            problem=problem,
            component_factory=component_factory,
            n_encode_layers=n_encode_layers,
            n_encode_sublayers=n_encode_sublayers,
            n_decode_layers=n_decode_layers,
            dropout_rate=dropout_rate,
            normalization=normalization,
            n_heads=n_heads,
            **kwargs,
        )

    @property
    def total_experts(self) -> int:
        """Retrieves the global count of experts active in the encoder stack.

        Returns:
            int: Number of expert modules.
        """
        return cast(int, self.encoder.num_experts if hasattr(self.encoder, "num_experts") else 0)
