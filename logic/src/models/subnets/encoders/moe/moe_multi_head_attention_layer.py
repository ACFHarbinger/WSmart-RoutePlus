"""MoE Multi-Head Attention Layer.

Attributes:
    MoEMultiHeadAttentionLayer: Single layer of the MoE Graph Attention Encoder.

Example:
    >>> from logic.src.models.subnets.encoders.moe.moe_multi_head_attention_layer import MoEMultiHeadAttentionLayer
    >>> layer = MoEMultiHeadAttentionLayer(n_heads=8, embed_dim=128, feed_forward_hidden=512, ...)
"""

from __future__ import annotations

from typing import List, Optional, Tuple, cast

from torch import nn

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.configs.models.normalization import NormalizationConfig
from logic.src.models.subnets.encoders.common import MultiHeadAttentionLayerBase
from logic.src.models.subnets.modules.connections import get_connection_module
from logic.src.models.subnets.modules.moe_feed_forward import MoEFeedForward


class MoEMultiHeadAttentionLayer(MultiHeadAttentionLayerBase):
    """Single layer of the MoE Graph Attention Encoder.

    Inherits from MultiHeadAttentionLayerBase but overrides the feed-forward
    component to use Mixture-of-Experts (MoE) routing instead of standard FFN.

    The MoE variant allows the model to specialize different experts for different
    input patterns, potentially improving capacity and expressiveness.

    Attributes:
        num_experts (int): Number of expert networks in MoE.
        k (int): Number of experts to activate per input (top-k routing).
        noisy_gating (bool): Whether to use noisy gating for exploration.
        activation (str): Activation function name.
        af_param (float): Activation function parameter.
        threshold (float): Activation threshold.
        replacement_value (float): Activation replacement value.
        n_params (int): Number of activation parameters.
        uniform_range (List[float]): Uniform range for activation.
    """

    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        feed_forward_hidden: int,
        normalization: str,
        epsilon_alpha: float,
        learn_affine: bool,
        track_stats: bool,
        mbeta: float,
        lr_k: float,
        n_groups: int,
        activation: str,
        af_param: float,
        threshold: float,
        replacement_value: float,
        n_params: int,
        uniform_range: List[float],
        connection_type: str = "skip",
        expansion_rate: int = 4,
        num_experts: int = 4,
        k: int = 2,
        noisy_gating: bool = True,
    ) -> None:
        """Initializes the MoEMultiHeadAttentionLayer.

        Args:
            n_heads: Number of attention heads.
            embed_dim: Embedding dimension.
            feed_forward_hidden: Hidden dimension for each expert's feed-forward network.
            normalization: Normalization type.
            epsilon_alpha: Epsilon for normalization stability.
            learn_affine: Whether to learn affine parameters in normalization.
            track_stats: Whether to track running stats in batch norm.
            mbeta: Momentum for batch norm running stats.
            lr_k: Local response normalization parameter.
            n_groups: Number of groups for group normalization.
            activation: Activation function name.
            af_param: Activation function parameter.
            threshold: Activation threshold.
            replacement_value: Activation replacement value.
            n_params: Number of activation parameters.
            uniform_range: Uniform range for activation.
            connection_type: Connection type ("skip", "dense", or "hyper").
            expansion_rate: Expansion factor for hyper-connections.
            num_experts: Number of expert networks in MoE.
            k: Number of experts to activate per input (top-k routing).
            noisy_gating: Whether to use noisy gating for exploration.
        """
        # Store MoE-specific parameters for _create_feed_forward()
        self.num_experts = num_experts
        self.k = k
        self.noisy_gating = noisy_gating
        self.activation = activation
        self.af_param = af_param
        self.threshold = threshold
        self.replacement_value = replacement_value
        self.n_params = n_params
        self.uniform_range = uniform_range

        # Create config objects
        norm_config = NormalizationConfig(
            norm_type=normalization,
            epsilon=epsilon_alpha,
            learn_affine=learn_affine,
            track_stats=track_stats,
            momentum=mbeta,
            k_lrnorm=lr_k,
            n_groups=n_groups,
        )

        activation_config = ActivationConfig(
            name=activation,
            param=af_param,
            threshold=threshold,
            replacement_value=replacement_value,
            n_params=n_params,
            range=uniform_range,
        )

        # Delegate to base class
        super(MoEMultiHeadAttentionLayer, self).__init__(
            n_heads=n_heads,
            embed_dim=embed_dim,
            feed_forward_hidden=feed_forward_hidden,
            norm_config=norm_config,
            activation_config=activation_config,
            connection_type=connection_type,
            expansion_rate=expansion_rate,
        )

    def _create_feed_forward(
        self,
        embed_dim: int,
        feed_forward_hidden: int,
        activation_config: ActivationConfig,
        connection_type: str,
        expansion_rate: int,
    ) -> nn.Module:
        """Override to use MoEFeedForward instead of standard feed-forward.

        Args:
            embed_dim: Embedding dimension.
            feed_forward_hidden: Hidden dimension for each expert.
            activation_config: Activation function configuration (already stored in self).
            connection_type: Connection type wrapper.
            expansion_rate: Expansion factor for hyper-connections.

        Returns:
            nn.Module: MoEFeedForward module wrapped with connection.
        """
        return get_connection_module(
            module=MoEFeedForward(
                embed_dim,
                feed_forward_hidden,
                self.activation,
                self.af_param,
                self.threshold,
                self.replacement_value,
                self.n_params,
                cast(Optional[Tuple[float, float]], self.uniform_range),
                num_experts=self.num_experts,
                k=self.k,
                noisy_gating=self.noisy_gating,
            ),
            embed_dim=embed_dim,
            connection_type=connection_type,
            expansion_rate=expansion_rate,
        )
