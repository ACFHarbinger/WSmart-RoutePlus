"""Normalization layers (Batch, Layer, Instance, Group).

This module provides the Normalization wrapper, which unifies various PyTorch
normalization strategies into a single interface compatible with the project's
configuration system.

Attributes:
    Normalization: Unified wrapper for various normalization layers.

Example:
    >>> import torch
    >>> from logic.src.models.subnets.modules.normalization import Normalization
    >>> norm = Normalization(128, norm_name="layer")
    >>> x = torch.randn(1, 10, 128)
    >>> x_norm = norm(x)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from logic.src.configs.models.normalization import NormalizationConfig


class Normalization(nn.Module):
    """Wrapper for various Normalization layers.

    Supports Batch, Layer, Instance, Group, and Local Response Normalization.
    It automatically routes inputs to the appropriate underlying PyTorch module
    and handles tensor reshaping for compatibility with graph and sequence data.

    Attributes:
        normalizer (nn.Module): The underlying PyTorch normalization layer.
    """

    normalizer: nn.Module

    def __init__(
        self,
        embed_dim: int,
        norm_name: Optional[str] = None,
        eps_alpha: Optional[float] = None,
        learn_affine: Optional[bool] = None,
        track_stats: Optional[bool] = None,
        mbval: Optional[float] = None,
        n_groups: Optional[int] = None,
        kval: Optional[float] = None,
        bias: Optional[bool] = True,
        norm_config: Optional[NormalizationConfig] = None,
    ) -> None:
        """Initializes Normalization.

        Args:
            embed_dim: dimensionality of the input feature space.
            norm_name: Normalization strategy ('batch', 'layer', 'instance',
                'group', 'local_response').
            eps_alpha: Numerical stability epsilon (or alpha for LRNorm).
            learn_affine: Whether to learn gain and bias parameters.
            track_stats: Whether to track running mean/variance (for BN/IN).
            mbval: Momentum for stats or beta for LRNorm.
            n_groups: Number of groups for GroupNorm.
            kval: Constant k for LocalResponseNorm.
            bias: Whether to add a bias term in LayerNorm.
            norm_config: Pre-populated configuration object. Overrides other args if
                provided.
        """
        super().__init__()

        # Use norm_config if provided, otherwise create from individual args
        if norm_config is None:
            norm_config = NormalizationConfig(
                norm_type=norm_name if norm_name is not None else "batch",
                epsilon=eps_alpha if eps_alpha is not None else 1e-05,
                learn_affine=learn_affine if learn_affine is not None else True,
                track_stats=track_stats if track_stats is not None else False,
                momentum=mbval if mbval is not None else 0.1,
                n_groups=n_groups if n_groups is not None else 3,
                k_lrnorm=kval if kval is not None else 1.0,
            )

        norm_name = norm_config.norm_type
        eps_alpha = norm_config.epsilon
        learn_affine = norm_config.learn_affine
        track_stats = norm_config.track_stats
        mbval = norm_config.momentum
        n_groups = norm_config.n_groups
        kval = norm_config.k_lrnorm

        if norm_name == "instance":
            self.normalizer = nn.InstanceNorm1d(
                embed_dim,
                eps=eps_alpha,
                affine=learn_affine if learn_affine is not None else True,
                track_running_stats=track_stats if track_stats is not None else False,
                momentum=mbval if mbval is not None else 0.1,
            )
        elif norm_name == "batch":
            self.normalizer = nn.BatchNorm1d(
                embed_dim,
                eps=eps_alpha,
                affine=learn_affine if learn_affine is not None else True,
                track_running_stats=track_stats if track_stats is not None else False,
                momentum=mbval if mbval is not None else 0.1,
            )
        elif norm_name == "layer":
            self.normalizer = nn.LayerNorm(
                embed_dim,
                eps=eps_alpha,
                elementwise_affine=learn_affine if learn_affine is not None else True,
                bias=bias if bias is not None else True,
            )
        elif norm_name == "group":
            if n_groups is None or embed_dim % n_groups != 0:
                n_groups = 1
            self.normalizer = nn.GroupNorm(
                num_groups=n_groups,
                num_channels=embed_dim,
                eps=eps_alpha,
                affine=learn_affine if learn_affine is not None else True,
            )
        elif norm_name == "local_response":
            self.normalizer = nn.LocalResponseNorm(
                embed_dim,
                alpha=eps_alpha,
                beta=mbval if mbval is not None else 0.75,
                k=kval if kval is not None else 1.0,
            )
        else:
            raise ValueError(f"Unknown normalization method: {norm_name}")

        if learn_affine:
            self.init_parameters()

    def init_parameters(self) -> None:
        """Initializes the affine parameters (weights) using uniform distribution."""
        for param in self.parameters():
            if param.dim() > 0:
                stdv: float = 1.0 / math.sqrt(param.size(-1))
                param.data.uniform_(-stdv, stdv)

    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Normalizes the input tensor.

        Handles reshaping for 1D normalization layers (Batch, Instance, Group)
        automatically to support various sequence/graph shapes.

        Args:
            input: Tensor to normalize of shape (..., embed_dim).
            mask: Optional mask (currently ignored by standard norm layers).

        Returns:
            torch.Tensor: Normalized tensor with same shape as input.
        """
        if isinstance(self.normalizer, nn.BatchNorm1d):
            flat = input.view(-1, input.size(-1))
            if flat.size(0) == 1 and not self.normalizer.track_running_stats:
                if self.normalizer.affine and self.normalizer.weight is not None and self.normalizer.bias is not None:
                    return (flat * self.normalizer.weight + self.normalizer.bias).view(*input.size())
                return flat.view(*input.size())
            return self.normalizer(flat).view(*input.size())
        elif isinstance(self.normalizer, (nn.InstanceNorm1d, nn.GroupNorm)):
            # Normalize over embedding dimension (last)
            orig_shape = input.shape
            dims = input.dim()
            if dims > 3:
                # view as (B, N1*N2*..., D)
                curr = input.view(orig_shape[0], -1, orig_shape[-1])
                # permute to (B, D, N_total)
                curr = curr.permute(0, 2, 1)
                curr = self.normalizer(curr)
                # permute back and reshape
                return curr.permute(0, 2, 1).view(*orig_shape)
            elif dims == 3:
                return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                return self.normalizer(input)
        elif isinstance(self.normalizer, nn.LayerNorm):
            return self.normalizer(input)
        else:
            return self.normalizer(input)
