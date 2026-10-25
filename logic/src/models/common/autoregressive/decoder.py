"""Autoregressive Decoder base module.

This module provides the abstract base class for decoders that construct
solutions step-by-step by sequentially selecting actions based on
the current state and encoded embeddings.

Attributes:
    AutoregressiveDecoder: Base class for autoregressive decoders.

Example:
    >>> decoder = AutoregressiveDecoder()
    >>> log_p, actions = decoder(td, embeddings, env)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Union

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase


class AutoregressiveDecoder(nn.Module, ABC):
    """Base class for autoregressive decoders.

    AR decoders select actions sequentially until a termination condition is met,
    using the previous selection and environment state to inform the next.

    Attributes:
        embed_dim: Dimensionality of the input embeddings.
    """

    def __init__(self, embed_dim: int = 128, **kwargs: Any) -> None:
        """Initialize the AutoregressiveDecoder.



        Args:
            embed_dim: Internal dimensionality for decoding features.
            kwargs: Additional parameters passed to the parent Module.
        """
        super().__init__()
        self.embed_dim = embed_dim

    @abstractmethod
    def forward(
        self,
        td: TensorDict,
        embeddings: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        env: Optional[RL4COEnvBase] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct solutions step-by-step.

        Args:
            td: TensorDict containing the current problem/environment state.
            embeddings: Encoded state representation(s) from the encoder.
            env: Environment object used for state transitions and masking.
            kwargs: Control parameters for decoding (e.g., strategy, mask).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - log_p (torch.Tensor): Log-likelihood for the predicted actions.
                - actions (torch.Tensor): Sequence of selected node indices.
        """
        raise NotImplementedError
