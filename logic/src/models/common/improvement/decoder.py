"""Improvement Decoder base module.

This module provides the abstract base class for decoders that predict
local search moves or optimization operators to refine an existing solution.

Attributes:
    ImprovementDecoder: Abstract base class for all improvement decoders.

Example:
    >>> from logic.src.models.common.improvement.decoder import ImprovementDecoder
    >>> # subclass and implement forward...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple, Union

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase


class ImprovementDecoder(nn.Module, ABC):
    """Base class for improvement decoders.

    Improvement decoders analyze the current solution state and predicted
    graph features to decide on the next refinement move (e.g., node swap,
    re-insertion).

    Attributes:
        embed_dim (int): Dimensionality of the latent representation.
    """

    def __init__(self, embed_dim: int = 128, **kwargs: Any) -> None:
        """Initializes the ImprovementDecoder.



        Args:
            embed_dim: Dimensionality of latent embeddings.
            kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.embed_dim = embed_dim

    @property
    def device(self) -> torch.device:
        """Retrieves the device on which the model's parameters are stored.

        Returns:
            torch.device: The computing device ('cpu' or 'cuda').
        """
        return next(self.parameters()).device

    @abstractmethod
    def forward(
        self,
        td: TensorDict,
        embeddings: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        env: RL4COEnvBase,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts improvement moves for the current solution.



        Args:
            td: TensorDict containing problem state and solution context.
            embeddings: Encoded representations of nodes and tours.
            env: Environment providing valid move masks and logic.
            kwargs: Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - log_p: Log-likelihood of the selected moves.
                - actions: Selected refinement operators or node indices.

        """
        raise NotImplementedError
