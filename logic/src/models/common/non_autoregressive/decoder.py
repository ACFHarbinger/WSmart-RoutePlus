"""Non-autoregressive Decoder base module.

This module provides the abstract base class for decoders that construct
solutions from pre-computed edge or node heatmaps, typical in one-shot
prediction models.

Attributes:
    NonAutoregressiveDecoder: Abstract base class for heatmap-based decoders.

Example:
    >>> from logic.src.models.common.non_autoregressive.decoder import NonAutoregressiveDecoder
    >>> # subclass and implement forward...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase


class NonAutoregressiveDecoder(nn.Module, ABC):
    """Base class for non-autoregressive decoders.

    NAR decoders construct solutions from heatmaps using various methods
    like Ant Colony Optimization, search heuristics, greedy decoding,
    or stochastic sampling.

    Attributes:
        kwargs (Dict[str, Any]): Additional parameters for the decoder.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initializes the NonAutoregressiveDecoder.

        Args:
            kwargs: Additional configuration parameters for the decoder.
        """
        super().__init__()

    @abstractmethod
    def forward(
        self,
        td: TensorDict,
        heatmap: torch.Tensor,
        env: RL4COEnvBase,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Produces logits and mask for the current construction step.

        Args:
            td: TensorDict with current environment state.
            heatmap: Pre-computed edge probability matrix.
            env: Environment providing step logic and constraints.
            kwargs: Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - logits: Predicted action logits.
                - mask: Valid action mask for the current state.
        """
        raise NotImplementedError

    def construct(
        self,
        td: TensorDict,
        heatmap: torch.Tensor,
        env: RL4COEnvBase,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Performs full solution construction from the provided heatmap.

        Subclasses should override this method to implement specific routing
        algorithms (e.g., ACO, Guided Search).

        Args:
            td: TensorDict with initial environment state.
            heatmap: Pre-computed edge probability matrix.
            env: Environment providing step logic.
            kwargs: Additional keyword arguments.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of construction results,
                typically including 'actions', 'log_likelihood', and 'reward'.
        """
        return {}
