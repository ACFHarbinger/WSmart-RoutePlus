"""
Base class for decoding strategies.

Attributes:
    DecodingStrategy: Base class for all decoding strategies.

Example:
    >>> from logic.src.utils.decoding import DecodingStrategy
    >>> strategy = DecodingStrategy()
    >>> strategy.step(torch.tensor([[0.1, 0.2], [0.3, 0.4]]), torch.tensor([[True, True], [True, True]]))
    (tensor([0, 0]), tensor([0.1, 0.3]), tensor([0.1, 0.3]))
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, cast

import torch
from tensordict import TensorDict

from .decoding_utils import (
    batchify,
    top_k_filter,
    top_p_filter,
    unbatchify,
)


class DecodingStrategy(ABC):
    """
    Abstract base class for decoding strategies.

    Decoding strategies control how actions are selected from
    the logits produced by the policy decoder.

    Attributes:
        temperature (float): Temperature for softmax scaling. Higher = more random.
        top_k (Optional[int]): Keep only top-k logits before sampling.
        top_p (Optional[float]): Keep smallest set of logits with cumsum >= top_p (nucleus sampling).
        tanh_clipping (float): Apply tanh clipping to logits (Bello et al., 2016).
        mask_logits (bool): Whether to apply masking to invalid actions.
        multistart (bool): Whether to use multiple starting points.
        num_starts (int): Number of starting points for multistart.
        select_best (bool): Whether to select best solution from multiple starts.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        tanh_clipping: float = 0.0,
        mask_logits: bool = True,
        multistart: bool = False,
        num_starts: int = 1,
        select_best: bool = False,
        **kwargs,
    ):
        """
        Initialize DecodingStrategy.

        Args:
            temperature: Temperature for softmax scaling. Higher = more random.
            top_k: Keep only top-k logits before sampling.
            top_p: Keep smallest set of logits with cumsum >= top_p (nucleus sampling).
            tanh_clipping: Apply tanh clipping to logits (Bello et al., 2016).
            mask_logits: Whether to apply masking to invalid actions.
            multistart: Whether to use multiple starting points.
            num_starts: Number of starting points for multistart.
            select_best: Whether to select best solution from multiple starts.
            kwargs: Additional arguments.
        """
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.tanh_clipping = tanh_clipping
        self.mask_logits = mask_logits
        self.multistart = multistart
        self.num_starts = num_starts
        self.select_best = select_best

    @abstractmethod
    def step(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        td: Optional[TensorDict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select action from logits.

        Args:
            logits: Action logits [batch, num_nodes]
            mask: Valid action mask [batch, num_nodes]
            td: Optional TensorDict with additional state

        Returns:
            Tuple of (action, log_prob, entropy)
        """
        raise NotImplementedError

    def pre_decoder_hook(
        self,
        td: TensorDict,
        env: Any,
    ) -> Tuple[TensorDict, Any, int]:
        """
        Hook called before decoding starts.

        Used for multistart expansion.

        Args:
            td: TensorDict with problem instance
            env: Environment

        Returns:
            Tuple of (td, env, num_starts)
        """
        num_starts = 1
        if self.multistart and self.num_starts > 1:
            td = batchify(td, self.num_starts)
            num_starts = self.num_starts
        return td, env, num_starts

    def post_decoder_hook(
        self,
        td: TensorDict,
        env: Any,
        log_likelihood: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict, Any]:
        """
        Hook called after decoding completes.

        Used for best selection from multistart.

        Args:
            td: TensorDict with final state
            env: Environment
            log_likelihood: Log probabilities [batch, ...]
            actions: Selected actions [batch, seq_len]

        Returns:
            Tuple of (log_likelihood, actions, td, env)
        """
        if self.select_best and self.multistart:
            log_likelihood, actions, td = self._select_best(td, log_likelihood, actions, self.num_starts)
        return log_likelihood, actions, td, env

    def _select_best(
        self,
        td: TensorDict,
        log_likelihood: torch.Tensor,
        actions: torch.Tensor,
        num_starts: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        """
        Select best solution from multiple starts.

        Args:
            td: TensorDict with final state
            log_likelihood: Log probabilities [batch, ...]
            actions: Selected actions [batch, seq_len]
            num_starts: Number of starting points

        Returns:
            Tuple of (log_likelihood, actions, td)
        """
        reward = td.get("reward", None)
        if reward is None:
            # Use log_likelihood as proxy
            reward = log_likelihood

        # Reshape to [batch // num_starts, num_starts]
        batch_size = reward.shape[0] // num_starts
        reward = reward.view(batch_size, num_starts)

        # Get best indices [batch_size]
        best_idx = reward.argmax(dim=1)

        # Gather best
        actions = actions.view(batch_size, num_starts, -1)
        # best_idx is [batch_size], we need [batch_size, 1, seq_len]
        gather_idx = best_idx.view(batch_size, 1, 1).expand(-1, -1, actions.shape[-1])
        actions = actions.gather(1, gather_idx)
        actions = actions.squeeze(1)

        log_likelihood = log_likelihood.view(batch_size, num_starts)
        log_likelihood = log_likelihood.gather(1, best_idx.unsqueeze(-1)).squeeze(-1)

        # Unbatchify td
        td = cast(TensorDict, unbatchify(td, num_starts))

        return log_likelihood, actions, td

    def _process_logits(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process logits with masking, clipping, and temperature.

        Args:
            logits: Raw logits [batch, num_nodes]
            mask: Valid action mask [batch, num_nodes]

        Returns:
            Processed logits.
        """
        # Apply tanh clipping (Bello et al., 2016)
        if self.tanh_clipping > 0:
            logits = self.tanh_clipping * torch.tanh(logits)

        # Apply mask
        if self.mask_logits:
            logits = logits.masked_fill(~mask, float("-inf"))

        # Apply temperature
        if self.temperature != 1.0:
            logits = logits / self.temperature

        # Apply top-k filtering
        if self.top_k is not None and self.top_k > 0:
            logits = top_k_filter(logits, self.top_k)

        # Apply top-p (nucleus) filtering
        if self.top_p is not None and 0 < self.top_p < 1:
            logits = top_p_filter(logits, self.top_p)

        return logits
