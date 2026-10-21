"""
Greedy decoding strategy implementation.

Attributes:
    Greedy: Greedy decoding strategy.

Example:
    >>> from logic.src.utils.decoding import Greedy
    >>> strategy = Greedy()
    >>> strategy.step(torch.tensor([[0.1, 0.2], [0.3, 0.4]]), torch.tensor([[True, True], [True, True]]))
    (tensor([1, 1]), tensor([0.2, 0.4]), tensor([0.2, 0.4]))
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from logic.src.constants.models import NUMERICAL_EPSILON

from .base import DecodingStrategy


class Greedy(DecodingStrategy):
    """
    Greedy decoding: always select the action with highest probability.

    Attributes:
        temperature: Temperature for sampling.
        top_k: Top-k filtering.
        top_p: Top-p filtering.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize Greedy decoding.

        Args:
            kwargs: Passed to super class.
        """
        # Greedy ignores temperature and sampling parameters
        super().__init__(temperature=1.0, top_k=None, top_p=None, **kwargs)

    def step(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        td: Optional[TensorDict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select action with highest probability.

        Args:
            logits: Logits tensor [batch, num_nodes]
            mask: Mask tensor [batch, num_nodes]
            td: TensorDict (unused)

        Returns:
            Tuple of (action, log_prob, entropy)
        """
        logits = self._process_logits(logits, mask)
        probs = F.softmax(logits, dim=-1)

        action = probs.argmax(dim=-1)
        eps = 1e-4 if probs.dtype == torch.float16 else NUMERICAL_EPSILON
        log_prob = torch.log(probs.gather(1, action.unsqueeze(-1)) + eps).squeeze(-1)

        # Entropy for greedy is 0 since it's deterministic
        entropy = torch.zeros_like(log_prob)

        return action, log_prob, entropy
