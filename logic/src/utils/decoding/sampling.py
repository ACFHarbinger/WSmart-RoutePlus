"""
Sampling decoding strategy implementation.

Attributes:
    Sampling: Sampling decoding strategy.

Example:
    >>> from logic.src.utils.decoding import Sampling
    >>> strategy = Sampling()
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


class Sampling(DecodingStrategy):
    """
    Sampling decoding: sample from the probability distribution.

    Attributes:
        temperature: Temperature for sampling.
        top_k: Top-k filtering.
        top_p: Top-p filtering.
    """

    def step(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        td: Optional[TensorDict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from distribution.

        Args:
            logits: Logits tensor [batch, num_nodes]
            mask: Mask tensor [batch, num_nodes]
            td: TensorDict (unused)

        Returns:
            Tuple of (action, log_prob, entropy)
        """
        logits = self._process_logits(logits, mask)
        if logits.dtype in (torch.float16, torch.bfloat16):
            # Half-precision epsilons either underflow or materially bias scores.
            # Work in log space in fp32; masked entries retain exactly zero mass.
            log_probs = F.log_softmax(logits.float(), dim=-1)
            probs = log_probs.exp()
            action = torch.multinomial(probs, num_samples=1).squeeze(-1)
            log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
            finite_logs = log_probs.masked_fill(~torch.isfinite(log_probs), 0.0)
            entropy = -(probs * finite_logs).sum(dim=-1)
            return action, log_prob, entropy

        probs = F.softmax(logits, dim=-1)

        # Ensure masked/invalid nodes have exactly zero probability
        if self.mask_logits and mask is not None:
            probs = probs.masked_fill(~mask, 0.0)

        probs_sum = probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        probs = probs / probs_sum

        action = torch.multinomial(probs, num_samples=1).squeeze(-1)
        eps = 1e-4 if probs.dtype == torch.float16 else NUMERICAL_EPSILON
        log_prob = torch.log(probs.gather(1, action.unsqueeze(-1)) + eps).squeeze(-1)

        # Calculate entropy = -sum(p * log(p)) over non-zero probabilities
        entropy = -(torch.where(probs > 0, probs * torch.log(probs.clamp(min=eps)), torch.zeros_like(probs))).sum(
            dim=-1
        )

        return action, log_prob, entropy
