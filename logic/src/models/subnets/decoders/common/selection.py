"""Node selection utilities for autoregressive decoders.

This module provides unified action selection logic used across multiple
decoder implementations to consolidate redundant greedy and sampling logic.

Attributes:
    select_action: Select action using greedy or sampling strategy.
    select_action_log_prob: Select action from log-probability distribution.

Example:
    >>> from logic.src.models.subnets.decoders.common.selection import select_action
    >>> probs = torch.tensor([[0.1, 0.7, 0.2]])
    >>> action = select_action(probs, strategy="greedy")
"""

from __future__ import annotations

from typing import Optional

import torch

from logic.src.constants.models import NUMERICAL_EPSILON


def select_action(
    probs: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    strategy: str = "greedy",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Select action from probability distribution using greedy or sampling strategy.

    This function is used in autoregressive decoders to select the next node
    to visit based on the attention-derived probability distribution over nodes.

    Args:
        probs: Probability distribution over actions/nodes.
            Shape: (batch_size, num_nodes) or (batch_size, num_actions).
        mask: Boolean mask indicating valid actions.
            Shape: (batch_size, num_nodes). True = valid, False = invalid.
        strategy: Selection strategy: "greedy" or "sampling".
        generator: Generator for reproducible sampling.

    Returns:
        torch.Tensor: Selected action indices of shape (batch_size,).
    """
    # Apply mask if provided
    if mask is not None:
        # Zero out invalid actions
        probs = probs.masked_fill(~mask, 0.0)
        # Renormalize to valid probability distribution
        probs = probs / (probs.sum(dim=-1, keepdim=True) + NUMERICAL_EPSILON)

    # Select action based on strategy
    if strategy == "greedy":
        # Greedy: Select action with highest probability
        return probs.argmax(dim=-1)
    else:
        # Sampling: Sample from probability distribution
        # Clamp to prevent numerical issues in multinomial
        probs = probs.clamp(min=NUMERICAL_EPSILON)
        return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def select_action_log_prob(
    log_probs: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    strategy: str = "greedy",
) -> torch.Tensor:
    """Select action from log-probability distribution.

    Convenience wrapper around select_action() that accepts log-probabilities
    instead of probabilities. Useful when working with log-softmax outputs.

    Args:
        log_probs: Log-probability distribution over actions.
            Shape: (batch_size, num_nodes).
        mask: Boolean mask indicating valid actions.
        strategy: Selection strategy: "greedy" or "sampling".

    Returns:
        torch.Tensor: Selected action indices of shape (batch_size,).
    """
    # Convert log-probabilities to probabilities
    probs = log_probs.exp()
    return select_action(probs, mask=mask, strategy=strategy)
