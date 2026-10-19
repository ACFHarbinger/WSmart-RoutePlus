"""
Modular loss functions for imitation learning and policy optimization.

This module provides reusable loss functions that can be composed
to build custom training objectives for RL and IL algorithms.

Attributes:
    js_divergence_loss: JS divergence loss function.
    kl_divergence_loss: KL divergence loss function.
    nll_loss: Negative log-likelihood loss function.
    reverse_kl_divergence_loss: Reverse KL divergence loss function.
    weighted_nll_loss: Weighted negative log-likelihood loss function.

Example:
    >>> from logic.src.pipeline.learning.core.losses import js_divergence_loss, kl_divergence_loss
    >>> js_divergence_loss
    <function js_divergence_loss at 0x...>
    >>> kl_divergence_loss
    <function kl_divergence_loss at 0x...>
"""

from .js_divergence_loss import js_divergence_loss
from .kl_divergence_loss import kl_divergence_loss, reverse_kl_divergence_loss
from .nll_loss import nll_loss
from .weighted_nll_loss import weighted_nll_loss

__all__ = [
    "js_divergence_loss",
    "kl_divergence_loss",
    "nll_loss",
    "reverse_kl_divergence_loss",
    "weighted_nll_loss",
]
