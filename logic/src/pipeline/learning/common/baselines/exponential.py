"""
Exponential moving average baseline.

Attributes:
    ExponentialBaseline: Exponential moving average baseline.

Example:
    >>> from logic.src.pipeline.learning.common.baselines import ExponentialBaseline
    >>> baseline = ExponentialBaseline()
    >>> baseline.eval()
    tensor(0.0)
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from tensordict import TensorDict

from .base import Baseline


class ExponentialBaseline(Baseline):
    """
    Moving average (exponential) baseline.

    Attributes:
        beta: Decay rate of the exponential moving average.
        running_mean: Running mean of the rewards.
    """

    def __init__(self, beta: float = 0.8, exp_beta: Optional[float] = None, **kwargs):
        """
        Initialize ExponentialBaseline.

        Args:
            beta: Decay factor for exponential moving average.
            exp_beta: Alternative name for beta (matching RLConfig).
            kwargs: Additional arguments.
        """
        super().__init__()
        self.beta = exp_beta if exp_beta is not None else beta
        self.running_mean: Optional[torch.Tensor] = None

    def eval(self, td: Optional[TensorDict], reward: torch.Tensor, env: Optional[Any] = None) -> torch.Tensor:  # type: ignore[override]
        """
        Compute baseline value using exponential moving average.

        Args:
            td: TensorDict with environment state.
            reward: Current batch rewards.
            env: Environment (unused).

        Returns:
            torch.Tensor: Baseline value expanded to match reward shape.
        """
        if self.running_mean is None:
            self.running_mean = reward.mean().detach()
        else:
            self.running_mean = self.beta * self.running_mean + (1 - self.beta) * reward.mean().detach()
        return self.running_mean.expand_as(reward)
