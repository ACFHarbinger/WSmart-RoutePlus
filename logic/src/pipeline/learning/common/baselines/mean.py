"""mean.py module.

Attributes:
    MeanBaseline: Mean baseline.

Example:
    >>> from logic.src.pipeline.learning.common.baselines import MeanBaseline
    >>> baseline = MeanBaseline()
    >>> baseline.eval()
    tensor(0.0)
"""

from typing import Any, Optional

import torch
from tensordict import TensorDict

from .base import Baseline, global_batch_mean


class MeanBaseline(Baseline):
    """
    Simple batch-mean baseline.

    Uses the mean reward of the current batch as the baseline value.
    Zero computational overhead but limited variance reduction.
    Useful as a simple default or for debugging.

    Attributes:
        None
    """

    def __init__(self, **kwargs):
        """Initialize MeanBaseline.

        Args:
            kwargs: Description of kwargs.
        """
        super().__init__()

    def eval(self, td: TensorDict, reward: torch.Tensor, env: Optional[Any] = None) -> torch.Tensor:  # type: ignore[override]
        """
        Compute baseline as mean of current batch rewards.

        Args:
            td: TensorDict with environment state (unused).
            reward: Current batch rewards.
            env: Environment (unused).

        Returns:
            torch.Tensor: Mean reward expanded to match reward shape.
        """
        # The full batch, across devices when training on several.
        return (
            global_batch_mean(reward, dim=0).expand_as(reward)
            if reward.dim() > 1
            else global_batch_mean(reward).expand_as(reward)
        )
