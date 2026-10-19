"""none.py module.

Attributes:
    NoBaseline: No baseline (vanilla REINFORCE).

Example:
    >>> from logic.src.pipeline.learning.common.baselines import NoBaseline
    >>> baseline = NoBaseline()
    >>> baseline.eval()
    tensor(0.0)
"""

from typing import Any, Optional

import torch
from tensordict import TensorDict

from .base import Baseline


class NoBaseline(Baseline):
    """No baseline (vanilla REINFORCE).

    Attributes:
        None
    """

    def __init__(self, **kwargs):
        """Initialize NoBaseline.

        Args:
            kwargs: Additional arguments (unused).
        """
        super().__init__()

    def eval(self, td: TensorDict, reward: torch.Tensor, env: Optional[Any] = None) -> torch.Tensor:  # type: ignore[override]
        """
        Return zero baseline (no variance reduction).

        Args:
            td: TensorDict with environment state (unused).
            reward: Current batch rewards.
            env: Environment (unused).

        Returns:
            torch.Tensor: Zeros matching the reward shape.
        """
        return torch.zeros_like(reward)
