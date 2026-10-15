"""
Base class for evaluation strategies.

Attributes:
    EvalBase: Base class for evaluation strategies.

Example:
    >>> evaluator = EvalBase(env)
    >>> evaluator(policy, data_loader)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader


class EvalBase(ABC):
    """Base class for evaluation strategies.

    Attributes:
        None
    """

    def __init__(self, env: Any, progress: bool = True, device: str | torch.device = "cpu", **kwargs):
        """Initialize Class.

        Args:
            env: Description of env.
            progress: Description of progress.
            device: Description of device.
            kwargs: Description of kwargs.
        """
        self.env = env
        self.progress = progress
        self.device = device

    @abstractmethod
    def __call__(self, policy: Any, data_loader: DataLoader, return_results: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Evaluate policy on dataset.

        Args:
            policy: Description of policy.
            data_loader: Description of data_loader.
            return_results: Description of return_results.
            kwargs: Description of kwargs.

        Returns:
            Dict with metrics (reward, etc.)
        """
        pass
