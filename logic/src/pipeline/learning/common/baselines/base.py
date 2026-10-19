"""
Base abstract class and simple baseline implementations.

Attributes:
    Baseline: Abstract base class for baselines.

Example:
    None
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.data.datasets import BaselineDataset
from logic.src.interfaces import ITensorDictLike
from logic.src.tracking.logging.pylogger import get_pylogger

logger = get_pylogger(__name__)


class Baseline(nn.Module, ABC):
    """Base class for baselines.

    Attributes:
        None
    """

    def __init__(self):
        """Initialize the baseline."""
        super().__init__()

    @abstractmethod
    def eval(self, td: TensorDict, reward: torch.Tensor, env: Optional[Any] = None) -> torch.Tensor:  # type: ignore[override]
        """Compute baseline value.

        Args:
            td: TensorDict with problem instance data.
            reward: Reward tensor.
            env: Environment for data generation.

        Returns:
            Baseline value.
        """
        raise NotImplementedError

    def unwrap_batch(self, batch: Any) -> Tuple[Any, Optional[torch.Tensor]]:
        """Unwrap the batch if it's wrapped with baseline values.

        Args:
            batch: Wrapped batch.

        Returns:
            Tuple: Unwrapped batch data and optional baseline.
        """
        if isinstance(batch, ITensorDictLike) and "data" in list(batch.keys()) and "baseline" in list(batch.keys()):
            return batch["data"], batch["baseline"]
        return batch, None

    def unwrap_dataset(self, dataset: Any) -> Any:
        """Unwrap the dataset if it's wrapped.

        Args:
            dataset: Wrapped dataset.

        Returns:
            Unwrapped dataset.
        """
        if isinstance(dataset, BaselineDataset):
            return dataset.dataset
        return dataset

    def epoch_callback(
        self,
        policy: nn.Module,
        epoch: int,
        val_dataset: Optional[Any] = None,
        env: Optional[Any] = None,
    ):
        """Optional callback at epoch end.

        Args:
            policy: RL policy module.
            epoch: Epoch number.
            val_dataset: Validation dataset.
            env: RL environment.
        """
        pass

    def wrap_dataset(
        self,
        dataset: Any,
        policy: Optional[nn.Module] = None,
        env: Optional[Any] = None,
    ) -> Any:
        """Optional wrap dataset.

        Args:
            dataset: Dataset to wrap.
            policy: RL policy module.
            env: RL environment.

        Returns:
            Wrapped dataset.
        """
        return dataset

    def setup(self, policy: nn.Module):
        """Optional setup with policy reference.

        Args:
            policy: RL policy module.
        """
        pass

    def get_learnable_parameters(self) -> list:
        """Get learnable parameters for the optimizer.

        Returns:
            List of learnable parameters.
        """
        return []
