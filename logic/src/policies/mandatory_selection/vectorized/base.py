"""Base selector module.

This module defines the abstract interface for vectorized bin selection
strategies, including an instrumentation mechanism for automatic telemetry
recording across all subclasses.

Attributes:
    VectorizedSelector: Abstract interface for batch-wise bin selection.

Example:
    >>> class MySelector(VectorizedSelector):
    >>>     def select(self, fill_levels, **kwargs):
    >>>         return fill_levels > 0.8
"""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from typing import Any

import torch

from logic.src.tracking.viz_mixin import PolicyVizMixin


class VectorizedSelector(PolicyVizMixin, ABC):
    """Abstract base class for vectorized bin selection strategies.

    This class serves as the foundation for all selection heuristics that
    determine which bins should be collected based on their current fill levels
    and other environmental features.

    Attributes:
        None
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-wrap ``select`` in every concrete subclass to record telemetry.

        Args:
            kwargs: Class creation arguments passed to super.

        Returns:
            None.
        """
        super().__init_subclass__(**kwargs)
        if "select" in cls.__dict__:
            original = cls.__dict__["select"]

            @functools.wraps(original)
            def _instrumented(self: Any, fill_levels: torch.Tensor, **kw: Any) -> torch.Tensor:
                mask = original(self, fill_levels, **kw)
                day = kw.get("current_day")
                self._viz_record(
                    n_selected=int(mask[:, 1:].sum().item()),
                    mean_fill=float(fill_levels[:, 1:].mean().item()),
                    day=int(day.max().item()) if day is not None else -1,
                )
                return mask

            cls.select = _instrumented  # type: ignore[method-assign]

    @abstractmethod
    def select(
        self,
        fill_levels: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Select bins that must be collected.

        Args:
            fill_levels: Current fill levels of all bins [B, N].
            kwargs: Additional environmental features (history, coordinates, etc.).

        Returns:
            torch.Tensor: Boolean mask [B, N] where True indicates collection.
                Note: Index 0 (depot) should always be False.
        """
        pass
