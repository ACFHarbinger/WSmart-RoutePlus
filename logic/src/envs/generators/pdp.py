"""
Pickup and Delivery Problem (PDP) generator.

Attributes:
    PDPGenerator: PDPGenerator class.

Example:
    >>> from logic.src.envs.generators import PDPGenerator
    >>> td = PDPGenerator(num_loc=20)(batch_size=2)
    >>> td["locs"].shape, td["capacity"]
    (torch.Size([2, 20, 2]), tensor([1., 1.]))
"""

from __future__ import annotations

from typing import Any, Callable, Union

import torch
from tensordict import TensorDict

from .base import RoutingGenerator


class PDPGenerator(RoutingGenerator):
    """
    Generator for the Pickup and Delivery Problem (PDP).

    Produces instances with:
    - locs     : [*B, num_loc, 2]  — pickup and delivery coordinates
    - depot    : [*B, 2]           — depot coordinate
    - capacity : [*B]              — vehicle capacity (1.0)

    Node layout (after depot prepend in the environment):
      [0] = depot
      [1  …  num_loc/2]        = pickup nodes
      [num_loc/2+1  …  num_loc] = delivery nodes

    Each pickup node i has a paired delivery at i + num_loc/2.
    Note: ``num_loc`` must be even.  If odd, it is incremented by 1.
    """

    def __init__(
        self,
        num_loc: int = 50,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: Union[str, Callable] = "uniform",
        depot_type: str = "random",
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        """
        Initialise PDPGenerator.

        Args:
            num_loc: Number of customer locations (excl. depot, must be even).
            min_loc: Lower bound for node coordinates.
            max_loc: Upper bound for node coordinates.
            loc_distribution: Spatial distribution.
            depot_type: Depot placement ("random", "center", "corner").
            device: Target device.
            kwargs: Forwarded to the RoutingGenerator base (``dm_filepath``, ``rng``, ...).
        """
        if num_loc % 2 != 0:
            num_loc += 1  # PDP requires even number of customer nodes
        super().__init__(
            num_loc=num_loc,
            min_loc=min_loc,
            max_loc=max_loc,
            loc_distribution=loc_distribution,
            depot_type=depot_type,
            device=device,
            **kwargs,
        )

    def _generate(self, batch_size: Union[int, tuple[int, ...]]) -> TensorDict:
        """Generate a batch of PDP instances: ``locs``, ``depot`` and ``capacity``."""
        td = super()._generate(batch_size)
        td["capacity"] = torch.full(tuple(td.batch_size), 1.0, device=self.device)
        return td


__all__ = ["PDPGenerator"]
