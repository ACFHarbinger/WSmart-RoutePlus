"""
Orienteering Problem (OP) generator.

Prize types follow Fischetti et al. (1998) and Kool et al. (2019).

Attributes:
    MAX_LENGTHS: Default route-length budget per instance size (Kool et al. 2019).
    OPGenerator: OPGenerator class.

Example:
    >>> from logic.src.envs.generators import OPGenerator
    >>> td = OPGenerator(num_loc=20, prize_type="dist")(batch_size=2)
    >>> td["prize"].shape, td["max_length"].shape
    (torch.Size([2, 20]), torch.Size([2]))
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Union

import torch
from tensordict import TensorDict

from .base import RoutingGenerator

# Default max-length table (from Kool et al. 2019)
MAX_LENGTHS: dict[int, float] = {20: 2.0, 50: 3.0, 100: 4.0}


class OPGenerator(RoutingGenerator):
    """
    Generator for the Orienteering Problem (OP).

    Produces instances with:
    - locs       : [*B, num_loc, 2]  — customer coordinates
    - depot      : [*B, 2]           — depot coordinate
    - prize      : [*B, num_loc]     — prize at each customer
    - max_length : [*B]              — maximum path length allowed

    Attributes:
        prize_type: Prize generation method ("uniform", "dist", "unif" or "const").
        max_length: Maximum allowed path length budget.
    """

    def __init__(
        self,
        num_loc: int = 50,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: Union[str, Callable] = "uniform",
        prize_type: str = "uniform",
        max_length: Optional[float] = 2.0,
        depot_type: str = "random",
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        """
        Initialise OPGenerator.

        Args:
            num_loc: Number of customer locations (excluding depot).
            min_loc: Lower bound for node coordinates.
            max_loc: Upper bound for node coordinates.
            loc_distribution: Spatial distribution.
            prize_type: Prize generation method:
                "uniform" — continuous uniform in [0, 1),
                "dist" — proportional to distance from depot,
                "unif" — uniform over {0.01, ..., 1.0},
                "const" — constant 1.0.
            max_length: Maximum allowed path length; None takes the ``MAX_LENGTHS``
                entry closest to ``num_loc``.
            depot_type: Depot placement ("random", "center", "corner").
            device: Target device.
            kwargs: Forwarded to the RoutingGenerator base (``dm_filepath``, ``rng``, ...).
        """
        super().__init__(
            num_loc=num_loc,
            min_loc=min_loc,
            max_loc=max_loc,
            loc_distribution=loc_distribution,
            depot_type=depot_type,
            device=device,
            **kwargs,
        )
        self.prize_type = prize_type
        if max_length is None:
            closest = min(MAX_LENGTHS.keys(), key=lambda x: abs(x - num_loc))
            max_length = MAX_LENGTHS[closest]
        self.max_length = max_length

    def _generate(self, batch_size: Union[int, Sequence[int]]) -> TensorDict:
        """Generate a batch of OP instances.

        Args:
            batch_size: Batch size.

        Returns:
            TensorDict with ``locs``, ``depot``, ``prize`` and ``max_length``.
        """
        td = super()._generate(batch_size)
        bs = tuple(td.batch_size)
        td["prize"] = self._generate_prize(bs, td["locs"], td["depot"])
        td["max_length"] = torch.full(bs, self.max_length, device=self.device)
        return td

    def _generate_prize(self, batch_size: tuple[int, ...], locs: torch.Tensor, depot: torch.Tensor) -> torch.Tensor:
        """Prizes ``[*B, num_loc]`` for ``prize_type``."""
        if self.prize_type == "uniform":
            return torch.rand((*batch_size, self.num_loc), device=self.device)
        if self.prize_type == "const":
            return torch.ones(*batch_size, self.num_loc, device=self.device, dtype=torch.float32)
        if self.prize_type == "unif":
            draws = torch.randint(0, 100, (*batch_size, self.num_loc), device=self.device, generator=self.generator)
            return (1 + draws.float()) / 100.0
        if self.prize_type == "dist":
            dist_to_depot = (depot.unsqueeze(-2) - locs).norm(p=2, dim=-1)
            return (1 + (dist_to_depot / dist_to_depot.max(dim=-1, keepdim=True)[0] * 99).int().float()) / 100.0
        raise ValueError(f"Unknown prize type: {self.prize_type}")


__all__ = ["MAX_LENGTHS", "OPGenerator"]
