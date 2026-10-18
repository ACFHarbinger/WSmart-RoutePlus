"""
CVRP problem generator.

Generates Capacitated Vehicle Routing Problem instances following the
benchmark conventions of Kool et al. (2019).

Attributes:
    CAPACITIES: dict[int, float]: Vehicle capacity table from Kool et al. (2019), Hottung et al. (2022), Kim et al. (2023)
    CVRPGenerator: Generator for the Capacitated Vehicle Routing Problem (CVRP).
        Produces instances with:
        - locs             : [*B, num_loc, 2]  — customer coordinates
        - depot            : [*B, 2]           — depot coordinate
        - demand           : [*B, num_loc]     — normalised demand in (0, 1]
        - vehicle_capacity : [*B]              — vehicle capacity (normalised to 1.0)

    Example:
        >>> from src.envs.generators import get_generator
        >>> generator = get_generator("cvrp", num_loc=20)
        >>> problem = generator.generate()
        >>> problem
        <ProblemInstance: ...>
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Union

import torch
from tensordict import TensorDict

from .base import RoutingGenerator

# Vehicle capacity table from Kool et al. (2019), Hottung et al. (2022), Kim et al. (2023)
CAPACITIES: dict[int, float] = {
    10: 20.0,
    15: 25.0,
    20: 30.0,
    30: 33.0,
    40: 37.0,
    50: 40.0,
    60: 43.0,
    75: 45.0,
    100: 50.0,
    125: 55.0,
    150: 60.0,
    200: 70.0,
    500: 100.0,
    1000: 150.0,
}


class CVRPGenerator(RoutingGenerator):
    """
    Generator for the Capacitated Vehicle Routing Problem (CVRP).

    Produces instances with:
    - locs             : [*B, num_loc, 2]  — customer coordinates
    - depot            : [*B, 2]           — depot coordinate
    - demand           : [*B, num_loc]     — integer demand divided by ``capacity``
    - vehicle_capacity : [*B]              — normalised vehicle capacity
    - capacity         : [*B]              — same as ``vehicle_capacity``

    Attributes:
        min_demand: Minimum demand (integer, before normalisation).
        max_demand: Maximum demand (integer, before normalisation).
        vehicle_capacity: Normalised vehicle capacity.
        capacity: Raw capacity used for normalisation.
    """

    def __init__(
        self,
        num_loc: int = 50,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: Union[str, Callable] = "uniform",
        min_demand: int = 1,
        max_demand: int = 9,
        vehicle_capacity: float = 1.0,
        capacity: Optional[float] = None,
        depot_type: str = "random",
        device: Union[str, torch.device] = "cpu",
        dm_filepath: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialise CVRPGenerator.

        Args:
            num_loc: Number of customer locations (excluding depot).
            min_loc: Lower bound for node coordinates.
            max_loc: Upper bound for node coordinates.
            loc_distribution: Spatial distribution ("uniform", "normal", "clustered").
            min_demand: Minimum integer demand (before normalisation).
            max_demand: Maximum integer demand (before normalisation).
            vehicle_capacity: Normalised vehicle capacity (default 1.0).
            capacity: Raw vehicle capacity used for normalisation. If None, the
                ``CAPACITIES`` entry for ``num_loc`` (Kool et al. 2019), or 50.
            depot_type: Depot placement ("random", "center", "corner").
            device: Target device.
            dm_filepath: Road distance matrix file.
            kwargs: Forwarded to the Generator base (``rng``, ``generator``, ...).
        """
        super().__init__(
            num_loc=num_loc,
            min_loc=min_loc,
            max_loc=max_loc,
            loc_distribution=loc_distribution,
            depot_type=depot_type,
            device=device,
            dm_filepath=dm_filepath,
            **kwargs,
        )
        self.min_demand = min_demand
        self.max_demand = max_demand
        self.vehicle_capacity = vehicle_capacity
        self.capacity = capacity if capacity is not None else CAPACITIES.get(num_loc, 50.0)

    def _generate(self, batch_size: Union[int, Sequence[int]]) -> TensorDict:
        """Generate a batch of CVRP instances.

        Args:
            batch_size: Batch size.

        Returns:
            TensorDict with ``locs``, ``depot``, ``demand``, ``vehicle_capacity`` and ``capacity``.
        """
        td = super()._generate(batch_size)
        bs = tuple(td.batch_size)
        # Integer demands in [min_demand, max_demand], normalised by capacity
        td["demand"] = (
            torch.randint(
                self.min_demand,
                self.max_demand + 1,
                size=(*bs, self.num_loc),
                device=self.device,
                dtype=torch.float32,
            )
            / self.capacity
        )
        td["vehicle_capacity"] = torch.full(bs, self.vehicle_capacity, device=self.device, dtype=torch.float32)
        td["capacity"] = td["vehicle_capacity"]
        return td


__all__ = ["CAPACITIES", "CVRPGenerator"]
