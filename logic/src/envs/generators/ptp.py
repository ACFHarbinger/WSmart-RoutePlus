"""
PTP problem generator.

Attributes:
    PTPGenerator: PTPGenerator class.

Example:
    >>> from logic.src.envs.generators import PTPGenerator
    >>> generator = PTPGenerator(num_loc=50)
    >>> instance = generator.generate()
    >>> instance
    TensorDict(
        {
            'locs': TensorDict(
                # ... customer locations (shape: [*B, 50, 2]) ...
            ),
            'depot': TensorDict(
                # ... depot location (shape: [*B, 2]) ...
            ),
            'waste': TensorDict(
                # ... waste at each location (shape: [*B, 50]) ...
            )
        }
    )
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
from tensordict import TensorDict

from .base import Generator


class PTPGenerator(Generator):
    """
    Generator for Profitable Tour Problem (PTP) instances.

    Generates instances with:
    - Depot location
    - Customer locations
    - Waste at each location
    - Waste values for visiting each location
    - Vehicle capacity
    - Maximum route length

    Attributes:
        num_loc: Number of customer locations.
        min_loc: Lower bound for node coordinates.
        max_loc: Upper bound for node coordinates.
        loc_distribution: Spatial distribution strategy.
        min_waste: Minimum waste value.
        max_waste: Maximum waste value.
        waste_distribution: Waste value distribution strategy.
        capacity: Vehicle capacity.
        depot_type: Depot placement strategy.
    """

    def __init__(
        self,
        num_loc: int = 50,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: Union[str, Callable] = "uniform",
        min_waste: float = 0.0,
        max_waste: float = 1.0,
        waste_distribution: str = "uniform",
        capacity: float = 1.0,
        depot_type: str = "corner",
        device: Union[str, torch.device] = "cpu",
        area: Optional[str] = None,
        data_dir: Optional[str] = None,
        indices: Optional[list[int]] = None,
        generator: Optional[np.random.Generator] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize PTP generator.

        Args:
            num_loc: Number of customer locations.
            min_loc: Minimum coordinate value.
            max_loc: Maximum coordinate value.
            loc_distribution: Distribution for location generation.
            min_waste: Minimum waste value.
            max_waste: Maximum waste value.
            waste_distribution: Distribution for waste generation.
            capacity: Vehicle capacity.
            depot_type: Depot placement ("center", "corner", "random").
            device: Device to place tensors on.
            area: Area for grid generation.
            data_dir: Directory for grid data.
            indices: Indices for grid generation.
            generator: Generator for random number generation.
            kwargs: Additional keyword arguments.
        """
        super().__init__(num_loc, min_loc, max_loc, loc_distribution, device, generator, **kwargs)

        self.min_waste = min_waste
        self.max_waste = max_waste
        self.waste_distribution = kwargs.get("data_distribution", waste_distribution)
        self.capacity = capacity if capacity is not None else 1.0
        self.depot_type = depot_type
        self.data_dir = data_dir
        self.area = area
        self.indices = indices if indices is not None else list(range(0, num_loc))
        self.grid: Any = None
        try:
            from logic.src.utils.data.loader import load_grid_base

            self.grid = load_grid_base(self.indices, self.area, self.data_dir)
        except FileNotFoundError:
            pass

    def _generate(self, batch_size: tuple[int, ...]) -> TensorDict:
        """Generate PTP instances.

        Args:
            batch_size: Batch size.

        Returns:
            TensorDict with:
                locs: [*B, num_loc, 2]     — customer locations
                depot: [*B, 2]              — depot location
                waste: [*B, num_loc]        — waste at each location
                capacity: [*B]               — vehicle capacity
                max_waste: [*B]             — max waste per vehicle
        """
        # Generate locations
        locs = self._generate_locations(batch_size)

        # Generate depot
        depot = self._generate_depot(batch_size)

        # Generate waste
        waste = self._generate_fill_levels(batch_size)

        # Compute max_waste per instance (for normalization)
        max_waste = torch.full((*batch_size,), self.capacity, device=self.device)
        return TensorDict(
            {
                "locs": locs,
                "depot": depot,
                "waste": waste,
                "capacity": torch.full((*batch_size,), self.capacity, device=self.device),
                "max_waste": max_waste,
            },
            batch_size=batch_size,
            device=self.device,
        )

    def _generate_depot(self, batch_size: tuple[int, ...]) -> torch.Tensor:
        """Generate depot location based on depot_type.

        Args:
            batch_size: Batch size.

        Returns:
            TensorDict with depot location (shape: [*B, 2]).
        """
        if self.depot_type == "center":
            center = (self.max_loc + self.min_loc) / 2
            return torch.full((*batch_size, 2), center, device=self.device)
        elif self.depot_type == "corner":
            return torch.full((*batch_size, 2), self.min_loc, device=self.device)
        elif self.depot_type == "random":
            return self._uniform_locations(batch_size, 1).squeeze(-2)
        else:
            raise ValueError(f"Unknown depot type: {self.depot_type}")

    def _generate_fill_levels(self, batch_size: tuple[int, ...]) -> torch.Tensor:
        """Generate waste values.

        Args:
            batch_size: Batch size.

        Returns:
            TensorDict with waste values (shape: [*B, num_loc]).
        """
        # Use common utility for consistency
        from logic.src.data.generators.waste import generate_waste

        bs = math.prod(batch_size) if batch_size else 1
        coords = (
            self._generate_depot(batch_size).view(bs, 2),
            self._generate_locations(batch_size).view(bs, self.num_loc, 2),
        )
        waste = generate_waste(
            self.num_loc,
            self.waste_distribution,
            coords,
            bs,
            grid=self.grid,
            rng=self.rng,
            sample_method="sample_array",
        )
        if isinstance(waste, np.ndarray):
            waste = torch.from_numpy(waste).float()
        return waste.to(self.device).view(*batch_size, self.num_loc)
