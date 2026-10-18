"""
PTP problem generator.

Attributes:
    PTPGenerator: Profitable Tour Problem instances (also the base of the CPTP and TCPTP generators).

Example:
    >>> from logic.src.envs.generators import PTPGenerator
    >>> td = PTPGenerator(num_loc=50)(batch_size=4)
    >>> td["locs"].shape, td["depot"].shape, td["waste"].shape
    (torch.Size([4, 50, 2]), torch.Size([4, 2]), torch.Size([4, 50]))
"""

from __future__ import annotations

import math
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from tensordict import TensorDict

from .base import RoutingGenerator


class PTPGenerator(RoutingGenerator):
    """
    Generator for Profitable Tour Problem (PTP) instances.

    Generates instances with:
    - Depot location
    - Customer locations
    - Waste at each location (sampled from the area's bins when a grid is available)
    - Vehicle capacity (``capacity`` and ``max_waste``)

    Attributes:
        min_waste: Minimum waste value (uniform and normal waste distributions).
        max_waste: Maximum waste value (uniform and normal waste distributions).
        waste_distribution: Waste distribution: "uniform" or "normal" (drawn here), any other
            name of :func:`logic.src.data.generators.waste.generate_waste` (e.g. "gamma1",
            "dist"), or the empirical bins of ``area`` when its grid loads.
        capacity: Vehicle capacity in bin fills.
        area: Geographic area of the empirical grid.
        data_dir: Directory of the grid data.
        indices: Bin indices of the empirical grid.
        grid: The loaded grid, or None.
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
        capacity: Optional[float] = None,
        depot_type: str = "random",
        device: Union[str, torch.device] = "cpu",
        area: Optional[str] = None,
        data_dir: Optional[str] = None,
        indices: Optional[List[int]] = None,
        dm_filepath: Optional[str] = None,
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
            waste_distribution: Distribution for waste generation; ``data_distribution`` in
                ``kwargs`` overrides it.
            capacity: Vehicle capacity in bin fills. When omitted it is the simulator's
                capacity for the area and waste type (percent of a full bin, divided by 100),
                or 1.0 when those are not known.
            depot_type: Depot placement ("random", "center", "corner").
            device: Device to place tensors on.
            area: Area for grid generation.
            data_dir: Directory for grid data.
            indices: Indices for grid generation.
            dm_filepath: Road distance matrix file.
            kwargs: Additional keyword arguments (``waste_type``, ``data_distribution``, ...).
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
        self.min_waste = min_waste
        self.max_waste = max_waste
        self.waste_distribution = kwargs.get("data_distribution", waste_distribution)
        self.capacity = capacity if capacity is not None else self._simulator_capacity(area, kwargs.get("waste_type"))
        self.data_dir = data_dir
        self.area = area
        self.indices = indices if indices is not None else list(range(0, num_loc))
        self.grid: Any = None
        try:
            from logic.src.utils.data.loader import load_grid_base

            self.grid = load_grid_base(self.indices, self.area, self.data_dir)
        except (FileNotFoundError, ImportError):
            pass

    @staticmethod
    def _simulator_capacity(area: Optional[str], waste_type: Optional[str]) -> float:
        """Vehicle capacity in bin fills for an area and waste type, as the simulator uses it.

        Args:
            area: Geographic area.
            waste_type: Waste stream.

        Returns:
            The simulator's capacity (percent of a full bin) divided by 100, or 1.0 when the
            area or waste type is missing or unknown.
        """
        if not area or not waste_type:
            return 1.0
        try:
            from logic.src.constants import MAX_CAPACITY_PERCENT
            from logic.src.pipeline.simulations.repository import load_area_and_waste_type_params

            return float(load_area_and_waste_type_params(area, waste_type)[0]) / MAX_CAPACITY_PERCENT
        except (AssertionError, ImportError, KeyError, TypeError, ValueError):
            return 1.0

    def _generate(self, batch_size: Union[int, Sequence[int]]) -> TensorDict:
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
        td = super()._generate(batch_size)
        bs = tuple(td.batch_size)
        td["waste"] = self._generate_fill_levels(bs, td["locs"], td["depot"])
        td["capacity"] = torch.full(bs, self.capacity, device=self.device)
        td["max_waste"] = torch.full(bs, self.capacity, device=self.device)
        return td

    def _generate_fill_levels(
        self, batch_size: Tuple[int, ...], locs: torch.Tensor, depot: torch.Tensor
    ) -> torch.Tensor:
        """Waste values ``[*B, num_loc]`` for the instance coordinates.

        Args:
            batch_size: Batch size.
            locs: Customer locations ``[*B, num_loc, 2]``.
            depot: Depot location ``[*B, 2]``.

        Returns:
            Waste values.
        """
        size = (*batch_size, self.num_loc)
        drawn_here = self.waste_distribution in ("uniform", "normal")
        if self.grid is not None or not drawn_here:
            try:
                return self._sample_waste(batch_size, locs, depot)
            except Exception:
                # The empirical grid is best effort: fall back to the parametric draw.
                if self.grid is None:
                    raise
        if self.waste_distribution == "normal":
            mean = (self.min_waste + self.max_waste) / 2
            std = (self.max_waste - self.min_waste) / 6
            return (torch.randn(size, device=self.device) * std + mean).clamp(self.min_waste, self.max_waste)
        return torch.empty(size, device=self.device).uniform_(self.min_waste, self.max_waste)

    def _sample_waste(self, batch_size: Tuple[int, ...], locs: torch.Tensor, depot: torch.Tensor) -> torch.Tensor:
        """Waste from :func:`generate_waste` (empirical grid or a named distribution)."""
        from logic.src.data.generators.waste import generate_waste

        bs = math.prod(batch_size) if batch_size else 1
        coords = (depot.reshape(bs, 2), locs.reshape(bs, self.num_loc, 2))
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


__all__ = ["PTPGenerator"]
