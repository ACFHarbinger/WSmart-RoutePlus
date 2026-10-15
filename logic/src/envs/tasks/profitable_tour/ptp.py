"""Profitable Tour Problem (PTP) environment, generator, and task specification.

In PTP, the agent routes a vehicle from a depot to collect prizes / waste from an optional
subset of customer nodes, maximizing collected profit minus travel distance cost.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from logic.src.envs.tasks.base import BaseProblem
from logic.src.envs.tasks.common.released_specs import PTP_SPEC
from logic.src.envs.tasks.espprc import ESPPRCEnv, ESPPRCGenerator
from logic.src.envs.tasks.profitable_tour.common import evaluate_profitable_tour
from logic.src.envs.tasks.spec import PROBLEMS
from tensordict import TensorDict


class PTPGenerator(ESPPRCGenerator):
    """Generator for Profitable Tour Problem (PTP) instances.

    Generates customer locations, depot coordinates, and waste / profit quantities.
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
        depot_type: str = "corner",
        device: Union[str, torch.device] = "cpu",
        area: Optional[str] = None,
        data_dir: Optional[str] = None,
        indices: Optional[List[int]] = None,
        generator: Optional[np.random.Generator] = None,
        dm_filepath: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_loc=num_loc,
            min_loc=min_loc,
            max_loc=max_loc,
            loc_distribution=loc_distribution,
            device=device,
            dm_filepath=dm_filepath,
            **kwargs,
        )
        self.min_waste = min_waste
        self.max_waste = max_waste
        self.waste_distribution = kwargs.get("data_distribution", waste_distribution)
        self.capacity = capacity if capacity is not None else self._simulator_capacity(area, kwargs.get("waste_type"))
        self.depot_type = depot_type
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
        if not area or not waste_type:
            return 1.0
        try:
            from logic.src.constants import MAX_CAPACITY_PERCENT
            from logic.src.pipeline.simulations.repository import load_area_and_waste_type_params

            return float(load_area_and_waste_type_params(area, waste_type)[0]) / MAX_CAPACITY_PERCENT
        except (AssertionError, ImportError, KeyError, TypeError, ValueError):
            return 1.0

    def _generate(self, batch_size: Union[int, Sequence[int]]) -> TensorDict:
        td = super()._generate(batch_size)
        bs = list(batch_size) if isinstance(batch_size, (list, tuple)) else [batch_size]

        waste = self._generate_fill_levels(tuple(bs))
        max_waste = torch.full((*bs,), self.capacity, device=self.device)

        td["waste"] = waste
        td["capacity"] = torch.full((*bs,), self.capacity, device=self.device)
        td["max_waste"] = max_waste
        return td

    def _generate_fill_levels(self, batch_size: Tuple[int, ...]) -> torch.Tensor:
        size = (*batch_size, self.num_loc)
        if self.grid is not None:
            try:
                import math

                import numpy as np
                from logic.src.data.generators.waste import generate_waste

                bs = math.prod(batch_size) if batch_size else 1
                depot = (
                    self._generate_depot(batch_size).view(bs, 2)
                    if hasattr(self, "_generate_depot")
                    else torch.zeros(bs, 2, device=self.device)
                )
                coords = (depot, self._generate_locations(batch_size, self.num_loc).view(bs, self.num_loc, 2))
                waste = generate_waste(
                    self.num_loc,
                    self.waste_distribution,
                    coords,
                    bs,
                    grid=self.grid,
                    rng=getattr(self, "rng", None),
                    sample_method="sample_array",
                )
                if isinstance(waste, np.ndarray):
                    waste = torch.from_numpy(waste).float()
                return waste.to(self.device).view(*batch_size, self.num_loc)
            except Exception:
                pass

        if self.waste_distribution == "uniform":
            return torch.empty(size, device=self.device).uniform_(self.min_waste, self.max_waste)
        elif self.waste_distribution == "normal":
            mean = (self.min_waste + self.max_waste) / 2
            std = (self.max_waste - self.min_waste) / 6
            return (torch.randn(size, device=self.device) * std + mean).clamp(self.min_waste, self.max_waste)
        return torch.empty(size, device=self.device).uniform_(self.min_waste, self.max_waste)


class PTPEnv(ESPPRCEnv):
    """Profitable Tour Problem Environment.

    Selects an optimal subset of profitable nodes to visit on an ESPPRC base
    without resource constraints.
    """

    NAME: str = "ptp"
    name: str = "ptp"
    node_dim: int = 4
    explicit_stop: bool = False
    require_all_customers: bool = False

    def __init__(
        self,
        generator: Optional[PTPGenerator] = None,
        generator_params: Optional[dict] = None,
        waste_weight: float = 1.0,
        cost_weight: float = 1.0,
        revenue_kg: Optional[float] = None,
        cost_km: Optional[float] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        generator_params = generator_params or kwargs
        if generator is None:
            generator = PTPGenerator(**generator_params, device=device)
        super().__init__(
            generator=generator,
            generator_params=generator_params,
            resources=(),
            waste_weight=waste_weight,
            cost_weight=cost_weight,
            revenue_kg=revenue_kg,
            cost_km=cost_km,
            device=device,
            **kwargs,
        )


class PTP(BaseProblem):
    """Profitable Tour Problem (PTP) task definition."""

    NAME: str = "ptp"

    @staticmethod
    def evaluate_tour(
        dataset: Dict[str, Any],
        pi: torch.Tensor,
        dist_matrix: Optional[torch.Tensor] = None,
        cw_dict: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Evaluate a completed PTP tour sequence."""
        return evaluate_profitable_tour(dataset, pi, dist_matrix=dist_matrix, cw_dict=cw_dict, **kwargs)


PROBLEMS.register(PTP_SPEC)

__all__ = ["PTPGenerator", "PTPEnv", "PTP", "PTP_SPEC"]
