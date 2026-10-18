"""Capacitated Vehicle Routing Problem (CVRP) environment, generator, and task specification.

The vehicle routes from a depot to serve all customers, returning to the depot
whenever capacity is exhausted. The objective is to minimise total distance.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch

from logic.src.envs.base.resources import LoadResource
from logic.src.envs.base.routing import RoutingEnv
from logic.src.envs.generators.cvrp import CAPACITIES, CVRPGenerator
from logic.src.envs.tasks.base import BaseProblem
from logic.src.envs.tasks.common.objectives import distance_objective
from logic.src.envs.tasks.common.released_specs import CVRP_SPEC
from logic.src.envs.tasks.core.ptp import calc_cumulative_trip_load, closed_tour_length
from logic.src.envs.tasks.spec import PROBLEMS


class CVRPEnv(RoutingEnv):
    """Capacitated Vehicle Routing Problem Environment on the routing base.

    Attributes:
        NAME: "cvrp"
        require_all_customers: True
    """

    NAME: str = "cvrp"
    name: str = "cvrp"
    node_dim: int = 2
    require_all_customers: bool = True

    def __init__(
        self,
        generator: Optional[CVRPGenerator] = None,
        generator_params: Optional[dict] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        generator_params = generator_params or kwargs
        if generator is None:
            generator = CVRPGenerator(**generator_params, device=device)

        super().__init__(
            generator=generator,
            generator_params=generator_params,
            resources=(LoadResource(limit_key="vehicle_capacity", demand_key="waste"),),
            waste_weight=0.0,
            cost_weight=1.0,
            device=device,
            **kwargs,
        )


def evaluate_cvrp_tour(
    dataset: Dict[str, Any],
    pi: torch.Tensor,
    dist_matrix: Optional[torch.Tensor] = None,
    capacity: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, torch.Tensor]:
    """Vectorized evaluation of a CVRP tour sequence."""
    if pi.dim() == 1:
        pi = pi.unsqueeze(0)

    bs, seq_len = pi.shape
    device = pi.device

    if seq_len == 0:
        z = torch.zeros(bs, device=device)
        return {
            "cost": z,
            "reward": z,
            "length": z,
            "demand": z,
            "overflows": z,
            "valid": torch.ones(bs, dtype=torch.bool, device=device),
            "total": z,
        }

    locs = dataset.get("locs") if "locs" in dataset else dataset.get("loc")
    depot = dataset.get("depot")

    if locs is not None:
        if locs.dim() == 2:
            locs = locs.unsqueeze(0).expand(bs, -1, -1)
        if depot is not None and (
            locs.size(1) == dataset.get("demand", locs).size(-1)
            and not torch.allclose(locs[:, 0, :], depot if depot.dim() == 2 else depot.unsqueeze(0), atol=1e-4)
        ):
            depot_2d = depot.unsqueeze(1) if depot.dim() == 2 else depot.unsqueeze(0).unsqueeze(1)
            if depot_2d.size(0) != bs:
                depot_2d = depot_2d.expand(bs, -1, -1)
            full_locs = torch.cat([depot_2d, locs], dim=1)
        else:
            full_locs = locs
    else:
        full_locs = None

    dm_input = dist_matrix if dist_matrix is not None else dataset.get("dist_matrix", dataset.get("dm"))
    if dm_input is not None and isinstance(dm_input, torch.Tensor):
        length = closed_tour_length(dm_input, pi)
    elif full_locs is not None:
        coords = full_locs.gather(1, pi.unsqueeze(-1).expand(*pi.shape, full_locs.size(-1)))
        depot_coord = full_locs[:, 0:1, :]
        length = (
            (coords[:, 1:] - coords[:, :-1]).norm(p=2, dim=-1).sum(dim=1)
            + (coords[:, 0:1] - depot_coord).norm(p=2, dim=-1).squeeze(1)
            + (coords[:, -1:] - depot_coord).norm(p=2, dim=-1).squeeze(1)
        )
    else:
        length = torch.zeros(bs, device=device)

    # Capacity check per trip
    raw_demand = dataset.get("demand")
    overflows = torch.zeros(bs, device=device)
    if raw_demand is not None:
        if raw_demand.dim() == 1:
            raw_demand = raw_demand.unsqueeze(0).expand(bs, -1)
        num_loc = (
            full_locs.size(1)
            if full_locs is not None
            else dm_input.size(-1)
            if dm_input is not None
            else raw_demand.size(1)
        )
        if raw_demand.size(1) == num_loc - 1:
            full_demand = torch.cat([torch.zeros(bs, 1, device=device), raw_demand], dim=-1)
        else:
            full_demand = raw_demand
        d_seq = full_demand.gather(1, pi)
        total_demand = d_seq.sum(dim=-1)

        cap_val = capacity if capacity is not None else dataset.get("vehicle_capacity", dataset.get("capacity", 1.0))
        cap_t = torch.as_tensor(cap_val, dtype=torch.float32, device=device).reshape(-1)
        if cap_t.size(0) != bs:
            cap_t = cap_t.expand(bs)

        trip_loads = calc_cumulative_trip_load(d_seq, pi)
        max_trip_load = trip_loads.max(dim=-1).values
        overflows = (max_trip_load - cap_t).clamp(min=0.0)
    else:
        total_demand = torch.zeros(bs, device=device)

    valid = overflows <= 1e-5

    return {
        "cost": length,
        "reward": distance_objective(length),
        "length": length,
        "demand": total_demand,
        "overflows": overflows,
        "valid": valid,
        "total": length,
    }


class CVRP(BaseProblem):
    """Capacitated Vehicle Routing Problem (CVRP) task definition."""

    NAME: str = "cvrp"

    @staticmethod
    def evaluate_tour(
        dataset: Dict[str, Any],
        pi: torch.Tensor,
        dist_matrix: Optional[torch.Tensor] = None,
        capacity: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Evaluate a completed CVRP tour sequence."""
        return evaluate_cvrp_tour(dataset, pi, dist_matrix=dist_matrix, capacity=capacity, **kwargs)


PROBLEMS.register(CVRP_SPEC)

__all__ = ["CAPACITIES", "CVRPGenerator", "CVRPEnv", "CVRP", "evaluate_cvrp_tour", "CVRP_SPEC"]
