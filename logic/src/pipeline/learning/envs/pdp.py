"""Pickup and Delivery Problem (PDP) task and environment on the routing base.

The module also holds ``PDPGenerator``, the problem's instance generator.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Union

import torch
from tensordict import TensorDict

from logic.src.pipeline.learning.envs.base.generator import RoutingGenerator
from logic.src.pipeline.learning.envs.base.ops import as_batch_nodes
from logic.src.pipeline.learning.envs.base.problem import BaseProblem
from logic.src.pipeline.learning.envs.base.routing import RoutingEnv
from logic.src.pipeline.learning.envs.spec import PROBLEMS, ProblemSpec


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


class PDPEnv(RoutingEnv):
    """Pickup and Delivery Problem Environment on the routing base."""

    NAME: str = "pdp"
    name: str = "pdp"
    node_dim: int = 3
    require_all_customers: bool = True

    def __init__(
        self,
        generator: Optional[PDPGenerator] = None,
        generator_params: Optional[dict] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        generator_params = generator_params or kwargs
        if generator is None:
            generator = PDPGenerator(**generator_params, device=device)
        super().__init__(
            generator=generator,
            generator_params=generator_params,
            resources=(),
            waste_weight=0.0,
            cost_weight=1.0,
            device=device,
            **kwargs,
        )

    # PDP dynamics of the pre-WP-07 envs/routing/pdp.py (review CR-12): customers
    # 1..k are pickups and k+1..2k their deliveries; a delivery opens only after
    # its pickup, the depot is never an action, and the day ends when every
    # customer is served. The reward is minus the closed tour length.

    def _reset_instance(self, tensordict: TensorDict) -> TensorDict:
        if "visited" in tensordict.keys():
            return tensordict
        tensordict = super()._reset_instance(tensordict)
        bs, device = tensordict.batch_size, tensordict.device
        num_loc = tensordict["locs"].shape[-2] - 1
        k = num_loc // 2
        tensordict["to_deliver"] = torch.cat(
            [
                torch.ones(*bs, k + 1, dtype=torch.bool, device=device),  # depot and pickups
                torch.zeros(*bs, num_loc - k, dtype=torch.bool, device=device),  # deliveries, locked
            ],
            dim=-1,
        )
        available = torch.ones(*bs, num_loc + 1, dtype=torch.bool, device=device)
        available[..., 0] = False
        tensordict["available"] = available
        return tensordict

    def _step_instance(self, tensordict: TensorDict) -> TensorDict:
        tensordict = super()._step_instance(tensordict)
        action = as_batch_nodes(tensordict["action"])
        num_loc = tensordict["locs"].shape[-2] - 1
        k = num_loc // 2
        tensordict["available"] = tensordict["available"].scatter(-1, action.unsqueeze(-1), False)
        # Visiting pickup i unlocks delivery i + k.
        unlocked = (action + k) % (num_loc + 1)
        tensordict["to_deliver"] = tensordict["to_deliver"].scatter(-1, unlocked.unsqueeze(-1), True)
        return tensordict

    def _get_action_mask(self, tensordict: TensorDict) -> torch.Tensor:
        return tensordict["available"] & tensordict["to_deliver"]

    def _check_done(self, tensordict: TensorDict) -> torch.Tensor:
        return ~tensordict["available"][..., 1:].any(dim=-1)


class PDP(BaseProblem):
    """Pickup and Delivery Problem (PDP) task definition."""

    NAME = "pdp"

    @staticmethod
    def evaluate_tour(
        dataset: Dict[str, Any],
        pi: torch.Tensor,
        dist_matrix: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Compute PDP tour length and outcomes."""
        if pi.size(-1) <= 1:
            z = torch.zeros(pi.size(0), device=pi.device)
            return {"cost": z, "reward": z, "length": z, "valid": torch.ones_like(z, dtype=torch.bool), "total": z}

        length = PDP.get_tour_length(dataset, pi, dist_matrix)
        return {
            "cost": length,
            "reward": -length,
            "length": length,
            "valid": torch.ones_like(length, dtype=torch.bool),
            "total": length,
        }


PDP_SPEC = ProblemSpec(
    name="pdp",
    objective=lambda distance_km, **kw: -float(distance_km),
    constraints=(),
    metrics=("cost", "reward", "length"),
    env_target="logic.src.pipeline.learning.envs.pdp:PDPEnv",
    expand_pool=False,
)
PROBLEMS.register(PDP_SPEC)

__all__ = ["PDPGenerator", "PDPEnv", "PDP", "PDP_SPEC"]
