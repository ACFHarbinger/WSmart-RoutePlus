"""Orienteering Problem (OP) task and environment on the routing base.

The module also holds ``OPGenerator``, the problem's instance generator.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Union

import torch
from tensordict import TensorDict

from logic.src.pipeline.learning.envs.base.generator import RoutingGenerator
from logic.src.pipeline.learning.envs.base.ops import as_batch_nodes
from logic.src.pipeline.learning.envs.base.problem import BaseProblem
from logic.src.pipeline.learning.envs.base.routing import RoutingEnv
from logic.src.pipeline.learning.envs.spec import PROBLEMS, ProblemSpec

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


class OPEnv(RoutingEnv):
    """Orienteering Problem Environment on the routing base."""

    NAME: str = "op"
    name: str = "op"
    node_dim: int = 3
    require_all_customers: bool = False

    def __init__(
        self,
        generator: Optional[OPGenerator] = None,
        generator_params: Optional[dict] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        generator_params = generator_params or kwargs
        if generator is None:
            generator = OPGenerator(**generator_params, device=device)
        super().__init__(
            generator=generator,
            generator_params=generator_params,
            resources=(),
            waste_weight=1.0,
            cost_weight=0.0,
            device=device,
            **kwargs,
        )

    # OP dynamics of the pre-WP-07 envs/routing/op.py (review CR-12): a customer is
    # offered only if the tour can reach it and still return within ``max_length``,
    # and the reward is the collected prize.

    def _reset_instance(self, tensordict: TensorDict) -> TensorDict:
        if "visited" in tensordict.keys():
            return tensordict
        prize = tensordict["prize"]
        tensordict = super()._reset_instance(tensordict)
        locs = tensordict["locs"]
        if prize.shape[-1] == locs.shape[-2] - 1:
            prize = torch.cat([torch.zeros_like(prize[..., :1]), prize], dim=-1)
        tensordict["prize"] = prize
        max_length = tensordict["max_length"]
        if max_length.dim() > 1:
            max_length = max_length.squeeze(-1)
        # Budget left at each node once the return to the depot is reserved.
        dist_to_depot = (locs[..., 0:1, :] - locs).norm(p=2, dim=-1)
        tensordict["max_length"] = max_length.unsqueeze(-1) - dist_to_depot - 1e-6
        tensordict["current_total_prize"] = torch.zeros(*tensordict.batch_size, device=tensordict.device)
        return tensordict

    def _step_instance(self, tensordict: TensorDict) -> TensorDict:
        tensordict = super()._step_instance(tensordict)
        action = as_batch_nodes(tensordict["action"])
        prize_here = tensordict["prize"].gather(-1, action.unsqueeze(-1)).squeeze(-1)
        tensordict["current_total_prize"] = tensordict["current_total_prize"] + prize_here
        return tensordict

    def _get_action_mask(self, tensordict: TensorDict) -> torch.Tensor:
        current = tensordict["current_node"].squeeze(-1)
        locs = tensordict["locs"]
        current_loc = locs.gather(1, current[:, None, None].expand(-1, -1, 2)).squeeze(1)
        dist_current_to_node = (locs - current_loc.unsqueeze(1)).norm(p=2, dim=-1)
        exceeds_budget = tensordict["tour_length"].unsqueeze(-1) + dist_current_to_node > tensordict["max_length"]
        mask = ~(tensordict["visited"] | exceeds_budget)
        mask[..., 0] = True
        return mask

    def _get_reward(self, tensordict: TensorDict, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        return tensordict["current_total_prize"]


class OP(BaseProblem):
    """Orienteering Problem (OP) task definition."""

    NAME = "op"

    @staticmethod
    def evaluate_tour(
        dataset: Dict[str, Any],
        pi: torch.Tensor,
        dist_matrix: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Compute OP prize, length, and outcomes."""
        if pi.size(-1) <= 1:
            z = torch.zeros(pi.size(0), device=pi.device)
            return {
                "cost": z,
                "reward": z,
                "prize": z,
                "length": z,
                "valid": torch.ones_like(z, dtype=torch.bool),
                "total": z,
            }

        prize_all = dataset.get("prize", torch.zeros(pi.size(0), pi.size(1), device=pi.device))
        if prize_all.shape[-1] == pi.shape[-1] - 1:
            prize_all = torch.cat([torch.zeros_like(prize_all[:, :1]), prize_all], dim=-1)

        collected_prize = prize_all.gather(1, pi).sum(-1)
        length = OP.get_tour_length(dataset, pi, dist_matrix)
        neg_prize = -collected_prize

        max_len = dataset.get("max_length", 2.0)
        max_len_t = torch.as_tensor(max_len, dtype=torch.float32, device=pi.device).reshape(-1)
        if max_len_t.size(0) != pi.size(0):
            max_len_t = max_len_t.expand(pi.size(0))
        valid = length <= max_len_t + 1e-4

        return {
            "cost": neg_prize,
            "reward": collected_prize,
            "prize": collected_prize,
            "length": length,
            "valid": valid,
            "total": neg_prize,
        }


OP_SPEC = ProblemSpec(
    name="op",
    objective=lambda prize, **kw: float(prize),
    constraints=(),
    metrics=("cost", "reward", "prize", "length"),
    env_target="logic.src.pipeline.learning.envs.op:OPEnv",
    expand_pool=True,
)
PROBLEMS.register(OP_SPEC)

__all__ = ["OPGenerator", "OPEnv", "OP", "OP_SPEC"]
