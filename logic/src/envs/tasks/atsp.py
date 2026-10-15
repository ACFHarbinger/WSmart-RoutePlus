"""Asymmetric Traveling Salesman Problem (ATSP) task and environment."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
from tensordict import TensorDict, TensorDictBase

from logic.src.envs.base.ops import OpsMixin
from logic.src.envs.generators.atsp import ATSPGenerator
from logic.src.envs.tasks.base import BaseProblem
from logic.src.envs.tasks.espprc import ESPPRCEnv
from logic.src.envs.tasks.spec import PROBLEMS, ProblemSpec


class ATSPEnv(ESPPRCEnv):
    """Asymmetric Traveling Salesman Problem Environment."""

    NAME: str = "atsp"
    name: str = "atsp"
    node_dim: int = 0
    require_all_customers: bool = True

    def __init__(
        self,
        generator: Optional[ATSPGenerator] = None,
        generator_params: Optional[dict] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        generator_params = generator_params or kwargs
        if generator is None:
            generator = ATSPGenerator(**generator_params, device=device)
        super().__init__(
            generator=generator,
            generator_params=generator_params,
            resources=(),
            waste_weight=0.0,
            cost_weight=1.0,
            device=device,
            **kwargs,
        )

    # Dynamics of the pre-WP-07 envs/routing/atsp.py, kept verbatim (review CR-12, Q-8b).

    def _reset_instance(self, td: TensorDict) -> TensorDict:
        """Initialise ATSP episode state.

        Args:
            td: Input TensorDict containing the environment state.

        Returns:
            Updated TensorDict or tensor containing the result.
        """
        if "visited" in td.keys():
            return td

        device = td.device
        bs = td.batch_size
        if self.generator is None:
            raise ValueError("Generator is not initialized.")
        n = td["cost_matrix"].shape[-1]

        td["current_node"] = torch.zeros(*bs, dtype=torch.long, device=device)
        td["first_node"] = torch.zeros(*bs, dtype=torch.long, device=device)
        # All nodes start unvisited; node 0 is marked as starting point
        td["visited"] = torch.zeros(*bs, n, dtype=torch.bool, device=device)
        td["visited"][..., 0] = True

        td["tour"] = torch.zeros(*bs, 0, dtype=torch.long, device=device)
        td["tour_length"] = torch.zeros(*bs, dtype=torch.float32, device=device)

        td["reward"] = torch.zeros(*bs, dtype=torch.float32, device=device)
        td["terminated"] = torch.zeros(*bs, dtype=torch.bool, device=device)
        td["truncated"] = torch.zeros(*bs, dtype=torch.bool, device=device)

        return td

    def _step(self, td: TensorDict) -> TensorDict:
        """Delegate to OpsMixin._step, which calls _step_instance.

        Args:
            td: Input TensorDict containing the environment state.

        Returns:
            Updated TensorDict or tensor containing the result.
        """
        return OpsMixin._step(self, td)

    def _step_instance(self, td: TensorDict) -> TensorDict:
        """
        Execute one ATSP action.

        Uses the asymmetric cost matrix to look up the travel cost from
        the current node to the selected action node.
        Args:
            td: Input TensorDict containing the environment state.

        Returns:
            Updated TensorDict or tensor containing the result.
        """
        action = td["action"]
        if action.dim() > 1:
            action = action.squeeze(-1)
        if action.dim() == 0:
            action = action.unsqueeze(0)

        current_node = td["current_node"]
        if current_node.dim() > 1:
            current_node = current_node.squeeze(-1)
        if current_node.dim() == 0:
            current_node = current_node.unsqueeze(0)

        # Batch-safe cost matrix lookup: cost[b, current[b], action[b]]
        cost_matrix = td["cost_matrix"]  # [*B, N, N]
        bs = td.batch_size
        flat_bs = bs.numel() if len(bs) > 0 else 1
        flat_cost = cost_matrix.reshape(flat_bs, cost_matrix.shape[-2], cost_matrix.shape[-1])
        batch_idx = torch.arange(flat_bs, device=td.device)
        dist = flat_cost[batch_idx, current_node.reshape(flat_bs), action.reshape(flat_bs)].reshape(*bs)

        td["tour_length"] = td["tour_length"] + dist
        td["visited"] = td["visited"].scatter(-1, action.unsqueeze(-1), True)
        td["current_node"] = action
        td["tour"] = torch.cat([td["tour"], action.unsqueeze(-1)], dim=-1)

        return td

    def _check_done(self, td: TensorDict) -> torch.Tensor:
        """Done when all N nodes have been visited.

        Args:
            td: Input TensorDict containing the environment state.

        Returns:
            Updated TensorDict or tensor containing the result.
        """
        return td["visited"].all(dim=-1)

    def _get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """All unvisited nodes are valid actions.

        Args:
            td: Input TensorDict containing the environment state.

        Returns:
            Updated TensorDict or tensor containing the result.
        """
        return ~td["visited"]

    def _get_reward(self, td: TensorDictBase, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Reward = -(accumulated tour cost + closing edge cost).

        The closing edge goes from the last visited node back to the first
        node selected (stored in ``td["first_node"]``).
        Args:
            td: Input TensorDict containing the environment state.
            actions: Tensor of actions to take.

        Returns:
            Updated TensorDict or tensor containing the result.
        """
        cost_matrix = td["cost_matrix"]
        bs = td.batch_size
        flat_bs = bs.numel() if len(bs) > 0 else 1
        flat_cost = cost_matrix.reshape(flat_bs, cost_matrix.shape[-2], cost_matrix.shape[-1])

        # Closing edge: current_node → first_node
        current = td["current_node"]
        if current.dim() > 1:
            current = current.squeeze(-1)
        first = td["first_node"]
        if first.dim() > 1:
            first = first.squeeze(-1)

        batch_idx = torch.arange(flat_bs, device=td.device)
        closing = flat_cost[batch_idx, current.reshape(flat_bs), first.reshape(flat_bs)].reshape(*bs)

        total = td["tour_length"] + closing
        return -total


class ATSP(BaseProblem):
    """Asymmetric Traveling Salesman Problem (ATSP) task definition."""

    NAME = "atsp"

    @staticmethod
    def evaluate_tour(
        dataset: Dict[str, Any],
        pi: torch.Tensor,
        dist_matrix: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Compute ATSP tour cost and outcomes."""
        cost_matrix = dist_matrix if dist_matrix is not None else dataset.get("cost_matrix")
        if cost_matrix is None:
            raise ValueError("ATSP evaluation requires 'cost_matrix'")
        bs = cost_matrix.shape[0]
        if pi.dim() == 1:
            pi = pi.unsqueeze(0).expand(bs, -1)

        nodes_src = pi
        nodes_tgt = torch.roll(pi, -1, dims=-1)
        batch_idx = torch.arange(bs, device=cost_matrix.device).unsqueeze(1)
        cost = cost_matrix[batch_idx, nodes_src, nodes_tgt].sum(-1)

        return {
            "cost": cost,
            "reward": -cost,
            "length": cost,
            "valid": torch.ones_like(cost, dtype=torch.bool),
            "total": cost,
        }


ATSP_SPEC = ProblemSpec(
    name="atsp",
    objective=lambda distance_km, **kw: -float(distance_km),
    constraints=(),
    metrics=("cost", "reward", "length"),
    env_target="logic.src.envs.tasks.atsp:ATSPEnv",
    expand_pool=False,
)
PROBLEMS.register(ATSP_SPEC)

__all__ = ["ATSPGenerator", "ATSPEnv", "ATSP", "ATSP_SPEC"]
