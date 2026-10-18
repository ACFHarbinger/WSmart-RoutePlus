"""Prize-Collecting Traveling Salesman Problem (PCTSP) task and environment on the routing base."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
from tensordict import TensorDict, TensorDictBase

from logic.src.envs.base.ops import OpsMixin
from logic.src.envs.base.routing import RoutingEnv
from logic.src.envs.generators.pctsp import PCTSPGenerator
from logic.src.envs.tasks.base import BaseProblem
from logic.src.envs.tasks.spec import PROBLEMS, ProblemSpec


class PCTSPEnv(RoutingEnv):
    """Prize-Collecting TSP Environment on the routing base."""

    NAME: str = "pctsp"
    name: str = "pctsp"
    node_dim: int = 2
    _stochastic: bool = False
    require_all_customers: bool = False

    def __init__(
        self,
        generator: Optional[PCTSPGenerator] = None,
        generator_params: Optional[dict] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        generator_params = generator_params or kwargs
        if generator is None:
            generator = PCTSPGenerator(**generator_params, device=device)
        super().__init__(
            generator=generator,
            generator_params=generator_params,
            resources=(),
            waste_weight=1.0,
            cost_weight=1.0,
            device=device,
            **kwargs,
        )

    # Dynamics of the pre-WP-07 envs/routing/pctsp.py, kept verbatim (review CR-12, Q-8b).

    def _reset_instance(self, td: TensorDict) -> TensorDict:
        """Initialize PCTSP episode state.

        Args:
            td: Input TensorDict containing graph structure and node properties.

        Returns:
            TensorDict: Initialized PCTSP state with per-episode mutable fields.
        """
        if self.generator is None:
            raise ValueError(f"Generator for {self.NAME} is not initialized. Initialize with an instance first.")
        if "visited" in td.keys():
            return td

        device = td.device
        bs = td.batch_size

        # Prepend depot to locs when customer-only coords are provided
        locs = td["locs"]
        gen_n = getattr(self.generator, "num_loc", None)
        if "depot" in td.keys() and (gen_n is None or locs.shape[-2] == gen_n):
            td["locs"] = torch.cat([td["depot"].unsqueeze(-2), locs], dim=-2)

        num_nodes = td["locs"].shape[-2]  # N+1

        # Real prize: stochastic (SPCTSP) or deterministic (PCTSP)
        real_prize_cust = td["stochastic_prize"] if self._stochastic else td["deterministic_prize"]
        # Prepend 0 for depot
        td["real_prize"] = torch.cat([torch.zeros(*bs, 1, device=device), real_prize_cust], dim=-1)  # [B, N+1]

        # Penalty: prepend 0 for depot
        td["penalty"] = torch.cat([torch.zeros(*bs, 1, device=device), td["penalty"]], dim=-1)  # [B, N+1]

        # Expected prize is the deterministic prize (for policy conditioning)
        td["expected_prize"] = td["deterministic_prize"]  # [B, N]

        td["current_node"] = torch.zeros(*bs, dtype=torch.long, device=device)
        td["visited"] = torch.zeros(*bs, num_nodes, dtype=torch.bool, device=device)
        td["cur_total_prize"] = torch.zeros(*bs, dtype=torch.float32, device=device)
        # Total penalty = sum of all customer penalties (they're "owed" until visited)
        td["cur_total_penalty"] = td["penalty"][..., 1:].sum(-1)  # [B]

        if self.generator is None:
            raise ValueError("Generator must be initialized for PCTSP environment.")
        td["prize_required"] = torch.full((*bs,), self.generator.prize_required, device=device, dtype=torch.float32)

        td["tour"] = torch.zeros(*bs, 0, dtype=torch.long, device=device)
        td["tour_length"] = torch.zeros(*bs, dtype=torch.float32, device=device)

        td["reward"] = torch.zeros(*bs, dtype=torch.float32, device=device)
        td["terminated"] = torch.zeros(*bs, dtype=torch.bool, device=device)
        td["truncated"] = torch.zeros(*bs, dtype=torch.bool, device=device)

        return td

    def _step(self, td: TensorDict) -> TensorDict:
        """Execute one step in the environment.

        Args:
            td: Input TensorDict containing action and state information.

        Returns:
            TensorDict: Updated TensorDict with new state and reward information.
        """
        return OpsMixin._step(self, td)

    def _step_instance(self, td: TensorDict) -> TensorDict:
        """
        Execute one PCTSP action.

        When visiting a customer node, the real prize is collected and the
        node's penalty is removed from the running total (it is "saved").

        Args:
            td: Input TensorDict containing action and state information.

        Returns:
            TensorDict: Updated TensorDict with new state and reward information.
        """
        action = td["action"]
        if action.dim() > 1:
            action = action.squeeze(-1)
        if action.dim() == 0:
            action = action.unsqueeze(0)

        current = td["current_node"]
        if current.dim() > 1:
            current = current.squeeze(-1)

        locs = td["locs"]
        current_loc = locs.gather(1, current[:, None, None].expand(-1, -1, 2)).squeeze(1)
        next_loc = locs.gather(1, action[:, None, None].expand(-1, -1, 2)).squeeze(1)
        distance = torch.norm(next_loc - current_loc, dim=-1)

        td["tour_length"] = td["tour_length"] + distance

        # Collect real prize and remove penalty for visited node
        prize_here = td["real_prize"].gather(-1, action.unsqueeze(-1)).squeeze(-1)
        penalty_here = td["penalty"].gather(-1, action.unsqueeze(-1)).squeeze(-1)

        td["cur_total_prize"] = td["cur_total_prize"] + prize_here
        td["cur_total_penalty"] = td["cur_total_penalty"] - penalty_here

        td["visited"] = td["visited"].scatter(1, action.unsqueeze(-1), True)
        td["current_node"] = action
        td["tour"] = torch.cat([td["tour"], action.unsqueeze(-1)], dim=-1)

        return td

    def _check_done(self, td: TensorDict) -> torch.Tensor:
        """Done when agent returns to depot after at least one step.

        Args:
            td: Input TensorDict containing action and state information.

        Returns:
            BoolTensor [*B] — True for completed episodes.
        """
        current = td["current_node"]
        if current.dim() > 1:
            current = current.squeeze(-1)
        step = td["i"].squeeze(-1) if td["i"].dim() > 1 else td["i"]
        return (current == 0) & (step > 0)

    def _get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """
        Depot (index 0) is blocked until the agent has collected at least
        ``prize_required`` total prize, unless all customers are visited.

        All already-visited nodes are masked out.

        Args:
            td: Input TensorDict containing action and state information.

        Returns:
            BoolTensor [*B, num_nodes] — True indicates a selectable action.
        """
        visited = td["visited"]  # [B, N+1]
        # Also mask out nodes reachable from an already-done episode (depot visited)
        depot_done = visited[..., 0:1]  # [B, 1]
        mask = ~(visited | depot_done)  # [B, N+1]

        # Depot: only open when prize_required is met (or all customers visited)
        prize_met = td["cur_total_prize"] >= td["prize_required"]
        all_visited = visited[..., 1:].all(dim=-1)
        depot_open = prize_met | all_visited
        mask[..., 0] = depot_open

        return mask

    def _get_reward(self, td: TensorDictBase, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute reward for PCTSP: saved_penalties - (tour_length + remaining_penalties).

        ``saved_penalties`` are the penalties of visited customer nodes.
        ``remaining_penalties`` = total penalty of all unvisited customer nodes
        (still owed at episode end).

        Args:
            td: Final state TensorDict containing ``penalty``, ``tour_length``,
                and ``cur_total_penalty``.
            actions: Optional tour action sequence ``(batch, tour_len)``;
                when provided, penalties are gathered directly from this tensor.

        Returns:
            torch.Tensor: Scalar reward per batch element, shape ``(batch,)``.
        """
        penalty = td["penalty"]  # [B, N+1] with depot=0

        # OpsMixin._step passes the single step's action; only a [batch, steps] sequence is gathered.
        if actions is not None and actions.dim() == 2 and actions.size(-1) > 0:
            # Gather penalties of visited customers
            saved_pen = penalty.gather(1, actions).sum(-1)
            # Remaining penalties (all customer penalties - saved)
            remaining_pen = penalty[..., 1:].sum(-1) - saved_pen
        else:
            saved_pen = penalty[..., 1:].sum(-1) - td["cur_total_penalty"]
            remaining_pen = td["cur_total_penalty"]

        return saved_pen - (td["tour_length"] + remaining_pen)


class PCTSP(BaseProblem):
    """Prize-Collecting TSP (PCTSP) task definition."""

    NAME = "pctsp"

    @staticmethod
    def evaluate_tour(
        dataset: Dict[str, Any],
        pi: torch.Tensor,
        dist_matrix: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Compute PCTSP objective and outcomes."""
        if pi.size(-1) <= 1:
            pen = dataset["penalty"].sum(-1) if "penalty" in dataset else torch.zeros(pi.size(0), device=pi.device)
            return {
                "cost": pen,
                "reward": -pen,
                "saved_penalty": torch.zeros_like(pen),
                "length": torch.zeros_like(pen),
                "valid": torch.ones_like(pen, dtype=torch.bool),
                "total": pen,
            }

        penalty = dataset.get("penalty", torch.zeros(pi.size(0), pi.size(1), device=pi.device))
        if penalty.shape[-1] == pi.max().item():
            penalty = torch.cat([torch.zeros_like(penalty[:, :1]), penalty], dim=-1)

        saved_penalty = penalty.gather(1, pi).sum(-1)
        length = PCTSP.get_tour_length(dataset, pi, dist_matrix)
        total_penalty = penalty.sum(-1)
        remaining_penalty = total_penalty - saved_penalty
        neg_reward = length + remaining_penalty

        return {
            "cost": neg_reward,
            "reward": -neg_reward,
            "saved_penalty": saved_penalty,
            "length": length,
            "valid": torch.ones_like(neg_reward, dtype=torch.bool),
            "total": neg_reward,
        }


PCTSP_SPEC = ProblemSpec(
    name="pctsp",
    objective=lambda neg_reward, **kw: -float(neg_reward),
    constraints=(),
    metrics=("cost", "reward", "saved_penalty", "length"),
    env_target="logic.src.envs.tasks.pctsp:PCTSPEnv",
    expand_pool=True,
)
PROBLEMS.register(PCTSP_SPEC)

__all__ = ["PCTSPGenerator", "PCTSPEnv", "PCTSP", "PCTSP_SPEC"]
