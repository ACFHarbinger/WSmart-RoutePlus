"""
TSPkopt Environment implementation.

Improvement-based Traveling Salesman Problem (TSP) supporting iterative
local search moves (e.g. 2-opt).

Attributes:
    TSPkoptEnv: Class definition of TSP Environment for improvement-based methods.

Example:
    >>> from logic.src.envs.generators import TSPGenerator
    >>> generator = TSPGenerator(num_nodes=10, generator_params={"num_nodes": 10})
    >>> env = TSPkoptEnv(generator)
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from tensordict import TensorDict

from logic.src.envs.base.improvement import ImprovementEnvBase
from logic.src.envs.generators import TSPGenerator


class TSPkoptEnv(ImprovementEnvBase):
    """
    TSP Environment for improvement-based methods (k-opt).

    State contains a 'solution' field (tour) which is modified by actions.
    Actions are typically pairs of indices to swap or reverse sections.

    Attributes:
        generator: TSPGenerator object
        generator_params: Dictionary of generator parameters
        device: Device to run the environment on
    """

    name: str = "tsp_kopt"

    def __init__(
        self,
        generator: Optional[TSPGenerator] = None,
        generator_params: Optional[dict] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs,
    ):
        """Initialize TSPkoptEnv.

        Args:
            generator: TSPGenerator object
            generator_params: Dictionary of generator parameters
            device: Device to run the environment on
            kwargs: Additional keyword arguments
        """
        generator_params = generator_params or kwargs
        if generator is None:
            generator = TSPGenerator(**generator_params, device=device)

        super().__init__(generator, generator_params, device, **kwargs)

    def _step(self, td: TensorDict) -> TensorDict:
        """Execute step via OpsMixin implementation.

        Args:
            td: TensorDict containing the state.

        Returns:
            TensorDict containing the next state.
        """
        from logic.src.envs.base.ops import OpsMixin

        return OpsMixin._step(self, td)

    def _get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """For improvement moves, all nodes are typically valid targets.
        Return a mask that allows all nodes.

        Args:
            td: TensorDict containing the state.

        Returns:
            torch.Tensor: Action mask.
        """
        bs = td.batch_size
        num_nodes = td["locs"].shape[-2] + 1
        return torch.ones(*bs, num_nodes, dtype=torch.bool, device=self.device)

    def _get_initial_solution(self, td: TensorDict, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Generate random initial tour starting at depot.

        Args:
            td: TensorDict containing the state.
            generator: PyTorch random generator.

        Returns:
            torch.Tensor: Initial tour.
        """
        if generator is None:
            generator = torch.Generator(device=self.device)
        bs = td.batch_size[0]
        num_nodes = td["locs"].shape[-2] + 1

        # Shuffle nodes 1 to N
        tour = torch.stack(
            [torch.randperm(num_nodes - 1, device=self.device, generator=generator) + 1 for _ in range(bs)]
        )

        # Prepend depot (0)
        depot = torch.zeros(bs, 1, dtype=torch.long, device=self.device)
        full_tour = torch.cat([depot, tour], dim=1)

        return full_tour

    def _step_instance(self, td: TensorDict) -> TensorDict:
        """Apply a 2-opt local search move to the solution.

        Action contract:
            The action tensor `td["action"]` of shape `[batch, 2]` specifies either:
            1. Node IDs (standard for neural improvement decoders): `action[b] = (node_u, node_v)`.
               The environment resolves each node ID to its current index in `solution[b]` and
               reverses the subpath between them (reversing slice `idx_min + 1 : idx_max + 1`).
            2. Tour positions (explicit positional indexing): when `td["action_is_position"] = True`
               for that batch element. No implicit positional fallback is allowed.
            Only two-column actions are supported. NeuOpt basis sequences and N2S
            request reinsertion require their own transition implementation.

        Args:
            td: TensorDict containing 'solution' [batch, N] and 'action' [batch, 2].

        Returns:
            TensorDict: Updated state with the modified 'solution'.
        """
        action = td["action"]  # [batch, 2]
        solution = td["solution"]

        if action.ndim != 2 or action.shape != (solution.size(0), 2):
            raise ValueError("TSPkoptEnv supports exactly two action columns; basis sequences require a separate executor")
        if action.is_floating_point() or action.is_complex() or action.dtype == torch.bool:
            raise ValueError("2-opt actions must contain integer node IDs or positions")
        i, j = action[:, 0], action[:, 1]
        is_pos_val = td.get("action_is_position", None)
        if isinstance(is_pos_val, torch.Tensor):
            modes = is_pos_val.reshape(-1)
            if modes.numel() not in (1, solution.size(0)):
                raise ValueError("action_is_position must be scalar or contain one flag per instance")
            modes = modes.expand(solution.size(0))
        else:
            modes = [bool(is_pos_val)] * solution.size(0)

        new_solution = solution.clone()
        for b in range(solution.size(0)):
            sol_b = solution[b]
            val_i = i[b].item()
            val_j = j[b].item()

            if bool(modes[b]):
                if not (0 <= val_i < sol_b.numel() and 0 <= val_j < sol_b.numel()):
                    raise ValueError("2-opt action positions are outside the tour")
                idx_i, idx_j = min(val_i, val_j), max(val_i, val_j)
            else:
                # Map node IDs to positions in current tour
                pos_i_match = (sol_b == val_i).nonzero(as_tuple=True)[0]
                pos_j_match = (sol_b == val_j).nonzero(as_tuple=True)[0]
                if len(pos_i_match) > 0 and len(pos_j_match) > 0:
                    pos_i = pos_i_match[0].item()
                    pos_j = pos_j_match[0].item()
                    idx_i, idx_j = min(pos_i, pos_j), max(pos_i, pos_j)
                else:
                    raise ValueError("2-opt action node IDs must be present in the tour")

            if idx_i < idx_j:
                new_solution[b, idx_i + 1 : idx_j + 1] = solution[b, idx_i + 1 : idx_j + 1].flip(0)

        td["solution"] = new_solution
        return td

    def _get_reward(self, td: TensorDict, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return negative tour length of current solution.

        Args:
            td: TensorDict containing the state.
            actions: Optional torch.Tensor containing the actions.

        Returns:
            torch.Tensor: Negative tour length.
        """
        solution = td["solution"]

        depot = td["depot"]
        customers = td["locs"]
        locs = torch.cat([depot.unsqueeze(1), customers], dim=1)

        # Gather coordinates in tour order
        d = locs.gather(1, solution.unsqueeze(-1).expand(-1, -1, 2))

        # Total distance: sum segments + closed loop
        length = (d[:, 1:] - d[:, :-1]).norm(p=2, dim=-1).sum(1) + (d[:, -1] - d[:, 0]).norm(p=2, dim=-1)

        return -length

    def _get_initial_reward(self, td: TensorDict) -> torch.Tensor:
        """Compute reward for initial solution.

        Args:
            td: TensorDict containing the state.

        Returns:
            torch.Tensor: Initial reward.
        """
        return self._get_reward(td)

    def _check_done(self, td: TensorDict) -> torch.Tensor:
        """
        Usually improvement continues for fixed steps or until converge.

        Here we define a max_steps entry in td if present.

        Args:
            td: TensorDict containing the state.

        Returns:
            torch.Tensor: Done mask.
        """
        if "max_steps" in td.keys():
            return td["i"] >= td["max_steps"]
        return torch.zeros(td.batch_size, dtype=torch.bool, device=self.device)
