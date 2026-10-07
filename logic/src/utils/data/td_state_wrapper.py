"""
TensorDict State Wrapper for legacy decoder compatibility.

Attributes:
    TensorDictStateWrapper: A wrapper for TensorDict that exposes methods expected by legacy model components (Decoders).

Example:
    >>> td = TensorDict({"action_mask": torch.tensor([[True, False], [False, True]])}, batch_size=[2])
    >>> wrapper = TensorDictStateWrapper(td)
    >>> wrapper.get_mask()
    tensor([[False,  True], [ True, False]])
"""

from typing import Optional

import torch
from tensordict import TensorDict


class TensorDictStateWrapper:
    """
    Wraps a TensorDict to expose methods expected by legacy model components (Decoders).

    Acts as a bridge between the new TensorDict state management and the
    legacy object-oriented state access patterns (e.g. state.get_mask()).

    Attributes:
        td: The TensorDict containing the state.
        problem_name: The name of the problem.
        env: The environment.
        ids: The IDs for batching.
        dist_matrix: The distance matrix.
        waste_with_depot: The waste with depot.
    """

    def __init__(self, td: TensorDict, problem_name: str = "vrpp", env=None):
        """Initialize TensorDictStateWrapper.

        Args:
            td: The TensorDict containing the state.
            problem_name: The name of the problem.
            env: The environment.
        """
        self.td = td
        self.problem_name = problem_name
        self.env = env

        # AttentionDecoder loop expects these
        if "ids" in td.keys():
            self.ids = td["ids"]
        else:
            # Default IDs for batching
            bs = td.batch_size[0] if len(td.batch_size) > 0 else 1
            self.ids = torch.arange(bs, device=td.device).unsqueeze(-1)

        # Expose common properties directly
        self.dist_matrix = td.get("dist", None)

        # Handle 'waste_with_depot' for partial updates (now standardized to 'waste')
        self.waste_with_depot = td.get("waste")

    def get_mask(self) -> Optional[torch.Tensor]:
        """
        Get action mask from TensorDict.

        Returns:
            The action mask.
        """
        # RL4CO envs provide "action_mask" where True means VALID.
        # AttentionDecoder expects mask where True means INVALID (masked out).
        # So we invert it.
        if "action_mask" in self.td.keys():
            mask = ~self.td["action_mask"]
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            return mask
        return None

    def get_edges_mask(self) -> Optional[torch.Tensor]:
        """
        Get edge mask from TensorDict.

        Returns:
            The edge mask.
        """
        return self.td.get("graph_mask", None)

    def get_current_node(self) -> torch.Tensor:
        """
        Get current node (last visited).

        Returns:
            The current node.
        """
        return self.td["current_node"].long()

    def get_current_profit(self) -> torch.Tensor:
        """
        For VRPP: get cumulative collected waste.

        Returns:
            The cumulative collected waste.
        """
        # This is used for context embedding.
        val = self.td.get("collected_waste", torch.zeros(self.td.batch_size, device=self.td.device))
        if val.dim() == 1:
            val = val.unsqueeze(-1)
        return val

    def get_current_efficiency(self) -> torch.Tensor:
        """
        Get current efficiency (legacy placeholder).

        Returns:
            The current efficiency.
        """
        # Legacy placeholder
        val = torch.zeros(self.td.batch_size, device=self.td.device)
        return val.unsqueeze(-1)

    def get_remaining_overflows(self) -> torch.Tensor:
        """
        Get remaining overflows (legacy placeholder).

        Returns:
            The remaining overflows.
        """
        # Legacy placeholder or value from td
        val = self.td.get("remaining_overflows", torch.zeros(self.td.batch_size, device=self.td.device))
        if val.dim() == 1:
            val = val.unsqueeze(-1)
        return val

    def update(self, action: torch.Tensor) -> "TensorDictStateWrapper":
        """
        Update state by taking an action in the environment.

        Args:
            action: The action to take.

        Returns:
            The updated state.
        """
        if self.env is None:
            raise ValueError("Environment (env) must be provided to TensorDictStateWrapper for updates.")

        # Prepare action for environment
        self.td["action"] = action
        next_td = self.env.step(self.td)["next"]
        return TensorDictStateWrapper(next_td, self.problem_name, self.env)

    def all_finished(self) -> bool:
        """
        Check if all instances in batch are finished.

        Returns:
            True if all instances are finished, False otherwise.
        """
        if "done" in self.td.keys():
            return self.td["done"].all().item()
        return False

    def get_finished(self) -> torch.Tensor:
        """
        Get finished mask for the batch.

        Returns:
            The finished mask.
        """
        return self.td.get("done", torch.zeros(self.td.batch_size, dtype=torch.bool, device=self.td.device))

    def __getitem__(self, key):
        """
        Allow slicing or indexing the state.

        Args:
            key: The key to index by.

        Returns:
            The indexed state.
        """
        return TensorDictStateWrapper(self.td[key], self.problem_name, self.env)
