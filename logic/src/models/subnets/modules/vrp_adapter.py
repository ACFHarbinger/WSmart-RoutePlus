"""VRP Adapter for GLOP.

This module provides the VRPAdapter, which specializes in converting partitioned
global nodes into independent Vehicle Routing Problem (VRP) subproblems,
specifically handling depot insertion and capacity management.

Attributes:
    VRPAdapter: Mapping system for Vehicle Routing Problem subproblems.

Example:
    >>> import torch
    >>> from tensordict import TensorDict
    >>> from logic.src.models.subnets.modules.vrp_adapter import VRPAdapter
    >>> td = TensorDict({"locs": torch.randn(1, 10, 2)}, batch_size=[1])
    >>> actions = torch.zeros(1, 10)
    >>> adapter = VRPAdapter(td, actions)
    >>> # Iterate through subproblems...
"""

from __future__ import annotations

from typing import Any, Iterator, List, Optional, cast

import torch
from tensordict import TensorDict

from .adapter_base import SubproblemAdapter
from .subproblem_mapping import SubproblemMapping


class VRPAdapter(SubproblemAdapter):
    """Adapter for extracting and reintegrating VRP-style subproblems.

    Groups global nodes into clusters and prepends the common depot to each cluster
    to form independent CVRP instances. Solved routes are consolidated back into
    the global plan.

    Attributes:
        subproblem_env_name (str): Identifier for the target CVRP solver environment.
        capacity (float): Uniform vehicle capacity constraint for all sub-instances.
        _subproblems (List[SubproblemMapping]): Internal cache of extracted segments.
    """

    subproblem_env_name: str = "cvrp"

    def __init__(
        self,
        td: TensorDict,
        partition_actions: torch.Tensor,
        subprob_batch_size: int = 2000,
        capacity: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes VRPAdapter.

        Args:
            td: TensorDict containing the global problem state and constraints.
            partition_actions: Cluster labels for each node in the scene.
            subprob_batch_size: Threshold for batching subproblems during optimization.
            capacity: Specific capacity value to enforce (defaults to `td` value).
            kwargs: Additional configuration parameters for subproblem extraction.
        """
        super().__init__(td, partition_actions, subprob_batch_size)

        capacity_value = cast(torch.Tensor, td.get("vehicle_capacity", torch.tensor(1.0)))
        self.capacity = capacity or capacity_value.item()
        self._subproblems: List[SubproblemMapping] = []
        self._extract_subproblems()

    def _extract_subproblems(self) -> None:
        """Extracts subproblems and ensures each receives a depot at index 0."""
        for batch_idx in range(self.batch_size):
            coords = cast(torch.Tensor, self.coordinates)[batch_idx]  # coordinates are stored as a tensor
            actions = self.partition_actions[batch_idx]

            # First node is assumed to be the depot
            depot = coords[0:1]

            # Group nodes by route assignment
            unique_routes = actions.unique()

            for route_idx, route_id in enumerate(unique_routes):
                mask = actions == route_id
                indices = torch.where(mask)[0]

                # Skip empty or depot-only clusters
                if len(indices) < 1:
                    continue

                # Prepare the local VRP instance coordinates
                subprob_coords = torch.cat([depot, coords[indices]], dim=0)

                self._subproblems.append(
                    SubproblemMapping(
                        batch_idx=batch_idx,
                        sample_idx=0,
                        subprob_coordinates=subprob_coords,
                        original_indices=indices,
                        route_idx=route_idx,
                    )
                )

    def get_batched_subprobs(self) -> Iterator[SubproblemMapping]:
        """Yields extracted VRP subproblems.

        Yields:
            SubproblemMapping: Mapping for a single VRP subproblem.
        """
        for subprob in self._subproblems:
            yield subprob

    def update_actions(
        self,
        mapping: SubproblemMapping,
        subprob_actions: torch.Tensor,
    ) -> None:
        """Reintegrates an optimized local VRP route back into the global plan.

        Args:
            mapping: The mapping descriptor for the solved subproblem.
            subprob_actions: Ordered node indices within the subproblem space.
        """
        original_indices = mapping.original_indices
        batch_idx = mapping.batch_idx

        # Map subproblem actions (which include depot) back to original labels
        for pos, node_idx in enumerate(subprob_actions):
            if node_idx == 0:
                # Local index 0 is always the depot
                self.final_actions[batch_idx, pos] = 0
            elif node_idx <= len(original_indices):
                # Map local node back to original node ID
                original_node = original_indices[node_idx - 1]
                self.final_actions[batch_idx, pos] = original_node
