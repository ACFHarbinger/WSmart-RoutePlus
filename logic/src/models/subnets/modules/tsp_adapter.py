"""TSP Adapter for GLOP.

This module provides the TSPAdapter, which specializes in converting partitioned
global nodes into independent Traveling Salesman Problem (TSP) subproblems for
efficient localized optimization.

Attributes:
    TSPAdapter: Mapping system for Traveling Salesman Problem subproblems.

Example:
    >>> import torch
    >>> from tensordict import TensorDict
    >>> from logic.src.models.subnets.modules.tsp_adapter import TSPAdapter
    >>> td = TensorDict({"locs": torch.randn(1, 10, 2)}, batch_size=[1])
    >>> actions = torch.zeros(1, 10)
    >>> adapter = TSPAdapter(td, actions)
    >>> # Iterate through subproblems...
"""

from __future__ import annotations

from typing import Any, Iterator, List

import torch
from tensordict import TensorDict

from .adapter_base import SubproblemAdapter
from .subproblem_mapping import SubproblemMapping


class TSPAdapter(SubproblemAdapter):
    """Adapter for extracting and reintegrating TSP subproblems.

    Groups global nodes into clusters based on partition labels and prepares
    them as standalone TSP instances. Solved tours are mapped back to the
    global sequence.

    Attributes:
        subproblem_env_name (str): Identifier for the target TSP solver environment.
        _subproblems (List[SubproblemMapping]): Internal cache of extracted segments.
    """

    subproblem_env_name: str = "tsp"

    def __init__(
        self,
        td: TensorDict,
        partition_actions: torch.Tensor,
        subprob_batch_size: int = 2000,
        **kwargs: Any,
    ) -> None:
        """Initializes TSPAdapter.

        Args:
            td: TensorDict containing the global problem state and coordinates.
            partition_actions: Cluster labels for each node in the scene.
            subprob_batch_size: Threshold for batching subproblems during optimization.
            kwargs: Additional configuration parameters for subproblem extraction.
        """
        super().__init__(td, partition_actions, subprob_batch_size)

        # Extract subproblems from partitions
        self._subproblems: List[SubproblemMapping] = []
        self._extract_subproblems()

    def _extract_subproblems(self) -> None:
        """Parses unique partition IDs to build localized SubproblemMapping objects."""
        for batch_idx in range(self.batch_size):
            coords = self.coordinates[batch_idx]  # coordinates are stored as a tensor
            actions = self.partition_actions[batch_idx]

            # Group nodes by partition
            unique_partitions = actions.unique()

            for route_idx, partition_id in enumerate(unique_partitions):
                # Get nodes in this partition
                mask = actions == partition_id
                indices = torch.where(mask)[0]

                if len(indices) < 2:
                    continue

                # Get coordinates for this partition
                subprob_coords = coords[indices]

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
        """Yields extracted TSP subproblems in sequential batches.

        Yields:
            SubproblemMapping: Mapping for a single TSP subproblem.
        """
        batch = []
        for subprob in self._subproblems:
            batch.append(subprob)
            if len(batch) >= self.subprob_batch_size:
                for item in batch:
                    yield item
                batch = []

        # Yield remaining
        for item in batch:
            yield item

    def update_actions(
        self,
        mapping: SubproblemMapping,
        subprob_actions: torch.Tensor,
    ) -> None:
        """Reintegrates an optimized local TSP tour back into the global plan.

        Args:
            mapping: The mapping descriptor for the solved subproblem.
            subprob_actions: Ordered node indices within the subproblem space.
        """
        # Map subproblem solution back to original indices
        original_indices = mapping.original_indices
        batch_idx = mapping.batch_idx

        # The subprob_actions gives tour order within the subproblem
        for pos, node_idx in enumerate(subprob_actions):
            if node_idx < len(original_indices):
                original_node = original_indices[node_idx]
                self.final_actions[batch_idx, pos] = original_node
