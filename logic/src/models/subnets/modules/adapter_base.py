"""Base class for GLOP subproblem adapters.

This module defines the SubproblemAdapter abstract base class, which provides
a standard interface for converting global problem partitions into local,
solvable subproblems used in decomposition-based solvers (e.g., GLOP).

Attributes:
    SubproblemAdapter: Abstract base for mapping global partitions to subproblems.

Example:
    >>> # SubproblemAdapter is an abstract class. Implementations define
    >>> # how to slice specific problems (e.g., TSPAdapter).
    >>> class MyAdapter(SubproblemAdapter):
    ...     def get_batched_subprobs(self): yield from []
    ...     def update_actions(self, mapping, actions): pass
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, cast

import torch
from tensordict import TensorDict

from .subproblem_mapping import SubproblemMapping


class SubproblemAdapter(ABC):
    """Abstract base class for partition-to-subproblem adapters.

    Provides a contract for extracting local subproblems from a partitioned global
    problem instance and reintegrating solved sub-decisions back into the
    global action space.

    Attributes:
        subproblem_env_name (str): Type identifier of the subproblem solver environment.
        td (TensorDict): The original global problem instance data.
        partition_actions (torch.Tensor): Cluster assignments for each node.
        subprob_batch_size (int): Maximum batch size used when querying sub-solvers.
        coordinates (torch.Tensor): Location coordinates of all nodes in the scene.
        batch_size (int): Number of independent problem instances in the batch.
        final_actions (torch.Tensor): The evolving global sequence/assignment plan.
    """

    subproblem_env_name: str = "tsp"

    def __init__(
        self,
        td: TensorDict,
        partition_actions: torch.Tensor,
        subprob_batch_size: int = 2000,
    ) -> None:
        """Initializes SubproblemAdapter.

        Args:
            td: TensorDict containing the global problem state.
            partition_actions: Tensor of initial partition/clustering labels.
            subprob_batch_size: Threshold for batching subproblems during inference.
        """
        self.td = td
        self.partition_actions = partition_actions
        self.subprob_batch_size = subprob_batch_size

        # Store original coordinates
        self.coordinates = cast(torch.Tensor, td.get("locs", td.get("locations")))  # locs key holds coordinates
        self.batch_size = self.coordinates.size(0)

        # Initialize storage for final actions
        self.final_actions = torch.zeros_like(partition_actions)

    @abstractmethod
    def get_batched_subprobs(self) -> Iterator[SubproblemMapping]:
        """Iteratively yields metadata for batches of local subproblems.

        Yields:
            SubproblemMapping: Mapping descriptor for a local problem segment.
        """
        pass

    @abstractmethod
    def update_actions(
        self,
        mapping: SubproblemMapping,
        subprob_actions: torch.Tensor,
    ) -> None:
        """Reintegrates subproblem decisions back into the global decision tensor.

        Args:
            mapping: The mapping used to generate the local subproblem.
            subprob_actions: The optimized actions returned by the sub-solver.
        """
        pass

    def get_actions(self) -> torch.Tensor:
        """Retrieves the finalized global decision tensor.

        Returns:
            torch.Tensor: The integrated global action plan.
        """
        return self.final_actions
