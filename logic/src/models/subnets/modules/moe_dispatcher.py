"""Sparse Dispatcher for Mixture of Experts.

This module provides the SparseDispatcher, a utility for efficient routing of
batch elements to different experts in a Mixture-of-Experts (MoE) layer and
the subsequent combination of their results.

Attributes:
    SparseDispatcher: Helper for routing inputs to experts and combining outputs.

Example:
    >>> import torch
    >>> from logic.src.models.subnets.modules.moe_dispatcher import SparseDispatcher
    >>> gates = torch.tensor([[0.8, 0.2, 0.0], [0.0, 0.5, 0.5]])
    >>> dispatcher = SparseDispatcher(num_experts=3, gates=gates)
    >>> x = torch.randn(2, 128)
    >>> expert_inputs = dispatcher.dispatch(x)
    >>> # Processes expert_inputs...
"""

from __future__ import annotations

from typing import List

import torch


class SparseDispatcher:
    """Efficient batch routing and combining for Mixture-of-Experts.

    Creates specialized micro-batches for each expert by filtering the master batch
    based on gating decisions. It handles the mapping of samples to active experts
    and provides a mechanism to aggregate the weighted outputs.

    Attributes:
        _gates (torch.Tensor): Raw gating weights for the entire batch.
        _num_experts (int): Total number of parallel expert networks.
        _expert_index (torch.Tensor): Sorted indices of experts selected for each sample.
        _batch_index (torch.Tensor): Mapping from expert-local indices back to global.
        _part_sizes (List[int]): Number of samples assigned to each expert.
        _nonzero_gates (torch.Tensor): Gating weights for exactly the active selections.
    """

    def __init__(self, num_experts: int, gates: torch.Tensor) -> None:
        """Initializes SparseDispatcher.

        Args:
            num_experts: Number of экспертов in the MoE layer.
            gates: Sparse gating weights of shape (batch, num_experts).
        """
        self._gates = gates
        self._num_experts = num_experts

        # Sort experts to create contiguous micro-batches
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)

        # Separate sorted results
        _, self._expert_index = sorted_experts.split(1, dim=1)

        # Extract according global batch index for each expert assignment
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]

        # Calculate number of samples that each expert receives
        self._part_sizes = (gates > 0).sum(0).tolist()

        # Gather gating weights for exactly these active pairs
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp: torch.Tensor) -> List[torch.Tensor]:
        """Creates specialized input tensors for each expert.

        Expert `i` receives a tensor containing only the slices of `inp` where
        `gates[b, i] > 0`.

        Args:
            inp: Global input tensor of shape (batch_size, feature_dim).

        Returns:
            List[torch.Tensor]: A list of length `num_experts` containing tensors of
                expert-dependent batch sizes.
        """
        # Assign samples to experts whose gate is nonzero
        inp_exp = inp[self._batch_index].squeeze(1)
        return list(torch.split(inp_exp, self._part_sizes, dim=0))

    def combine(self, expert_out: List[torch.Tensor], multiply_by_gates: bool = True) -> torch.Tensor:
        """Aggregates expert outputs into a single batch tensor.

        Results are weighted by their corresponding gate values before summation
        at overlap positions.

        Args:
            expert_out: List of output tensors from each expert.
            multiply_by_gates: If True, scales outputs by gating intensity.

        Returns:
            torch.Tensor: Combined output tensor of shape (batch_size, feature_dim).
        """
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)

        # Prepare accumulator
        zeros = torch.zeros(
            self._gates.size(0),
            expert_out[-1].size(-1),
            requires_grad=True,
            device=stitched.device,
        )

        # Sum contributions based on original batch indices
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined
