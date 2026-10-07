"""Lookahead selection strategy.

This module provides a predictive selection strategy that marks bins for
collection based on when they are expected to overflow, using current fill
levels and accumulation rates to calculate a dynamic lookahead horizon.

The algorithm mirrors the scalar reference implementation in
``logic/src/policies/mandatory_selection/selection_lookahead.py``:

1. Mark bins whose fill level will reach capacity **today** (fill + rate ≥ max).
2. If none are found, return an empty mask — no constraints this step.
3. Simulate collection of the mandatory set (reset their fill to 0) and
   compute the absolute day each bin would overflow again from empty.
4. The earliest of those days is the **next collection day** (horizon anchor).
5. Include every non-mandatory bin that would overflow *before* that day,
   i.e. bins where ``fill + (next_collection_day − today − 1) × rate ≥ max``.

Attributes:
    LookaheadSelector: Vectorised predictive selection policy.

Example:
    >>> selector = LookaheadSelector(current_collection_day=0)
    >>> mask = selector.select(fill_levels, accumulation_rates=rates)
"""

from __future__ import annotations

from typing import Any, Optional, Union

import torch

from logic.src.constants import MAX_WASTE

from .base import VectorizedSelector


class LookaheadSelector(VectorizedSelector):
    """Vectorised predictive selection looking ahead to the next collection day.

    Selects bins that overflow today *and* any additional bins that would
    overflow before the next expected collection opportunity, mirroring the
    logic in ``LookaheadSelection.select_bins``.

    Attributes:
        current_collection_day: Default reference day for lookahead simulation.
    """

    def __init__(self, current_collection_day: int = 0, **kwargs: Any) -> None:
        """Initialise the lookahead selector.

        Args:
            current_collection_day: Current day in the collection cycle.
            kwargs: Additional keyword arguments (ignored).
        """
        self.current_collection_day = current_collection_day

    def select(
        self,
        fill_levels: torch.Tensor,
        accumulation_rates: Optional[torch.Tensor] = None,
        current_collection_day: Optional[Union[int, torch.Tensor]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Select bins predicted to overflow within the dynamic lookahead horizon.

        Args:
            fill_levels: Current fill levels, shape ``[B, N]``.
            accumulation_rates: Mean daily waste generation per node, shape ``[B, N]``.
                When ``None`` no prediction is possible and an empty mask is returned.
            current_collection_day: Override for the reference day; may be an
                ``int``, a scalar ``Tensor``, or a per-batch ``Tensor`` of shape ``[B]``.
            kwargs: Additional keyword arguments (ignored).

        Returns:
            torch.Tensor: Boolean mask ``[B, N]`` where ``True`` means the bin
            must be collected.  Depot (index 0) is always ``False``.
        """
        overflow_thresh: float = MAX_WASTE

        if accumulation_rates is None:
            # Without rates no prediction is possible — select nothing.
            mandatory = torch.zeros_like(fill_levels, dtype=torch.bool)
            mandatory[:, 0] = False
            return mandatory

        # ------------------------------------------------------------------
        # Step 1: Bins that overflow *today* (fill + one day of accumulation
        # already reaches capacity).  Mirrors ``_should_bin_be_collected``.
        # ------------------------------------------------------------------
        initial_mandatory = (fill_levels + accumulation_rates) >= overflow_thresh  # [B, N]

        # ------------------------------------------------------------------
        # Step 2: Compute the day each mandatory bin (reset to 0 after
        # collection) would overflow again.
        # Mirrors ``_calculate_next_collection_days``:
        #   starting from fill = 0, accumulate `rate` per day until overflow.
        #   → days needed = ceil(MAX_CAPACITY / rate)  (rate > 0)
        # Only bins that are both mandatory AND have a positive rate count;
        # mandatory bins with rate ≤ 0 can be ignored for the horizon anchor.
        # ------------------------------------------------------------------
        day: Union[int, torch.Tensor]
        day = current_collection_day if current_collection_day is not None else self.current_collection_day

        # Broadcast day to [B, 1] when it is a per-batch tensor.
        if isinstance(day, torch.Tensor):
            day_bc = (
                day.unsqueeze(1).to(dtype=fill_levels.dtype, device=fill_levels.device)
                if day.dim() > 0
                else day.to(dtype=fill_levels.dtype, device=fill_levels.device)
            )
        else:
            day_bc = float(day)  # scalar — broadcasts freely

        # Absolute day of re-overflow from zero fill.
        # Only mandatory bins with rate > 0 can anchor the horizon; bins whose
        # rate is zero would never overflow again and must not set the anchor.
        valid_mandatory = initial_mandatory & (accumulation_rates > 0)  # [B, N]
        has_valid_mandatory = valid_mandatory.any(dim=1)  # [B]

        rates_safe = accumulation_rates.clamp(min=1e-9)
        days_from_zero = torch.ceil(
            torch.tensor(overflow_thresh, device=fill_levels.device, dtype=fill_levels.dtype) / rates_safe
        )  # [B, N]
        abs_reoverflow_day = day_bc + days_from_zero  # [B, N]

        # ------------------------------------------------------------------
        # Step 3: Earliest re-overflow among the valid mandatory set.
        # Mirrors ``_get_next_collection_day``.
        # For batch instances with no valid mandatory bin (all rates ≤ 0),
        # set the anchor to LARGE so the expansion horizon collapses to zero.
        # ------------------------------------------------------------------
        LARGE = 1e9
        anchor_per_bin = torch.where(
            valid_mandatory,
            abs_reoverflow_day,
            torch.full_like(abs_reoverflow_day, LARGE),
        )
        next_collection_day = anchor_per_bin.min(dim=1).values  # [B]

        # ------------------------------------------------------------------
        # Step 4: Expand — add non-mandatory bins that would overflow
        # *before* the next collection day.
        # Mirrors ``_add_bins_to_collect``:
        #   for j in range(today+1, next_collection_day):
        #       if fill[i] + j * rate[i] >= MAX_CAPACITY: add i
        # The worst-case day in that range is (next_collection_day − 1).
        # Using relative days from today:
        #   additional accumulation = (next_collection_day − 1 − today) × rate
        # ------------------------------------------------------------------
        # Relative horizon (days between today and next_collection_day − 1).
        relative_horizon = (next_collection_day.unsqueeze(1) - day_bc - 1.0).clamp(min=0.0)  # [B, 1]

        predicted_fill = fill_levels + relative_horizon * accumulation_rates  # [B, N]
        candidates = predicted_fill >= overflow_thresh  # [B, N]

        # Expansion only applies to instances that have a valid horizon anchor
        # (i.e. at least one mandatory bin with rate > 0).  Instances whose
        # only mandatory bins have rate = 0 get no expansion — the reference
        # returns only the zero-rate mandatory bin in that case.
        candidates = candidates & has_valid_mandatory.unsqueeze(1)  # [B, N]

        mandatory = initial_mandatory | candidates  # [B, N]

        # Depot is never mandatory.
        mandatory = mandatory.clone()
        mandatory[:, 0] = False
        return mandatory
