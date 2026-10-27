"""Batched Torch adapters of the released packing rules (P-37).

These functions implement ``packing.spec`` on tensors of shape ``[B, N]``
with the depot at index 0. Fill levels, rates and standard deviations are
fractions in ``[0, 1]``; overflow is ``MAX_WASTE``. There are no per-instance
Python loops (RQ-gemini-1 applies to this adapter).

Attributes:
    last_minute: Fill-threshold rule.
    regular: Periodic collection on scheduled days.
    service_level: Linear confidence-bound overflow projection.
    lookahead: Seed today's overflows and bundle the next trip.
"""

from typing import Optional, Union

import torch

from logic.src.constants import MAX_WASTE


def _clear_depot(mask: torch.Tensor) -> torch.Tensor:
    """Return a clone with depot column 0 set to False."""
    out = mask.clone()
    out[:, 0] = False
    return out


def last_minute(fill_levels: torch.Tensor, threshold: float) -> torch.Tensor:
    """Collect customers whose fill is at or above ``threshold`` percent.

    Args:
        fill_levels: Current fills ``[B, N]`` as fractions.
        threshold: Inclusive trigger in percent (70 means 0.70).

    Returns:
        torch.Tensor: Boolean mask ``[B, N]``; depot is False.
    """
    return _clear_depot(fill_levels >= float(threshold) / 100.0)


def regular(
    fill_levels: torch.Tensor,
    current_day: Optional[Union[torch.Tensor, int]],
    frequency: float,
    min_fill: float = 0.0,
) -> torch.Tensor:
    """Collect eligible customers on scheduled days.

    Scheduled days are ``1, 1+period, …``. A missing day or a period below
    one day (non-positive included) collects every day. Eligibility is ``fill >= min_fill`` (fraction).

    Args:
        fill_levels: Current fills ``[B, N]`` as fractions.
        current_day: Day index, scalar or ``[B]``.
        frequency: Collection period in days.
        min_fill: Minimum fill fraction on a collection day.

    Returns:
        torch.Tensor: Boolean mask ``[B, N]``; depot is False.
    """
    batch_size, _num_nodes = fill_levels.shape
    device = fill_levels.device
    eligible = fill_levels >= float(min_fill)
    period = float(frequency)

    if period < 1 or current_day is None:  # int(period) would be 0 for a sub-daily period
        return _clear_depot(eligible)

    day = current_day if isinstance(current_day, torch.Tensor) else torch.tensor(current_day, device=device)
    if day.dim() == 0:
        day = day.expand(batch_size)
    is_collection_day = ((day - 1) % int(period)) == 0
    return _clear_depot(eligible & is_collection_day.unsqueeze(-1))


def service_level(
    fill_levels: torch.Tensor,
    rates: Optional[torch.Tensor],
    stds: Optional[torch.Tensor],
    z: float,
    horizon_days: int,
    capacity: float = MAX_WASTE,
) -> torch.Tensor:
    """Collect bins whose worst-case linear projection reaches capacity.

    Args:
        fill_levels: Current fills ``[B, N]`` as fractions.
        rates: Daily accumulation ``[B, N]``.
        stds: Daily accumulation std ``[B, N]``.
        z: Confidence factor.
        horizon_days: Projection length in days.
        capacity: Overflow level in the same units as ``fill_levels``.

    Returns:
        torch.Tensor: Boolean mask ``[B, N]``; depot is False.
    """
    if rates is None or stds is None:
        return _clear_depot(torch.zeros_like(fill_levels, dtype=torch.bool))
    horizon = float(horizon_days)
    predicted = fill_levels + horizon * rates + horizon * float(z) * stds
    return _clear_depot(predicted >= float(capacity))


def lookahead(
    fill_levels: torch.Tensor,
    rates: Optional[torch.Tensor],
    current_collection_day: Union[int, torch.Tensor] = 0,
    capacity: float = MAX_WASTE,
) -> torch.Tensor:
    """Seed today's overflows and bundle bins that overflow before the next trip.

    Same four steps as ``packing.spec.lookahead``, in fraction units.

    Args:
        fill_levels: Current fills ``[B, N]`` as fractions.
        rates: Daily accumulation ``[B, N]``.
        current_collection_day: Reference day.
        capacity: Overflow level in the same units as ``fill_levels``.

    Returns:
        torch.Tensor: Boolean mask ``[B, N]``; depot is False.
    """
    if rates is None:
        return _clear_depot(torch.zeros_like(fill_levels, dtype=torch.bool))

    cap = float(capacity)
    initial = (fill_levels + rates) >= cap
    valid = initial & (rates > 0)
    has_valid = valid.any(dim=1)

    day_bc: Union[float, torch.Tensor]
    if isinstance(current_collection_day, torch.Tensor):
        day_bc = (
            current_collection_day.unsqueeze(1).to(dtype=fill_levels.dtype, device=fill_levels.device)
            if current_collection_day.dim() > 0
            else current_collection_day.to(dtype=fill_levels.dtype, device=fill_levels.device)
        )
    else:
        day_bc = float(current_collection_day)

    rates_safe = rates.clamp(min=1e-9)
    days_from_zero = torch.ceil(torch.tensor(cap, device=fill_levels.device, dtype=fill_levels.dtype) / rates_safe)
    abs_reoverflow = day_bc + days_from_zero
    large = torch.full_like(abs_reoverflow, 1e9)
    next_day = torch.where(valid, abs_reoverflow, large).min(dim=1).values
    relative_horizon = (next_day.unsqueeze(1) - day_bc - 1.0).clamp(min=0.0)
    candidates = ((fill_levels + relative_horizon * rates) >= cap) & has_valid.unsqueeze(1)
    return _clear_depot(initial | candidates)
