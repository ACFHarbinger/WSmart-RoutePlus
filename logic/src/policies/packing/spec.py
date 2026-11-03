"""Canonical NumPy definitions of the released packing rules (P-37).

Each function is the single classical definition of a packing rule. The
simulator wrappers in this package call these functions; the batched Torch
adapters in ``vectorized/`` implement the same predicates on ``[B, N]``
tensors (depot at index 0, fills as fractions in ``[0, 1]``). Parity tests
keep the two backends in agreement (RQ-codex-1).

Units on this side match the simulator: fill levels, rates and standard
deviations are percent of capacity (0–100). Overflow is
``MAX_CAPACITY_PERCENT``.

Attributes:
    last_minute: Fill-threshold rule (1.0.0 last-minute; no EOQ).
    regular: Periodic collection on days first_day, first_day+period, …
    service_level: Linear confidence-bound overflow projection.
    lookahead: Seed today's overflows and bundle bins that overflow first.
    selected_ids: Convert a customer-level mask to 1-based bin IDs.
"""

from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from logic.src.constants import MAX_CAPACITY_PERCENT


def selected_ids(mask: NDArray[np.bool_]) -> List[int]:
    """Return 1-based customer IDs for a customer-level boolean mask.

    Args:
        mask: Per-customer decision, length ``N``.

    Returns:
        List[int]: Selected bin IDs in index order.
    """
    return (np.nonzero(np.asarray(mask, dtype=bool))[0] + 1).tolist()


def last_minute(fills: NDArray[np.floating], threshold: float) -> NDArray[np.bool_]:
    """Collect a bin when its fill is at or above the uniform threshold.

    1.0.0 last-minute is this fill-threshold rule only (RQ-cursor-3). EOQ
    last-minute is a separate unreleased key.

    Args:
        fills: Current fill levels, percent of capacity.
        threshold: Inclusive trigger, same units as ``fills``.

    Returns:
        NDArray[np.bool_]: Customer-level mask.
    """
    return np.asarray(fills, dtype=np.float64) >= float(threshold)


def regular(
    fills: NDArray[np.floating],
    current_day: int,
    frequency: float,
    min_fill: float = 0.0,
    first_day: int = 1,
) -> NDArray[np.bool_]:
    """Collect every eligible bin on scheduled days.

    Scheduled days are ``first_day, first_day+period, first_day+2*period, …`` and no day
    before ``first_day`` is scheduled. A ``frequency`` below one day (non-positive included)
    means every day from ``first_day``. Eligibility is ``fill >= min_fill``.

    Args:
        fills: Current fill levels, percent of capacity.
        current_day: Simulation day (1-based in the paper runs).
        frequency: Collection period in days (``context.threshold``).
        min_fill: Minimum fill to include on a collection day.
        first_day: First scheduled day (1 starts the calendar on the first simulated day).

    Returns:
        NDArray[np.bool_]: Customer-level mask.
    """
    values = np.asarray(fills, dtype=np.float64)
    eligible = values >= float(min_fill)
    period = float(frequency)
    offset = int(current_day) - int(first_day)
    if offset < 0:
        return np.zeros(values.shape, dtype=bool)
    if period < 1:  # int(period) would be 0 for a sub-daily period
        return eligible
    if offset % int(period) != 0:
        return np.zeros(values.shape, dtype=bool)
    return eligible


def service_level(
    fills: NDArray[np.floating],
    rates: Optional[NDArray[np.floating]],
    stds: Optional[NDArray[np.floating]],
    z: float,
    horizon_days: int,
    capacity: float = MAX_CAPACITY_PERCENT,
) -> NDArray[np.bool_]:
    """Collect bins whose worst-case linear projection reaches capacity.

    ``predicted = fill + horizon * rate + horizon * z * std``. Missing
    statistics yield an empty mask.

    Args:
        fills: Current fill levels, percent.
        rates: Daily accumulation, percent.
        stds: Daily accumulation std, percent.
        z: Confidence factor (``context.threshold``).
        horizon_days: Projection length in days.
        capacity: Overflow level in the same units as ``fills``.

    Returns:
        NDArray[np.bool_]: Customer-level mask.
    """
    values = np.asarray(fills, dtype=np.float64)
    if rates is None or stds is None:
        return np.zeros(values.shape, dtype=bool)
    horizon = float(horizon_days)
    predicted = (
        values + horizon * np.asarray(rates, dtype=np.float64) + horizon * float(z) * np.asarray(stds, dtype=np.float64)
    )
    return predicted >= float(capacity)


def lookahead(
    fills: NDArray[np.floating],
    rates: Optional[NDArray[np.floating]],
    current_collection_day: float = 0.0,
    capacity: float = MAX_CAPACITY_PERCENT,
) -> NDArray[np.bool_]:
    """Seed today's overflows and bundle bins that overflow before the next trip.

    1. Mark bins with ``fill + rate >= capacity`` (overflow today).
    2. If none of those have a positive rate, return the seed set.
    3. After collecting a seed bin its fill is 0; it overflows again in
       ``ceil(capacity / rate)`` days. The earliest of those days is the
       next-collection anchor.
    4. Add every other bin that would overflow strictly before that day:
       ``fill + (anchor - today - 1) * rate >= capacity``.

    Args:
        fills: Current fill levels, percent.
        rates: Daily accumulation, percent.
        current_collection_day: Reference day for the horizon.
        capacity: Overflow level in the same units as ``fills``.

    Returns:
        NDArray[np.bool_]: Customer-level mask.
    """
    values = np.asarray(fills, dtype=np.float64)
    if rates is None:
        return np.zeros(values.shape, dtype=bool)
    rates_arr = np.asarray(rates, dtype=np.float64)
    cap = float(capacity)
    day = float(current_collection_day)

    initial = (values + rates_arr) >= cap
    valid = initial & (rates_arr > 0.0)
    if not np.any(valid):
        return initial

    rates_safe = np.clip(rates_arr, 1e-9, None)
    days_from_zero = np.ceil(cap / rates_safe)
    abs_reoverflow = day + days_from_zero
    next_day = float(np.min(np.where(valid, abs_reoverflow, np.inf)))
    relative_horizon = max(next_day - day - 1.0, 0.0)
    candidates = (values + relative_horizon * rates_arr) >= cap
    return initial | candidates
