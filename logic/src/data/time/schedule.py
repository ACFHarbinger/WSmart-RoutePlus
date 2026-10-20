"""Schedule representations and pattern generation for simulation time windows.

This module provides the core data structures and generator functions for time-window
routing availability across simulation days.

Attributes:
    TimeWindowSchedule: Class representing day-level routing availability.
    build_time_window_pattern: Generate a periodic TimeWindowSchedule from 3 parameters.
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def _parse_bool(value: Any) -> bool:
    """Parse various representations of booleans into a native bool.

    Args:
        value: Scalar value to interpret (bool, int, float, str).

    Returns:
        Interpreted boolean value.

    Raises:
        ValueError: If the value cannot be parsed as a boolean.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            raise ValueError("Boolean value cannot be NaN")
        if value in (0.0, 1.0):
            return bool(int(value))
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in ("true", "t", "yes", "y", "1", "1.0", "allowed", "open", "active"):
            return True
        if cleaned in ("false", "f", "no", "n", "0", "0.0", "blocked", "closed", "forbidden", "none"):
            return False
    raise ValueError(f"Cannot parse value '{value}' (type {type(value).__name__}) as a boolean")


class TimeWindowSchedule:
    """Represents a day-by-day routing availability schedule over a simulation horizon.

    Simulation days are 1-indexed (day 1 to day n_days).

    Attributes:
        n_days: Total number of days in the simulation horizon.
        mask: 1D boolean numpy array of shape (n_days + 1,) where mask[d] indicates
            whether routing is allowed on day d (1 <= d <= n_days).
    """

    def __init__(self, allowed_mask: Sequence[bool], n_days: Optional[int] = None) -> None:
        """Initialize TimeWindowSchedule from an allowed-day mask.

        Args:
            allowed_mask: Sequence of booleans. Can be of length n_days (0-indexed for days 1..n_days)
                or length n_days + 1 (1-indexed, index 0 ignored).
            n_days: Optional explicit horizon length. If not specified, inferred from allowed_mask.
        """
        raw = [_parse_bool(v) for v in allowed_mask]
        if n_days is not None:
            if n_days <= 0:
                raise ValueError("n_days must be a positive integer (>= 1)")
            if len(raw) == n_days:
                self._n_days = n_days
                self._mask = np.zeros(n_days + 1, dtype=bool)
                self._mask[1 : n_days + 1] = raw
            elif len(raw) == n_days + 1:
                self._n_days = n_days
                self._mask = np.asarray(raw, dtype=bool)
            elif len(raw) > n_days:
                self._n_days = n_days
                self._mask = np.zeros(n_days + 1, dtype=bool)
                self._mask[1 : n_days + 1] = raw[:n_days]
            else:
                raise ValueError(
                    f"allowed_mask has {len(raw)} entries, which is fewer than requested n_days={n_days}"
                )
        else:
            if not raw:
                raise ValueError("allowed_mask cannot be empty")
            self._n_days = len(raw)
            self._mask = np.zeros(self._n_days + 1, dtype=bool)
            self._mask[1 : self._n_days + 1] = raw

    @property
    def n_days(self) -> int:
        """Total number of simulation days."""
        return self._n_days

    @property
    def mask(self) -> np.ndarray:
        """1-indexed boolean array of shape (n_days + 1,) where mask[day] is routing availability."""
        return self._mask

    @property
    def allowed_days(self) -> List[int]:
        """List of 1-based day numbers on which routing is allowed."""
        return [d for d in range(1, self._n_days + 1) if self._mask[d]]

    @property
    def blocked_days(self) -> List[int]:
        """List of 1-based day numbers on which routing is prohibited."""
        return [d for d in range(1, self._n_days + 1) if not self._mask[d]]

    def is_routing_allowed(self, day: int) -> bool:
        """Check whether routing is permitted on a specific simulation day.

        Args:
            day: Simulation day index (1-based, 1 <= day <= n_days).

        Returns:
            True if routing is allowed, False if the day is closed.

        Raises:
            IndexError: If day is outside the valid range [1, n_days].
        """
        if not (1 <= day <= self._n_days):
            raise IndexError(f"Simulation day {day} out of range [1, {self._n_days}]")
        return bool(self._mask[day])

    def as_array(self) -> np.ndarray:
        """Return a 0-indexed 1D boolean numpy array of shape (n_days,) for days 1..n_days."""
        return self._mask[1 : self._n_days + 1].copy()

    def to_dict(self) -> Dict[int, bool]:
        """Return a mapping of day numbers (1..n_days) to allowed booleans."""
        return {d: bool(self._mask[d]) for d in range(1, self._n_days + 1)}

    def to_series(self) -> pd.Series:
        """Return a pandas Series indexed by day (1..n_days)."""
        days = list(range(1, self._n_days + 1))
        return pd.Series(self.as_array(), index=days, name="routing_allowed")

    def __getitem__(self, day: int) -> bool:
        """1-based day lookup for routing availability."""
        return self.is_routing_allowed(day)

    def __contains__(self, day: int) -> bool:
        """Check if routing is allowed on day."""
        if 1 <= day <= self._n_days:
            return self.is_routing_allowed(day)
        return False

    def __len__(self) -> int:
        """Length of the simulation horizon in days."""
        return self._n_days

    def __repr__(self) -> str:
        allowed_count = sum(1 for d in range(1, self._n_days + 1) if self._mask[d])
        blocked_count = self._n_days - allowed_count
        return (
            f"TimeWindowSchedule(n_days={self._n_days}, allowed={allowed_count}, "
            f"blocked={blocked_count}, blocked_days={self.blocked_days})"
        )


def build_time_window_pattern(
    start_day: int,
    days_after: int,
    interval_between: int,
    n_days: int,
) -> TimeWindowSchedule:
    """Construct a periodic time-window schedule from 3 parameters.

    Args:
        start_day: The first day of the simulation where a window with no collections occurs.
            Must be >= 1.
        days_after: Number of consecutive days with no collections immediately after the start day.
            Must be >= 0. (A window duration is 1 + days_after days).
        interval_between: Number of days between each time window with no collections
            where routing is allowed. Must be >= 0.
        n_days: Total number of days in the simulation temporal horizon. Must be >= 1.

    Returns:
        A TimeWindowSchedule instance covering days 1..n_days.

    Raises:
        ValueError: If any parameter is out of valid bounds.
    """
    if start_day < 1:
        raise ValueError(f"start_day must be >= 1 (got {start_day})")
    if days_after < 0:
        raise ValueError(f"days_after must be >= 0 (got {days_after})")
    if interval_between < 0:
        raise ValueError(f"interval_between must be >= 0 (got {interval_between})")
    if n_days < 1:
        raise ValueError(f"n_days must be >= 1 (got {n_days})")

    mask = [True] * n_days
    current_window_start = start_day
    window_length = 1 + days_after

    while current_window_start <= n_days:
        window_end = min(current_window_start + window_length - 1, n_days)
        for day in range(current_window_start, window_end + 1):
            mask[day - 1] = False

        current_window_start = current_window_start + window_length + interval_between
        if window_length + interval_between <= 0:
            break

    return TimeWindowSchedule(mask, n_days=n_days)


__all__ = [
    "_parse_bool",
    "TimeWindowSchedule",
    "build_time_window_pattern",
]
