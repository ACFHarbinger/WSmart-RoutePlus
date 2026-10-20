"""Simulation time windows module for day-by-day routing availability constraints.

This module provides data structures and loading utilities for optional time window
constraints where no routing/collections can occur on designated simulation days.

Time windows can be specified via:
1. A CSV or Excel file associating each simulation day with a boolean flag
   indicating whether routing is allowed.
2. A periodic pattern defined by 3 parameters:
   - ``start_day``: The first simulation day where a no-collection window occurs.
   - ``days_after``: The number of consecutive no-collection days following the start day.
   - ``interval_between``: The number of routing-allowed days between no-collection windows.

Example:
    >>> from logic.src.data.time.time_windows import load_time_windows, build_time_window_pattern
    >>> schedule = build_time_window_pattern(start_day=5, days_after=2, interval_between=4, n_days=30)
    >>> schedule.is_routing_allowed(day=5)
    False
    >>> schedule.is_routing_allowed(day=1)
    True
"""

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from logic.src.constants import ROOT_DIR


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


def _resolve_time_window_file(filepath: Union[str, Path]) -> Path:
    """Resolve a file path for time windows against standard simulator locations."""
    path = Path(filepath)
    if path.is_file():
        return path.resolve()

    if path.parent == Path("."):
        candidates = [
            Path(ROOT_DIR) / "data/simulator/time_windows" / path,
            Path(ROOT_DIR) / "data/simulator" / path,
            Path(ROOT_DIR) / path,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()

    root_relative = Path(ROOT_DIR) / path
    if root_relative.is_file():
        return root_relative.resolve()

    raise FileNotFoundError(f"Time windows file not found: {filepath}")


def _detect_series(df: pd.DataFrame, resolved: Path) -> Tuple[pd.Series, pd.Series]:
    """Detect day and availability series from a loaded DataFrame."""
    possible_day_names = {"day", "simulation_day", "sim_day", "date", "step", "idx", "day_id"}
    possible_bool_names = {
        "routing_allowed",
        "allowed",
        "can_route",
        "route",
        "routing",
        "is_allowed",
        "collection",
        "active",
        "open",
        "status",
        "value",
    }
    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    day_col = next((cols_lower[name] for name in possible_day_names if name in cols_lower), None)
    bool_col = next(
        (cols_lower[name] for name in possible_bool_names if name in cols_lower and cols_lower[name] != day_col),
        None,
    )

    if day_col is not None and bool_col is not None:
        return df[day_col], df[bool_col]
    if len(df.columns) == 2:
        return df.iloc[:, 0], df.iloc[:, 1]
    if len(df.columns) == 1:
        if df.index.name and str(df.index.name).lower() in possible_day_names:
            day_s = pd.Series(df.index)
        else:
            day_s = pd.Series(range(1, len(df) + 1))
        return day_s, df.iloc[:, 0]
    if bool_col is not None:
        return pd.Series(range(1, len(df) + 1)), df[bool_col]

    raise ValueError(
        f"Could not identify day and routing availability columns in '{resolved}'. "
        f"Columns present: {list(df.columns)}"
    )


def load_time_windows_file(
    filepath: Union[str, Path],
    n_days: Optional[int] = None,
) -> TimeWindowSchedule:
    """Load a time window schedule from a CSV or Excel file.

    The file should associate each simulation day with a true/false value
    indicating whether routing is allowed.

    Args:
        filepath: Path to the CSV or Excel file.
        n_days: Expected simulation horizon length. If provided, validates coverage.

    Returns:
        TimeWindowSchedule instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file content cannot be interpreted or is invalid.
    """
    resolved = _resolve_time_window_file(filepath)
    ext = resolved.suffix.lower()

    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(resolved)
    else:
        try:
            df = pd.read_csv(resolved)
        except Exception:
            df = pd.read_excel(resolved)

    if df.empty:
        raise ValueError(f"Time windows file '{resolved}' is empty")

    day_series, val_series = _detect_series(df, resolved)

    day_mapping: Dict[int, bool] = {}
    for raw_d, raw_v in zip(day_series, val_series, strict=False):
        try:
            d = int(float(raw_d))
        except (ValueError, TypeError):
            continue
        day_mapping[d] = _parse_bool(raw_v)

    if not day_mapping:
        raise ValueError(f"No valid day records found in '{resolved}'")

    if min(day_mapping.keys()) == 0 and 0 in day_mapping:
        day_mapping = {d + 1: v for d, v in day_mapping.items()}

    target_n_days = n_days if n_days is not None else max(day_mapping.keys())
    if target_n_days < 1:
        raise ValueError(f"Target n_days must be >= 1 (got {target_n_days})")

    missing = set(range(1, target_n_days + 1)) - set(day_mapping.keys())
    if missing:
        raise ValueError(
            f"Time windows file '{resolved}' is missing entries for days: {sorted(missing)[:10]} "
            f"(total missing: {len(missing)})"
        )

    mask = [day_mapping[d] for d in range(1, target_n_days + 1)]
    return TimeWindowSchedule(mask, n_days=target_n_days)


def _parse_pattern_mapping(source: Mapping, n_days: Optional[int], **kwargs: Any) -> Optional[TimeWindowSchedule]:
    """Parse a dictionary specifying pattern or file parameters."""
    s_day = source.get("start_day", source.get("start_date", source.get("start")))
    d_after = source.get("days_after", source.get("no_collection_days", source.get("duration_after", source.get("after"))))
    i_between = source.get("interval_between", source.get("days_between", source.get("gap", source.get("between"))))
    if s_day is not None and d_after is not None and i_between is not None:
        horizon = n_days if n_days is not None else int(source.get("n_days", kwargs.get("n_days", 31)))
        return build_time_window_pattern(
            start_day=int(s_day),
            days_after=int(d_after),
            interval_between=int(i_between),
            n_days=horizon,
        )
    if "filepath" in source or "file" in source:
        fp = source.get("filepath") or source.get("file")
        return load_time_windows_file(fp, n_days=n_days)
    return None


def load_time_windows(
    source: Optional[Any] = None,
    n_days: Optional[int] = None,
    **kwargs: Any,
) -> Optional[TimeWindowSchedule]:
    """Load or build a time window schedule from various supported sources.

    Args:
        source: Primary source specification for time windows.
        n_days: Total number of days in the simulation horizon.
        **kwargs: Fallback or additional parameters.

    Returns:
        TimeWindowSchedule instance if active, else None.
    """
    if source is None:
        if "filepath" in kwargs or "file" in kwargs:
            fp = kwargs.get("filepath") or kwargs.get("file")
            return load_time_windows_file(fp, n_days=n_days)
        if all(k in kwargs for k in ("start_day", "days_after", "interval_between")):
            horizon = n_days if n_days is not None else int(kwargs.get("n_days", 31))
            return build_time_window_pattern(
                start_day=int(kwargs["start_day"]),
                days_after=int(kwargs["days_after"]),
                interval_between=int(kwargs["interval_between"]),
                n_days=horizon,
            )
        return None

    if isinstance(source, TimeWindowSchedule):
        return source

    if isinstance(source, (tuple, list)):
        if len(source) == 3 and all(isinstance(x, (int, np.integer, float)) for x in source):
            horizon = n_days if n_days is not None else int(kwargs.get("n_days", 31))
            return build_time_window_pattern(
                start_day=int(source[0]),
                days_after=int(source[1]),
                interval_between=int(source[2]),
                n_days=horizon,
            )
        return TimeWindowSchedule(source, n_days=n_days)

    if isinstance(source, Mapping):
        parsed = _parse_pattern_mapping(source, n_days, **kwargs)
        if parsed is not None:
            return parsed

    if isinstance(source, (str, Path)):
        s_str = str(source).strip()
        if s_str.endswith((".csv", ".xlsx", ".xls")) or Path(s_str).is_file():
            return load_time_windows_file(s_str, n_days=n_days)
        parts = s_str.replace(";", ",").replace(" ", "").split(",")
        if len(parts) == 3:
            try:
                nums = [int(p) for p in parts]
                horizon = n_days if n_days is not None else int(kwargs.get("n_days", 31))
                return build_time_window_pattern(nums[0], nums[1], nums[2], n_days=horizon)
            except ValueError:
                pass
        try:
            return load_time_windows_file(s_str, n_days=n_days)
        except (FileNotFoundError, ValueError) as err:
            raise ValueError(f"Cannot interpret time_windows source: '{source}'") from err

    raise TypeError(f"Unsupported time_windows source type: {type(source).__name__}")


__all__ = [
    "TimeWindowSchedule",
    "build_time_window_pattern",
    "load_time_windows_file",
    "load_time_windows",
]
