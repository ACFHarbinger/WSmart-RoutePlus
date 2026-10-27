"""Simulation time windows module for day-by-day routing availability constraints.

This module provides file loading and unified dispatching utilities for optional time
window constraints where no routing/collections can occur on designated simulation days.

Time windows can be specified via:
1. A CSV or Excel file associating each simulation day with a boolean flag
   indicating whether routing is allowed.
2. A periodic pattern defined by 3 parameters:
   - ``start_day``: The first simulation day where a no-collection window occurs.
   - ``days_after``: The number of consecutive no-collection days following the start day.
   - ``interval_between``: The number of routing-allowed days between no-collection windows.

Example:
    >>> from logic.src.data.time.schedule import build_time_window_pattern
    >>> from logic.src.data.time.time_windows import load_time_windows
    >>> schedule = build_time_window_pattern(start_day=5, days_after=2, interval_between=4, n_days=30)
    >>> schedule.is_routing_allowed(day=5)
    False
    >>> schedule.is_routing_allowed(day=1)
    True
"""

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union, cast

import pandas as pd

from logic.src.constants import ROOT_DIR
from logic.src.data.time.schedule import (
    TimeWindowSchedule,
    _parse_bool,
    build_time_window_pattern,
)


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
    possible_day_names = {"day", "simulation_day", "sim_day", "daily", "date", "step", "idx", "day_id"}
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
        return load_time_windows_file(cast(Union[str, Path], fp), n_days=n_days)
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
            return load_time_windows_file(cast(Union[str, Path], fp), n_days=n_days)
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
        import numpy as np

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
    "load_time_windows_file",
    "load_time_windows",
]
