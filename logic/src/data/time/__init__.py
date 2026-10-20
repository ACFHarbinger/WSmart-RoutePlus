"""Temporal data loading for the simulator.

This package provides loaders for time-related simulation constraints:

- ``travel_matrix``: travel-time matrices between bins (directed, in hours).
- ``time_windows``:  day-level routing windows that mark simulation days as
  open (collections allowed) or closed (no routing).

Submodules
----------
- ``travel_matrix``:  ``compute_time_matrix`` — load or compute a travel-time
  matrix aligned to simulation coordinates.
- ``time_windows``:   ``load_time_windows``, ``build_time_window_pattern``,
  ``load_time_windows_file``, ``TimeWindowSchedule`` — build day-level routing schedules
  from a CSV / Excel schedule file or from a periodic pattern specification.

Attributes:
    compute_time_matrix: Build a travel-time matrix from distances and speed, or
        from an ID-labelled CSV file.
    load_time_windows: Build a TimeWindowSchedule from a file, pattern, or dictionary.
    build_time_window_pattern: Generate a periodic TimeWindowSchedule from 3 parameters.
    load_time_windows_file: Load a TimeWindowSchedule from CSV or Excel.
    TimeWindowSchedule: Class representing day-level routing availability.
"""

from logic.src.utils.registry import lazy_getattr

_LAZY = {
    "compute_time_matrix": "logic.src.data.time.travel_matrix:compute_time_matrix",
    "load_time_windows": "logic.src.data.time.time_windows:load_time_windows",
    "build_time_window_pattern": "logic.src.data.time.time_windows:build_time_window_pattern",
    "load_time_windows_file": "logic.src.data.time.time_windows:load_time_windows_file",
    "TimeWindowSchedule": "logic.src.data.time.time_windows:TimeWindowSchedule",
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    return lazy_getattr(globals(), _LAZY, name)
