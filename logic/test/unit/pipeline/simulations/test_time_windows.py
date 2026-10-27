"""Unit tests for simulation time windows and non-routing day constraints."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from logic.src.configs.envs.graph import GraphConfig
from logic.src.configs.tasks.sim import SimConfig
from logic.src.data.time.schedule import build_time_window_pattern
from logic.src.data.time.time_windows import (
    load_time_windows,
    load_time_windows_file,
)
from logic.src.pipeline.simulations.daily import SimulationDayContext, run_day
from logic.src.pipeline.simulations.repository import load_time_windows as repo_load_time_windows
from logic.src.pipeline.simulations.states.initializing import InitializingState

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_build_time_window_pattern_basic():
    """Verify 3-parameter pattern generator with start_day, days_after, interval_between."""
    schedule = build_time_window_pattern(start_day=5, days_after=2, interval_between=4, n_days=30)
    assert len(schedule) == 30
    assert schedule.n_days == 30

    expected_blocked = [5, 6, 7, 12, 13, 14, 19, 20, 21, 26, 27, 28]
    assert schedule.blocked_days == expected_blocked

    expected_allowed = [d for d in range(1, 31) if d not in expected_blocked]
    assert schedule.allowed_days == expected_allowed

    for day in range(1, 31):
        assert schedule.is_routing_allowed(day) == (day not in expected_blocked)
        assert schedule[day] == (day not in expected_blocked)
        assert (day in schedule) == (day not in expected_blocked)

    # Conversion helpers
    arr = schedule.as_array()
    assert len(arr) == 30
    assert arr[0] is np.True_
    assert arr[4] is np.False_

    s_dict = schedule.to_dict()
    assert len(s_dict) == 30
    assert s_dict[5] is False
    assert s_dict[8] is True

    series = schedule.to_series()
    assert len(series) == 30
    assert bool(series.loc[5]) is False
    assert bool(series.loc[8]) is True


def test_build_time_window_pattern_zero_days_after():
    """When days_after=0, only the start day of each window is closed."""
    schedule = build_time_window_pattern(start_day=2, days_after=0, interval_between=1, n_days=5)
    assert schedule.blocked_days == [2, 4]
    assert schedule.allowed_days == [1, 3, 5]


def test_build_time_window_pattern_zero_interval():
    """When interval_between=0, windows touch."""
    schedule = build_time_window_pattern(start_day=2, days_after=1, interval_between=0, n_days=6)
    assert schedule.blocked_days == [2, 3, 4, 5, 6]
    assert schedule.allowed_days == [1]


def test_build_time_window_pattern_start_beyond_horizon():
    """When start_day > n_days, all days remain allowed."""
    schedule = build_time_window_pattern(start_day=15, days_after=2, interval_between=3, n_days=10)
    assert len(schedule.blocked_days) == 0
    assert schedule.allowed_days == list(range(1, 11))


@pytest.mark.parametrize(
    "start_day,days_after,interval_between,n_days",
    [
        (0, 1, 1, 10),
        (-1, 1, 1, 10),
        (1, -1, 1, 10),
        (1, 1, -1, 10),
        (1, 1, 1, 0),
        (1, 1, 1, -5),
    ],
)
def test_build_time_window_pattern_invalid_parameters(start_day, days_after, interval_between, n_days):
    """Invalid parameter bounds must raise ValueError."""
    with pytest.raises(ValueError):
        build_time_window_pattern(start_day, days_after, interval_between, n_days)


def test_time_window_schedule_bounds():
    """Day lookups outside [1, n_days] must raise IndexError."""
    schedule = build_time_window_pattern(start_day=2, days_after=1, interval_between=2, n_days=5)
    with pytest.raises(IndexError):
        schedule.is_routing_allowed(0)
    with pytest.raises(IndexError):
        schedule.is_routing_allowed(6)
    assert (0 in schedule) is False
    assert (6 in schedule) is False


def test_load_time_windows_csv_two_columns(tmp_path):
    """Load time windows from a 2-column CSV file."""
    csv_file = tmp_path / "windows.csv"
    csv_file.write_text("day,routing_allowed\n1,True\n2,False\n3,True\n4,False\n5,True\n")
    schedule = load_time_windows_file(csv_file, n_days=5)
    assert schedule.allowed_days == [1, 3, 5]
    assert schedule.blocked_days == [2, 4]


def test_load_time_windows_csv_single_column(tmp_path):
    """Load time windows from a 1-column CSV file."""
    csv_file = tmp_path / "single_col.csv"
    csv_file.write_text("routing_allowed\nTrue\nFalse\nFalse\nTrue\n")
    schedule = load_time_windows_file(csv_file, n_days=4)
    assert schedule.allowed_days == [1, 4]
    assert schedule.blocked_days == [2, 3]


def test_load_time_windows_excel(tmp_path):
    """Load time windows from an Excel file."""
    excel_file = tmp_path / "windows.xlsx"
    df = pd.DataFrame({"day": [1, 2, 3, 4], "allowed": [True, True, False, True]})
    df.to_excel(excel_file, index=False)

    schedule = load_time_windows_file(excel_file, n_days=4)
    assert schedule.allowed_days == [1, 2, 4]
    assert schedule.blocked_days == [3]


def test_load_time_windows_file_missing_days(tmp_path):
    """A file missing days required by n_days must raise ValueError."""
    csv_file = tmp_path / "incomplete.csv"
    csv_file.write_text("day,allowed\n1,True\n2,False\n")
    with pytest.raises(ValueError, match="missing entries"):
        load_time_windows_file(csv_file, n_days=5)


def test_load_time_windows_flexible_sources(tmp_path):
    """Verify load_time_windows accepts tuples, dicts, strings, and files."""
    assert load_time_windows(None) is None

    sched1 = load_time_windows((3, 1, 2), n_days=10)
    assert sched1 is not None
    assert sched1.blocked_days == [3, 4, 7, 8]

    sched2 = load_time_windows([3, 1, 2], n_days=10)
    assert sched2 is not None
    assert sched2.blocked_days == [3, 4, 7, 8]

    sched3 = load_time_windows("3, 1, 2", n_days=10)
    assert sched3 is not None
    assert sched3.blocked_days == [3, 4, 7, 8]

    sched4 = load_time_windows({"start_day": 3, "days_after": 1, "interval_between": 2}, n_days=10)
    assert sched4 is not None
    assert sched4.blocked_days == [3, 4, 7, 8]

    assert load_time_windows(sched1) is sched1

    csv_file = tmp_path / "test.csv"
    csv_file.write_text("day,routing\n1,1\n2,0\n3,1\n")
    sched5 = load_time_windows(filepath=str(csv_file), n_days=3)
    assert sched5 is not None
    assert sched5.allowed_days == [1, 3]
    assert sched5.blocked_days == [2]

    sched6 = repo_load_time_windows(source=(2, 0, 1), n_days=4)
    assert sched6 is not None
    assert sched6.blocked_days == [2, 4]


def test_run_day_on_blocked_day():
    """Verify that on a day with routing_allowed=False, no routing occurs but bins fill."""
    bins = MagicMock()
    bins.n = 3
    bins.real_c = np.array([20.0, 30.0, 40.0])
    bins.c = np.array([20.0, 30.0, 40.0])
    bins.revenue = 0.5
    bins.expenses = 1.0
    bins.means = np.array([5.0, 5.0, 5.0])
    bins.load_filling.return_value = (1, np.array([10.0, 10.0, 10.0]), np.array([30.0, 40.0, 50.0]), 5.0)

    coords = pd.DataFrame({"ID": [0, 101, 102, 103], "Lat": [0, 1, 2, 3], "Lng": [0, 1, 2, 3]})
    dist_matrix = np.zeros((4, 4))

    context = SimulationDayContext(
        graph_size=4,
        policy_name="test_policy",
        bins=bins,
        coords=coords,
        distance_matrix=dist_matrix,
        day=2,
        routing_allowed=False,
    )

    result_ctx = run_day(context)

    bins.load_filling.assert_called_once_with(2)
    bins.collect.assert_not_called()

    assert result_ctx.tour == [0]
    assert result_ctx.cost == 0.0
    assert result_ctx.profit == 0.0
    assert result_ctx.ncol == 0
    assert result_ctx.total_collected == 0.0
    assert result_ctx.solver_status == "no_routing"
    assert result_ctx.mandatory == []

    dlog = result_ctx.daily_log
    assert dlog is not None
    assert dlog["day"] == 2
    assert dlog["solver_status"] == "no_routing"
    assert dlog["ncol"] == 0
    assert dlog["km"] == 0.0
    assert dlog["kg"] == 0.0
    assert dlog["collection_days"] == 0.0
    assert dlog["overflows"] == 1


def test_initialization_state_loads_time_windows(tmp_path, monkeypatch):
    """Verify InitializingState configures ctx.time_windows from config."""
    graph = GraphConfig(num_loc=3, n_days=10)
    sim = SimConfig(seed=42, graph=graph, time_windows=(4, 1, 2))
    ctx = SimpleNamespace(
        cfg=SimpleNamespace(sim=sim),
        data_dir=str(tmp_path),
        pol_name="tsp",
        transition_to=MagicMock(),
        results_dir=str(tmp_path),
        pol_id_orig="tsp",
        sample_id=0,
    )
    state = InitializingState()
    for name in ("_setup_logging_and_dirs", "_load_all_configurations", "_setup_models", "_load_checkpoint_if_needed"):
        monkeypatch.setattr(state, name, lambda ctx, n=name: (None, 0) if "checkpoint" in n else None)
    monkeypatch.setattr(
        "logic.src.pipeline.simulations.states.initializing.setup_basedata", lambda *args: (None, None, None)
    )
    monkeypatch.setattr(
        "logic.src.pipeline.simulations.states.initializing.load_area_and_waste_type_params",
        lambda *args: (100.0, 0.5, 20.0, 1.0, 2.5),
    )

    def restore_or_init(ctx, *args):
        ctx.coords = pd.DataFrame({"ID": [0, 1, 2, 3]})
        ctx.dist_tup = (np.zeros((4, 4)), None, None, None)

    monkeypatch.setattr(state, "_initialize_new_state", restore_or_init)
    state.handle(ctx)

    assert ctx.time_windows is not None
    assert ctx.time_windows.is_routing_allowed(1) is True
    assert ctx.time_windows.is_routing_allowed(4) is False
    assert ctx.time_windows.is_routing_allowed(5) is False
    assert ctx.time_windows.is_routing_allowed(6) is True
