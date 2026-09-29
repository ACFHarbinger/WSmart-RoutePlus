"""Regression tests for issue #41: solver status on empty tours."""

import pandas as pd
import pytest
from logic.src.pipeline.simulations.day_context import get_daily_results, record_aborted_day, resolve_solver_status
from logic.src.pipeline.simulations.solver_status import (
    current_solver_status,
    format_backend_status,
    note_solver_status,
    reset_solver_status,
)
from logic.src.tracking.logging.modules.analysis import daily_row_is_visible

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_empty_tour_keeps_mandatory_set_and_solver_status():
    """An empty tour must keep the mandatory set and the solver status."""
    coords = pd.DataFrame({"ID": [0, 101, 202]}, index=[0, 1, 2])
    res = get_daily_results(
        total_collected=0.0,
        ncol=0,
        cost=0.0,
        tour=[0],
        day=16,
        new_overflows=4,
        sum_lost=10.0,
        coordinates=coords,
        profit=0.0,
        time=12.0,
        mandatory_nodes=[1, 2],
        solver_status="gurobi:INFEASIBLE",
    )
    assert res["tour"] == [0]
    assert res["kg"] == 0
    assert res["mandatory_nodes"] == [101, 202]
    assert res["solver_status"] == "gurobi:INFEASIBLE"


def test_resolve_solver_status_prefers_the_published_code():
    """A published backend code wins over the empty-tour label and the exception."""
    assert resolve_solver_status("gurobi:TIME_LIMIT", [0, 0], RuntimeError("boom")) == "gurobi:TIME_LIMIT"
    assert resolve_solver_status(None, [0, 0], RuntimeError("boom")) == "RuntimeError: boom"
    assert resolve_solver_status(None, [0, 1, 0], None) == "ok"
    assert resolve_solver_status(None, [0], None) == "empty_tour"


def test_backend_status_names():
    """Gurobi and OR-Tools codes used by the Figueira rerun decode to stable names."""
    assert format_backend_status("gurobi", 3) == "gurobi:INFEASIBLE"
    assert format_backend_status("gurobi", 9) == "gurobi:TIME_LIMIT"
    assert format_backend_status("ortools", 2) == "ortools:INFEASIBLE"
    reset_solver_status()
    assert current_solver_status() is None
    note_solver_status(format_backend_status("gurobi", 3))
    assert current_solver_status() == "gurobi:INFEASIBLE"
    reset_solver_status()


def test_aborted_day_records_mandatory_set_and_status():
    """A constructor exception still leaves a daily record for that day."""
    coords = pd.DataFrame({"ID": [0, 11, 22]})
    ctx = {
        "mandatory": [1, 2],
        "coords": coords,
        "tour": [0, 0],
        "day": 16,
        "solver_status": "gurobi:INFEASIBLE",
    }
    dlog = record_aborted_day(ctx, RuntimeError("SWC-TCF model is infeasible (Gurobi status 3)."))
    assert dlog["mandatory_nodes"] == [11, 22]
    assert dlog["solver_status"] == "gurobi:INFEASIBLE"
    assert dlog["tour"] == [0]
    assert ctx["daily_log"]["mandatory_nodes"] == [11, 22]


def test_zero_km_day_stays_visible_when_it_has_a_mandatory_set_or_status():
    """The printed daily table must not hide the day that stopped collecting."""
    assert daily_row_is_visible(0.0, [101, 202], "gurobi:INFEASIBLE") is True
    assert daily_row_is_visible(0.0, [], "gurobi:TIME_LIMIT") is True
    assert daily_row_is_visible(0.0, [], "skipped") is False
    assert daily_row_is_visible(3.0, [], "ok") is True
