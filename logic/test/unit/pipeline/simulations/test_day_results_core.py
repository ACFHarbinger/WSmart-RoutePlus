

import pandas as pd
import pytest
from logic.src.pipeline.simulations.day_context import get_daily_results

pytestmark = [pytest.mark.unit, pytest.mark.fast]





class TestDayResults:
    """Class for daily results calculation tests."""

    @pytest.mark.unit
    def test_get_daily_results_happy_path(self):
        """Test successful results aggregation."""
        coords = pd.DataFrame({"ID": ["D", "B1", "B2"]}, index=[0, 1, 2])
        res = get_daily_results(
            total_collected=500.0,
            ncol=2,
            cost=20.0,
            tour=[0, 1, 2, 0],
            day=1,
            new_overflows=0,
            sum_lost=0.0,
            coordinates=coords,
            profit=480.0,
            time=0.1
        )
        assert res["day"] == 1
        assert res["kg"] == 500.0
        assert res["km"] == 20.0
        assert res["ncol"] == 2
        assert res["tour"] == [0, "B1", "B2", 0]

    @pytest.mark.unit
    def test_get_daily_results_empty_tour(self):
        """An empty solver tour must not erase the pre-solver mandatory set."""
        coords = pd.DataFrame({"ID": [0, 101, 202]}, index=[0, 1, 2])
        res = get_daily_results(
            total_collected=0.0,
            ncol=0,
            cost=0.0,
            tour=[0],
            day=1,
            new_overflows=1,
            sum_lost=10.0,
            coordinates=coords,
            profit=0.0,
            time=0.1,
            mandatory_nodes=[1, 2],
        )
        assert res["kg"] == 0
        assert res["km"] == 0
        assert res["overflows"] == 1
        assert res["tour"] == [0]
        assert res["mandatory_nodes"] == [101, 202]
