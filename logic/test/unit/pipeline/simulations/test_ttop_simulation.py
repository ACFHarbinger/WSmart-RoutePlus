"""Unit tests for Temporal Team Orienteering Problem (TTOP) simulation support.

Validates:
- Validation logic accepting 'ttop' as a supported problem type.
- Route splitting with dual constraints (vehicle capacity + working-shift time budget).
- CollectAction computing exact operational time spent (driving + service).
- SimulationDayContext and get_daily_results logging time_spent additively.
"""

from typing import Any, Dict
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from logic.src.configs import Config
from logic.src.constants.simulation import PROBLEMS
from logic.src.pipeline.features.test.validation import validate_sim_config
from logic.src.pipeline.simulations.actions.collection import CollectAction
from logic.src.pipeline.simulations.day_context import SimulationDayContext, get_daily_results
from logic.src.policies.route_construction.other_algorithms.travelling_salesman_problem.tsp import (
    get_multi_tour,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]


class TestTTOPValidation:
    """Tests that TTOP is a registered and validated problem variant."""

    def test_ttop_in_problems(self):
        assert "ttop" in PROBLEMS

    def test_validate_sim_config_accepts_ttop(self):
        cfg = Config()
        cfg.sim.problem = "ttop"
        cfg.sim.graph.n_days = 5
        cfg.sim.graph.n_samples = 1
        cfg.sim.graph.area = "figueiradafoz"
        cfg.sim.graph.waste_type = "plastic"
        cfg.sim.cpu_cores = 1

        validate_sim_config(cfg)
        assert cfg.sim.problem == "ttop"

    def test_validate_sim_config_rejects_unknown_problem(self):
        cfg = Config()
        cfg.sim.problem = "invalid_problem_xyz"
        cfg.sim.graph.n_days = 5
        cfg.sim.graph.n_samples = 1
        cfg.sim.graph.area = "figueiradafoz"

        with pytest.raises(AssertionError, match="Unknown problem"):
            validate_sim_config(cfg)


class TestTTOPRouteSplitting:
    """Tests dual-constraint splitting (capacity + time budget) in get_multi_tour."""

    def test_capacity_splitting_only(self):
        # 3 customer bins with 60 kg each, vehicle capacity 100 kg
        tour = [0, 1, 2, 3, 0]
        bins_waste = np.array([60.0, 60.0, 60.0])
        dist_matrix = np.ones((4, 4))
        np.fill_diagonal(dist_matrix, 0)

        # High shift hours (unconstrained time)
        split_tour = get_multi_tour(
            tour,
            bins_waste,
            max_capacity=100.0,
            distance_matrix=dist_matrix,
            shift_hours=100.0,
            avg_speed_kmh=35.0,
            service_time_h=0.025,
        )
        assert split_tour == [0, 1, 0, 2, 0, 3, 0]

    def test_time_budget_splitting_under_capacity(self):
        # 2 customer bins with 10 kg each (total 20 kg << 100 kg capacity)
        # Shift limit = 3.0 hours.
        tour = [0, 1, 2, 0]
        bins_waste = np.array([10.0, 10.0])
        # dist_matrix: 0<->1 is 35 km (1h), 1<->2 is 35 km (1h), 2<->0 is 35 km (1h)
        dist_matrix = np.array(
            [
                [0.0, 35.0, 35.0],
                [35.0, 0.0, 35.0],
                [35.0, 35.0, 0.0],
            ]
        )
        # Trip 0->1->2->0: dist = 35 + 35 + 35 = 105 km -> 105 / 35 = 3.0h + 2 * 0.1h = 3.2h > 3.0h
        # Trip 0->1->0: dist = 35 + 35 = 70 km -> 2.0h + 1 * 0.1h = 2.1h <= 3.0h
        split_tour = get_multi_tour(
            tour,
            bins_waste,
            max_capacity=100.0,
            distance_matrix=dist_matrix,
            shift_hours=3.0,
            avg_speed_kmh=35.0,
            service_time_h=0.1,
        )
        assert split_tour == [0, 1, 0, 2, 0]


class TestTTOPCollectionAction:
    """Tests operational time spent calculation and TTOP constraint validation in CollectAction."""

    def test_time_spent_calculation(self):
        bins = MagicMock()
        bins.collect.return_value = ([1, 2], 120.0, 2, 200.0)

        dist_matrix = np.array(
            [
                [0.0, 10.0, 20.0],
                [10.0, 0.0, 15.0],
                [20.0, 15.0, 0.0],
            ]
        )
        tour = [0, 1, 2, 0]  # raw_km = 10 + 15 + 20 = 45 km
        context: Dict[str, Any] = {
            "bins": bins,
            "tour": tour,
            "distance_matrix": dist_matrix,
            "avg_speed_kmh": 30.0,  # 45 / 30 = 1.5 h driving
            "service_time_h": 0.25,  # 2 bins * 0.25 = 0.5 h service
            "shift_hours": 7.0,
            "problem": "ttop",
        }

        action = CollectAction()
        action.execute(context)

        assert context["cost"] == 45.0
        assert context["ncol"] == 2
        assert context["time_spent"] == pytest.approx(1.5 + 0.5, rel=1e-5)

    def test_ttop_shift_violation_raises(self):
        bins = MagicMock()
        bins.collect.return_value = ([1], 50.0, 1, 100.0)

        dist_matrix = np.array(
            [
                [0.0, 100.0],
                [100.0, 0.0],
            ]
        )
        tour = [0, 1, 0]  # raw_km = 200 km
        context: Dict[str, Any] = {
            "bins": bins,
            "tour": tour,
            "distance_matrix": dist_matrix,
            "avg_speed_kmh": 20.0,  # 200 / 20 = 10.0 h driving
            "service_time_h": 0.5,  # 1 bin * 0.5 = 0.5 h -> total 10.5 h
            "shift_hours": 8.0,  # budget 8.0 h < 10.5 h
            "problem": "ttop",
        }

        action = CollectAction()
        with pytest.raises(AssertionError, match="TTOP violation: trip duration"):
            action.execute(context)


class TestTTOPLoggingAndContext:
    """Tests daily logging and context tracking of TTOP metrics."""

    def test_get_daily_results_with_time_spent(self):
        coords = pd.DataFrame({"ID": [0, 101, 102]}, index=[0, 1, 2])
        res = get_daily_results(
            total_collected=250.0,
            ncol=2,
            cost=35.0,
            tour=[0, 1, 2, 0],
            day=1,
            new_overflows=0,
            sum_lost=0.0,
            coordinates=coords,
            profit=200.0,
            time=0.05,
            time_spent=2.35,
        )
        assert res["day"] == 1
        assert res["km"] == 35.0
        assert res["time_spent"] == 2.35

    def test_simulation_day_context_defaults(self):
        ctx = SimulationDayContext(problem="ttop", shift_hours=7.5, avg_speed_kmh=40.0)
        assert ctx.problem == "ttop"
        assert ctx.shift_hours == 7.5
        assert ctx.avg_speed_kmh == 40.0
