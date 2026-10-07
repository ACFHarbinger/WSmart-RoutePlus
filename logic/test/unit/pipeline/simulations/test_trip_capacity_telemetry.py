"""Per-trip payload telemetry at execution (#78: vrpp capacity was never checked)."""

from types import SimpleNamespace

import numpy as np
import pytest
from logic.src.pipeline.simulations.actions.collection import capacity_report, trip_loads

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_trip_loads_split_on_internal_depot_visits():
    fill = np.array([80.0, 50.0, 30.0, 90.0])
    assert trip_loads([0, 1, 2, 0, 3, 4, 0], fill) == [130.0, 120.0]
    assert trip_loads([0, 0], fill) == []
    assert trip_loads([0, 2, 0, 0, 4, 0], fill) == [50.0, 90.0]


def test_capacity_report_counts_over_capacity_trips_and_converts_to_kg():
    bins = SimpleNamespace(real_c=np.array([80.0, 50.0, 30.0, 90.0]), volume=2.5, density=20.0)
    loads_pct, loads_kg, violations = capacity_report([0, 1, 2, 0, 3, 4, 0], bins, capacity_pct=125.0)
    assert loads_pct == [130.0, 120.0]
    assert loads_kg == pytest.approx([65.0, 60.0])  # 1 % of a 2.5 x 20 = 50 kg bin is 0.5 kg
    assert violations == 1


def test_over_capacity_trips_are_split_on_the_true_fill():
    """Owner decision 2026-09-27: non-TCMVPTP routes that overflow the vehicle are split, not executed as is."""
    from logic.src.pipeline.simulations.actions.time_constraints import TimeConstraintAction

    bins = SimpleNamespace(real_c=np.array([80.0, 50.0, 30.0, 90.0]), c=np.zeros(4), volume=2.5, density=20.0)
    dist = np.ones((5, 5)) - np.eye(5)
    ctx = {"problem": "ptp", "tour": [0, 1, 2, 3, 4, 0], "vehicle_capacity": 150.0, "bins": bins, "distance_matrix": dist}
    TimeConstraintAction().execute(ctx)
    assert ctx["tour"] == [0, 1, 2, 0, 3, 4, 0]
    assert ctx["capacity_splits"] == 1
    assert all(load <= 150.0 for load in trip_loads(ctx["tour"], bins.real_c))


def test_feasible_route_is_left_alone():
    from logic.src.pipeline.simulations.actions.time_constraints import TimeConstraintAction

    bins = SimpleNamespace(real_c=np.array([10.0, 10.0]), c=np.zeros(2), volume=2.5, density=20.0)
    ctx = {"problem": "ptp", "tour": [0, 2, 1, 0], "vehicle_capacity": 150.0, "bins": bins, "distance_matrix": np.ones((3, 3))}
    TimeConstraintAction().execute(ctx)
    assert ctx["tour"] == [0, 2, 1, 0] and ctx["capacity_splits"] == 0
