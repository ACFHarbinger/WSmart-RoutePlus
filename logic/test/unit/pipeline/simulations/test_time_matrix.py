"""Directed travel-time loading and CTOP simulation regression tests."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from logic.src.configs.envs.graph import GraphConfig
from logic.src.configs.tasks.sim import SimConfig
from logic.src.constants import ROOT_DIR
from logic.src.data.time import compute_time_matrix
from logic.src.pipeline.simulations.actions.collection import CollectAction
from logic.src.pipeline.simulations.actions.time_constraints import TimeConstraintAction
from logic.src.pipeline.simulations.repository import load_temporal_params
from logic.src.pipeline.simulations.states.initializing import InitializingState
from logic.src.policies.route_construction.other_algorithms.travelling_salesman_problem.tsp import get_multi_tour

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_default_speed_applies_to_every_pair():
    distances = np.array([[0, 35, 70], [14, 0, 21], [7, 28, 0]])
    shift, times, service = load_temporal_params(coords=pd.DataFrame({"ID": [0, 262, 225]}), distance_matrix=distances)
    np.testing.assert_allclose(times, distances / 35.0)
    assert (shift, service) == (7.0, 0.025)
    assert load_temporal_params() == (7.0, 35.0, 0.025)


def test_dashboard_seconds_reorder_each_axis_and_preserve_depot_fallback(tmp_path):
    path = tmp_path / "times.csv"
    path.write_text(",262 - 262,225 - 225\n225 - 225,774,0\n262 - 262,0,778\n")
    coords = pd.DataFrame({"ID": [0.0, 262.0, 225.0]})
    distances = np.array([[0, 35, 70], [14, 0, 21], [7, 28, 0]])
    times = compute_time_matrix(coords, distances, path)
    np.testing.assert_allclose(times, [[0, 1, 2], [0.4, 0, 778 / 3600], [0.2, 774 / 3600, 0]])


@pytest.mark.parametrize("unit,factor", [("seconds", 3600), ("minutes", 60), ("hours", 1)])
def test_file_depot_times_override_speed(tmp_path, unit, factor):
    path = tmp_path / "times.csv"
    path.write_text(",0,101\n0,0,2\n101,3,0\n")
    times = compute_time_matrix(pd.DataFrame({"ID": [0, 101]}), np.zeros((2, 2)), path, time_unit=unit)
    np.testing.assert_allclose(times, np.array([[0, 2], [3, 0]]) / factor)


@pytest.mark.parametrize(
    "csv,match",
    [
        (",0,102\n0,0,2\n102,3,0\n", "missing bin IDs"),
        (",0,101\n0,0,-2\n101,3,0\n", "nonnegative"),
        (",0,101\n0,0,nan\n101,3,0\n", "finite"),
        (",0,101\n0,1,2\n101,3,0\n", "diagonal"),
        (",0,101\n0,0,2\n102,3,0\n", "matching row"),
        (",0,101\n0,0,2\n0,3,0\n", "unique"),
    ],
)
def test_invalid_matrix_is_rejected(tmp_path, csv, match):
    path = tmp_path / "times.csv"
    path.write_text(csv)
    with pytest.raises(ValueError, match=match):
        compute_time_matrix(pd.DataFrame({"ID": [0, 101]}), np.zeros((2, 2)), path)


@pytest.mark.parametrize("kwargs", [{"avg_speed_kmh": 0}, {"shift_hours": -1}, {"service_time_h": -1}])
def test_invalid_temporal_resources_rejected(kwargs):
    with pytest.raises(ValueError):
        load_temporal_params(**kwargs)


def test_matrix_repair_and_collection_use_same_directed_times():
    bins = MagicMock()
    bins.c = np.array([10.0, 10.0])
    bins.collect.return_value = (np.array([10.0, 10.0]), 20.0, 2, 12.0)
    context = {
        "problem": "ctop",
        "tour": [0, 1, 2, 0],
        "bins": bins,
        "distance_matrix": np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
        "time_matrix": np.array([[0, 1, 0.5], [0.5, 0, 2], [1, 0.5, 0]]),
        "vehicle_capacity": 100.0,
        "shift_hours": 2.0,
        "service_time_h": 0.0,
    }
    TimeConstraintAction().execute(context)
    assert context["tour"] == [0, 1, 0, 2, 0]
    CollectAction().execute(context)
    assert context["time_spent"] == 3.0
    assert context["cost"] == 4.0
    bins.collect.assert_called_once_with([0, 1, 0, 2, 0], 4.0)


@pytest.mark.parametrize("tour", [[0, 1, 0], [1]])
def test_infeasible_return_does_not_mutate_bins(tour):
    bins = MagicMock()
    context = {
        "problem": "ctop",
        "tour": tour,
        "bins": bins,
        "distance_matrix": np.zeros((2, 2)),
        "time_matrix": np.array([[0, 0.5], [2, 0]]),
        "shift_hours": 2.0,
        "service_time_h": 0.0,
    }
    with pytest.raises(AssertionError, match="trip duration"):
        CollectAction().execute(context)
    bins.collect.assert_not_called()


def test_capacity_violation_does_not_mutate_bins():
    bins = MagicMock()
    bins.c = np.array([60, 60])
    context = {
        "problem": "ctop",
        "tour": [0, 1, 2, 0],
        "bins": bins,
        "distance_matrix": np.zeros((3, 3)),
        "vehicle_capacity": 100.0,
    }
    with pytest.raises(AssertionError, match="capacity"):
        CollectAction().execute(context)
    bins.collect.assert_not_called()


def test_impossible_single_customer_rejected_and_existing_depots_preserved():
    distances = np.ones((3, 3)) - np.eye(3)
    with pytest.raises(ValueError, match="on its own"):
        get_multi_tour([0, 1, 0], np.array([10]), 100, distances, shift_hours=1, time_matrix=distances)
    tour = [0, 1, 0, 2, 0]
    assert get_multi_tour(tour, np.array([10, 10]), 100, distances) == tour


def test_real_dashboard_file_loads_by_customer_id():
    if not (Path(ROOT_DIR) / "data/simulator/time_matrix/matriz_c7_dashboard_tempo_seg.csv").is_file():
        pytest.skip("Local dashboard dataset is not distributed with the repository")
    times = compute_time_matrix(
        pd.DataFrame({"ID": [0, 262, 225]}),
        np.zeros((3, 3)),
        "matriz_c7_dashboard_tempo_seg.csv",
    )
    assert times[1, 2] == pytest.approx(778 / 3600)
    assert times[2, 1] == pytest.approx(774 / 3600)


@pytest.mark.parametrize("resume", [False, True])
def test_initialization_loads_times_after_new_or_restored_coordinates(tmp_path, monkeypatch, resume):
    path = tmp_path / "times.csv"
    path.write_text(",0,101\n0,0,360\n101,720,0\n")
    graph = GraphConfig(num_loc=1, area="riomaior", waste_type="plastic", tm_filepath=str(path))
    sim = SimConfig(seed=42, graph=graph, resume=resume, problem="ctop")
    ctx = SimpleNamespace(
        cfg=SimpleNamespace(sim=sim),
        data_dir=str(tmp_path),
        pol_name="tsp",
        transition_to=MagicMock(),
        shift_hours=3.0,
        avg_speed_kmh=35.0,
        service_time_h=0.0,
        results_dir=str(tmp_path),
        pol_id_orig="tsp",
        sample_id=0,
    )
    state = InitializingState()
    for name in ("_setup_logging_and_dirs", "_load_all_configurations", "_setup_capacities", "_setup_models"):
        monkeypatch.setattr(state, name, lambda ctx: None)
    monkeypatch.setattr(
        "logic.src.pipeline.simulations.states.initializing.setup_basedata", lambda *args: (None, None, None)
    )
    monkeypatch.setattr(state, "_load_checkpoint_if_needed", lambda ctx: (object(), 1))

    def restore_or_initialize(ctx, *args):
        ctx.coords = pd.DataFrame({"ID": [0, 101]})
        ctx.dist_tup = (np.zeros((2, 2)), None, None, None)

    monkeypatch.setattr(state, "_restore_state", restore_or_initialize)
    monkeypatch.setattr(state, "_initialize_new_state", restore_or_initialize)
    state.handle(ctx)
    np.testing.assert_allclose(ctx.time_matrix, [[0, 0.1], [0.2, 0]])
    assert ctx.shift_hours == 3.0
    assert ctx.service_time_h == 0.0
    ctx.transition_to.assert_called_once()
