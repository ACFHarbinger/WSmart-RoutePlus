"""
Fixtures for the Simulator pipeline - Day fixtures.
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch
from logic.src.pipeline.simulations.day_context import SimulationDayContext


@pytest.fixture
def mock_run_day_deps(mocker):
    """
    Mocks all dependencies for the run_day function with REAL numpy arrays
    instead of MagicMocks to allow for math operations and slicing.
    """
    # 1. Setup Logic Data (Graph size 5)
    n_nodes = 5

    # Mock Bins object with real arrays
    mock_bins = MagicMock()
    mock_bins.is_stochastic.return_value = False
    mock_bins.load_filling.return_value = (0, np.zeros(n_nodes), np.zeros(n_nodes), 0)
    mock_bins.load_filling.return_value = (
        0,
        np.zeros(n_nodes),
        np.zeros(n_nodes),
        0,
    )
    mock_bins.c = np.full(n_nodes, 50.0)  # 50% fill
    mock_bins.real_c = np.full(n_nodes, 50.0)
    mock_bins.means = np.full(n_nodes, 10.0)
    mock_bins.std = np.full(n_nodes, 1.0)
    mock_bins.collectlevl = np.full(n_nodes, 90.0)  # For last_minute policy
    mock_bins.n = n_nodes
    mock_bins.collect.return_value = (100.0, 2, 10.0, 5.0)

    # 2. Mock DataFrames
    mock_new_data = pd.DataFrame(
        {
            "ID": range(1, n_nodes + 1),
            "Stock": [0] * n_nodes,
            "Accum_Rate": [0] * n_nodes,
        }
    )

    # Create coordinates and set 'ID' as the index so .loc[id] works
    mock_coords = pd.DataFrame(
        {
            "ID": range(1, n_nodes + 1),
            "Lat": [40.0 + i * 0.1 for i in range(n_nodes)],
            "Lng": [-8.0 - i * 0.1 for i in range(n_nodes)],
        }
    )
    # We keep 'ID' as a column but also set it as index
    mock_coords.set_index("ID", drop=False, inplace=True)

    # 3. Real Numpy Distance Matrix (6x6: Depot=0, Nodes=1..5)
    matrix_size = n_nodes + 1
    real_dist_matrix = np.full((matrix_size, matrix_size), 10.0)
    np.fill_diagonal(real_dist_matrix, 0.0)

    # Int matrix for TSP
    real_distancesC = real_dist_matrix.astype(np.int32)

    mock_model_env = MagicMock()
    # Ensure graph is iterable: (edges, dist_matrix). dist_matrix must be a Tensor
    mock_graph = (MagicMock(), torch.tensor(real_dist_matrix))
    mock_model_ls = (MagicMock(), mock_graph, MagicMock())

    # 4. Construct distpath_tup with REAL arrays
    mock_dist_tup = (
        real_dist_matrix,  # Real float matrix
        MagicMock(),  # paths
        MagicMock(),  # dm_tensor
        real_distancesC,  # Real int matrix
    )

    mocker.patch("logic.src.pipeline.simulations.actions.logging.send_daily_output_to_gui")
    mocker.patch("logic.src.utils.routing.tours.get_route_cost", return_value=50.0)
    mocker.patch("logic.src.policies.route_construction.other_algorithms.travelling_salesman_problem.tsp.find_route", return_value=[0, 1, 0])

    return {
        "bins": mock_bins,
        "model_env": mock_model_env,
        "model_ls": mock_model_ls,
        "distpath_tup": mock_dist_tup,
        "new_data": mock_new_data,
        "coords": mock_coords,
        "distance_matrix": real_dist_matrix,
        "distancesC": real_distancesC,
        "mock_send_output": mocker.patch("logic.src.pipeline.simulations.actions.logging.send_daily_output_to_gui"),
    }

@pytest.fixture
def make_day_context():
    """Fixture providing a factory to create a default SimulationDayContext with overrides."""

    def _make(**kwargs):
        full_policy = kwargs.get("full_policy", "regular_lvl3_cvrp_ortools_gamma1")

            # Determine policy_name and policy if not provided
        if "policy_name" not in kwargs:
            # Simple policy (without gamma/threshold)
            p_simple = full_policy.split("_gamma")[0] if "_gamma" in full_policy else full_policy

            kwargs.setdefault("policy_name", p_simple)

        defaults = {
            "graph_size": 3,
            "full_policy": "regular_lvl3_cvrp_ortools_gamma1",

            "policy_name": "regular_lvl3_cvrp_ortools",
            "bins": MagicMock(),
            "new_data": MagicMock(),
            "coords": MagicMock(),
            "distance_matrix": MagicMock(),
            "distpath_tup": (None, None, None, None),
            "distancesC": MagicMock(),
            "paths_between_states": MagicMock(),
            "dm_tensor": MagicMock(),
            "sample_id": 0,
            "overflows": 0,
            "day": 1,
            "model_env": MagicMock(),
            "model_ls": MagicMock(),
            "n_vehicles": 1,
            "area": "riomaior",
            "realtime_log_path": "mock_log.json",
            "waste_type": "paper",
            "current_collection_day": 1,
            "cached": None,
            "device": torch.device("cpu"),
            "lock": None,
            "config": {},
        }
        defaults.update(kwargs)
        return SimulationDayContext(**defaults)  # type: ignore

    return _make
