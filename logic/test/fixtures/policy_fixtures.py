"""
Fixtures for Policy Unit Tests.
"""


import numpy as np
import pytest


@pytest.fixture
def policy_deps(mocker):
    """
    Provides a comprehensive set of mocks for unit-testing individual policy functions.
    """

    # 1. Mock common data (5 bins + 1 depot)
    n_bins = 5
    bins_waste = np.array([10.0, 95.0, 30.0, 85.0, 50.0])

    # Distances (Depot=0, Bins=1-5)
    distancesC = np.array(
        [
            [0, 10, 10, 15, 20, 15],  # Depot 0
            [10, 0, 5, 10, 15, 10],  # Bin 1
            [10, 5, 0, 10, 15, 10],  # Bin 2
            [15, 10, 10, 0, 5, 5],  # Bin 3
            [20, 15, 15, 5, 0, 5],  # Bin 4
            [15, 10, 10, 5, 5, 0],  # Bin 5
        ],
        dtype=np.int32,
    )

    # 2. Mock dependent functions
    mock_load_params = mocker.patch(
        "logic.src.pipeline.simulations.repository.load_area_and_waste_type_params",
        return_value=(4000, 0.16, 21.0, 1.0, 2.5),  # Q, R, B, C, V
    )

    # Mock TSP solver
    mocker.patch(
        "logic.src.policies.routing.other_algorithms.travelling_salesman_problem.tsp.find_route",
        return_value=[0, 1, 3, 0],
    )

    return {
        "n_bins": n_bins,
        "bins_waste": bins_waste,
        "distancesC": distancesC,
        "mock_load_params": mock_load_params,
    }


@pytest.fixture
def mock_policy_common_data():
    """Provides common data structures for policy tests."""
    n_bins = 5
    dist_matrix = np.ones((n_bins + 1, n_bins + 1))
    np.fill_diagonal(dist_matrix, 0)

    class MockBins:
        def __init__(self, n):
            self.n = n
            self.c = np.array([10.0, 95.0, 30.0, 85.0, 50.0])
            self.means = np.full(n, 10.0)
            self.std = np.full(n, 1.0)
            self.collectlevl = 90.0

    return {
        "n_bins": n_bins,
        "bins_waste": MockBins(n_bins),
        "distance_matrix": dist_matrix,
        "distancesC": dist_matrix.astype(np.int32),
    }


@pytest.fixture
def mock_vrpp_inputs(mock_policy_common_data):
    """Provides data structures needed for VRPP policies."""
    data = mock_policy_common_data
    media = np.full(data["n_bins"], 10.0)
    std = np.full(data["n_bins"], 1.0)

    return {
        "bins": data["bins_waste"],
        "distances": data["distance_matrix"].tolist(),
        "distance_matrix": data["distance_matrix"],
        "media": media,
        "std": std,
        "mandatory_bins": [1, 3],
        "binsids": list(range(data["n_bins"])),
    }


@pytest.fixture
def mock_optimizer_data(mock_policy_common_data):
    """Provides data structures needed for VRPP policies."""
    data = mock_policy_common_data
    media = np.full(data["n_bins"], 10.0)
    std = np.full(data["n_bins"], 1.0)

    return {
        "bins": data["bins_waste"],
        "distances": data["distance_matrix"].tolist(),
        "media": media,
        "std": std,
    }


@pytest.fixture
def hgs_inputs():
    """Provide standard inputs for HGS tests."""
    dist_matrix = [
        [0, 10, 20, 30, 40],
        [10, 0, 10, 20, 30],
        [20, 10, 0, 10, 20],
        [30, 20, 10, 0, 10],
        [40, 30, 20, 10, 0],
    ]
    waste = {1: 10, 2: 10, 3: 10, 4: 10}
    capacity = 100
    R = 1.0
    C = 1.0
    global_mandatory = {2, 4}
    local_to_global = {0: 1, 1: 2, 2: 3, 3: 4}
    vrpp_tour_global = [2, 4, 1, 3]

    return (
        dist_matrix,
        waste,
        capacity,
        R,
        C,
        global_mandatory,
        local_to_global,
        vrpp_tour_global,
    )
