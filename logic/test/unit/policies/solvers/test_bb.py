"""Tests for Branch-and-Bound policy."""

from typing import Callable, cast
from unittest.mock import patch

import numpy as np
import pytest
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry
from logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_bound.policy_bb import (
    BranchAndBoundPolicy,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]



class MockBins:
    def __init__(self, n=3):
        self.n = n
        self.c = np.array([50.0, 50.0, 50.0])
        self.means = np.ones(n) * 50.0
        self.std = np.ones(n) * 1.0


@pytest.fixture
def bb_test_data():
    n_bins = 3
    bins = MockBins(n_bins)

    # Simple triangle distance matrix
    # 0 (depot), 1, 2, 3
    dist_matrix = np.array([
        [0, 10, 10, 10],
        [10, 0, 5, 15],
        [10, 5, 0, 5],
        [10, 15, 5, 0]
    ])

    return {
        "bins": bins,
        "distance_matrix": dist_matrix.tolist(),
        "waste_type": "plastic",
        "area": "riomaior",
        "config": {"bb": {"time_limit": 10, "vrpp": True}},
        "mandatory": [1],
    }


class TestBBPolicy:

    @pytest.fixture(autouse=True)
    def mock_loader(self):
        with patch("logic.src.pipeline.simulations.repository.load_area_and_waste_type_params") as mock_load:
            # Q, R, B, C, V
            mock_load.return_value = (100.0, 1.0, 1.0, 1.0, 100.0)
            yield mock_load

    @pytest.mark.unit
    def test_bb_adapter_registration(self):
        policy = RoutingAlgorithmRegistry.get("bb")
        assert policy is not None
        instance = policy()
        assert isinstance(instance, BranchAndBoundPolicy)

    @pytest.mark.unit
    def test_bb_execution_flow(self, bb_test_data):
        # We mock gurobipy to avoid license issues in CI/Test env if needed,
        # but here we want to test if the logic holds.
        # However, for a unit test of the ADAPTER, we can mock run_bb.
        with patch("logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_bound.policy_bb.run_bb_optimizer") as mock_run:
            mock_run.return_value = ([[1, 2]], 25.0)
            policy = cast(Callable, RoutingAlgorithmRegistry.get("bb"))()
            tour, cost, extra, *_ = policy.execute(**bb_test_data)

            assert tour[0] == 0
            assert tour[-1] == 0
            assert 1 in tour
            assert mock_run.called

    @pytest.mark.parametrize("backend", ["gurobi"])
    @pytest.mark.integration
    def test_bb_solver_integration(self, bb_test_data, check_license, backend):
        """Integration test with Gurobi (requires license)."""
        policy = RoutingAlgorithmRegistry.get("bb")
        assert policy is not None
        instance = policy()
        tour, cost, profit, *_ = instance.execute(**bb_test_data)

        assert isinstance(tour, list)
        assert tour[0] == 0
        assert tour[-1] == 0
        assert cost > 0
        assert profit > 0
