"""Tests for policy implementations and solver engines."""

from typing import Callable, cast
from unittest.mock import patch

import numpy as np
import pytest
from logic.src.policies import run_hgs
from logic.src.policies.routing.base import RoutingAlgorithmRegistry
from logic.src.policies.routing.base.sub_problem import build_subproblem
from logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_price_and_cut.policy_bpc import (
    BPCPolicy,
)
from logic.src.policies.routing.matheuristics.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.policy_ils_rvnd_sp import (
    ILSRVNDSPPolicy,
)
from logic.src.policies.routing.meta_heuristics.adaptive_large_neighborhood_search.policy_alns import (
    ALNSPolicy,
)
from logic.src.policies.routing.meta_heuristics.hybrid_genetic_search.policy_hgs import HGSPolicy

pytestmark = [pytest.mark.unit, pytest.mark.slow]





class MockBins:
    """Mock bins helper for policy tests."""
    def __init__(self, n=5):
        self.n = n
        self.c = np.zeros(n)
        self.means = np.ones(n) * 10.0
        self.std = np.ones(n) * 1.0
        self.collectlevl = 90.0


@pytest.fixture
def mock_policy_data():
    """Mock data for policy adapter tests."""
    n_bins = 5
    bins = MockBins(n_bins)
    bins.c[0] = 95.0

    dist_matrix = np.ones((n_bins + 1, n_bins + 1))
    np.fill_diagonal(dist_matrix, 0)

    return {
        "bins": bins,
        "distance_matrix": dist_matrix.tolist(),
        "waste_type": "plastic",
        "area": "riomaior",
        "config": {},
        "distancesC": np.zeros((n_bins + 1, n_bins + 1), dtype=np.int32),
        "mandatory": [1],  # Pre-selected bin index 1 (ID 1)
    }


class TestPolicyAdapters:
    """
    Unit tests for standardized policy adapters and registry integration.
    """

    @pytest.fixture(autouse=True)
    def mock_loader(self):
        with patch("logic.src.pipeline.simulations.repository.load_area_and_waste_type_params") as mock_load:
            # Q, R, B, C, V
            mock_load.return_value = (100.0, 1.0, 1.0, 1.0, 1100.0)
            yield mock_load

    @pytest.mark.unit
    def test_alns_adapter(self, mock_policy_data):

        with patch.object(ALNSPolicy, "_solver") as mock_run:
            mock_run.return_value = ([[1]], 10.0, 5.0)
            policy = cast(Callable, RoutingAlgorithmRegistry.get("alns"))()
            assert isinstance(policy, ALNSPolicy)
            tour, cost, extra, *_ = policy.execute(policy="alns_1.0", **mock_policy_data)
            assert tour == [0, 1, 0]
            assert mock_run.called

    @pytest.mark.unit
    def test_bpc_adapter(self, mock_policy_data):

        with patch.object(BPCPolicy, "_solver") as mock_run:
            mock_run.return_value = ([[1]], 10.0, 5.0)
            policy = cast(Callable, RoutingAlgorithmRegistry.get("bpc"))()
            assert isinstance(policy, BPCPolicy)
            tour, cost, extra, *_ = policy.execute(policy="bpc_1.0", **mock_policy_data)
            assert tour == [0, 1, 0]
            assert mock_run.called

    @pytest.mark.unit
    def test_ils_rvnd_sp_adapter(self, mock_policy_data):
        with patch("logic.src.policies.routing.matheuristics.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.policy_ils_rvnd_sp.ILSRVNDSPSolver") as mock_solver_cls:
            mock_solver_instance = mock_solver_cls.return_value
            mock_solver_instance.solve.return_value = ([[1]], 10.0, 5.0)
            policy = cast(Callable, RoutingAlgorithmRegistry.get("ils_rvnd_sp"))()
            assert isinstance(policy, ILSRVNDSPPolicy)
            tour, cost, extra, *_ = policy.execute(policy="ils_rvnd_sp_1.0", **mock_policy_data)
            assert tour == [0, 1, 0]
            assert mock_solver_instance.solve.called

    @pytest.mark.unit
    def test_hgs_adapter(self, mock_policy_data):
        with patch.object(HGSPolicy, "_solver") as mock_run:
            mock_run.return_value = ([[1]], 10.0, 5.0)
            policy = cast(Callable, RoutingAlgorithmRegistry.get("hgs"))()
            assert isinstance(policy, HGSPolicy)
            tour, cost, extra, *_ = policy.execute(policy="hgs_1.0", **mock_policy_data)
            assert tour == [0, 1, 0]
            assert mock_run.called


class TestAdvancedSolverEngines:
    """
    Unit tests for core solver engines (HGS, ALNS, BCP).
    """

    @pytest.mark.unit
    def test_hgs_custom(self, hgs_inputs):
        """Tests the custom HGS engine integration."""
        (
            dist_matrix,
            waste,
            capacity,
            R,
            C,
            global_mandatory,
            local_to_global,
            vrpp_tour_global,
        ) = hgs_inputs
        values = {"time_limit": 5, "n_iterations_no_improvement": 10}
        routes, fitness, cost = run_hgs(
            build_subproblem(dist_matrix, waste, capacity, R, C, mandatory_nodes=list(global_mandatory)),
            values,
        )
        assert isinstance(routes, list)
        assert len(routes) > 0
        for r in routes:
            assert isinstance(r, list)
