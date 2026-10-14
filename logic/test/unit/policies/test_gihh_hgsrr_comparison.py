"""
Test for HGS-RR vs GIHH policies.

This module tests the Hybrid Genetic Search with Ruin and Recreate (HGS-RR) and
Hyper-Heuristic with Two Guidance Indicators (GIHH) policies.
"""

import numpy as np
import pytest
from logic.src.policies.routing.hyper_heuristics.guided_indicators_hyper_heuristic import (
    GIHHParams,
    GIHHSolver,
)
from logic.src.policies.routing.meta_heuristics.hybrid_genetic_search_with_ruin_and_recreate import (
    HGSRRParams,
    HGSRRSolver,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]



class TestPolicyComparison:
    """Compare HGS-RR and GIHH on the same instances."""

    @pytest.fixture
    def test_instance(self):
        """Create a test instance for comparison."""
        np.random.seed(42)
        n = 10
        dist_matrix = np.random.uniform(0.5, 5.0, (n + 1, n + 1))
        dist_matrix = (dist_matrix + dist_matrix.T) / 2  # Symmetric
        np.fill_diagonal(dist_matrix, 0.0)

        wastes = {i: np.random.uniform(0.1, 0.4) for i in range(1, n + 1)}
        capacity = 1.0
        R = 10.0
        C = 1.0

        return dist_matrix, wastes, capacity, R, C

    def test_both_policies_feasible(self, test_instance):
        """Test that both policies produce feasible solutions."""
        dist_matrix, wastes, capacity, R, C = test_instance

        params_hgs_rr = HGSRRParams(
            time_limit=2.0,
            population_size=20,
            n_iterations_no_improvement=50,
            min_removal_pct=0.05,  # Reduce removal to avoid edge cases
            max_removal_pct=0.2,
            seed=42,
        )

        params_gihh = GIHHParams(
            time_limit=2.0,
            max_iterations=100,
            seed=42,
        )

        solver_hgs_rr = HGSRRSolver(dist_matrix, wastes, capacity, R, C, params_hgs_rr, None)
        routes_hgs_rr, profit_hgs_rr, cost_hgs_rr = solver_hgs_rr.solve()

        solver_gihh = GIHHSolver(dist_matrix, wastes, capacity, R, C, params_gihh, None)
        arch_gihh = solver_gihh.solve()
        assert arch_gihh, "GIHH Solver returned empty archive"
        best_sol_gihh = max(arch_gihh, key=lambda s: s.profit)
        routes_gihh, _profit_gihh, _cost_gihh = best_sol_gihh.routes, best_sol_gihh.profit, solver_gihh._cost(best_sol_gihh.routes)

        # Both should produce valid solutions
        assert isinstance(routes_hgs_rr, list)
        assert isinstance(routes_gihh, list)

        # Check feasibility for HGS-RR
        for route in routes_hgs_rr:
            route_load = sum(wastes.get(node, 0.0) for node in route)
            assert route_load <= capacity

        # Check feasibility for GIHH
        for route in routes_gihh:
            route_load = sum(wastes.get(node, 0.0) for node in route)
            assert route_load <= capacity
