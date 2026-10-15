"""
Tests for GIHH policies.

This module tests the Hyper-Heuristic with Two Guidance Indicators (GIHH) policy.
"""

import numpy as np
import pytest
from logic.src.policies.routing.base.factory import RoutingAlgorithmFactory
from logic.src.policies.routing.hyper_heuristics.guided_indicators_hyper_heuristic import (
    GIHHConfig,
    GIHHSolver,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]



class TestGIHHPolicy:
    """Test suite for GIHH policy."""

    @pytest.fixture
    def simple_instance(self):
        """Create a simple test instance."""
        # 5 nodes + depot
        dist_matrix = np.array([
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 0.0, 1.5, 2.5, 3.5, 4.5],
            [2.0, 1.5, 0.0, 1.0, 2.0, 3.0],
            [3.0, 2.5, 1.0, 0.0, 1.0, 2.0],
            [4.0, 3.5, 2.0, 1.0, 0.0, 1.0],
            [5.0, 4.5, 3.0, 2.0, 1.0, 0.0],
        ])
        wastes = {1: 0.3, 2: 0.4, 3: 0.5, 4: 0.3, 5: 0.4}
        capacity = 1.0
        R = 10.0  # Revenue per unit
        C = 1.0   # Cost per distance
        return dist_matrix, wastes, capacity, R, C

    def test_gihh_basic_execution(self, simple_instance):
        """Test that GIHH executes without errors."""
        dist_matrix, wastes, capacity, R, C = simple_instance
        params = GIHHConfig(time_limit=1.0, max_iterations=5, seed=42)
        solver = GIHHSolver(dist_matrix, wastes, capacity, R, C, params, None)
        arch = solver.solve()
        assert isinstance(arch, list)
        if arch:
            best_sol = max(arch, key=lambda s: s.profit)
            routes, profit, cost = best_sol.routes, best_sol.profit, solver._cost(best_sol.routes)
        else:
            routes, profit, cost = [], 0.0, 0.0

        assert isinstance(routes, list)
        assert isinstance(profit, float)
        assert isinstance(cost, float)

    def test_gihh_respects_capacity(self, simple_instance):
        """Test that GIHH respects capacity constraints."""
        dist_matrix, wastes, capacity, R, C = simple_instance
        params = GIHHConfig(time_limit=1.0, max_iterations=5, seed=42)
        solver = GIHHSolver(dist_matrix, wastes, capacity, R, C, params, None)
        arch = solver.solve()
        assert isinstance(arch, list)
        if arch:
            best_sol = max(arch, key=lambda s: s.profit)
            routes, _profit, _cost = best_sol.routes, best_sol.profit, solver._cost(best_sol.routes)
        else:
            routes, _profit, _cost = [], 0.0, 0.0

        # Check capacity for each route
        for route in routes:
            route_load = sum(wastes.get(node, 0.0) for node in route)
            assert route_load <= capacity, f"Route {route} exceeds capacity"

    def test_gihh_mandatory_nodes(self, simple_instance):
        """Test that GIHH visits mandatory nodes."""
        dist_matrix, wastes, capacity, R, C = simple_instance
        mandatory_nodes = [1, 3]  # Local indices
        params = GIHHConfig(time_limit=1.0, max_iterations=5, seed=42)
        solver = GIHHSolver(dist_matrix, wastes, capacity, R, C, params, mandatory_nodes)
        arch = solver.solve()
        assert isinstance(arch, list)
        if arch:
            best_sol = max(arch, key=lambda s: s.profit)
            routes, _profit, _cost = best_sol.routes, best_sol.profit, solver._cost(best_sol.routes)
        else:
            routes, _profit, _cost = [], 0.0, 0.0

        # Check that mandatory nodes are visited
        visited_nodes = set()
        for route in routes:
            visited_nodes.update(route)

        for node in mandatory_nodes:
            assert node in visited_nodes, f"Mandatory node {node} not visited"

    def test_gihh_empty_instance(self):
        """Test GIHH with empty instance."""
        dist_matrix = np.array([[0.0]])
        wastes = {}
        capacity = 1.0
        R = 10.0
        C = 1.0
        params = GIHHConfig(seed=42)
        solver = GIHHSolver(dist_matrix, wastes, capacity, R, C, params, None)
        arch = solver.solve()
        assert isinstance(arch, list)
        if arch:
            best_sol = max(arch, key=lambda s: s.profit)
            routes, profit, cost = best_sol.routes, best_sol.profit, solver._cost(best_sol.routes)
        else:
            routes, profit, cost = [], 0.0, 0.0

        assert routes == []
        assert profit == 0.0
        assert cost == 0.0

    def test_gihh_single_node(self):
        """Test GIHH with single node."""
        dist_matrix = np.array([
            [0.0, 1.0],
            [1.0, 0.0]
        ])
        wastes = {1: 0.5}
        capacity = 1.0
        R = 10.0
        C = 1.0
        params = GIHHConfig(seed=42)
        solver = GIHHSolver(dist_matrix, wastes, capacity, R, C, params, None)
        arch = solver.solve()
        assert isinstance(arch, list)
        if arch:
            best_sol = max(arch, key=lambda s: s.profit)
            routes, profit, cost = best_sol.routes, best_sol.profit, solver._cost(best_sol.routes)
        else:
            routes, profit, cost = [], 0.0, 0.0

        assert len(routes) == 1
        assert routes[0] == [1]
        assert profit > 0
        assert cost > 0

    def test_gihh_indicator_weights(self, simple_instance):
        """Test GIHH with custom indicator weights."""
        dist_matrix, wastes, capacity, R, C = simple_instance
        params = GIHHConfig(time_limit=1.0, max_iterations=5, seg=80, alpha=0.5, beta=0.4, gamma=0.1, min_prob=0.05, seed=42)
        solver = GIHHSolver(dist_matrix, wastes, capacity, R, C, params, None)
        arch = solver.solve()
        assert isinstance(arch, list)
        if arch:
            best_sol = max(arch, key=lambda s: s.profit)
            routes, profit, _cost = best_sol.routes, best_sol.profit, solver._cost(best_sol.routes)
        else:
            routes, profit, _cost = [], 0.0, 0.0

        assert isinstance(routes, list)
        assert isinstance(profit, float)

    def test_gihh_policy_adapter(self, simple_instance):
        """Test GIHH through RoutingAlgorithmFactory."""
        dist_matrix, wastes, capacity, R, C = simple_instance

        config = {
            "gihh": {
                "time_limit": 1.0,
                "max_iterations": 5,
                "seed": 42,
            }
        }

        policy = RoutingAlgorithmFactory.get_adapter("gihh", config=config)
        assert policy is not None
