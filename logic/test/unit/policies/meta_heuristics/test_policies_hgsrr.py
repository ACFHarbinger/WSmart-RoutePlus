"""
Tests for HGS-RR policy.

This module tests the Hybrid Genetic Search with Ruin-and-Recreate (HGS-RR) policy.
"""

import numpy as np
import pytest
from logic.src.policies.routing.base.factory import RoutingAlgorithmFactory
from logic.src.policies.routing.meta_heuristics.hybrid_genetic_search_with_ruin_and_recreate import (
    HGSRRConfig,
    HGSRRSolver,
)

pytestmark = [pytest.mark.unit, pytest.mark.slow]



class TestHGSRRPolicy:
    """Test suite for HGS-RR policy."""

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

    def test_hgs_rr_basic_execution(self, simple_instance):
        """Test that HGS-RR executes without errors."""
        dist_matrix, wastes, capacity, R, C = simple_instance
        params = HGSRRConfig(time_limit=1.0, population_size=10, n_iterations_no_improvement=20, mutation_rate=0.3, seed=42)
        solver = HGSRRSolver(dist_matrix, wastes, capacity, R, C, params, None)
        routes, profit, cost = solver.solve()

        assert isinstance(routes, list)
        assert isinstance(profit, float)
        assert isinstance(cost, float)

    def test_hgs_rr_respects_capacity(self, simple_instance):
        """Test that HGS-RR respects capacity constraints."""
        dist_matrix, wastes, capacity, R, C = simple_instance
        params = HGSRRConfig(time_limit=1.0, population_size=10, n_iterations_no_improvement=20, mutation_rate=0.3, seed=42)
        solver = HGSRRSolver(dist_matrix, wastes, capacity, R, C, params, None)
        routes, profit, cost = solver.solve()

        # Check capacity for each route
        for route in routes:
            route_load = sum(wastes.get(node, 0.0) for node in route)
            assert route_load <= capacity, f"Route {route} exceeds capacity"

    def test_hgs_rr_mandatory_nodes(self, simple_instance):
        """Test that HGS-RR visits mandatory nodes."""
        dist_matrix, wastes, capacity, R, C = simple_instance
        mandatory_nodes = [1, 3]  # Local indices
        params = HGSRRConfig(time_limit=1.0, population_size=10, n_iterations_no_improvement=20, mutation_rate=0.3, seed=42)
        solver = HGSRRSolver(dist_matrix, wastes, capacity, R, C, params, mandatory_nodes)
        routes, profit, cost = solver.solve()

        # Check that mandatory nodes are visited
        visited_nodes = set()
        for route in routes:
            visited_nodes.update(route)

        for node in mandatory_nodes:
            assert node in visited_nodes, f"Mandatory node {node} not visited"

    def test_hgs_rr_empty_instance(self):
        """Test HGS-RR with empty instance."""
        dist_matrix = np.array([[0.0]])
        wastes = {}
        capacity = 1.0
        R = 10.0
        C = 1.0
        params = HGSRRConfig(seed=42)

        solver = HGSRRSolver(dist_matrix, wastes, capacity, R, C, params, None)
        routes, profit, cost = solver.solve()

        assert routes == []
        assert profit == 0.0
        assert cost == 0.0

    def test_hgs_rr_single_node(self):
        """Test HGS-RR with single node."""
        dist_matrix = np.array([
            [0.0, 1.0],
            [1.0, 0.0]
        ])
        wastes = {1: 0.5}
        capacity = 1.0
        R = 10.0
        C = 1.0
        params = HGSRRConfig(seed=42)
        solver = HGSRRSolver(dist_matrix, wastes, capacity, R, C, params, None)
        routes, profit, cost = solver.solve()

        assert len(routes) == 1
        assert routes[0] == [1]
        assert profit > 0
        assert cost > 0

    def test_hgs_rr_policy_adapter(self, simple_instance):
        """Test HGS-RR through RoutingAlgorithmFactory."""
        dist_matrix, wastes, capacity, R, C = simple_instance

        config = {
            "hgs_rr": {
                "time_limit": 1.0,
                "population_size": 10,
                "n_iterations_no_improvement": 20,
                "seed": 42,
            }
        }

        policy = RoutingAlgorithmFactory.get_adapter("hgs_rr", config=config)
        assert policy is not None
