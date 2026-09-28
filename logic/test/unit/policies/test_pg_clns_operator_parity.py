"""
Parity tests for PG-CLNS operator switch to shared operators.

These tests verify that the PG-CLNS operators (now using shared helpers/operators/)
produce correct results for:
1. Operators that should stay identical (random_removal, cluster_removal)
2. Mandatory node handling
3. Capacity checks (demand > capacity rejection)
4. Directed-distance cases (profit-aware insertion)
"""

import random
from typing import Dict, List

import numpy as np
import pytest

from logic.src.policies.helpers.operators.destroy_ruin.cluster import cluster_removal
from logic.src.policies.helpers.operators.destroy_ruin.random import random_removal
from logic.src.policies.helpers.operators.destroy_ruin.worst import worst_removal
from logic.src.policies.helpers.operators.recreate_repair.greedy import (
    greedy_insertion,
    greedy_profit_insertion,
)
from logic.src.policies.helpers.operators.recreate_repair.regret import (
    regret_2_insertion,
    regret_2_profit_insertion,
)


@pytest.fixture
def simple_dist_matrix():
    """4-node problem: depot (0) + 3 customers."""
    return np.array([
        [0.0, 10.0, 20.0, 30.0],
        [10.0, 0.0, 15.0, 25.0],
        [20.0, 15.0, 0.0, 10.0],
        [30.0, 25.0, 10.0, 0.0],
    ])


@pytest.fixture
def wastes() -> Dict[int, float]:
    return {1: 5.0, 2: 8.0, 3: 3.0}


class TestRandomRemovalParity:
    """random_removal should produce consistent results with the same seed."""

    def test_deterministic_with_seed(self, simple_dist_matrix):
        routes = [[1, 2], [3]]
        rng = random.Random(42)
        result1, removed1 = random_removal(routes, 2, rng=rng)

        rng = random.Random(42)
        result2, removed2 = random_removal(routes, 2, rng=rng)

        assert result1 == result2
        assert removed1 == removed2

    def test_removes_correct_count(self, simple_dist_matrix):
        routes = [[1, 2, 3]]
        rng = random.Random(42)
        result, removed = random_removal(routes, 2, rng=rng)
        assert len(removed) == 2
        total_nodes = sum(len(r) for r in result) + len(removed)
        assert total_nodes == 3


class TestClusterRemovalParity:
    """cluster_removal should remove geographically clustered nodes."""

    def test_removes_at_least_one_node(self, simple_dist_matrix):
        routes = [[1, 2, 3]]
        nodes = [1, 2, 3]
        rng = random.Random(42)
        result, removed = cluster_removal(routes, 2, simple_dist_matrix, nodes, rng=rng)
        # Cluster removal should remove at least 1 node (may remove fewer if cluster is small)
        assert len(removed) >= 1
        assert len(removed) <= 2


class TestWorstRemovalRandomization:
    """worst_removal with p>1 should use Ropke-Pisinger randomization."""

    def test_randomized_with_p3(self, simple_dist_matrix):
        """p=3.0 should randomize (different results with different seeds)."""
        routes = [[1, 2, 3]]
        # Run multiple times with different seeds to verify randomization
        results = set()
        for seed in range(10):
            rng = np.random.default_rng(seed)
            _, removed = worst_removal(routes, 2, simple_dist_matrix, p=3.0, rng=rng)
            results.add(tuple(sorted(removed)))
        # With p=3.0, we should see at least 2 different removal patterns
        assert len(results) >= 2


class TestCapacityChecks:
    """Nodes with demand > capacity should be rejected."""

    def test_greedy_profit_rejects_over_capacity(self, simple_dist_matrix):
        """greedy_profit_insertion should reject nodes exceeding capacity."""
        routes = [[]]
        removed_nodes = [1]
        wastes = {1: 20.0}  # demand=20 > capacity=10
        capacity = 10.0
        R = 1.0
        C = 1.0

        result = greedy_profit_insertion(
            routes, removed_nodes, simple_dist_matrix, wastes, capacity, R, C
        )
        # Node 1 should NOT be inserted (demand > capacity)
        assert 1 not in [n for r in result for n in r]

    def test_greedy_profit_accepts_within_capacity(self, simple_dist_matrix):
        """greedy_profit_insertion should accept nodes within capacity."""
        routes = [[]]
        removed_nodes = [1]
        wastes = {1: 5.0}  # demand=5 <= capacity=10
        capacity = 10.0
        R = 10.0  # high revenue to make it profitable
        C = 0.1

        result = greedy_profit_insertion(
            routes, removed_nodes, simple_dist_matrix, wastes, capacity, R, C
        )
        # Node 1 should be inserted
        assert 1 in [n for r in result for n in r]

    def test_mandatory_node_over_capacity_skipped(self, simple_dist_matrix):
        """Mandatory nodes exceeding capacity should be skipped."""
        routes = [[]]
        removed_nodes = [1]
        wastes = {1: 20.0}  # demand=20 > capacity=10
        capacity = 10.0
        R = 1.0
        C = 1.0
        mandatory_nodes = [1]

        result = greedy_profit_insertion(
            routes, removed_nodes, simple_dist_matrix, wastes, capacity, R, C,
            mandatory_nodes=mandatory_nodes
        )
        # Even though mandatory, node 1 can't be served (demand > capacity)
        assert 1 not in [n for r in result for n in r]


class TestMandatoryNodeHandling:
    """Mandatory nodes should be prioritized and force-inserted."""

    def test_mandatory_node_inserted_even_if_unprofitable(self, simple_dist_matrix):
        """Mandatory nodes should be inserted even with negative profit."""
        routes = [[]]
        removed_nodes = [1, 2]
        wastes = {1: 1.0, 2: 1.0}
        capacity = 100.0
        R = 0.01  # very low revenue
        C = 10.0  # very high cost
        mandatory_nodes = [1]

        result = greedy_profit_insertion(
            routes, removed_nodes, simple_dist_matrix, wastes, capacity, R, C,
            mandatory_nodes=mandatory_nodes
        )
        # Mandatory node 1 should be inserted
        assert 1 in [n for r in result for n in r]

    def test_optional_node_skipped_if_unprofitable(self, simple_dist_matrix):
        """Optional nodes with negative profit should be skipped."""
        routes = [[]]
        removed_nodes = [1]
        wastes = {1: 1.0}
        capacity = 100.0
        R = 0.01  # very low revenue
        C = 10.0  # very high cost

        result = greedy_profit_insertion(
            routes, removed_nodes, simple_dist_matrix, wastes, capacity, R, C
        )
        # Optional node 1 should NOT be inserted (unprofitable)
        assert 1 not in [n for r in result for n in r]


class TestDirectedDistance:
    """Profit-aware insertion should use directed distances."""

    def test_profit_insertion_uses_revenue_minus_cost(self, simple_dist_matrix):
        """greedy_profit_insertion should maximize (waste*R - dist*C)."""
        routes = [[1]]
        removed_nodes = [2, 3]
        wastes = {1: 5.0, 2: 8.0, 3: 3.0}
        capacity = 100.0
        R = 1.0
        C = 0.1

        result = greedy_profit_insertion(
            routes, removed_nodes, simple_dist_matrix, wastes, capacity, R, C
        )
        # Both nodes should be inserted (both profitable)
        all_nodes = [n for r in result for n in r]
        assert 2 in all_nodes
        assert 3 in all_nodes


class TestRegretInsertion:
    """regret_2_insertion should use regret-based insertion."""

    def test_regret_profit_rejects_over_capacity(self, simple_dist_matrix):
        """regret_2_profit_insertion should reject nodes exceeding capacity."""
        routes = [[]]
        removed_nodes = [1]
        wastes = {1: 20.0}  # demand=20 > capacity=10
        capacity = 10.0
        R = 1.0
        C = 1.0

        result = regret_2_profit_insertion(
            routes, removed_nodes, simple_dist_matrix, wastes, capacity, R, C
        )
        # Node 1 should NOT be inserted
        assert 1 not in [n for r in result for n in r]
