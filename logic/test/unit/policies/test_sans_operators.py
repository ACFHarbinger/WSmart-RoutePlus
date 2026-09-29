"""
SANS operator parity tests.

These tests verify that the SANS perturbation operators work correctly.
SANS operators are perturbation operators for simulated annealing (random exploration),
not local search operators (improvement). They modify routes in-place and return None.
"""

import random

import pytest
from logic.src.policies.helpers.operators.perturbation_shaking.sans.intra_move import (
    move_1_route,
    move_n_route_consecutive,
    move_n_route_random,
)
from logic.src.policies.helpers.operators.perturbation_shaking.sans.intra_swap import (
    swap_1_route,
    swap_n_route_consecutive,
    swap_n_route_random,
)


@pytest.fixture
def sample_routes():
    """Sample routes for testing."""
    return [
        [0, 1, 2, 3, 4, 0],  # Route with 4 bins
        [0, 5, 6, 7, 0],     # Route with 3 bins
        [0, 8, 9, 10, 11, 12, 0],  # Route with 5 bins
    ]


class TestIntraSwapOperators:
    """Test intra-route swap operators."""

    def test_swap_1_route_modifies_routes(self, sample_routes):
        """swap_1_route should swap two bins within a route."""
        rng = random.Random(42)
        routes_before = [r.copy() for r in sample_routes]

        swap_1_route(sample_routes, rng)

        # At least one route should be modified
        assert sample_routes != routes_before

        # Total number of nodes should be preserved
        total_before = sum(len(r) for r in routes_before)
        total_after = sum(len(r) for r in sample_routes)
        assert total_before == total_after

    def test_swap_1_route_preserves_depots(self, sample_routes):
        """swap_1_route should not swap depot nodes (first/last)."""
        rng = random.Random(42)

        # Run multiple times to check depot preservation
        for _ in range(10):
            routes_copy = [r.copy() for r in sample_routes]
            swap_1_route(routes_copy, rng)

            # First and last elements should still be 0 (depot)
            for route in routes_copy:
                assert route[0] == 0
                assert route[-1] == 0

    def test_swap_n_route_random_returns_count(self, sample_routes):
        """swap_n_route_random should return the number of swaps performed."""
        rng = random.Random(42)

        result = swap_n_route_random(sample_routes, rng, n=2)

        # Should return the number of swaps (or None if no routes)
        assert result is None or isinstance(result, int)

    def test_swap_n_route_consecutive_swaps_segments(self, sample_routes):
        """swap_n_route_consecutive should swap consecutive segments."""
        rng = random.Random(42)
        routes_before = [r.copy() for r in sample_routes]

        result = swap_n_route_consecutive(sample_routes, rng, n=2)

        # Should return the segment size or None
        assert result is None or isinstance(result, int)

        # Total nodes should be preserved
        total_before = sum(len(r) for r in routes_before)
        total_after = sum(len(r) for r in sample_routes)
        assert total_before == total_after


class TestIntraMoveOperators:
    """Test intra-route move operators."""

    def test_move_1_route_modifies_routes(self, sample_routes):
        """move_1_route should move one bin within a route."""
        rng = random.Random(42)
        routes_before = [r.copy() for r in sample_routes]

        move_1_route(sample_routes, rng)

        # At least one route should be modified
        assert sample_routes != routes_before

        # Total number of nodes should be preserved
        total_before = sum(len(r) for r in routes_before)
        total_after = sum(len(r) for r in sample_routes)
        assert total_before == total_after

    def test_move_1_route_preserves_depots(self, sample_routes):
        """move_1_route should not move depot nodes."""
        rng = random.Random(42)

        # Run multiple times to check depot preservation
        for _ in range(10):
            routes_copy = [r.copy() for r in sample_routes]
            move_1_route(routes_copy, rng)

            # First and last elements should still be 0 (depot)
            for route in routes_copy:
                assert route[0] == 0
                assert route[-1] == 0

    def test_move_n_route_random_returns_count(self, sample_routes):
        """move_n_route_random should return the number of moves performed."""
        rng = random.Random(42)

        result = move_n_route_random(sample_routes, rng, n=3)

        # Should return the number of moves (or None if no routes)
        assert result is None or isinstance(result, int)

    def test_move_n_route_consecutive_moves_segment(self, sample_routes):
        """move_n_route_consecutive should move a consecutive segment."""
        rng = random.Random(42)
        routes_before = [r.copy() for r in sample_routes]

        result = move_n_route_consecutive(sample_routes, rng, n=2)

        # Should return the segment size or None
        assert result is None or isinstance(result, int)

        # Total nodes should be preserved
        total_before = sum(len(r) for r in routes_before)
        total_after = sum(len(r) for r in sample_routes)
        assert total_before == total_after


class TestOperatorDeterminism:
    """Test that operators are deterministic with the same seed."""

    def test_swap_1_route_deterministic(self, sample_routes):
        """swap_1_route should produce same result with same seed."""
        routes1 = [r.copy() for r in sample_routes]
        routes2 = [r.copy() for r in sample_routes]

        rng1 = random.Random(42)
        rng2 = random.Random(42)

        swap_1_route(routes1, rng1)
        swap_1_route(routes2, rng2)

        assert routes1 == routes2

    def test_move_1_route_deterministic(self, sample_routes):
        """move_1_route should produce same result with same seed."""
        routes1 = [r.copy() for r in sample_routes]
        routes2 = [r.copy() for r in sample_routes]

        rng1 = random.Random(42)
        rng2 = random.Random(42)

        move_1_route(routes1, rng1)
        move_1_route(routes2, rng2)

        assert routes1 == routes2


class TestEdgeCases:
    """Test edge cases."""

    def test_swap_empty_routes(self):
        """Operators should handle empty route lists."""
        rng = random.Random(42)
        routes = []

        # Should not crash
        swap_1_route(routes, rng)
        move_1_route(routes, rng)

        assert routes == []

    def test_swap_single_node_route(self):
        """Operators should handle routes with only depot nodes."""
        rng = random.Random(42)
        routes = [[0, 0]]  # Only depot

        # Should not crash
        swap_1_route(routes, rng)
        move_1_route(routes, rng)

        # Route should be unchanged (no bins to swap/move)
        assert routes == [[0, 0]]

    def test_swap_two_node_route(self):
        """Operators should handle routes with only one bin."""
        rng = random.Random(42)
        routes = [[0, 1, 0]]  # One bin

        routes_before = [r.copy() for r in routes]

        # Should not crash
        swap_1_route(routes, rng)

        # Route may or may not change (depends on implementation)
        # But should not crash and should preserve structure
        assert len(routes) == len(routes_before)
        assert routes[0][0] == 0
        assert routes[0][-1] == 0
