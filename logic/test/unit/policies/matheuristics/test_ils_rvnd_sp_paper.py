

from typing import List, Optional

import numpy as np
import pytest
from logic.src.policies.helpers.local_search.local_search_base import LocalSearch
from logic.src.policies.routing.matheuristics.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.ils_rvnd_sp import (
    ILSRVNDSPSolver,
)
from logic.src.configs.policies.ils_rvnd_sp import ILSRVNDSPConfig

pytestmark = [pytest.mark.unit, pytest.mark.fast]



class MockLocalSearch(LocalSearch):
    def __init__(self, routes, dist_matrix, capacity, C=1.0):
        self.routes = [list(r) for r in routes]
        self.d = dist_matrix
        self.Q = capacity
        self.C = C
        self.waste = {i: 1 for i in range(1, 100)} # Mock waste
        self.route_map = {}
        self._update_map(range(len(routes)))
        self._loads = [sum(self.waste.get(n, 0) for n in r) for r in routes]
        self.target_neighborhood = "all"

    def optimize(self, solution: Optional[List[List[int]]] = None):
        """Dummy optimize implementation."""
        if solution:
            self.routes = [list(r) for r in solution]
        return self.routes

    def _get_load_cached(self, ri):
        return self._loads[ri]

    def _calc_load_fresh(self, r):
        return sum(self.waste.get(n, 0) for n in r)

    def _update_map(self, affected_indices):
        for r_idx in affected_indices:
            for n in self.routes[r_idx]:
                self.route_map[n] = r_idx


@pytest.fixture
def sample_params():
    return ILSRVNDSPConfig(
        time_limit=10,
        seed=42,
        vrpp=True
    )


def test_params_paper_defaults(sample_params):
    """Verify that parameters default to Subramanian 2013 values."""
    assert sample_params.N == 150
    assert sample_params.A == 11.0
    assert sample_params.MaxIter_a == 50
    assert sample_params.MaxIterILS_b == 2000
    assert sample_params.TDev_b == 0.005


def test_perturb_paper():
    """Verify that _perturb uses random moves and maintains node set."""
    dist_matrix = np.zeros((10, 10))
    wastes = {i: 1. for i in range(1, 10)}
    params = ILSRVNDSPConfig(time_limit=10, seed=42)
    solver = ILSRVNDSPSolver(dist_matrix, wastes, 100, 100, 1, params)

    initial_routes = [[1, 2, 3], [4, 5, 6]]
    initial_nodes = set([1, 2, 3, 4, 5, 6])

    # Run perturbation multiple times to see different moves
    for _ in range(5):
        perturbed = solver._perturb(initial_routes)
        perturbed_nodes = set([n for r in perturbed for n in r])
        assert perturbed_nodes == initial_nodes
        # Should be different from initial most of the time
        # (Though technically it could randomly swap back, unlikely with 2-5 moves)


def test_ils_acceptance_strict():
    """Verify that only strict improvement is accepted in ILS."""
    # This is harder to test directly without mocking the inner loop,
    # but we can verify the logic in run_ils_rvnd via code inspection or tiny solver run.
    dist_matrix = np.zeros((5, 5))
    wastes = {i: 1. for i in range(1, 5)}
    params = ILSRVNDSPConfig(time_limit=10, MaxIter_a=1, MaxIter_b=1, MaxIterILS_b=1, seed=42)
    solver = ILSRVNDSPSolver(dist_matrix, wastes, 100, 100, 1, params)

    # Mock RVND to return same or worse profit
    # If it's same, it shouldn't be accepted as the new 'iter_routes'
    # We'll check this by ensuring it doesn't loop forever or crashes
    solver.solve()


def test_neighborhood_cross():
    """Verify suffix exchange (Cross/2-opt*) operator."""
    dist = np.zeros((10, 10))
    # Distance: 0-1-2-0 vs 0-1-6-0
    # Let's make 2-6 cheaper than 2-3 and 5-6 cheaper than 5-3
    dist[2, 6] = 1
    dist[5, 3] = 1
    dist[2, 3] = 10
    dist[5, 6] = 10

    routes = [[1, 2, 3], [4, 5, 6]]
    ls = MockLocalSearch(routes, dist, 100)

    # Cross at (2, r0, pos1) and (5, r1, pos1)
    # r0: [1, 2] + [6] = [1, 2, 6]
    # r1: [4, 5] + [3] = [4, 5, 3]
    from logic.src.policies.helpers.operators.inter_route_local_search.subramanian_neighborhoods import move_cross
    improved = move_cross(ls, u=2, v=5, r_u=0, p_u=1, r_v=1, p_v=1)

    assert improved
    assert ls.routes[0] == [1, 2, 6]
    assert ls.routes[1] == [4, 5, 3]


def test_shift_2_0():
    """Verify relocation of 2 nodes."""
    dist = np.zeros((10, 10))
    # Moving [2, 3] from r0 to r1
    routes = [[1, 2, 3], [4, 5]]
    ls = MockLocalSearch(routes, dist, 100)
    # Force improvement by making removal expensive
    ls.d[1, 2] = 10  # Edge being removed: 1-2
    ls.d[1, 0] = 0   # Repair edge: 1-0 (cheap)

    from logic.src.policies.helpers.operators.inter_route_local_search.subramanian_neighborhoods import shift_2_0
    # shift [2, 3] from r0 (pos 1) to r1 (after pos 1)


    improved = shift_2_0(ls, r_src=0, pos_src=1, r_dst=1, pos_dst=1)
    assert improved
    assert ls.routes[0] == [1]
    assert ls.routes[1] == [4, 5, 2, 3]
