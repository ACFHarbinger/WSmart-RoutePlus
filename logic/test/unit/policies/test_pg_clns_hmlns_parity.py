"""PG-CLNS on the shared operators and HMLNS on the canonical ALNS (M-qwen-03).

PG-CLNS runs on ``policies/helpers/operators`` (owner ruling 2026-10-24; the 1.0.2 release keeps the
PG-CLNS operator copies the paper ran on). Random removal is the one operator whose behaviour is unchanged:
it is checked against outputs recorded from the PG-CLNS implementation (commit 63f116656) on seeded
asymmetric instances. Cluster removal, worst removal, and greedy and regret-2 repair are the shared variants.
"""

import copy
import math
import random

import numpy as np
import pytest
from logic.src.policies.acceptance_criteria.boltzmann_metropolis_criterion import BoltzmannAcceptance
from logic.src.policies.helpers.operators.destroy_ruin.random import random_removal
from logic.src.policies.routing.base.sub_problem import build_subproblem
from logic.src.policies.routing.meta_heuristics.hybrid_memetic_large_neighborhood_search.params import (
    HybridMemeticLargeNeighborhoodSearchParams,
)
from logic.src.policies.routing.meta_heuristics.hybrid_memetic_large_neighborhood_search.solver import (
    HybridMemeticLargeNeighborhoodSearchSolver,
)
from logic.src.policies.routing.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.lns import (
    LNSSolver,
)

pytestmark = [pytest.mark.unit]

# Outputs of the previous PG-CLNS random_removal (routes, sorted removed) for seeds 0..5, recorded at 63f116656.
PREVIOUS_RANDOM_REMOVAL = {
    "0": [[[2, 3, 4], [6], [8, 9, 10, 11]], [1, 5, 7, 12]],
    "1": [[[1, 4], [6, 7], [8, 9, 11, 12]], [2, 3, 5, 10]],
    "2": [[[3, 4], [5, 7], [8, 9, 10, 12]], [1, 2, 6, 11]],
    "3": [[[1, 2], [5, 6, 7], [8, 11, 12]], [3, 4, 9, 10]],
    "4": [[[1, 3], [6], [8, 9, 10, 11, 12]], [2, 4, 5, 7]],
    "5": [[[1, 2, 3, 4], [7], [8, 11, 12]], [5, 6, 9, 10]],
}


def _instance(seed, n=12):
    rng = np.random.default_rng(seed)
    pts = rng.random((n + 1, 2)) * 10
    d = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)
    d = d * (1 + 0.3 * rng.random((n + 1, n + 1)))  # asymmetric on purpose
    np.fill_diagonal(d, 0.0)
    return d, [[1, 2, 3, 4], [5, 6, 7], [8, 9, 10, 11, 12]]


@pytest.mark.parametrize("seed", range(6))
def test_random_removal_matches_the_previous_pg_clns_implementation(seed):
    d, routes = _instance(seed)
    assert not np.allclose(d, d.T), "the fixture must be asymmetric"
    new_routes, removed = random_removal(copy.deepcopy(routes), 4, rng=random.Random(seed))
    assert [new_routes, sorted(removed)] == PREVIOUS_RANDOM_REMOVAL[str(seed)]


def test_pg_clns_repair_rejects_a_bin_above_capacity_and_serves_mandatory_bins():
    """Through the PG-CLNS adapter (LNSSolver.repair_ops), not the operator functions directly."""
    d = np.array([[0, 1, 1, 9], [1, 0, 1, 9], [1, 1, 0, 9], [9, 9, 9, 0]], dtype=float)
    wastes = {1: 20.0, 2: 3.0, 3: 4.0}
    solver = LNSSolver(build_subproblem(d, wastes, 10.0, 1.0, 1.0, mandatory_nodes=[3]), {"max_iterations": 1}, seed=1)
    for repair in solver.repair_ops:
        routes = repair([], [1, 2, 3])
        served = {n for r in routes for n in r}
        assert 1 not in served, "a bin heavier than the vehicle must not be routed"
        assert 3 in served, "the mandatory bin must be served even though it is unprofitable"
        assert all(sum(wastes[n] for n in r) <= 10.0 for r in routes)


def test_hmlns_calibrates_through_the_canonical_alns_even_with_negative_profit():
    """The removed HMLNS ALNS copy skipped calibration when profit <= 0 and kept T = 100."""
    params = HybridMemeticLargeNeighborhoodSearchParams()
    params.alns_params.max_iterations = 0
    criterion = BoltzmannAcceptance(initial_temp=0.0, alpha=0.995, seed=0, start_temp_control=0.05)
    params.alns_params.acceptance_criterion = criterion
    d = np.array([[0, 5, 5], [5, 0, 1], [5, 1, 0]], dtype=float)
    solver = HybridMemeticLargeNeighborhoodSearchSolver(d, {1: 1.0, 2: 1.0}, 100.0, R=1.0, C=1.0, params=params)
    _routes, profit, _cost = solver.alns_solver.solve(initial_solution=[[1, 2]])
    assert profit < 0
    assert pytest.approx(0.05 * abs(profit) / math.log(2.0)) == solver.alns_solver.acceptance_criterion.T
