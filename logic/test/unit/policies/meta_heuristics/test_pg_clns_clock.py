"""PG-CLNS budgets its search in wall-clock time, like every other policy (B-qwen-01)."""

from unittest.mock import patch

import numpy as np
import pytest
from logic.src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params import (
    ACOParams,
    LNSParams,
    PGCLNSParams,
)
from logic.src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pg_clns import (
    PGCLNSSolver,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_pg_clns_never_reads_cpu_time():
    rng = np.random.default_rng(0)
    pts = rng.random((7, 2))
    dist = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)
    wastes = {i: 10.0 for i in range(1, 7)}
    params = PGCLNSParams(
        population_size=2,
        max_iterations=2,
        time_limit=5.0,
        aco=ACOParams(n_ants=2, max_iterations=2, local_search_iterations=2, time_limit=5.0),
        lns=LNSParams(max_iterations=5, time_limit=5.0),
    )
    solver = PGCLNSSolver(dist, wastes, capacity=100.0, R=1.0, C=1.0, params=params, seed=1)
    with patch("time.process_time", side_effect=AssertionError("PG-CLNS must use perf_counter")):
        routes, profit, cost = solver.solve()
    assert isinstance(routes, list)
