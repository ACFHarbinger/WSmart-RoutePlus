"""The SANS initial 2-opt must stop on an unrounded distance matrix (a zero tolerance looped on float noise)."""

import numpy as np
import pytest
from logic.src.policies.routing.meta_heuristics.simulated_annealing_neighborhood_search.common.routes import (
    uncross_arcs_in_sans_routes,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_uncross_arcs_terminates_on_unrounded_distances():
    rng = np.random.default_rng(1)
    n = 30
    xy = np.c_[39.35 + rng.random(n + 1) * 0.05, -8.93 + rng.random(n + 1) * 0.05]
    dist = np.linalg.norm(xy[:, None] - xy, axis=2) * 111.0
    route = [0, *rng.permutation(np.arange(1, 24)).tolist(), 0]
    out = uncross_arcs_in_sans_routes([route], {i: i for i in range(n + 1)}, dist)
    assert sorted(out[0][1:-1]) == list(range(1, 24)) and out[0][0] == out[0][-1] == 0
