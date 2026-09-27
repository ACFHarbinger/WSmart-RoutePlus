"""fast-tsp wrapper contract (B-cursor-05)."""

from unittest.mock import patch

import numpy as np
import pytest
from logic.src.policies.route_construction.other_algorithms.travelling_salesman_problem import tsp

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_library_failure_keeps_the_input_trip():
    dist = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
    with patch.object(tsp.fast_tsp, "find_tour", side_effect=RuntimeError("boom")):
        assert tsp.find_route(dist, [2, 1]) == [0, 2, 1, 0]


def test_scaled_distances_stay_within_uint16():
    """SCALE=10000 turned a 7 km edge into 70000, above the library's uint16 contract."""
    dist = np.array([[0, 7.0, 30.0], [7.0, 0, 25.0], [30.0, 25.0, 0]])
    seen = {}

    def fake(matrix, duration_seconds):
        seen["max"] = int(np.max(matrix))
        return list(range(len(matrix)))

    with patch.object(tsp.fast_tsp, "find_tour", side_effect=fake):
        tour = tsp.find_route(dist, [1, 2])
    assert seen["max"] <= 65535
    assert tour[0] == 0 and tour[-1] == 0 and sorted(tour[1:-1]) == [1, 2]
