"""get_route_cost reads the distance matrix in place (no full-matrix copy per call)."""

import numpy as np
import pytest
from logic.src.utils.routing.tours import get_route_cost


def _reference(dist, tour):
    return float(sum(dist[a][b] for a, b in zip(tour[:-1], tour[1:], strict=False)))


@pytest.mark.parametrize("as_list", [False, True])
def test_route_cost_matches_reference_and_leaves_inputs_untouched(as_list):
    rng = np.random.default_rng(0)
    dist = rng.random((30, 30))
    tour = [0, 4, 9, 17, 0, 3, 28, 0]
    before = dist.copy()
    d = dist.tolist() if as_list else dist
    t = tour if as_list else np.array(tour)
    assert get_route_cost(d, t) == pytest.approx(_reference(dist, tour))
    np.testing.assert_array_equal(dist, before)


def test_route_cost_does_not_copy_the_matrix():
    dist = np.random.default_rng(1).random((50, 50))

    class NoCopy(np.ndarray):
        def copy(self, *args, **kwargs):  # pragma: no cover - fails the test if reached
            raise AssertionError("distance matrix copied")

    view = dist.view(NoCopy)
    assert get_route_cost(view, [0, 1, 2, 0]) == pytest.approx(dist[0, 1] + dist[1, 2] + dist[2, 0])


def test_route_cost_empty_and_single_node_tours():
    dist = np.ones((3, 3))
    assert get_route_cost(dist, [0]) == 0.0
    assert get_route_cost(dist, [0, 0]) == 1.0
