"""two_opt_steepest checks matrix symmetry once per call, with unchanged results."""

import numpy as np
import pytest

from logic.src.policies.helpers.operators.improvement_descent import steepest_two_opt as mod

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _reference(routes, d, C=1.0, max_iter=500):
    """The pre-change loop: symmetry re-derived inside every search."""
    working = [list(r) for r in routes if r]
    for _ in range(max_iter):
        _, move = mod._find_best_2opt(working, d, C)
        if move is None:
            break
        r, i, j = move
        working[r][i:j] = working[r][i:j][::-1]
    return [r for r in working if r]


@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("seed", range(5))
def test_results_match_the_per_iteration_check(symmetric, seed):
    rng = np.random.default_rng(seed)
    pts = rng.random((25, 2))
    d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    if not symmetric:
        d = d + rng.random(d.shape) * 0.3
    routes = [list(rng.permutation(np.arange(1, 13))), list(rng.permutation(np.arange(13, 25)))]
    assert mod.two_opt_steepest(routes, d, {}, 100.0) == _reference(routes, d)


def test_symmetry_is_checked_once(monkeypatch):
    calls = []
    real = mod._is_symmetric
    monkeypatch.setattr(mod, "_is_symmetric", lambda d: calls.append(1) or real(d))
    rng = np.random.default_rng(3)
    pts = rng.random((15, 2))
    d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    mod.two_opt_steepest([list(rng.permutation(np.arange(1, 15)))], d, {}, 100.0)
    assert calls == [1]
