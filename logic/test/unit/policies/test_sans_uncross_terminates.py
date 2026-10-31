"""The SANS final 2-opt must terminate on a directed distance matrix."""

import random
import signal
from pathlib import Path

import numpy as np
import pytest

import logic.src.constants as udef
from logic.src.policies.routing.meta_heuristics.simulated_annealing_neighborhood_search.common.routes import (
    uncross_arcs_in_sans_routes,
)

MATRIX = Path(udef.ROOT_DIR) / "data" / "simulator" / "distance_matrix" / "gmaps_distmat_plastic[riomaior].csv"


class _Timeout(Exception):
    pass


def _raise(*_):
    raise _Timeout


@pytest.mark.skipif(not MATRIX.exists(), reason="Rio Maior distance matrix not available")
def test_uncross_terminates_on_asymmetric_matrix():
    raw = np.genfromtxt(MATRIX, delimiter=",")
    d = (raw[1:, 1:] if raw[0, 0] == -1 else raw[1:])[:101, :101]
    assert not np.allclose(d, d.T)
    ids = {i: i for i in range(d.shape[0])}
    rng = random.Random(0)
    signal.signal(signal.SIGALRM, _raise)
    for _ in range(60):  # with a zero tolerance about one route in five never finished
        inner = rng.sample(range(1, d.shape[0]), rng.randint(4, 40))
        signal.alarm(5)
        try:
            out = uncross_arcs_in_sans_routes([[0, *inner, 0]], ids, d)
        finally:
            signal.alarm(0)
        assert sorted(out[0][1:-1]) == sorted(inner)
