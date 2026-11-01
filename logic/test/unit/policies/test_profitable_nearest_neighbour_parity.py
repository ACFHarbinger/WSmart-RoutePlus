"""The batched expert gives the same tours as the original one-instance-at-a-time expert.

``_reference`` is the original implementation, kept verbatim (it trained the paper models).
The instances cover Euclidean and asymmetric distances, duplicate coordinates, distances with
many exact ties, bins without waste and instances where no tour pays.
"""

from typing import List

import numpy as np
import pytest
import torch
from logic.src.policies.vectorized.routing.profitable_nearest_neighbour import ProfitableNearestNeighbourPolicy

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _reference(D: np.ndarray, waste: np.ndarray, revenue: float, cost: float, passes: int = 5) -> List[int]:
    def tour_cost(tour: List[int]) -> float:
        path = [0] + tour + [0]
        return float(sum(D[a, b] for a, b in zip(path[:-1], path[1:], strict=True)))

    def two_opt(tour: List[int]) -> List[int]:
        path = [0] + tour + [0]

        def prefix(p: List[int]):
            fwd = np.concatenate([[0.0], np.cumsum(D[p[:-1], p[1:]])])
            bwd = np.concatenate([[0.0], np.cumsum(D[p[1:], p[:-1]])])
            return fwd, bwd

        for _ in range(passes):
            improved = False
            fwd, bwd = prefix(path)
            for i in range(1, len(path) - 2):
                for j in range(i + 1, len(path) - 1):
                    a, b, c, d = path[i - 1], path[i], path[j], path[j + 1]
                    old = D[a, b] + (fwd[j] - fwd[i]) + D[c, d]
                    new = D[a, c] + (bwd[j] - bwd[i]) + D[b, d]
                    if new < old - 1e-9:
                        path[i : j + 1] = path[i : j + 1][::-1]
                        fwd, bwd = prefix(path)
                        improved = True
            if not improved:
                break
        return path[1:-1]

    left = {int(j) for j in np.flatnonzero(waste[1:] > 0) + 1}
    tour: List[int] = []
    cur = 0
    while left:
        nxt = min(left, key=lambda x: D[cur, x])
        tour.append(nxt)
        left.remove(nxt)
        cur = nxt
    tour = two_opt(tour)
    while tour:
        path = [0] + tour + [0]
        gains = [
            cost * (D[path[k - 1], path[k]] + D[path[k], path[k + 1]] - D[path[k - 1], path[k + 1]])
            - revenue * waste[path[k]]
            for k in range(1, len(path) - 1)
        ]
        k = int(np.argmax(gains))
        if gains[k] <= 0:
            break
        del tour[k]
    if tour and revenue * waste[tour].sum() - cost * tour_cost(tour) <= 0:
        return []
    return tour


def _instances(n: int, kind: str, batch: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed + n)
    locs = rng.random((batch, n + 1, 2)).astype(np.float32)
    locs[:5, 2] = locs[:5, 1]  # duplicate coordinates
    dms = torch.cdist(torch.tensor(locs), torch.tensor(locs)).numpy()
    if kind == "asymmetric":
        dms = (dms * rng.uniform(0.7, 1.3, dms.shape)).astype(np.float32)
    elif kind == "ties":
        dms = rng.integers(1, 5, dms.shape).astype(np.float32)
    waste = rng.random((batch, n + 1)).astype(np.float32)
    waste[:, 0] = 0.0
    waste[rng.random(waste.shape) < 0.3] = 0.0
    waste[:3] = 0.0  # nothing to collect
    return dms, waste


@pytest.mark.parametrize("kind", ["euclidean", "asymmetric", "ties"])
@pytest.mark.parametrize("n", [3, 12, 40])
@pytest.mark.parametrize("revenue", [1.0, 27.7])
def test_batched_expert_matches_the_reference(kind: str, n: int, revenue: float) -> None:
    dms, waste = _instances(n, kind)
    expert = ProfitableNearestNeighbourPolicy("cmptp")

    batched = expert.solve(dms, waste, revenue, 1.0)

    assert batched == [_reference(dms[b], waste[b], revenue, 1.0) for b in range(len(waste))]


def test_a_shared_distance_matrix_is_not_copied_per_instance() -> None:
    # The environment expands one road matrix over the batch; the expert reads it as a view.
    dms, waste = _instances(30, "asymmetric", batch=8)
    shared = torch.tensor(dms[:1]).expand(8, -1, -1).numpy()

    batched = ProfitableNearestNeighbourPolicy("cmptp").solve(shared, waste, 5.0, 1.0)

    assert batched == [_reference(dms[0], waste[b], 5.0, 1.0) for b in range(8)]
