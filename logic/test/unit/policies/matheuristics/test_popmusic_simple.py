

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.append(os.path.abspath("."))

from logic.src.policies.routing.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver import (
    find_route_neighbors,
    run_popmusic,
)
from scipy.spatial import KDTree

pytestmark = [pytest.mark.unit, pytest.mark.fast]


# Mock missing dependencies to allow importing solver without triggering the whole tree.
# Save originals and restore them immediately after the import so the mocks do not
# leak into other test modules loaded later in the same pytest session.

_MOCK_MODULES = [
    "alns",
    "alns.accept",
    "alns.select",
    "alns.stop",
    "gurobipy",
    "hexaly",
    "pyvrp",
    "vrpy",
    "fast_tsp",
    "googlemaps",
    "geopy",
    "geopandas",
    "osmnx",
    "shapely",
]
_SAVED_MODULES = {m: sys.modules.pop(m, None) for m in _MOCK_MODULES}
for _m in _MOCK_MODULES:
    sys.modules[_m] = MagicMock()

# Restore original modules now that the import above has completed.
for _m, _orig in _SAVED_MODULES.items():
    if _orig is None:
        sys.modules.pop(_m, None)
    else:
        sys.modules[_m] = _orig
del _MOCK_MODULES, _SAVED_MODULES, _m, _orig


def test_find_route_neighbors_kdtree():
    print("Testing KD-Tree neighbor search...")
    centroids = [np.array([0, 0]), np.array([1, 1]), np.array([10, 10]), np.array([11, 11])]
    kdtree = KDTree(np.array(centroids))

    # k=2 neighbors of seed 0 should be [0, 1]
    neighbors = find_route_neighbors(0, centroids, k=2, kdtree=kdtree, k_prox=4)
    assert set(neighbors) == {0, 1}, f"Expected {0, 1}, got {set(neighbors)}"

    # k=2 neighbors of seed 2 should be [2, 3]
    neighbors = find_route_neighbors(2, centroids, k=2, kdtree=kdtree, k_prox=4)
    assert set(neighbors) == {2, 3}, f"Expected {2, 3}, got {set(neighbors)}"
    print("KD-Tree neighbor search passed!")


def test_popmusic_lifo_order():
    print("Testing POPMUSIC LIFO order...")
    coords = pd.DataFrame({"Lat": [0, 1, 10, 11], "Lng": [0, 10, 1, 11]})
    dist_matrix = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            dist_matrix[i, j] = np.linalg.norm(coords.iloc[i].values - coords.iloc[j].values)  # pyrefly: ignore [no-matching-overload, unsupported-operation]

    data = {
        "coords": coords,
        "mandatory": [1, 2, 3],
        "distance_matrix": dist_matrix,
        "n_vehicles": 4,
        "wastes": {1: 10.0, 2: 10.0, 3: 10.0},
        "capacity": 100,
        "R": 1.0,
        "C": 0.1,
    }

    call_order = []

    def mock_optimize(*args, **kwargs):
        seed_idx = kwargs["neighborhood_indices"][0]
        call_order.append(seed_idx)
        # Improve only on the first call to seed 3 to see if it re-pushes
        if seed_idx == 3 and len(call_order) == 1:
            return [[0, 1, 0]], 100.0  # Big improvement
        return [[0, seed_idx, 0]], 0.0  # No improvement

    def mock_init(*args, **kwargs):
        # Return 4 routes, dummy centers, and a G_cluster that links 3 to 2.
        routes = [[0, 1, 0], [0, 2, 0], [0, 3, 0], [0, 1, 0]]
        centers = np.zeros((4, 2))
        G = {0: [1], 1: [0], 2: [3], 3: [2]}
        return routes, centers, G

    with patch(
        "logic.src.policies.routing.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._optimize_subproblem",
        side_effect=mock_optimize,
    ), patch(
        "logic.src.policies.routing.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._initialize_pmedian",
        side_effect=mock_init,
    ):
        run_popmusic(**data, subproblem_size=2, seed_strategy="lifo")

    print(f"Call order: {call_order}")
    # Initial stack for 4 routes is [0, 1, 2, 3].
    # 1. Pop 3. R=[3, 2]. Improve! Re-push 3, 2. Stack: [0, 1, 3, 2]
    # 2. Pop 2. No improvement. Stack: [0, 1, 3]
    # 3. Pop 3. No improvement. Stack: [0, 1]
    # 4. Pop 1. No improvement. Stack: [0]
    # 5. Pop 0. No improvement. Stack: []
    assert call_order[0] == 3
    assert call_order[1] == 2
    assert call_order[2] == 3
    print("POPMUSIC LIFO order passed!")


if __name__ == "__main__":
    try:
        test_find_route_neighbors_kdtree()
        test_popmusic_lifo_order()
        print("\nALL STANDALONE TESTS PASSED!")
    except Exception as e:
        print(f"\nTESTS FAILED: {e}")
        import traceback



        traceback.print_exc()
        sys.exit(1)
