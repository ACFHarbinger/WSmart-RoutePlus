"""
Unit tests for GENI 3-opt and 4-opt topologies.
"""

import numpy as np
import pytest
from logic.src.policies.helpers.operators.recreate_repair.geni import _apply_geni_move, _evaluate_route

pytestmark = [pytest.mark.unit, pytest.mark.fast]





def test_geni_type_i_reconnection():
    """Verify Type I (3-opt) reconnection structure."""
    u = 6
    route = [1, 2, 3, 4, 5]
    # full: [0, 1, 2, 3, 4, 5, 0]
    i = 1 # v_i = 1
    j = 3 # v_j = 3
    k = 5 # v_k = 5
    label = -1

    new_route = _apply_geni_move(route, u, i, j, k, label, "TYPE_I")
    # Expected: [0, 1, 6, 3, 2, 5, 4, 0] -> without depots: [1, 6, 3, 2, 5, 4]
    expected = [1, 6, 3, 2, 5, 4]
    assert new_route == expected

def test_geni_type_ii_reconnection():
    """Verify Type II (4-opt) reconnection structure."""
    u = 8
    route = [1, 2, 3, 4, 5, 6, 7]
    # full: [0, 1, 2, 3, 4, 5, 6, 7, 0]
    i = 1 # v_i = 1
    label = 3 # v_l = 3
    j = 5 # v_j = 5
    k = 7 # v_k = 7

    new_route = _apply_geni_move(route, u, i, j, k, label, "TYPE_II")
    # New Expected: [0, 1, 8, 5, 4, 3, 6, 2, 7, 0] -> [1, 8, 5, 4, 3, 6, 2, 7]
    expected = [1, 8, 5, 4, 3, 6, 2, 7]
    assert new_route == expected

def test_geni_evaluate_delta():
    """Verify GENI evaluation cost delta."""
    dist_matrix = np.zeros((10, 10))
    # Fill distances such that (1,2), (3,4), (5,0) are deleted
    # and (1,6), (6,3), (2,5), (4,0) are inserted
    # For simplicity, 1.0 for all arcs
    dist_matrix.fill(10.0)

    # Arcs to delete (cost 1 each)
    dist_matrix[1, 2] = 1.0
    dist_matrix[3, 4] = 1.0
    dist_matrix[5, 0] = 1.0
    dist_matrix[0, 5] = 1.0 # Symmetric for simplicity

    # Arcs to insert (cost 2 each)
    dist_matrix[1, 6] = 2.0
    dist_matrix[6, 3] = 2.0
    dist_matrix[2, 5] = 2.0
    dist_matrix[4, 0] = 2.0

    # Reversed segments (cost 1 each)
    # v_{i+1}...v_j: [2, 3]. Reversed [3, 2].
    dist_matrix[2, 3] = 1.0
    dist_matrix[3, 2] = 1.0

    # v_{j+1}...v_k: [4, 5]. Reversed [5, 4].
    dist_matrix[4, 5] = 1.0
    dist_matrix[5, 4] = 1.0

    u = 6
    route = [1, 2, 3, 4, 5]
    neighborhood_size = 5

    val, move = _evaluate_route(u, route, dist_matrix, neighborhood_size, revenue=None, C=1.0, is_man=True)

    # Delta should be: (2+2+2+2) - (1+1+1) + (rev_s1=0) + (rev_s2=0)?
    # rev_s1 cost: dist(3,2) - dist(2,3) = 1 - 1 = 0.
    # rev_s2 cost: dist(5,4) - dist(4,5) = 1 - 1 = 0.
    # delta = 8 - 3 = 5.

    assert move is not None
    assert move[4] == "TYPE_I"
    assert val == 5.0
