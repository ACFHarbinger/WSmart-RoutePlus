"""
Unit tests for Unstringing and Stringing (US) operators.
"""

import pytest
from logic.src.policies.helpers.operators.generalized_insertion_and_deletion import (
    apply_type_i_us,
    apply_type_ii_us,
    apply_type_iii_us,
    apply_type_iv_us,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]



def test_type_i_unstringing():
    """
    Test Type I operator.
    Route: 0, 1, 2, 3, 4, 5, 0
    Remove V_i = 3 (index 3).
    Predecessor V_{i-1} = 2 (index 2).
    Successor V_{i+1} = 4 (index 4).

    Neighbors:
    V_j neighbor of V_{i+1} (4). Let j=5 (index 5).
    V_k neighbor of V_{i-1} (2). Let k=None?
    Prompt: k in (i+1...j-1).
    k should be between 4 and 5? No, indices.
    Route: 0, 1, 2, 3, 4, 5, 6, 7, 0.
    i = 3 (V_i = 3).
    i-1 = 2 (V_{i-1} = 2).
    i+1 = 4 (V_{i+1} = 4).

    Let j = 7 (V_j = 7).
    Let k = 5 (V_k = 5).
    k is in sub-tour (i+1...j-1) i.e. (4...6). Yes.

    Segments:
    S1: (4...5) = [4, 5]. Reversed -> [5, 4].
    S2: (6...7) = [6, 7]. Reversed -> [7, 6]
    Prefix: [0, 1, 2].
    Suffix: [0]. (After j=7).

    Expected:
    Prefix + S2 + S1_rev + Suffix
    [0, 1, 2] + [7, 6] + [5, 4] + [0]
    = [0, 1, 2, 7, 6, 5, 4, 0]
    """
    route = [0, 1, 2, 3, 4, 5, 6, 7, 0]
    i = 3
    j = 7
    k = 5

    new_route = apply_type_i_us(route, i, j, k)
    expected = [0, 1, 2, 7, 6, 5, 4, 0]

    assert new_route == expected, f"Expected {expected}, got {new_route}"


def test_type_ii_unstringing():
    """
    Test Type II operator (Corrected definition: involving 4 neighbor nodes).
    Route: 0, 1, 2, 3, 4, 5, 6, 7, 8, 0
    i = 3. V_i=3.
    i+1=4. i-1=2.

    Let k = 5. (Between 4 and j).
    Let j = 7.
    Let l = 8. (After j).

    Segments:
    S1: (4...5) = [4, 5]. Rev -> [5, 4].
    S2: (6...7) = [6, 7]. Rev -> [7, 6].
    S3: (8...8) = [8]. Rev -> [8].

    Construction:
    Prefix (0, 1, 2) + S1_rev + S2_rev + S3_rev + Suffix (0)
    [0, 1, 2, 5, 4, 7, 6, 8, 0]
    """
    route = [0, 1, 2, 3, 4, 5, 6, 7, 8, 0]
    i = 3
    k = 5
    j = 7
    label = 8

    new_route = apply_type_ii_us(route, i, j, k, label)
    expected = [0, 1, 2, 5, 4, 7, 6, 8, 0]

    assert new_route == expected, f"Expected {expected}, got {new_route}"


def test_type_iii_unstringing():
    """
    Test Type III operator (Corrected definition: involving 3 neighbor nodes).
    Route: 0, 1, 2, 3, 4, 5, 6, 7, 0
    i = 3 (V_i = 3).
    i+1 = 4.
    i-1 = 2.

    Let j = 5.
    Let k = 7. (k > j).

    Segments:
    S1: (4...5) = [4, 5]. Reversed -> [5, 4].
    S2: (6...7) = [6, 7]. Reversed -> [7, 6].

    Construction Type III:
    V_{i-1} -> S2_rev -> S1_rev -> Remainder
    Prefix ends at 2.
    [0, 1, 2] + [7, 6] + [5, 4] + [0]
    = [0, 1, 2, 7, 6, 5, 4, 0]
    """
    route = [0, 1, 2, 3, 4, 5, 6, 7, 0]
    i = 3
    j = 5
    k = 7

    new_route = apply_type_iii_us(route, i, j, k)
    expected = [0, 1, 2, 7, 6, 5, 4, 0]

    assert new_route == expected, f"Expected {expected}, got {new_route}"


def test_type_iv_unstringing():
    """
    Test Type IV operator.
    Route: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0
    Remove V_i = 9 (index 9). V_{i-1} = 8.
    Start rotated route at V_{i+1} = 0.
    Route (rotated): 0, 1, 2, 3, 4, 5, 6, 7, 8. (V_i removed).

    Indices in rotated route:
    0=0, 1=1... 8=8.

    Let j = 2.
    Let l = 5.
    Let k = 7.

    Segments:
    S_C: 0...1 -> [0, 1]
    S_D: 5...7 -> [5, 6, 7]
    S_A_rev (V_{i-1}...V_{k+1}): [8]. (Between 7 and 9). Rev -> [8].
    S_B_rev (V_{l-1}...V_j): [4, 3, 2]. (Between 5 and 2). Rev -> [4, 3, 2].

    Construction:
    S_C + S_D + S_A_rev + S_B_rev
    [0, 1] + [5, 6, 7] + [8] + [4, 3, 2]

    Append depot 0 at end?
    Expected: [0, 1, 5, 6, 7, 8, 4, 3, 2, 0]
    """
    route = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
    i = 9
    j = 2
    label = 5
    k = 7

    new_route = apply_type_iv_us(route, i, j, k, label)
    expected = [0, 1, 5, 6, 7, 8, 4, 3, 2, 0]

    assert new_route == expected, f"Expected {expected}, got {new_route}"
