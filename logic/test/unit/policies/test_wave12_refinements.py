

import numpy as np
import pytest
from logic.src.policies.helpers.operators.generalized_insertion_and_deletion import (
    apply_type_i_us,
    apply_type_iii_us,
)
from logic.src.configs.policies.de import DEConfig
from logic.src.policies.routing.meta_heuristics.differential_evolution.solver import DESolver

pytestmark = [pytest.mark.unit, pytest.mark.fast]





def test_de_binomial_crossover_j_rand():
    """Verify that j_rand component is always inherited from mutant in DE."""
    dist_matrix = np.zeros((3, 3))
    wastes = {1: 10.0, 2: 10.0}
    params = DEConfig(pop_size=5, crossover_rate=0.0, seed=42) # CR=0 ensures only j_rand comes from mutant

    solver = DESolver(dist_matrix, wastes, 100.0, 1.0, 1.0, params)

    target = np.array([0.5, 0.6])
    mutant = np.array([-0.5, -0.6])

    # Run crossover multiple times to ensure j_rand works
    for _ in range(10):
        trial = solver._binomial_crossover(target, mutant, 0.0)

        # With CR=0.0, exactly one component must be from mutant (j_rand)
        # and the other from target.
        diff_from_target = trial != target
        assert np.sum(diff_from_target) == 1

        # The changed component must match the mutant
        idx = np.where(diff_from_target)[0][0]
        assert trial[idx] == mutant[idx]

def test_hulk_type_i_reconnection():
    """Verify Müller & Bonilha (2022) Type I reconnection logic."""
    # route: 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 0
    route = [0, 1, 2, 3, 4, 5, 0]
    # Remove i=2 (node 2). i-1=1, i+1=3.
    # j=4 (node 4), k=3 (node 3).
    # Broken: (1,2), (2,3), (4,5), (3,4)
    # Reconnect: (1,4), (3,5), (4,3) -> Wait, s2 is (k+1...j) = (4...4) = [4]. s1 is (i+1...k) = (3...3) = [3].
    # new_rot: [1] + [4] + [3] + [5, 0] = [1, 4, 3, 5, 0]
    # Re-depot: [0, 1, 4, 3, 5, 0]
    new_route = apply_type_i_us(route, 2, 4, 3)
    assert new_route == [0, 1, 4, 3, 5, 0]

def test_hulk_type_iii_reconnection():
    """Verify Müller & Bonilha (2022) Type III reconnection logic (formerly Type II)."""
    # route: 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 0
    route = [0, 1, 2, 3, 4, 5, 0]
    # Remove i=2 (node 2). i-1=1, i+1=3.
    # j=3 (node 3), k=4 (node 4).
    # s1: (i+1...j) = [3]. s2: (j+1...k) = [4]. rem: [5, 0]
    # new_rot: [1] + [4]reversed + [3]reversed + [5, 0] = [1, 4, 3, 5, 0]
    new_route = apply_type_iii_us(route, 2, 3, 4)
    assert new_route == [0, 1, 4, 3, 5, 0]
