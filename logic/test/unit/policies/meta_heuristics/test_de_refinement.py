

import numpy as np
import pytest
from logic.src.configs.policies.de import DEConfig
from logic.src.policies.routing.meta_heuristics.differential_evolution.solver import DESolver

pytestmark = [pytest.mark.unit, pytest.mark.fast]



@pytest.fixture
def instance_data():
    dist_matrix = np.array([
        [0, 10, 20],
        [10, 0, 15],
        [20, 15, 0]
    ])
    wastes = {1: 100.0, 2: 200.0}
    capacity = 500.0
    return dist_matrix, wastes, capacity

def test_dynamic_population_sizing(instance_data):
    dist_matrix, wastes, capacity = instance_data
    # n_nodes = 2. Dynamic NP should be max(10 * 2, 4) = 20.
    params = DEConfig(pop_size=None)
    solver = DESolver(dist_matrix, wastes, capacity, R=1.0, C=1.0, params=params)
    assert solver.pop_size == 20

    # Test explicit pop_size
    params_explicit = DEConfig(pop_size=15)
    solver_explicit = DESolver(dist_matrix, wastes, capacity, R=1.0, C=1.0, params=params_explicit)
    assert solver_explicit.pop_size == 15

    # Test minimum axiom enforcement
    params_low = DEConfig(pop_size=2)
    with pytest.raises(ValueError, match="Population size NP must be at least 4"):
        DESolver(dist_matrix, wastes, capacity, R=1.0, C=1.0, params=params_low)

def test_bounce_back_boundary_handling(instance_data):
    dist_matrix, wastes, capacity = instance_data
    params = DEConfig(pop_size=4)
    solver = DESolver(dist_matrix, wastes, capacity, R=1.0, C=1.0, params=params)

    base_vector = np.array([0.5, -0.5])
    # Case 1: Upper bound violation
    mutant = np.array([1.5, 0.0])
    bounded = solver._apply_boundary_handling(mutant.copy(), base_vector)
    assert -1.0 <= bounded[0] <= 1.0
    assert bounded[1] == 0.0
    # Bounce back should be between base (0.5) and boundary (1.0)
    assert 0.5 < bounded[0] < 1.0

    # Case 2: Lower bound violation
    mutant_low = np.array([0.0, -2.0])
    bounded_low = solver._apply_boundary_handling(mutant_low.copy(), base_vector)
    assert -1.0 <= bounded_low[1] <= 1.0
    assert bounded_low[0] == 0.0
    # Bounce back should be between base (-0.5) and boundary (-1.0)
    assert -1.0 < bounded_low[1] < -0.5

def test_exponential_crossover_logic(instance_data):
    dist_matrix, wastes, capacity = instance_data
    params = DEConfig(pop_size=4, crossover_rate=0.5)
    solver = DESolver(dist_matrix, wastes, capacity, R=1.0, C=1.0, params=params)

    target = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    mutant = np.array([0.9, 0.9, 0.9, 0.9, 0.9])

    # We set n_nodes manually for this small test vector if needed,
    # but solver.n_nodes is determined by dist_matrix.
    # Let's fix the instance to match target size.
    solver.n_nodes = 5

    # Run crossover multiple times to ensure it inherits at least one from mutant
    for _ in range(100):
        trial = solver._exponential_crossover(target, mutant)
        diff_indices = np.where(trial != target)[0]
        assert len(diff_indices) >= 1
        # Check that mutant inheritance is consecutive (modulo n_nodes)
        if 0 < len(diff_indices) < 5:
            # Shift indices so they are strictly increasing without gap if consecutive
            # Find the "break" point where (idx[i]+1)%D != idx[i+1]
            breaks = []
            for k in range(len(diff_indices) - 1):
                if (diff_indices[k] + 1) % 5 != diff_indices[k+1]:
                    breaks.append(k)
            # There should be at most ONE break in a wrapped-around consecutive sequence
            assert len(breaks) <= 1
        elif len(diff_indices) == 5:
            # Full replacement is also possible and consecutive
            pass

def test_mde_full_solve(instance_data):
    dist_matrix, wastes, capacity = instance_data
    params = DEConfig(pop_size=10, max_iterations=5, crossover_rate=0.8)
    solver = DESolver(dist_matrix, wastes, capacity, R=1.0, C=1.0, params=params)

    routes, profit, cost = solver.solve()
    assert isinstance(routes, list)
    assert profit >= 0 # Should be at least 0 for this instance if it finds anything

def test_gaussian_initialization(instance_data):
    from logic.src.policies.helpers.operators.solution_initialization import build_greedy_routes
    dist_matrix, wastes, capacity = instance_data
    # Use NP=10 to have n_seeded=1 (10% of 10)
    params = DEConfig(pop_size=10, seed=42)
    solver = DESolver(dist_matrix, wastes, capacity, R=1.0, C=1.0, params=params)

    # Capture the greedy vector that would be generated
    # We need to use a fresh RNG to match what's inside solver
    py_rng_ref = np.random.RandomState(42) # Wait, solver uses random.Random for greedy
    import random
    py_rng_ref = random.Random(42)



    greedy_routes = build_greedy_routes(
        dist_matrix=dist_matrix,
        wastes=wastes,
        capacity=capacity,
        R=1.0,
        C=1.0,
        mandatory_nodes=[],
        rng=py_rng_ref,
    )
    greedy_vector = solver._encode_routes(greedy_routes)

    population = solver._initialize_population()

    # population[0] should be greedy_vector + Gaussian(0, 0.1)
    # Check that it's NOT the same but close
    assert not np.array_equal(population[0], greedy_vector)
    # Mean absolute error should be small (around 0.08 for normal(0, 0.1))
    mae = np.mean(np.abs(population[0] - greedy_vector))
    assert mae < 0.2

    # Check that other individuals are further away (random uniform)
    random_maes = [np.mean(np.abs(population[i] - greedy_vector)) for i in range(1, 10)]
    # Expected MAE for |Uniform(-1,1) - X| is larger than 0.2
    for r_mae in random_maes:
        assert r_mae > 0.2 or any(np.abs(population[i] - greedy_vector) > 0.5 for i in range(1, 10))

    assert np.all(population >= -1.0)
    assert np.all(population <= 1.0)
