"""
Tests for NDS-BRKGA.
"""

import os

import numpy as np
import pytest

# Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python for the entire test session
# to avoid descriptor creation errors if another test imports protobuf.
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from logic.src.policies.packing_and_routing.non_dominated_sorting_biased_random_key_genetic_algorithm.chromosome import (
    Chromosome,
    compute_adaptive_thresholds,
)
from logic.src.policies.packing_and_routing.non_dominated_sorting_biased_random_key_genetic_algorithm.crossover import (
    biased_crossover,
)
from logic.src.policies.packing_and_routing.non_dominated_sorting_biased_random_key_genetic_algorithm.nsga2 import (
    fast_non_dominated_sort,
)
from logic.src.policies.packing_and_routing.non_dominated_sorting_biased_random_key_genetic_algorithm.objectives import (
    compute_overflow_risk,
)
from logic.src.configs.policies.nds_brkga import NDSBRKGAConfig
from logic.src.policies.packing_and_routing.non_dominated_sorting_biased_random_key_genetic_algorithm.policy_nds_brkga import (
    NDSBRKGAPolicy,
)
from logic.src.policies.packing_and_routing.non_dominated_sorting_biased_random_key_genetic_algorithm.population import (
    Population,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]



@pytest.fixture
def mock_nds_problem():
    """Simple 5-bin problem for testing."""
    n_bins = 5
    dist_matrix = np.ones((n_bins + 1, n_bins + 1))
    np.fill_diagonal(dist_matrix, 0)
    # Make distance variation
    for i in range(1, n_bins + 1):
        dist_matrix[0, i] = dist_matrix[i, 0] = i * 0.1

    current_fill = np.array([10.0, 50.0, 110.0, 20.0, 80.0])  # % (one overflow bin)
    wastes = {i: current_fill[i - 1] for i in range(1, n_bins + 1)}

    return {
        "dist_matrix": dist_matrix,
        "current_fill": current_fill,
        "wastes": wastes,
        "capacity": 1000.0,
        "revenue": 2.0,
        "cost_unit": 1.0,
        "bin_density": 0.5,
        "bin_volume": 100.0,
        "max_fill": 100.0,
        "overflow_penalty_frac": 2.0,
    }


def test_chromosome_decoding(mock_nds_problem):
    """Test that chromosome decodes to valid selection and routes."""
    data = mock_nds_problem
    n_bins = len(data["current_fill"])
    NDSBRKGAConfig()

    # 2N keys
    keys = np.random.rand(2 * n_bins)
    # High risk bin (index 2, 95%) should be selected if key is reasonably low
    # Low risk bin (index 0, 10%) should be excluded if key is high
    keys[2] = 1.0  # Force select bin 3 (index 2)
    keys[0] = 0.0  # Force exclude bin 1 (index 0)

    chrom = Chromosome(keys, n_bins)
    thresholds = np.full(n_bins, 0.5)  # fixed thresholds for this test

    routes = chrom.to_routes(thresholds, data["wastes"], data["capacity"])
    selected = sorted({b for r in routes for b in r})

    assert isinstance(selected, list)
    assert 3 in selected  # bin 3 (index 2)
    assert 1 not in selected  # bin 1 (index 0)

    assert isinstance(routes, list)
    if selected:
        # Check that all selected bins are in the routes
        flat_route = [node for r in routes for node in r]
        assert set(flat_route) == set(selected)


def test_nsga2_sorting():
    """Test non-dominated sorting logic."""
    # 3 objectives (f1, f2, f3) - we minimize all
    # Obj: (-profit, overflow_cost, distance)

    # sol0 dominates sol1
    # sol2 is non-dominated with sol0 (different trade-off)
    objs = np.array([
        [10.0, 5.0, 100.0],  # sol 0
        [12.0, 6.0, 110.0],  # sol 1 (dominated by 0)
        [5.0, 15.0, 150.0],  # sol 2 (better f1, worse f2, f3)
    ])

    fronts = fast_non_dominated_sort(objs)

    assert 0 in fronts[0]
    assert 2 in fronts[0]
    assert 1 in fronts[1]


def test_crossover():
    """Test biased uniform crossover inheriting from elite."""
    rng = np.random.default_rng(42)
    p1 = Chromosome(np.zeros(10), 5)
    p2 = Chromosome(np.ones(10), 5)

    # p1 is elite, bias=0.7
    child = biased_crossover(p1, p2, 1.0, rng) # Always elite
    assert np.all(child.keys == 0)

    child = biased_crossover(p1, p2, 0.0, rng) # Always non-elite
    assert np.all(child.keys == 1)


def test_population_seeding(mock_nds_problem):
    """Test that seeding produces chromosomes."""
    data = mock_nds_problem
    n_bins = len(data["current_fill"])
    params = NDSBRKGAConfig(n_seed_solutions=2)

    # Evaluate threshold and risk for population init
    bin_mass = np.full(n_bins, 50.0) # 0.5 * 100
    overflow_risk = compute_overflow_risk(data["current_fill"], bin_mass, None, data["overflow_penalty_frac"])
    thresholds = compute_adaptive_thresholds(overflow_risk)

    pop = Population.initialise(
        n_bins=n_bins,
        thresholds=thresholds,
        dist_matrix=data["dist_matrix"],
        wastes=data["wastes"],
        capacity=data["capacity"],
        R=data["revenue"],
        C=data["cost_unit"],
        overflow_risk=overflow_risk,
        current_fill=data["current_fill"],
        bin_density=data["bin_density"],
        bin_volume=data["bin_volume"],
        max_fill=data["max_fill"],
        overflow_penalty_frac=data["overflow_penalty_frac"],
        scenario_tree=None,
        params=params
    )

    # Check chromosomes populated
    assert len(pop.chromosomes) >= 2
    assert pop.objectives.shape == (params.pop_size, 3)


@pytest.mark.integration
def test_policy_adapter_integration(mock_nds_problem):
    """Test the NDSBRKGAPolicy adapter with simulation-like inputs."""
    data = mock_nds_problem
    config = {
        "pop_size": 10,
        "max_generations": 2,
    }
    policy = NDSBRKGAPolicy(config)

    # sub_dist_matrix is 1..M + depot
    # wastes is {1..M: fill}
    # NDSBRKGAPolicy adapter expects these as args
    # _run_solver(self, sub_dist_matrix, sub_wastes, capacity, revenue, cost_unit, values, mandatory_nodes, **kwargs)

    values = {
        "B": data["bin_density"],
        "V": data["bin_volume"],
        "max_fill": 100.0,
        "overflow_penalty_frac": 1.0,
    }

    routes, profit, cost = policy._run_solver(
        sub_dist_matrix=data["dist_matrix"],
        sub_wastes=data["wastes"],
        capacity=data["capacity"],
        revenue=data["revenue"],
        cost_unit=data["cost_unit"],
        values=values,
        mandatory_nodes=[],
    )

    assert isinstance(routes, list)
    assert isinstance(profit, float)
    assert isinstance(cost, float)
