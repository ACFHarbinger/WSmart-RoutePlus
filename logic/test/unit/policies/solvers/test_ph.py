"""
Unit tests for the Progressive Hedging (PH) policy.
"""

import numpy as np
import pytest
from logic.src.configs.policies import PHConfig
from logic.src.interfaces.context.problem_context import ProblemContext
from logic.src.pipeline.simulations.bins.prediction import ScenarioTree, ScenarioTreeNode
from logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_cut.bc import GUROBI_AVAILABLE
from logic.src.policies.routing.exact_and_decomposition_solvers.progressive_hedging import (
    ProgressiveHedgingPolicy,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]



@pytest.mark.skipif(not GUROBI_AVAILABLE, reason="Gurobi not available for subproblems")
def test_ph_convergence_small():
    """Test Progressive Hedging on a small stochastic instance."""
    n_nodes = 5
    # Simple star distance matrix
    dist_matrix = np.zeros((n_nodes, n_nodes))
    for i in range(1, n_nodes):
        dist_matrix[0, i] = 1.0  # Depot to node i
        dist_matrix[i, 0] = 1.0
        for j in range(1, n_nodes):
            if i != j:
                dist_matrix[i, j] = 1.5

    # Define scenarios
    scenarios_data = [
        {1: 50.0, 2: 50.0, 3: 0.0, 4: 0.0},
        {1: 0.0, 2: 0.0, 3: 50.0, 4: 50.0},
        {1: 30.0, 2: 30.0, 3: 30.0, 4: 30.0},
    ]

    # Build ScenarioTree
    root = ScenarioTreeNode(day=0, wastes=np.zeros(n_nodes), probability=1.0)
    for _i, data in enumerate(scenarios_data):
        w = np.zeros(n_nodes)
        for node, val in data.items():
            w[node] = val
        child = ScenarioTreeNode(day=1, wastes=w, probability=1.0 / len(scenarios_data))
        root.children.append(child)

    tree = ScenarioTree(root=root, horizon=1, num_bins=n_nodes)

    capacity = 100.0
    revenue = 1.0
    cost_unit = 1.0

    config = PHConfig(rho=0.1, max_iterations=10, convergence_tol=0.05, sub_solver="bc", verbose=True, seed=42)

    policy = ProgressiveHedgingPolicy(config=config)

    # Wrap in ProblemContext
    problem = ProblemContext(
        distance_matrix=dist_matrix,
        wastes=scenarios_data[0].copy(),  # Root wastes
        fill_rate_means=np.zeros(n_nodes),
        fill_rate_stds=np.zeros(n_nodes),
        capacity=capacity,
        max_fill=100.0,
        revenue_per_kg=revenue,
        cost_per_km=cost_unit,
        horizon=1,
        mandatory=[],
        locations=np.zeros((n_nodes, 2)),
        scenario_tree=tree,
    )

    # Execute
    tour, cost, profit, search_ctx, md_ctx = policy.execute(problem=problem)

    # Basic validity checks
    assert len(tour) > 0
    assert profit > 0
    assert cost > 0
    assert tour[0] == 0
    assert tour[-1] == 0
