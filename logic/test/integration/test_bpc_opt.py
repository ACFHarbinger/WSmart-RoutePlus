

import logging

import numpy as np
import pytest
from logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_price_and_cut.bpc_engine import (
    run_bpc,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


# Configure logging to see branching decisions
logging.basicConfig(level=logging.INFO)

def run_mini_instance(strategy):
    print(f"\n--- Testing BPC with strategy: {strategy} ---")
    # 4 nodes: 0 (depot), 1, 2, 3
    # Distances: triangle 1-2-3 with depot in middle
    # Optimal should probably be visiting all nodes if revenue is high
    dist = np.array([
        [0.0, 10.0, 10.0, 10.0],
        [10.0, 0.0, 5.0, 5.0],
        [10.0, 5.0, 0.0, 5.0],
        [10.0, 5.0, 5.0, 0.0]
    ])
    wastes = {1: 100.0, 2: 100.0, 3: 100.0}
    capacity = 500.0 # Can visit all
    R = 1.0 # Revenue 300
    C = 1.0 # Distance approx 10 + 5 + 5 + 10 = 30. Cost 30. Profit 270.

    # Optional nodes
    mandatory = [] # All optional

    params = {
        "branching_strategy": strategy,
        "max_bb_nodes": 50,
        "search_strategy": "depth_first"
    }

    routes, profit = run_bpc(
        dist_matrix=dist,
        wastes=wastes,
        capacity=capacity,
        R=R,
        C=C,
        params=params,
        mandatory_nodes=mandatory
    )

    print(f"Strategy {strategy} - Profit: {profit}")
    print(f"Routes: {routes}")
    return profit

if __name__ == "__main__":
    p_edge = run_mini_instance("edge")
    p_rf = run_mini_instance("ryan_foster")

    # Since all nodes are optional, Ryan-Foster with the new mandatory filter
    # might find no branching pairs if it only branches on mandatory nodes.
    # Wait, if all nodes are optional, and mandatory_nodes=[], then RF will NOT branch
    # on node pairs. It will fall back to edge branching in bpc_engine.py!
    # Let's verify that too.
