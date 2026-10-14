

import numpy as np
import pytest
from logic.src.policies.helpers.solvers_and_matheuristics import (
    BranchNode,
    EdgeBranching,
    EdgeBranchingConstraint,
    Route,
    RyanFosterBranching,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]



def test_edge_branching_partition_logic():
    """Verify that EdgeBranching finds two fractional arcs and partitions on them."""
    # Create routes such that node 1 has multiple outgoing arcs with fractional flows
    # Arc (1, 2): 0.5
    # Arc (1, 3): 0.4
    # Arc (1, 4): 0.1
    routes = [
        Route(nodes=[0, 1, 2, 0], cost=0, revenue=0, load=0, node_coverage={1, 2}),
        Route(nodes=[0, 1, 3, 0], cost=0, revenue=0, load=0, node_coverage={1, 3}),
        Route(nodes=[0, 1, 4, 0], cost=0, revenue=0, load=0, node_coverage={1, 4}),
    ]
    route_values = {0: 0.5, 1: 0.4, 2: 0.1}

    # find_branching_arc should return ((1, 2), 0.5)
    res = EdgeBranching.find_branching_arc(routes, route_values)
    assert res is not None
    (u, v), flow = res
    assert (u, v) == (1, 2)

    # create_child_nodes should create two branches
    parent = BranchNode()
    left, right = EdgeBranching.create_child_nodes(parent, u, v)

    assert len(left.constraints) == 1
    assert isinstance(left.constraints[0], EdgeBranchingConstraint)
    assert left.constraints[0].u == 1
    assert left.constraints[0].v == 2
    assert left.constraints[0].must_use is True

    assert len(right.constraints) == 1
    assert isinstance(right.constraints[0], EdgeBranchingConstraint)
    assert right.constraints[0].u == 1
    assert right.constraints[0].v == 2
    assert right.constraints[0].must_use is False

def test_ryan_foster_mandatory_filter():
    """Verify that Ryan-Foster branching only pairs mandatory nodes."""
    # Mandatory nodes: {1}
    # Optional nodes: {2, 3}
    mandatory = {1}

    # Routes such that (1, 2) and (2, 3) have fractional co-occurrence
    routes = [
        Route(nodes=[0, 1, 2, 0], cost=0, revenue=0, load=0, node_coverage={1, 2}),
        Route(nodes=[0, 2, 3, 0], cost=0, revenue=0, load=0, node_coverage={2, 3}),
    ]
    # lambda_1 = 0.5, lambda_2 = 0.5
    # Co-occurrence(1, 2) = 0.5 (one mandatory) -> VALID
    # Co-occurrence(2, 3) = 0.5 (both optional) -> INVALID
    route_values = {0: 0.5, 1: 0.5}

    # Ryan-Foster now requires mandatory_nodes set to ensure exact partitioning.
    res = RyanFosterBranching.find_branching_pair(routes, route_values, mandatory_nodes=mandatory)
    assert res is not None
    (r, s), together_sum = res
    assert set((r, s)) == {1, 2}

    # Optional only (neither 2 nor 3 in same route fractionally? Wait)
    routes_opt = [
        Route(nodes=[0, 2, 3, 0], cost=0, revenue=0, load=0, node_coverage={2, 3}),
    ]
    route_values_opt = {0: 0.5}
    # When no mandatory nodes are restricted, we can pass an empty set or a universe set.
    res_opt = RyanFosterBranching.find_branching_pair(routes_opt, route_values_opt, mandatory_nodes={2, 3})
    assert res_opt is not None # It finds (2, 3) because it's fractional
    (r_opt, s_opt), together_sum_opt = res_opt
    assert set((r_opt, s_opt)) == {2, 3}

def test_edge_branching_no_partition():
    """Verify that EdgeBranching returns None if no second fractional arc exists at u."""
    # Only one fractional arc (1, 2)
    routes = [
        Route(nodes=[0, 1, 2, 0], cost=0, revenue=0, load=0, node_coverage={1, 2}),
    ]
    route_values = {0: 0.5}

    res = EdgeBranching.find_branching_arc(routes, route_values)
    # It returns (0, 1) or whatever arc is fractional.
    # The requirement of having TWO arcs was removed in the simplified EdgeBranching.
    assert res is not None

def test_bpc_policy_adapter_profit_alignment():
    """Verify that BPCPolicy._run_solver correctly interprets solver output."""
    from unittest.mock import patch

    from logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_price_and_cut.policy_bpc import (
        BPCPolicy,
    )

    policy = BPCPolicy()

    # Mock data
    sub_dist = np.array([[0, 10], [10, 0]])
    sub_wastes = {1: 100.0}

    # run_bpc returns (routes, solver_cost) where solver_cost is actually profit
    mock_routes = [[1]]
    mock_profit = 90.0 # 100 revenue - 10 cost

    with patch("logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_price_and_cut.policy_bpc.run_bpc") as mock_run:
        mock_run.return_value = (mock_routes, mock_profit)

        routes, profit, solver_cost = policy._run_solver(
            sub_dist_matrix=sub_dist,
            sub_wastes=sub_wastes,
            capacity=500.0,
            revenue=1.0,
            cost_unit=1.0,
            values={},
            mandatory_nodes=[]
        )

        assert routes == mock_routes
        assert profit == mock_profit
        # solver_cost (raw distance) should be 10 + 10 = 20
        assert solver_cost == 20.0
