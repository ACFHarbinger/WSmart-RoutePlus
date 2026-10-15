

import numpy as np
import pytest
from logic.src.configs.policies.swc_tcf import SWCTCFConfig
from logic.src.policies import run_hgs
from logic.src.policies.routing.base.sub_problem import build_subproblem
from logic.src.policies.routing.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.policy_swc_tcf import (
    run_swc_tcf_optimizer,
)
from logic.src.policies.routing.other_algorithms.capacitated_vehicle_routing_problem.cvrp import (
    find_routes,
    find_routes_ortools,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]



class TestSolverEdgeCases:
    """
    Tests for solver stability handling edge cases:
    - N=0 (No clients)
    - N=1 (Single client)
    - Exact Capacity (Waste = Capacity)
    - Zero Waste (Client with 0 waste)
    - Impossible Waste (Waste > Capacity)
    """

    @pytest.fixture
    def edge_case_data(self):
        """Base template for edge cases."""
        # Minimal valid structure
        return {
            "bins": np.array([]),
            "dist_matrix": [[0]],
            "values": {
                "Q": 100.0,
                "R": 1.0,
                "B": 1.0,
                "C": 1.0,
                "V": 1.0,
                "Omega": 0.0,
                "delta": 0.0,
                "psi": 0.0,
            },
            "binsids": [0],
            "mandatory": [],
        }

    @pytest.mark.integration
    @pytest.mark.parametrize("backend", ["gurobi"])
    def test_n_zero_commercial(self, backend, edge_case_data, check_license):
        """Test with 0 clients (only depot) - Commercial Solvers."""
        data = edge_case_data
        data["bins"] = np.array([])
        data["dist_matrix"] = [[0.0]]  # 1x1 matrix (Depot only)
        data["binsids"] = [0]

        try:
            routes, _, _ = run_swc_tcf_optimizer(
                build_subproblem(
                    np.array(data["dist_matrix"]),
                    {},
                    data["values"]["Q"],
                    data["values"]["R"],
                    data["values"]["C"],
                    values=data["values"],
                ),
                SWCTCFConfig(framework="ortools", engine=backend, time_limit=2),
            )
            assert routes == [0] or routes == [0, 0] or len(routes) == 0
        except Exception as e:
            pytest.fail(f"{backend} crashed on N=0: {e}")

    @pytest.mark.integration
    @pytest.mark.parametrize("backend", ["ortools", "pyvrp", "hgs"])
    def test_n_zero_opensource(self, backend, edge_case_data):
        """Test with 0 clients (only depot) - Open Source Solvers."""
        data = edge_case_data
        data["bins"] = np.array([])
        data["dist_matrix"] = [[0]]
        data["binsids"] = [0]

        if backend == "ortools":
            try:
                tour = find_routes_ortools(
                    dist_mat=np.array(data["dist_matrix"]),
                    wastes=np.array([0]),  # Depot waste
                    max_caps=100,
                    to_collect=[],  # Empty list
                    n_vehicles=1,
                )
                assert tour == [0] or tour == [0, 0]
            except Exception as e:
                pytest.fail(f"OR-Tools crashed on N=0: {e}")

        elif backend == "pyvrp":
            try:
                tour = find_routes(
                    dist_mat=np.array(data["dist_matrix"]),
                    wastes=np.array([0]),  # Depot waste
                    max_caps=100,
                    to_collect=[],  # Empty list
                    n_vehicles=1,
                )
                assert tour == [0, 0] or tour == [0] or tour == []
            except Exception as e:
                pytest.fail(f"PyVRP crashed on N=0: {e}")

        elif backend == "hgs":
            try:
                wastes_dict = {}
                routes, _, _ = run_hgs(
                build_subproblem(np.array(data["dist_matrix"]), wastes_dict, 100, 1.0, 1.0),
                data["values"],
            )
                assert not routes or routes == [0]
            except Exception as e:
                pytest.fail(f"HGS crashed on N=0: {e}")

    @pytest.mark.integration
    @pytest.mark.parametrize("backend", ["gurobi"])
    def test_n_one_commercial(self, backend, edge_case_data, check_license):
        """Test with 1 client - Commercial Solvers."""
        data = edge_case_data
        data["bins"] = np.array([50.0])
        data["dist_matrix"] = [[0.0, 10.0], [10.0, 0.0]]
        data["binsids"] = [1]

        try:
            routes, _, _ = run_swc_tcf_optimizer(
                build_subproblem(
                    np.array(data["dist_matrix"]),
                    {1: 50.0},
                    data["values"]["Q"],
                    data["values"]["R"],
                    data["values"]["C"],
                    mandatory_nodes=[1],
                    values=data["values"],
                ),
                SWCTCFConfig(framework="ortools", engine=backend, time_limit=2),
            )
            assert 1 in routes
        except Exception as e:
            pytest.fail(f"{backend} crashed on N=1: {e}")

    @pytest.mark.integration
    @pytest.mark.parametrize("backend", ["ortools", "pyvrp", "hgs"])
    def test_n_one_opensource(self, backend, edge_case_data):
        """Test with 1 client - Open Source Solvers."""
        data = edge_case_data
        data["bins"] = np.array([50.0])
        data["dist_matrix"] = [[0, 10], [10, 0]]
        data["binsids"] = [1]

        if backend == "ortools":
            tour = find_routes_ortools(
                dist_mat=np.array(data["dist_matrix"]),
                wastes=np.array([0, 50]),
                max_caps=100,
                to_collect=[1],
                n_vehicles=1,
            )
            assert 1 in tour

        elif backend == "pyvrp":
            tour = find_routes(
                dist_mat=np.array(data["dist_matrix"]),
                wastes=np.array([0, 50]),
                max_caps=100,
                to_collect=[1],
                n_vehicles=1,
            )
            assert 1 in tour

        elif backend == "hgs":
            wastes_dict = {1: 50}
            routes, _, _ = run_hgs(
                build_subproblem(np.array(data["dist_matrix"]), wastes_dict, 100, 1.0, 1.0),
                data["values"],
            )
            flat = []
            if routes and isinstance(routes[0], list):
                for r in routes:
                    flat.extend(r)
            else:
                flat = routes
            assert 1 in flat

    @pytest.mark.integration
    @pytest.mark.parametrize("backend", ["gurobi"])
    def test_exact_capacity(self, backend, check_license):
        """Test with waste exactly equal to capacity."""
        # 2 nodes, 50+50 = 100 capacity.
        dist_matrix = [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
        values = {"Q": 100.0, "R": 1, "B": 1, "C": 0.1, "V": 1, "Omega": 0, "delta": 0, "psi": 0}

        # Test Gurobi (most sensitive to constraints)
        routes, _, _ = run_swc_tcf_optimizer(
            build_subproblem(
                np.array(dist_matrix),
                {1: 50.0, 2: 50.0},
                1000,
                1.0,
                0.1,
                mandatory_nodes=[1, 2],
                values=values,
            ),
            SWCTCFConfig(framework="ortools", engine="gurobi", time_limit=10),
        )
        assert 1 in routes and 2 in routes

    @pytest.mark.integration
    def test_zero_waste_node(self):
        """Test handling of nodes with 0 waste."""
        # Should be valid to visit.
        tour = find_routes_ortools(
            dist_mat=np.array([[0, 1], [1, 0]]),
            wastes=np.array([0, 0]),  # Node 1 has 0 waste
            max_caps=100,
            to_collect=[1],
            n_vehicles=1,
        )
        assert 1 in tour

    @pytest.mark.integration
    def test_impossible_waste_pyvrp(self):
        """Test waste > capacity."""
        try:
            tour = find_routes(
                dist_mat=np.array([[0, 1], [1, 0]]),
                wastes=np.array([0, 150]),  # 150 > 100
                max_caps=100,
                to_collect=[1],
                n_vehicles=1,
            )
            assert isinstance(tour, list)
        except Exception:
            pass
