

import numpy as np
import pytest
from logic.src.policies import run_hgs
from logic.src.policies.routing.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.policy_swc_tcf import (
    run_swc_tcf_optimizer,
)
from logic.src.policies.routing.other_algorithms.capacitated_vehicle_routing_problem.cvrp import (
    find_routes,
    find_routes_ortools,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]



class TestSWCTCFOptimizerContract:
    """Contract tests for run_swc_tcf_optimizer (Gurobi)."""

    @pytest.mark.integration
    @pytest.mark.parametrize("backend", ["gurobi"])
    def test_swc_tcf_optimizer_basic_contract(self, base_vrpp_data, backend, check_license):
        data = base_vrpp_data
        routes, profit, cost = run_swc_tcf_optimizer(
            bins=data["bins"],
            distance_matrix=data["dist_matrix"],
            values=data["values"],
            binsids=data["binsids"],
            mandatory_nodes=data["mandatory_nodes"],
            optimizer=backend,
            time_limit=5,
        )

        # 1. Structure Check
        assert isinstance(routes, list), "Routes must be a list"
        assert routes[0] == 0, "Route must start at depot"
        assert routes[-1] == 0, "Route must end at depot"

        # 2. Results Check
        assert isinstance(profit, float), "Profit must be a float"
        assert isinstance(cost, float), "Cost must be a float"

        # 3. Constraint Check (Mandatory)
        # mandatory contains ID 2 and 4. They MUST be in the routes.
        # Note: ID 0 is depot. Bins are 1-5.
        for ms_id in data["mandatory_nodes"]:
            assert ms_id in routes, f"Mandatory bin {ms_id} was not collected by {backend}"

    @pytest.mark.integration
    @pytest.mark.parametrize("backend", ["gurobi"])
    def test_swc_tcf_optimizer_empty_bins(self, base_vrpp_data, backend, check_license):
        """Case where no bins should be collected."""
        data = base_vrpp_data
        data["bins"] = np.zeros(len(data["bins"]))  # all empty
        data["mandatory_nodes"] = []
        data["values"]["psi"] = 0.99

        routes, profit, cost = run_swc_tcf_optimizer(
            bins=data["bins"],
            distance_matrix=data["dist_matrix"],
            values=data["values"],
            binsids=data["binsids"],
            mandatory_nodes=data["mandatory_nodes"],
            optimizer=backend,
            time_limit=5,
        )
        # Should just return to depot [0] or [0, 0]
        assert set(routes) == {0}


class TestMultiVehicleContract:
    """Contract tests for find_routes (PyVRP/OR-Tools)."""

    @pytest.mark.integration
    def test_pyvrp_contract(self, base_vrpp_data):
        data = base_vrpp_data
        # find_routes expects wastes for clients (1..N)
        # our bins array has 5 elements for bins 1..5
        wastes = data["bins"]  # already 5 elements
        to_collect = [1, 2, 3, 4, 5]

        tour = find_routes(
            dist_mat=np.array(data["dist_matrix"]),
            wastes=wastes,
            max_caps=100,
            to_collect=to_collect,
            n_vehicles=0,  # Unlimited
        )

        assert isinstance(tour, list)
        assert tour[0] == 0
        assert tour[-1] == 0
        assert all(id in tour for id in to_collect), "All bins should be visited"

    @pytest.mark.integration
    def test_ortools_contract(self, base_vrpp_data):
        data = base_vrpp_data
        wastes = data["bins"]
        to_collect = [1, 2, 3, 4, 5]

        tour = find_routes_ortools(
            dist_mat=np.array(data["dist_matrix"]),
            wastes=wastes,
            max_caps=100,
            to_collect=to_collect,
            n_vehicles=0,  # Unlimited
        )

        assert isinstance(tour, list)
        assert tour.count(0) >= 3, "Should have at least Start-End and one separator for 2 vehicles or splits"
        assert all(id in tour for id in to_collect)


class TestHGSSolverContract:
    """Contract tests for run_hgs."""

    @pytest.mark.integration
    @pytest.mark.parametrize("engine", ["custom", "pyvrp"])
    def test_hgs_contract(self, base_vrpp_data, engine):
        data = base_vrpp_data
        # hgs expects wastes as dict
        wastes_dict = {i: data["bins"][i - 1] for i in range(1, 6)}

        # run_hgs(dist_matrix, wastes, capacity, R, C, values, *args)
        values = data["values"]
        values["engine"] = engine
        values["time_limit"] = 2

        routes, profit, cost = run_hgs(
            np.array(data["dist_matrix"]), wastes_dict, values["Q"], values["R"], values["C"], values
        )

        assert isinstance(routes, list)
        if routes and isinstance(routes[0], list):
            # nested routes [[...], [...]]
            flat = [0]
            for r in routes:
                flat.extend(r)
                flat.append(0)
            assert all(id in flat for id in [1, 2, 3, 4, 5])
        else:
            # flat route
            assert all(id in routes for id in [1, 2, 3, 4, 5])
