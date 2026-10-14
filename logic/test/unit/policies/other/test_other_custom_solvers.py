


import numpy as np
import pytest
from logic.src.policies.routing.other_algorithms.capacitated_vehicle_routing_problem.cvrp import find_routes
from logic.src.policies.routing.other_algorithms.travelling_salesman_problem.tsp import find_route

pytestmark = [pytest.mark.unit, pytest.mark.fast]





def test_custom_cvrp_clarke_wright():
    # Simple instance
    dist_mat = np.array([
        [0, 10, 10, 12],
        [10, 0, 5, 15],
        [10, 5, 0, 15],
        [12, 15, 15, 0]
    ])
    wastes = np.array([0, 10, 10, 10]) # node 0 is depot
    max_caps = 15
    to_collect = [1, 2, 3] # node indices

    # Internal engine (Clark-Wright)
    routes = find_routes(dist_mat, wastes, max_caps, to_collect, n_vehicles=2, engine="custom")

    # Expect at least two routes due to capacity (10+10 > 15)
    # Clark-Wright should group 1 and 2 if capacity allowed, but here it won't.
    # Actually, 1 and 2 are closer (5), so if cap was 20: [0, 1, 2, 0, 3, 0]
    # With cap 15, maybe [0, 1, 2, 0, 3, 0] is impossible.
    # Let's check: waste[1]=10, waste[2]=10. Sum=20 > 15.
    # So 1 and 2 MUST be in different routes.
    # Possible: [0, 1, 0, 2, 0, 3, 0] or similar.

    assert isinstance(routes, list)
    assert 0 in routes
    # All to_collect nodes should be present
    for node in to_collect:
        assert node in routes

def test_custom_tsp_2opt():
    # Simple instance
    dist_mat = np.array([
        [0, 10, 20, 10],
        [10, 0, 10, 20],
        [20, 10, 0, 10],
        [10, 20, 10, 0]
    ])
    to_collect = [1, 2, 3]

    # Optimal TSP should be 0-1-2-3-0 (cost 10+10+10+10=40)
    # vs 0-2-1-3-0 (cost 20+10+20+10=60)

    tour = find_route(dist_mat, to_collect, engine="custom")

    assert isinstance(tour, list)
    assert tour[0] == 0
    assert tour[-1] == 0
    assert len(tour) == 5 # [0, n1, n2, n3, 0]
    for node in to_collect:
        assert node in tour
