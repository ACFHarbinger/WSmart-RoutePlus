"""Tests for Route Improvement sub-system."""

import numpy as np
import pytest
from logic.src.policies.route_improvement import (
    ClassicalLocalSearchRouteImprover,
    FastTSPRouteImprover,
    RandomLocalSearchRouteImprover,
    RouteImproverFactory,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]


@pytest.fixture
def sample_route_data():
    """Sample data for route improvement tests."""
    dist_matrix = np.array(
        [
            [0, 10, 20, 10],
            [10, 0, 10, 20],
            [20, 10, 0, 10],
            [10, 20, 10, 0],
        ],
        dtype=np.int32,
    )
    tour = [0, 2, 1, 3, 0]  # Inefficient tour
    return dist_matrix, tour


class TestRouteImproverFactory:
    def test_factory_get_fast_tsp(self):
        proc = RouteImproverFactory.create("fast_tsp")
        assert isinstance(proc, FastTSPRouteImprover)

    def test_factory_get_classical_ls(self):
        proc = RouteImproverFactory.create("classical_local_search")
        assert isinstance(proc, ClassicalLocalSearchRouteImprover)

    def test_factory_invalid(self):
        with pytest.raises(ValueError):
            RouteImproverFactory.create("invalid_processor")


class TestFastTSPRouteImprover:
    def test_fast_tsp_refinement(self, sample_route_data):
        dist_matrix, tour = sample_route_data
        processor = FastTSPRouteImprover()
        refined_tour, _ = processor.process(tour, distance_matrix=dist_matrix)
        assert len(refined_tour) == len(tour)
        assert refined_tour[0] == 0
        assert refined_tour[-1] == 0


class TestClassicalLocalSearchRouteImprover:
    def test_2opt_refinement(self, sample_route_data):
        dist_matrix, tour = sample_route_data
        processor = ClassicalLocalSearchRouteImprover()
        refined_tour, _ = processor.process(tour, distance_matrix=dist_matrix, operator_name="2opt")
        assert len(refined_tour) == len(tour)
        assert refined_tour[0] == 0
        assert refined_tour[-1] == 0

    def test_multi_route_refinement(self):
        # 5 customer nodes + depot 0
        dist_matrix = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                dist_matrix[i, j] = abs(i - j)

        # Inefficient multi-route tour: [0, 2, 1, 0, 4, 3, 5, 0]
        # Route 1: 0-2-1-0 (length 2+1+1=4). Optimal: 0-1-2-0 (length 1+1+2=4) wait.
        # Actually with distance = |i-j|:
        # 0-2-1-0: d(0,2)=2, d(2,1)=1, d(1,0)=1. Total=4.
        # 0-4-3-5-0: d(0,4)=4, d(4,3)=1, d(3,5)=2, d(5,0)=5. Total=12.
        # Optimized 0-3-4-5-0: d(0,3)=3, d(3,4)=1, d(4,5)=1, d(5,0)=5. Total=10.

        tour = [0, 2, 1, 0, 4, 3, 5, 0]
        processor = ClassicalLocalSearchRouteImprover()
        refined_tour, _ = processor.process(tour, distance_matrix=dist_matrix, operator_name="2opt")

        assert refined_tour[0] == 0
        assert refined_tour[-1] == 0
        assert 0 in refined_tour[1:-1]  # Ensure it remains multi-route

    def test_multi_operator_sequential_refinement(self):
        # Setup an inefficient tour where a combination of operators is applied.
        dist_matrix = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                dist_matrix[i, j] = abs(i - j)

        # Inefficient tour: [0, 4, 3, 5, 2, 1, 0]
        # Distance: d(0,4)=4, d(4,3)=1, d(3,5)=2, d(5,2)=3, d(2,1)=1, d(1,0)=1 -> Total = 12
        tour = [0, 4, 3, 5, 2, 1, 0]
        processor = ClassicalLocalSearchRouteImprover()
        refined_tour, _ = processor.process(tour, distance_matrix=dist_matrix, operator_name="2opt", ls_operator="all")

        def get_tour_cost(t):
            return sum(dist_matrix[t[k], t[k + 1]] for k in range(len(t) - 1))

        assert refined_tour[0] == 0
        assert refined_tour[-1] == 0
        assert len(refined_tour) == len(tour)
        assert get_tour_cost(refined_tour) < get_tour_cost(tour)


class TestRandomLocalSearchRouteImprover:
    def test_random_ls_refinement(self, sample_route_data):
        dist_matrix, tour = sample_route_data
        processor = RandomLocalSearchRouteImprover()
        refined_tour, _ = processor.process(tour, distance_matrix=dist_matrix, iterations=10, op_probs={"two_opt": 1.0})
        assert len(refined_tour) == len(tour)
        assert refined_tour[0] == 0
        assert refined_tour[-1] == 0


class TestPathRouteImprover:
    def test_path_refinement_fills_gap(self):
        # Scenario: Tour 0 -> 1 -> 3 -> 0
        # Path 1->3 goes through 2: [1, 2, 3]
        # Bin 2 is not in tour. fits capacity.

        processor = RouteImproverFactory.create("path")

        tour = [0, 1, 3, 0]

        # Mock paths: paths[1][3] = [1, 2, 3]
        # We need a structure that supports paths[u][v]
        # Minimal mock using dict/list
        paths = [[[] for _ in range(4)] for _ in range(4)]
        paths[1][3] = [1, 2, 3]

        # Mock fill levels: bin 1=10, bin 2=10, bin 3=10
        # Indices in fill array are bin_id-1.
        # bin 1 -> idx 0, etc.
        current_fill = np.array([10.0, 10.0, 10.0, 10.0])  # 4 bins

        refined, _ = processor.process(
            tour, paths_between_states=paths, total_fill=current_fill, vehicle_capacity=100.0, bins=None
        )

        # Expect 2 to be inserted between 1 and 3
        expected = [0, 1, 2, 3, 0]
        assert refined == expected

    def test_path_refinement_skip_if_full(self):
        processor = RouteImproverFactory.create("path")
        tour = [0, 1, 3, 0]
        paths = [[[] for _ in range(4)] for _ in range(4)]
        paths[1][3] = [1, 2, 3]
        current_fill = np.array([50.0, 50.0, 50.0, 50.0])

        # Current load: 1(50) + 3(50) = 100. Capacity = 100.
        # Node 2(50) + 100 = 150 > 100. Should skip.

        refined, _ = processor.process(
            tour, paths_between_states=paths, total_fill=current_fill, vehicle_capacity=100.0
        )

        assert refined == tour

    def test_path_refinement_skip_existing(self):
        # If 2 is already in tour, don't double add
        processor = RouteImproverFactory.create("path")
        tour = [0, 1, 2, 3, 0]
        paths = [[[] for _ in range(4)] for _ in range(4)]
        paths[1][3] = [1, 2, 3]
        current_fill = np.array([10.0, 10.0, 10.0, 10.0])

        refined, _ = processor.process(
            tour, paths_between_states=paths, total_fill=current_fill, vehicle_capacity=100.0
        )

        assert refined == tour
