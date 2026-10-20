"""Tests for Imitation Learning training pipelines."""

import time
from typing import Any

import pytest
import torch
from logic.src.policies.vector.core.evaluation import calc_broken_pairs_distance
from logic.src.policies.vector.hybrid_genetic_search import (
    VectorizedHGS,
    VectorizedPopulation,
    vectorized_linear_split,
    vectorized_ordered_crossover,
)
from logic.src.policies.vector.operators import vectorized_two_opt
from logic.src.policies.vector.operators import (
    vectorized_relocate,
    vectorized_swap,
    vectorized_swap_star,
    vectorized_two_opt_star,
)

pytestmark = [pytest.mark.unit, pytest.mark.slow]


class TestHGS:
    """Tests for HGS policies."""

    @pytest.fixture
    def device(self):
        """Fixture for device."""
        return "cpu"

    @pytest.fixture
    def mock_dist_matrix(self, device):
        """Fixture for a mock distance matrix."""
        # Simple Euclidean distance for points (0,0), (0,1), (1,0), (1,1)
        pts = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], device=device)
        diff = pts.unsqueeze(1) - pts.unsqueeze(0)
        dist = torch.sqrt((diff**2).sum(dim=2))
        return dist

    @pytest.fixture
    def sample_data(self, device):
        """Fixture for sample data (distance matrix and wastes)."""
        # 5 nodes + depot (0)
        pts = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
            device=device,
        )
        diff = pts.unsqueeze(1) - pts.unsqueeze(0)
        dist = torch.sqrt((diff**2).sum(dim=2))
        wastes = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=device)
        return dist, wastes

    @pytest.mark.unit
    def test_broken_pairs_distance(self, device):
        """Test calculation of broken pairs distance."""
        B, P, N = 2, 4, 3
        pop = torch.tensor([[1, 2, 3]], device=device).expand(B, P, N)
        diversity = calc_broken_pairs_distance(pop)
        assert torch.allclose(diversity, torch.zeros_like(diversity))

        pop3 = torch.tensor([[1, 2, 3, 4], [1, 2, 4, 3]], device=device).unsqueeze(0)

        diversity = calc_broken_pairs_distance(pop3)
        assert torch.isclose(diversity[0, 0], torch.tensor(0.5, device=device))
        assert torch.isclose(diversity[0, 1], torch.tensor(0.5, device=device))

    @pytest.mark.unit
    def test_vectorized_swap(self, device, mock_dist_matrix):
        """Test vectorized swap operator."""
        tours = torch.tensor([[0, 1, 2, 3, 0]], device=device)
        dist = mock_dist_matrix.unsqueeze(0)
        new_tours = vectorized_swap(tours.clone(), dist, max_iterations=200)

        def calc_cost(t):
            """Calculate route cost."""
            c: Any = 0
            for k in range(len(t) - 1):
                c += dist[0, t[k], t[k + 1]]
            return c

        orig_cost = calc_cost(tours[0])
        new_cost = calc_cost(new_tours[0])
        assert new_cost <= orig_cost + 1e-5

    @pytest.mark.unit
    def test_vectorized_relocate(self, device, mock_dist_matrix):
        """Test vectorized relocate operator."""
        tours = torch.tensor([[0, 1, 2, 3, 0]], device=device)
        dist = mock_dist_matrix.unsqueeze(0)
        new_tours = vectorized_relocate(tours.clone(), dist, max_iterations=200)
        orig_cost = 4.414  # Known cost for 0-1-2-3-0

        def calc_cost(t):
            """Calculate route cost."""
            c: Any = 0
            for k in range(len(t) - 1):
                c += dist[0, t[k], t[k + 1]]
            return c

        new_cost = calc_cost(new_tours[0])
        assert new_cost <= orig_cost + 1e-5

    @pytest.mark.unit
    def test_vectorized_two_opt_star(self, device):
        """Test vectorized two-opt* operator."""
        # Crossing path scenario
        pts = torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [1.0, 1.0], [1.0, 2.0]], device=device)
        diff = pts.unsqueeze(1) - pts.unsqueeze(0)
        dist = torch.sqrt((diff**2).sum(dim=2)).unsqueeze(0)

        tours = torch.tensor([[0, 1, 4, 0, 3, 2, 0]], device=device)
        new_tours = vectorized_two_opt_star(tours.clone(), dist, max_iterations=100)

        def calc_cost(t):
            """Calculate route cost."""
            c: Any = 0
            for k in range(len(t) - 1):
                c += dist[0, t[k], t[k + 1]]
            return c

        orig = calc_cost(tours[0])
        new_c = calc_cost(new_tours[0])

        assert new_c < orig - 0.1
        t = new_tours[0]
        nodes = t[t != 0]
        assert len(nodes) == 4
        assert set(nodes.tolist()) == {1, 2, 3, 4}

    @pytest.mark.unit
    def test_vectorized_swap_star(self, device):
        """Test vectorized swap* operator."""
        pts = torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [1.0, 1.0], [1.0, 2.0]], device=device)
        diff = pts.unsqueeze(1) - pts.unsqueeze(0)
        dist = torch.sqrt((diff**2).sum(dim=2)).unsqueeze(0)

        tours = torch.tensor([[0, 1, 4, 0, 3, 2, 0]], device=device)
        new_tours = vectorized_swap_star(tours.clone(), dist, max_iterations=100)

        def calc_cost(t):
            """Calculate route cost."""
            c: Any = 0
            for k in range(len(t) - 1):
                c += dist[0, t[k], t[k + 1]]
            return c

        orig = calc_cost(tours[0])
        new_c = calc_cost(new_tours[0])

        assert new_c < orig - 0.1
        t = new_tours[0]
        nodes = t[t != 0]
        assert len(nodes) == 4
        assert set(nodes.tolist()) == {1, 2, 3, 4}

    @pytest.mark.unit
    def test_population_diversity(self, device):
        """Test population diversity calculation."""
        B, P, N = 1, 10, 5
        pop = VectorizedPopulation(size=5, device=device)
        population = torch.stack([torch.randperm(N, device=device) + 1 for _ in range(P)]).unsqueeze(0)
        costs = torch.rand((B, P), device=device)
        pop.initialize(population, costs, nb_elite=2)

        assert pop.biased_fitness is not None
        assert pop.diversity_scores is not None
        assert pop.diversity_scores.shape == (B, P)
        assert pop.biased_fitness.shape == (B, P)

    @pytest.mark.unit
    def test_vectorized_ordered_crossover(self):
        """Test vectorized ordered crossover operator."""
        B, N = 4, 10
        p1 = torch.stack([torch.randperm(N) + 1 for _ in range(B)])
        p2 = torch.stack([torch.randperm(N) + 1 for _ in range(B)])
        offspring = vectorized_ordered_crossover(p1, p2)
        assert offspring.size() == (B, N)
        for b in range(B):
            nodes = offspring[b]
            assert len(torch.unique(nodes)) == N
            assert nodes.min() == 1
            assert nodes.max() == N

    @pytest.mark.unit
    def test_vectorized_linear_split(self, device, sample_data):
        """Test vectorized linear split for routing."""
        dist, wastes = sample_data
        # 5 nodes + depot. wastes = 1 each.
        # Capacity = 2.
        # Expected split:
        # Route 1: 1-2 (Load 2)
        # Route 2: 3-4 (Load 2)
        # Route 3: 5 (Load 1)
        # Or similar.

        # Giant tour: 1-2-3-4-5
        tours = torch.tensor([[1, 2, 3, 4, 5]], device=device)
        B = 1
        dist_b = dist.unsqueeze(0)
        wastes_b = wastes.unsqueeze(0)

        capacity = 2.0

        routes, costs = vectorized_linear_split(tours, dist_b, wastes_b, capacity)

        # Check output structure
        assert len(routes) == B
        # routes[0] is a flat list [0, n, n, 0, n, n, 0...]
        r_flat = routes[0]
        assert isinstance(r_flat, list)

        # Parse into individual routes
        parsed_routes = []
        current_route = [0]
        for node_val in r_flat[1:]:
            current_route.append(node_val)
            if node_val == 0:
                if len(current_route) > 2:  # Ignore 0-0 sequence if any
                    parsed_routes.append(current_route)
                current_route = [0]

        # With capacity 2 and 5 items, we expect at least 3 routes.
        assert len(parsed_routes) == 3

        # Check validity
        for r in parsed_routes:
            # r is [0, n1, n2, ..., 0]
            assert r[0] == 0 and r[-1] == 0
            nodes = r[1:-1]
            load = wastes[nodes].sum()
            assert load <= capacity + 1e-5

        # Check total nodes cover
        all_nodes = []
        for r in parsed_routes:
            all_nodes.extend(r[1:-1])
        all_nodes = sorted(all_nodes)
        assert all_nodes == [1, 2, 3, 4, 5]

    @pytest.mark.unit
    def test_time_limit(self, device, sample_data):
        """Test time limit enforcement."""
        dist, wastes = sample_data
        B = 2
        dist_b = dist.unsqueeze(0).expand(B, -1, -1)
        wastes_b = wastes.unsqueeze(0).expand(B, -1)

        hgs = VectorizedHGS(dist_b, wastes_b, vehicle_capacity=10.0, time_limit=10.0, device=device)
        init_sol = torch.stack([torch.randperm(5, device=device) + 1 for _ in range(B)])

        start_t = time.time()
        hgs.solve(init_sol, n_generations=10000, population_size=10, time_limit=0.2)
        end_t = time.time()
        # Allow sufficient buffer for initialization overhead and generation granularity
        assert (end_t - start_t) < 3.0

    @pytest.mark.unit
    def test_max_vehicles_constraint(self, device, sample_data):
        """Test maximum vehicles constraint."""
        dist, wastes = sample_data
        dist_b = dist.unsqueeze(0)
        wastes_b = wastes.unsqueeze(0)
        tours = torch.tensor([[1, 2, 3, 4, 5]], device=device)

        hgs = VectorizedHGS(dist_b, wastes_b, vehicle_capacity=3.0, device=device)
        routes, costs = hgs.solve(tours, n_generations=1, population_size=5, max_vehicles=2)
        assert costs[0] < float("inf")

        routes_inf, costs_inf = hgs.solve(tours, n_generations=1, population_size=5, max_vehicles=1)
        assert costs_inf[0] == float("inf") or costs_inf[0] > 1e9

    @pytest.mark.unit
    def test_stagnation_restart_runs(self, mocker, device, sample_data):
        """Test restart mechanism on stagnation."""
        dist, wastes = sample_data
        B = 1
        dist_b = dist.unsqueeze(0)
        wastes_b = wastes.unsqueeze(0)
        hgs = VectorizedHGS(dist_b, wastes_b, vehicle_capacity=10.0, device=device)
        init_sol = torch.stack([torch.randperm(5, device=device) + 1 for _ in range(B)])

        # Run enough gens to trigger restart (50 is threshold)
        hgs.solve(init_sol, n_generations=60, population_size=10, elite_size=2)


class TestVectorized2OptIL:
    """Tests for Vectorized 2-Opt Imitation Learning."""

    @pytest.fixture
    def device(self):
        """Fixture for device."""
        return "cpu"

    @pytest.mark.unit
    def test_2opt_basic_improvement(self, device):
        """Test basic 2-opt improvement."""
        # Scenario: Crossing paths
        # 0 at (0,0), 1 at (0,1), 2 at (1,0), 3 at (1,1)
        # Tour: 0 -> 1 -> 2 -> 3 -> 0
        # Edges: (0,1), (1,2), (2,3), (3,0)
        # Dist: 1 + sqrt(2) + 1 + 1 = 3 + 1.414 = 4.414
        # Optimal: 0 -> 1 -> 3 -> 2 -> 0 (Square perimeter)
        # Edges: (0,1), (1,3), (3,2), (2,0)
        # Dist: 1 + 1 + 1 + 1 = 4.0

        pts = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], device=device)
        diff = pts.unsqueeze(1) - pts.unsqueeze(0)
        dist_matrix = torch.sqrt((diff**2).sum(dim=2)).unsqueeze(0)  # (1, 4, 4)

        # Initial tour: 0-1-2-3-0. But operator expects indices?
        # If input is (B, N) or (B, N+2) depending on if depot included.
        # vectorized_two_opt expects batch of tours.
        # It typically works on indices.

        # tour: [0, 1, 2, 3] (if depot implied?) or [0, 1, 2, 3, 0]?
        # The operator implementation looks for edges (i, i+1).
        # It requires indices to be valid for distance_matrix.
        # If N=4 (0..3), tour should just contain 0,1,2,3.
        # If it's a cycle, it usually ends with 0.

        tours = torch.tensor([[0, 1, 2, 3, 0]], device=device)

        optimized_tours = vectorized_two_opt(tours.clone(), dist_matrix, max_iterations=50)

        def calc_cost(t, d):
            """Calculate route cost."""
            c: Any = 0
            for k in range(len(t) - 1):
                c += d[0, t[k], t[k + 1]]
            return c

        orig_cost = calc_cost(tours[0], dist_matrix)
        opt_cost = calc_cost(optimized_tours[0], dist_matrix)

        assert opt_cost < orig_cost - 0.1
        assert abs(opt_cost - 4.0) < 1e-5

    @pytest.mark.unit
    def test_2opt_batch_consistency(self, device):
        """Test batch consistency of 2-opt."""
        # Two identical tours in batch should produce identical results
        pts = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], device=device)
        diff = pts.unsqueeze(1) - pts.unsqueeze(0)
        dist_matrix = torch.sqrt((diff**2).sum(dim=2)).unsqueeze(0)
        dist_batch = dist_matrix.expand(2, -1, -1)

        tours = torch.tensor([[0, 1, 2, 3, 0], [0, 1, 2, 3, 0]], device=device)

        optimized = vectorized_two_opt(tours.clone(), dist_batch, max_iterations=50)

        assert torch.all(optimized[0] == optimized[1])
        # Both should be improved
        assert (optimized[0] == torch.tensor([0, 1, 3, 2, 0], device=device)).all()

    @pytest.mark.unit
    def test_2opt_stability(self, device):
        """Test stability of 2-opt."""
        # Optimal tour shouldn't change
        # 0 -> 1 -> 3 -> 2 -> 0 is optimal 4.0

        pts = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], device=device)
        diff = pts.unsqueeze(1) - pts.unsqueeze(0)
        dist_matrix = torch.sqrt((diff**2).sum(dim=2)).unsqueeze(0)

        optimal_tour = torch.tensor([[0, 1, 3, 2, 0]], device=device)

        result = vectorized_two_opt(optimal_tour.clone(), dist_matrix, max_iterations=10)

        assert torch.all(result == optimal_tour)
