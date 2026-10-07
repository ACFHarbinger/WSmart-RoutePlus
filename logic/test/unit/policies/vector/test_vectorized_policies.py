"""
Unit tests for vectorized policy implementations (Local Search, HGS, Split).
"""

import pytest
import torch
from logic.src.policies.vector.adaptive_large_neighborhood_search import (
    VectorizedALNS,
)
from logic.src.policies.vector.hgs_core.evaluation import calc_broken_pairs_distance
from logic.src.policies.vector.hybrid_genetic_search import (
    VectorizedHGS,
    VectorizedPopulation,
    vectorized_ordered_crossover,
)
from logic.src.policies.vector.operators import (
    vectorized_relocate,
    vectorized_swap,
    vectorized_swap_star,
    vectorized_three_opt,
    vectorized_two_opt,
    vectorized_two_opt_star,
)
from logic.src.policies.vector.shared.linear import (
    vectorized_linear_split,
)
from tensordict.tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]



class TestVectorizedLocalSearch:
    """Tests for vectorized local search heuristics."""

    @pytest.mark.unit
    def test_vectorized_two_opt(self, v_device):
        """Test basic 2-opt improvement."""
        pts = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], device=v_device)
        diff = pts.unsqueeze(1) - pts.unsqueeze(0)
        dist_matrix = torch.sqrt((diff**2).sum(dim=2)).unsqueeze(0)
        tours = torch.tensor([[0, 1, 2, 3, 0]], device=v_device)
        optimized_tours = vectorized_two_opt(tours.clone(), dist_matrix, max_iterations=50)

        def calc_cost(t, d):
            """Manual cost calculation for verification."""
            c = 0
            for k in range(len(t) - 1):
                c += d[0, t[k], t[k + 1]]
            return c

        orig_cost = calc_cost(tours[0], dist_matrix)
        opt_cost = calc_cost(optimized_tours[0], dist_matrix)
        assert opt_cost < orig_cost - 0.1
        assert abs(opt_cost - 4.0) < 1e-5

    @pytest.mark.unit
    def test_vectorized_three_opt(self, v_device):
        """Test 3-opt improvement."""
        pts = torch.tensor(
            [[0, 0], [2, 0], [2, 2], [0, 2], [1, 3], [-1, 3]],
            dtype=torch.float32,
            device=v_device,
        )
        diff = pts.unsqueeze(1) - pts.unsqueeze(0)
        dist_matrix = torch.sqrt((diff**2).sum(dim=2)).unsqueeze(0)
        tours = torch.tensor([[0, 1, 3, 5, 2, 4, 0]], device=v_device)

        def calc_cost(t, d):
            """Manual cost calculation for verification."""
            c = 0
            for k in range(len(t) - 1):
                c += d[0, t[k], t[k + 1]]
            return c

        orig_cost = calc_cost(tours[0], dist_matrix)
        optimized_tours = vectorized_three_opt(tours.clone(), dist_matrix, max_iterations=200)
        opt_cost = calc_cost(optimized_tours[0], dist_matrix)

        assert opt_cost <= orig_cost + 1e-5
        t = optimized_tours[0]
        assert t[0] == 0 and t[-1] == 0
        assert len(torch.unique(t[1:-1])) == 5

    @pytest.mark.unit
    def test_vectorized_swap(self, v_device, v_mock_dist_matrix):
        """Test vectorized swap operator."""
        tours = torch.tensor([[0, 1, 2, 3, 0]], device=v_device)
        dist = v_mock_dist_matrix.unsqueeze(0)
        new_tours = vectorized_swap(tours.clone(), dist, max_iterations=200)
        assert new_tours.shape == tours.shape

    @pytest.mark.unit
    def test_vectorized_relocate(self, v_device, v_mock_dist_matrix):
        """Test vectorized relocate operator."""
        tours = torch.tensor([[0, 1, 2, 3, 0]], device=v_device)
        dist = v_mock_dist_matrix.unsqueeze(0)
        new_tours = vectorized_relocate(tours.clone(), dist, max_iterations=200)
        assert new_tours.shape == tours.shape

    @pytest.mark.unit
    def test_vectorized_two_opt_star(self, v_device):
        """Test vectorized two-opt* operator."""
        pts = torch.tensor(
            [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [1.0, 1.0], [1.0, 2.0]],
            device=v_device,
        )
        diff = pts.unsqueeze(1) - pts.unsqueeze(0)
        dist = torch.sqrt((diff**2).sum(dim=2)).unsqueeze(0)
        tours = torch.tensor([[0, 1, 4, 0, 3, 2, 0]], device=v_device)
        new_tours = vectorized_two_opt_star(tours.clone(), dist, max_iterations=100)
        assert new_tours.shape == tours.shape

    @pytest.mark.unit
    def test_vectorized_swap_star(self, v_device):
        """Test vectorized swap* operator."""
        pts = torch.tensor(
            [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [1.0, 1.0], [1.0, 2.0]],
            device=v_device,
        )
        diff = pts.unsqueeze(1) - pts.unsqueeze(0)
        dist = torch.sqrt((diff**2).sum(dim=2)).unsqueeze(0)
        tours = torch.tensor([[0, 1, 4, 0, 3, 2, 0]], device=v_device)
        new_tours = vectorized_swap_star(tours.clone(), dist, max_iterations=100)
        assert new_tours.shape == tours.shape


class TestVectorizedPolicies:
    """Tests for the Hybrid Genetic Search vectorized implementation."""

    @pytest.mark.unit
    def test_vectorized_linear_split(self, v_device, v_sample_data):
        """Test vectorized linear split."""
        dist, wastes = v_sample_data
        tours = torch.tensor([[1, 2, 3, 4, 5]], device=v_device)
        dist_b = dist.unsqueeze(0)
        wastes_b = wastes.unsqueeze(0)
        capacity = 2.0
        routes, costs = vectorized_linear_split(tours, dist_b, wastes_b, capacity)
        assert len(routes) == 1
        assert costs.shape == (1,)

    @pytest.mark.unit
    def test_broken_pairs_distance(self, v_device):
        """Test broken pairs distance diversity."""
        B, P = 1, 2
        pop = torch.tensor([[[1, 2, 3, 4], [1, 2, 4, 3]]], device=v_device)
        diversity = calc_broken_pairs_distance(pop)
        assert diversity.shape == (B, P)
        assert diversity[0, 0] > 0

    @pytest.mark.unit
    def test_vectorized_ordered_crossover(self, v_device):
        """Test OX1 crossover."""
        B, N = 2, 5
        p1 = torch.stack([torch.randperm(N) + 1 for _ in range(B)]).to(v_device)
        p2 = torch.stack([torch.randperm(N) + 1 for _ in range(B)]).to(v_device)
        offspring = vectorized_ordered_crossover(p1, p2, v_device, None)
        assert offspring.size() == (B, N)
        assert len(torch.unique(offspring[0])) == N

    @pytest.mark.unit
    def test_vectorized_hgs_solve(self, v_device, v_sample_data):
        """Test full HGS solve run."""
        dist, wastes = v_sample_data
        B = 1
        dist_b = dist.unsqueeze(0)
        wastes_b = wastes.unsqueeze(0)
        hgs = VectorizedHGS(dist_b, wastes_b, vehicle_capacity=3.0, device=v_device)
        init_sol = torch.tensor([[1, 2, 3, 4, 5]], device=v_device)
        routes, costs = hgs.solve(init_sol, n_generations=5, population_size=10)
        assert len(routes) == B
        assert costs.shape == (B,)
        assert costs[0] < 1e9

    @pytest.mark.unit
    def test_vectorized_alns_basic(self):
        """Test VectorizedALNS on a small synthetic batch."""
        device = "cpu"
        batch_size = 4
        num_nodes = 11  # 1 depot + 10 nodes

        # 1. Create synthetic data
        # Locations (B, N, 2)
        locs = torch.rand(batch_size, num_nodes, 2, device=device)
        # Compute Euclidean distance matrix
        diff = locs.unsqueeze(2) - locs.unsqueeze(1)
        dist_matrix = torch.sqrt((diff**2).sum(dim=-1))

        # Wastes (B, N)
        wastes = torch.rand(batch_size, num_nodes, device=device) * 0.5
        wastes[:, 0] = 0.0  # Depot has 0 waste

        # Capacity (B,)
        capacity = torch.ones(batch_size, device=device) * 1.5

        # Initial solutions (giant tours)
        initial_solutions = torch.stack([torch.randperm(num_nodes - 1, device=device) + 1 for _ in range(batch_size)])

        # 2. Initialize Solver
        solver = VectorizedALNS(
            dist_matrix=dist_matrix,
            wastes=wastes,
            vehicle_capacity=capacity, # pyrefly: ignore [bad-argument-type]
            time_limit=1.0,
            device=device,
        )

        # 3. Solve
        n_iterations = 10
        routes_list, costs = solver.solve(initial_solutions=initial_solutions, n_iterations=n_iterations)

        # 4. Verify output
        assert len(routes_list) == batch_size
        assert costs.shape == (batch_size,)

        for b in range(batch_size):
            route = routes_list[b]
            # Routes from split usually start and end with 0 and contain all nodes
            nodes_visited = set(n for n in route if n != 0)
            assert len(nodes_visited) == num_nodes - 1, f"Batch {b} missing nodes"

            # Check capacity constraints (optional but good)
            curr_load = 0.0
            for node_val in route:
                node_idx = node_val
                if node_idx == 0:
                    assert curr_load <= capacity[b].item() + 1e-5
                    curr_load = 0.0
                else:
                    curr_load += wastes[b, node_idx].item()

    @pytest.mark.unit
    def test_alns_policy_forward(self):
        """Test the ALNSPolicy wrapper forward function."""
        import torch.nn as nn
        from logic.src.policies.vector.adaptive_large_neighborhood_search import (
            VectorizedALNS,
        )

        class ALNSPolicyWrapper(nn.Module):
            def __init__(self, env_name="mvptp", time_limit=1.0, max_iterations=100):
                super().__init__()
                self.env_name = env_name
                self.time_limit = time_limit
                self.max_iterations = max_iterations

            def forward(self, td, env=None):
                # Extract data
                locs = td["locs"]
                batch_size, num_nodes, _ = locs.shape

                # Compute distance matrix
                diff = locs.unsqueeze(2) - locs.unsqueeze(1)
                dist_matrix = torch.sqrt((diff**2).sum(dim=-1))

                wastes = td["waste"]
                capacity = td["capacity"]

                # Create initial solutions (random)
                # Note: ALNS expects initial solutions, or we can create them.
                # Here we mock them for testing.
                initial_solutions = torch.stack([
                    torch.randperm(num_nodes - 1, device=locs.device) + 1
                    for _ in range(batch_size)
                ])

                solver = VectorizedALNS(
                    dist_matrix=dist_matrix,
                    wastes=wastes,
                    vehicle_capacity=capacity,
                    time_limit=self.time_limit,
                    device=locs.device
                )

                routes, costs = solver.solve(
                    initial_solutions=initial_solutions,
                    n_iterations=self.max_iterations
                )

                return {
                    "actions": routes, # Note: routes structure might differ from flat actions
                    "reward": -costs # Transformation to reward
                }

        device = "cpu"
        batch_size = 2
        num_nodes = 6

        td = TensorDict(
            {
                "locs": torch.rand(batch_size, num_nodes, 2, device=device),
                "waste": torch.rand(batch_size, num_nodes, device=device) * 0.2,
                "depot": torch.zeros(batch_size, 2, device=device),
                "capacity": torch.ones(batch_size, device=device),
            },
            batch_size=[batch_size],
        )
        td["waste"][:, 0] = 0.0

        policy = ALNSPolicyWrapper(env_name="mvptp", time_limit=1.0, max_iterations=5).to(device)

        # Mock environment
        class MockEnv:
            waste_weight = 1.0
            cost_weight = 1.0

        out = policy(td, env=MockEnv())

        assert "reward" in out
        assert out["reward"].shape == (batch_size,)
        # Note: 'actions' check might need adjustment depending on solve output structure
        assert len(out["actions"]) == batch_size


class TestVectorizedPopulation:
    """Detailed tests for VectorizedPopulation management."""

    @pytest.mark.unit
    def test_population_add_individuals_selection(self, v_device):
        """Test that add_individuals keeps only best survivors."""

        max_size = 3
        pop = VectorizedPopulation(size=max_size, device=v_device)

        # Initial pop (size 2)
        initial_pop = torch.tensor([[[1, 2, 3, 4], [4, 3, 2, 1]]], device=v_device)
        initial_costs = torch.tensor([[10.0, 20.0]], device=v_device)  # 10 is best
        pop.initialize(initial_pop, initial_costs, nb_elite=1)
        assert pop.population.size(1) == 2

        # Add 3 more (total 5 > max_size 3)
        candidates = torch.tensor([[[2, 3, 4, 1], [3, 4, 1, 2], [1, 4, 2, 3]]], device=v_device)
        candidate_costs = torch.tensor([[5.0, 15.0, 30.0]], device=v_device)  # 5 is now best

        # Expected survivors (sorted by cost since alpha=0): 5.0, 10.0, 15.0
        pop.add_individuals(candidates, candidate_costs, nb_elite=5)

        assert pop.population.size(1) == max_size
        assert torch.allclose(pop.costs.sort().values, torch.tensor([[5.0, 10.0, 15.0]], device=v_device))

    @pytest.mark.unit
    def test_population_get_parents_tournament(self, v_device):
        """Test parent selection (tournament shape and randomness)."""
        B, P, N = 2, 10, 5
        pop = VectorizedPopulation(size=P, device=v_device)
        population = torch.stack([torch.randperm(N) + 1 for _ in range(B * P)]).view(B, P, N).to(v_device)
        costs = torch.rand((B, P), device=v_device)
        pop.initialize(population, costs, nb_elite=2)

        n_off = 4
        p1, p2 = pop.get_parents(n_offspring=n_off)

        assert p1.shape == (B, n_off, N)
        assert p2.shape == (B, n_off, N)

        # Check that nodes are valid
        assert (p1 > 0).all() and (p1 <= N).all()
