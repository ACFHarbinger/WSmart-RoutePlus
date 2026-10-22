"""
Unit tests for vectorized policy implementations (Local Search, HGS, Split).
"""

import pytest
import torch
from logic.src.policies.vectorized.routing.core.crossover import vectorized_ordered_crossover
from logic.src.policies.vectorized.routing.operators import (
    vectorized_relocate,
    vectorized_swap,
    vectorized_swap_star,
    vectorized_three_opt,
    vectorized_two_opt,
    vectorized_two_opt_star,
)

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
    """Tests for the shared split and crossover helpers (the experts are in test_vector_experts.py)."""

    @pytest.mark.unit
    def test_vectorized_ordered_crossover(self, v_device):
        """Test OX1 crossover."""
        B, N = 2, 5
        p1 = torch.stack([torch.randperm(N) + 1 for _ in range(B)]).to(v_device)
        p2 = torch.stack([torch.randperm(N) + 1 for _ in range(B)]).to(v_device)
        offspring = vectorized_ordered_crossover(p1, p2, v_device, None)
        assert offspring.size() == (B, N)
        assert len(torch.unique(offspring[0])) == N
