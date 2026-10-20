"""Tests for Imitation Learning training pipelines."""

from typing import Any

import pytest
import torch
from logic.src.policies.vector.core.crossover import vectorized_ordered_crossover
from logic.src.policies.vector.operators import (
    vectorized_relocate,
    vectorized_swap,
    vectorized_swap_star,
    vectorized_two_opt,
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
