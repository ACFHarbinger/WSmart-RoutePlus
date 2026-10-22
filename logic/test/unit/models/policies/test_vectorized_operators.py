

import pytest
import torch
from logic.src.policies.vectorized.routing.operators import (
    vectorized_greedy_insertion,
    vectorized_random_removal,
    vectorized_regret_k_insertion,
    vectorized_worst_removal,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]



@pytest.fixture
def data():
    B, N = 5, 20
    tours = torch.zeros((B, N + 2), dtype=torch.long)
    tours[:, 1:-1] = torch.arange(1, N + 1)

    dist_matrix = torch.rand((B, N + 1, N + 1)) # nodes 0..N

    return tours, dist_matrix

def test_vectorized_random_removal(data):
    tours, _ = data
    B, N_total = tours.shape
    n_remove = 5

    new_tours, removed = vectorized_random_removal(tours, n_remove)

    assert new_tours.shape == (B, N_total - n_remove)
    assert removed.shape == (B, n_remove)

def test_vectorized_worst_removal(data):
    tours, dist_matrix = data
    n_remove = 5
    B, N_total = tours.shape

    new_tours, removed = vectorized_worst_removal(tours, dist_matrix, n_remove)

    assert new_tours.shape == (B, N_total - n_remove)
    assert removed.shape == (B, n_remove)

def test_vectorized_greedy_insertion(data):
    tours, dist_matrix = data
    B, N_total = tours.shape
    n_rem = 3

    # Fake removed nodes
    removed_nodes = torch.tensor([[100, 101, 102]] * B, dtype=torch.long)

    # We need dist_matrix to cover these new nodes.
    # Resize dist_matrix
    ext_dist = torch.rand((B, 200, 200)) # Enough space
    ext_dist[:, :21, :21] = dist_matrix[:, :21, :21]

    new_tours = vectorized_greedy_insertion(tours, removed_nodes, ext_dist)

    assert new_tours.shape == (B, N_total + n_rem)

def test_vectorized_regret_k_insertion(data):
    tours, dist_matrix = data
    B, N_total = tours.shape
    n_rem = 3

    removed_nodes = torch.tensor([[100, 101, 102]] * B, dtype=torch.long)
    ext_dist = torch.rand((B, 200, 200))

    new_tours = vectorized_regret_k_insertion(tours, removed_nodes, ext_dist, k=2)

    assert new_tours.shape == (B, N_total + n_rem)
