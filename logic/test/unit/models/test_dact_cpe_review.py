"""Independent scalar equation checks for DACT positional encoding."""

import math

import pytest
import torch
from logic.src.models.core.dact.encoder import DACTEncoder
from tensordict import TensorDict


@pytest.mark.parametrize("dim,n", [(16, 5), (64, 20)])
def test_cpe_matches_paper_scalar_equations(dim, n):
    encoder = DACTEncoder(embed_dim=dim, num_layers=1, num_heads=4)
    solution = torch.arange(n).roll(2).unsqueeze(0)
    expected = torch.empty(1, n, dim)
    for rank, node in enumerate(solution[0].tolist()):
        for d in range(dim):
            minimum = n ** (1 / (dim // 2))
            wavelength = ((3 * (d // 3) + 1) / dim) * (n - minimum) + minimum if d < dim // 2 else n
            omega = 2 * math.pi / wavelength
            z = rank / n * wavelength * math.ceil(n / wavelength)
            phase = omega * abs((z % (4 * math.pi / omega)) - 2 * math.pi / omega)
            expected[0, node, d] = math.sin(phase) if d % 2 == 0 else math.cos(phase)
    torch.testing.assert_close(encoder._compute_cyclic_positions(solution, n), expected, atol=2e-5, rtol=2e-5)


def test_encoder_without_solution_uses_identity_tour_cpe():
    encoder = DACTEncoder(embed_dim=16, num_layers=1, num_heads=4)
    td = TensorDict({"depot": torch.zeros(1, 2), "locs": torch.zeros(1, 4, 2)}, batch_size=[1])
    without_solution = encoder(td)
    td["solution"] = torch.arange(5).unsqueeze(0)
    with_solution = encoder(td)
    for aspect in range(2):
        torch.testing.assert_close(without_solution[aspect], with_solution[aspect])
