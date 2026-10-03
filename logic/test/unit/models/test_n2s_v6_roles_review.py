"""Request identity must survive changes to tour order."""

import torch
from logic.src.models.core.n2s.decoder import N2SDecoder


def test_delivery_identity_is_not_redefined_by_tour_order():
    decoder = N2SDecoder(embed_dim=16, num_heads=4)
    partners = torch.tensor([[0, 4, 5, 6, 1, 2, 3]])
    valid = torch.arange(7).unsqueeze(0)
    # An unrestricted TSP reset can generate this ordering for the same requests.
    reordered = torch.tensor([[0, 4, 2, 3, 1, 5, 6]])
    first, _ = decoder._identify_delivery_nodes(valid, partners, 1, 7, torch.device("cpu"))
    try:
        second, _ = decoder._identify_delivery_nodes(reordered, partners, 1, 7, torch.device("cpu"))
    except ValueError:
        return  # Rejecting an infeasible initial PDP tour is also an acceptable contract.
    assert torch.equal(first[:, 1:], second[:, 1:]), "pickup/delivery roles changed with tour order"
