"""Review probe for explicit pickup/delivery partners in removal scoring."""

import torch
from logic.src.models.core.n2s.decoder import N2SDecoder
from tensordict import TensorDict


def test_removal_features_use_request_partner_not_tour_successor():
    torch.manual_seed(123)
    decoder = N2SDecoder(embed_dim=16, num_heads=4)
    partners = torch.tensor([[0, 4, 5, 6, 1, 2, 3]])
    td = TensorDict({"solution": torch.arange(7).unsqueeze(0), "partner_ids": partners}, batch_size=[1])
    captured = []
    handle = decoder.mlp_lambda.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
    try:
        decoder(td, torch.randn(1, 7, 16), None)
    finally:
        handle.remove()
    features = captured[0]
    own_scores = features[:, :, :4]
    paired_scores = features[:, :, 4:8]
    expected = own_scores.gather(1, partners.unsqueeze(-1).expand(-1, -1, 4))
    torch.testing.assert_close(paired_scores, expected)
