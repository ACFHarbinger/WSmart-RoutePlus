"""End-to-end review probes beyond shape and selected happy-path examples."""

from unittest.mock import patch

import torch
from logic.src.models.core.n2s.decoder import N2SDecoder
from logic.src.models.core.n2s.policy import execute_n2s_request_move
from logic.src.models.core.neuopt.decoder import NeuOptDecoder
from logic.src.models.core.neuopt.policy import execute_neuopt_basis_sequence
from tensordict import TensorDict


def test_neuopt_wraparound_sequence_remains_a_permutation(monkeypatch):
    decoder = NeuOptDecoder(embed_dim=16, k_basis=3)
    targets = iter([0, 8, 4])

    def scores(q_mu, q_lambda, h):
        score = torch.zeros(1, 9)
        score[0, next(targets)] = 10
        return score, torch.zeros_like(score)

    monkeypatch.setattr(decoder, "_compute_stream_scores", scores)
    solution = torch.arange(9).unsqueeze(0)
    td = TensorDict({"solution": solution}, batch_size=[1])
    log_p, action = decoder(td, torch.zeros(1, 9, 16), None)
    assert action.tolist() == [[0, 8, 4]]
    assert torch.isfinite(log_p).all()
    result = execute_neuopt_basis_sequence(solution, action)
    assert torch.equal(result.sort(dim=1).values, solution)


def test_n2s_delivery_cannot_be_reinterpreted_as_pickup():
    decoder = N2SDecoder(embed_dim=16, num_heads=4)
    solution = torch.arange(7).unsqueeze(0)
    # Valid PDP: pickups 1,2,3, deliveries 4,5,6. All pickups initially precede deliveries.
    td = TensorDict({"solution": solution, "partner_ids": torch.tensor([[0,4,5,6,1,2,3]])}, batch_size=[1])
    # The unmasked removal distribution permits delivery node 4 as i+.
    with patch("torch.multinomial", side_effect=[torch.tensor([[4]]), torch.tensor([[2]])]):
        _, action = decoder(td, torch.randn(1, 7, 16), None, strategy="sampling")
    result = execute_n2s_request_move(solution, action)[0].tolist()
    assert result.index(1) < result.index(4), "original pickup now follows its delivery"
