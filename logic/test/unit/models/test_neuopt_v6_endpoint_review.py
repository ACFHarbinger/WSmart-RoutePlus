"""The edge GRU must follow the endpoint of the evolving open path."""

import torch
from logic.src.models.core.neuopt.decoder import NeuOptDecoder
from tensordict import TensorDict


def test_edge_stream_endpoint_after_repeated_wraparound(monkeypatch):
    decoder = NeuOptDecoder(embed_dim=16, k_basis=5)
    targets = iter([0, 3, 8, 8, 6])

    def scores(q_mu, q_lambda, h):
        result = torch.zeros(1, 9)
        result[0, next(targets)] = 10
        return result, torch.zeros_like(result)

    monkeypatch.setattr(decoder, "_compute_stream_scores", scores)
    inputs = []
    hook = decoder.gru_lambda.register_forward_pre_hook(lambda module, args: inputs.append(args[0].detach().clone()))
    h = torch.arange(144, dtype=torch.float).reshape(1, 9, 16)
    td = TensorDict({"solution": torch.arange(9).unsqueeze(0)}, batch_size=[1])
    try:
        log_p, action = decoder(td, h, None)
    finally:
        hook.remove()
    assert action.tolist() == [[0, 3, 8, 8, 6]]
    assert torch.isfinite(log_p).all()
    # After S(0), I(3), I(8), I(8), path is [4,5,6,7,8,0,3,2,1].
    # Endpoints are 4 and 1, so the next introduced edge must start at 1.
    torch.testing.assert_close(inputs[4], h[:, 1])
