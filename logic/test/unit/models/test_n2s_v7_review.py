"""Input and multi-start contracts for the PDP policy."""

import pytest
import torch
from logic.src.envs.tsp_kopt import TSPkoptEnv
from logic.src.models.core.n2s.decoder import N2SDecoder
from logic.src.models.core.n2s.policy import N2SPolicy, validate_pdp_solution
from tensordict import TensorDict


def test_multistart_preserves_caller_batch_and_returns_all_solutions():
    env = TSPkoptEnv(num_loc=6)
    td = env.reset(batch_size=[2])
    out = N2SPolicy(embed_dim=16, num_heads=2, k_neighbors=3)(td, env, max_steps=2, num_starts=3)
    assert out["solution"].shape == (2, 3, 7)
    assert out["partner_ids"].shape == (2, 3, 7)
    assert td["solution"].shape == (2, 7)
    best = out["reward"].argmax(1)
    torch.testing.assert_close(td["solution"], out["solution"][torch.arange(2), best])


def test_validator_rejects_nonpermutation():
    partners = torch.tensor([[0, 4, 5, 6, 1, 2, 3]])
    pickup = torch.tensor([[False, True, True, True, False, False, False]])
    with pytest.raises(ValueError, match="permutation"):
        validate_pdp_solution(torch.tensor([[0, 1, 2, 4, 5, 6, 6]]), partners, pickup, ~pickup)


@pytest.mark.parametrize("partners", [[0, 4, 5, 6, 2, 1, 3], [3, 4, 5, 0, 1, 2], [0., 4.5, 5., 6., 1., 2., 3.]])
def test_invalid_request_map_rejected(partners):
    decoder = N2SDecoder(embed_dim=16)
    n = len(partners)
    td = TensorDict({"partner_ids": torch.tensor([partners])}, batch_size=[1])
    with pytest.raises(ValueError):
        decoder._resolve_partner_ids(td, 1, n, torch.device("cpu"), {})


def test_conflicting_roles_rejected():
    decoder = N2SDecoder(embed_dim=16)
    partners = torch.tensor([[0, 4, 5, 6, 1, 2, 3]])
    td = TensorDict({"is_delivery": torch.zeros(1, 7, dtype=torch.bool)}, batch_size=[1])
    with pytest.raises(ValueError):
        decoder._identify_delivery_nodes(None, partners, 1, 7, torch.device("cpu"), td=td)
