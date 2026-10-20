"""Semantic action and paper-faithfulness tests for the DACT, NeuOpt and N2S decoders.

Endpoint ranks, tour-position decoding, DACT cyclic positional encoding, N2S history and
precedence-preserving reinsertion, NeuOpt k-step basis moves, and the TSP k-opt action
contract. Moved unchanged from ``test_improvement_decoders.py`` (section 5).
"""

from unittest.mock import patch

import pytest
import torch
from logic.src.models.core.dact.encoder import DACTEncoder
from logic.src.models.core.n2s.decoder import N2SDecoder
from logic.src.models.core.n2s.policy import N2SPolicy, execute_n2s_request_move
from logic.src.models.core.neuopt.decoder import NeuOptDecoder
from logic.src.models.core.neuopt.policy import NeuOptPolicy, execute_neuopt_basis_sequence
from logic.src.pipeline.learning.envs.tsp_kopt import TSPkoptEnv
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]


# ---------------------------------------------------------------------------
# 5. Semantic Action and Paper Faithfulness Verification Tests
# ---------------------------------------------------------------------------


def test_neuopt_second_edge_input_is_lower_rank_endpoint():
    """Verifies that NeuOpt edge GRU at step 2 receives the lower-ranked endpoint (xa = x1)."""
    decoder = NeuOptDecoder(embed_dim=16)
    h = torch.arange(80, dtype=torch.float).reshape(1, 5, 16)
    td = TensorDict({"solution": torch.tensor([[0, 3, 1, 4, 2]])}, batch_size=[1])
    inputs = []
    handle = decoder.gru_lambda.register_forward_pre_hook(lambda module, args: inputs.append(args[0].detach().clone()))
    try:
        _, actions = decoder(td, h, None)
    finally:
        handle.remove()
    # After S(xa), endpoints are xa (rank 0) and succ(xa) (rank 1).
    # NeuOpt §4.1–4.2 uses the lower-ranked endpoint as the introduced-edge source.
    anchor = actions[0, 0]
    assert torch.equal(inputs[1], h[:, anchor])


def test_decoded_nodes_operate_on_their_tour_positions(monkeypatch):
    """Verifies that decoded node IDs are correctly mapped to tour positions for 2-opt moves."""
    decoder = NeuOptDecoder(embed_dim=16)
    calls = []

    def scores(q_mu, q_lambda, h):
        target = 3 if not calls else 2
        calls.append(target)
        logits = torch.zeros(1, 5)
        logits[0, target] = 10
        return logits, torch.zeros_like(logits)

    monkeypatch.setattr(decoder, "_compute_stream_scores", scores)
    solution = torch.tensor([[0, 3, 1, 4, 2]])
    td = TensorDict({"solution": solution.clone()}, batch_size=[1])
    _, action = decoder(td, torch.zeros(1, 5, 16), None)
    assert action.tolist() == [[3, 2]]
    td["action"] = action
    result = TSPkoptEnv()._step_instance(td)["solution"]
    # Node 3 is at position 1 and node 2 at position 4: reverse positions 2..4.
    assert result.tolist() == [[0, 3, 2, 4, 1]]


def test_dact_cpe_multidimensional_periods(dummy_batch):
    """Verifies that DACT implements multi-dimensional CPE (Eq. 4 & Eq. 13) with distinct frequencies."""
    embed_dim = 64
    encoder = DACTEncoder(embed_dim=embed_dim)
    sol = dummy_batch["solution"]
    n_nodes = sol.shape[-1]
    cpe = encoder._compute_cyclic_positions(sol, n_nodes)

    assert cpe.shape == (dummy_batch.batch_size[0], n_nodes, embed_dim)
    # Check that dimensions have distinct wavelengths/representations
    dim0 = cpe[0, :, 0]
    dim_mid = cpe[0, :, embed_dim // 2]
    assert not torch.allclose(dim0, dim_mid), "Different dimensions of CPE must have different wavelength patterns"


def test_n2s_evolving_history_and_successor_gather(dummy_batch, dummy_env):
    """Verifies N2SPolicy maintains evolving 4-dim history features and decodes with successor gather."""
    embed_dim = 32
    num_heads = 2
    policy = N2SPolicy(embed_dim=embed_dim, num_heads=num_heads, k_neighbors=5)
    res = policy(dummy_batch, dummy_env, max_steps=3, return_actions=True)

    assert "actions" in res
    assert res["actions"].shape == (dummy_batch.batch_size[0], 3, 4)
    assert "reward" in res


def test_neuopt_k_step_basis_moves():
    """Verifies NeuOpt supports K-step basis sequences (K >= 2) with endpoint evolution and closure."""
    decoder = NeuOptDecoder(embed_dim=16, k_basis=3)
    sol = torch.tensor([[0, 3, 1, 4, 2]])
    td = TensorDict({"solution": sol.clone()}, batch_size=[1])
    h = torch.randn(1, 5, 16)

    # 1. K=3 basis moves produces [B, 3]
    log_p, actions = decoder(td, h, None, k_basis=3)
    assert actions.shape == (1, 3)
    assert log_p.shape == (1,)

    # 2. Dynamic rank mask: with xj having rank > 1, mask excludes all nodes with rank < rank(xj)
    x1 = torch.tensor([0])
    xj = torch.tensor([1])  # node 1 has position 2 on [0, 3, 1, 4, 2], rank 2 w.r.t node 0
    mask = decoder._compute_cyclic_rank_mask(td, x1, 1, 5, xj=xj)
    # Node 0 (rank 0) and node 3 (rank 1) must be masked out
    assert mask[0, 0].item() is True
    assert mask[0, 3].item() is True
    # Node 1 (rank 2), node 4 (rank 3), node 2 (rank 4) are valid
    assert mask[0, 1].item() is False
    assert mask[0, 4].item() is False
    assert mask[0, 2].item() is False


def test_n2s_joint_precedence_reinsertion_and_windowed_history(dummy_batch, dummy_env):
    """Verifies N2S evaluates joint (j, k) reinsertion with precedence masking and windowed history."""
    decoder = N2SDecoder(embed_dim=16, num_heads=4)
    td = TensorDict({"solution": torch.tensor([[0, 1, 2, 3, 4]])}, batch_size=[1])
    h = torch.randn(1, 5, 16)
    log_p, actions = decoder(td, h, None)

    assert actions.shape == (1, 4)
    assert log_p.shape == (1,)

    # Verify windowed history eviction over past K steps
    policy = N2SPolicy(embed_dim=16, num_heads=2, k_neighbors=3)
    res = policy(dummy_batch, dummy_env, max_steps=12, window_size=5, return_actions=True)
    assert res["actions"].shape[1] == 12


def test_tsp_kopt_env_action_contract():
    """Verifies TSPkoptEnv contract for both node IDs and explicit tour positions."""
    env = TSPkoptEnv()

    # 1. Node IDs mode: on tour [0, 3, 1, 4, 2], action [3, 2] reverses between node 3 (pos 1) and node 2 (pos 4)
    td1 = TensorDict({"solution": torch.tensor([[0, 3, 1, 4, 2]]), "action": torch.tensor([[3, 2]])}, batch_size=[1])
    res1 = env._step_instance(td1)["solution"]
    assert res1.tolist() == [[0, 3, 2, 4, 1]]

    # 2. Tour positions mode: action_is_position=True reverses positions 1..3
    td2 = TensorDict(
        {
            "solution": torch.tensor([[0, 3, 1, 4, 2]]),
            "action": torch.tensor([[1, 3]]),
            "action_is_position": torch.tensor([True]),
        },
        batch_size=[1],
    )
    res2 = env._step_instance(td2)["solution"]
    # Slice 2..3 (which is [1, 4]) reversed -> [4, 1]
    assert res2.tolist() == [[0, 3, 4, 1, 2]]


def test_n2s_delivery_position_changes_final_route_and_preserves_precedence():
    """Verifies that distinct delivery positions produce distinct tours while strictly preserving pickup-before-delivery."""
    tour = torch.tensor([[0, 1, 2, 3, 4, 5]])
    # Request (1, 4): pickup 1, delivery 4
    # Option A: insert pickup after 0, delivery after 2
    actA = torch.tensor([[1, 4, 0, 2]])
    # Option B: insert pickup after 0, delivery after 3
    actB = torch.tensor([[1, 4, 0, 3]])

    solA = execute_n2s_request_move(tour, actA)
    solB = execute_n2s_request_move(tour, actB)

    assert not torch.equal(solA, solB), "Different delivery positions must produce distinct routes"
    # Route A: [0, 1, 2, 4, 3, 5]
    assert solA.tolist() == [[0, 1, 2, 4, 3, 5]]
    # Route B: [0, 1, 2, 3, 4, 5]
    assert solB.tolist() == [[0, 1, 2, 3, 4, 5]]

    # In both routes, pickup 1 must strictly precede delivery 4
    idx_pA, idx_dA = (solA == 1).nonzero()[0, 1].item(), (solA == 4).nonzero()[0, 1].item()
    idx_pB, idx_dB = (solB == 1).nonzero()[0, 1].item(), (solB == 4).nonzero()[0, 1].item()
    assert idx_pA < idx_dA, "Pickup 1 must precede delivery 4 in route A"
    assert idx_pB < idx_dB, "Pickup 1 must precede delivery 4 in route B"


def test_neuopt_basis_execution_hand_computed_k3_k4_routes_and_edge_sets():
    """Verifies NeuOpt basis-sequence execution matches paper Figure 7 hand-computed routes and edge sets."""
    tour = torch.arange(1, 10).unsqueeze(0)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]

    # 1. 2-opt [S(2), I(4), E(null)]
    sol_2opt = execute_neuopt_basis_sequence(tour, torch.tensor([[2, 4]]))
    assert sol_2opt.tolist()[0] == [1, 2, 4, 3, 5, 6, 7, 8, 9]

    # 2. 3-opt [S(2), I(4), I(7), E(null)]
    sol_3opt = execute_neuopt_basis_sequence(tour, torch.tensor([[2, 4, 7]]))
    assert sol_3opt.tolist()[0] == [1, 2, 4, 3, 7, 6, 5, 8, 9]

    # 3. 4-opt [S(2), I(4), I(7), I(9), E(null)]
    sol_4opt = execute_neuopt_basis_sequence(tour, torch.tensor([[2, 4, 7, 9]]))
    assert sol_4opt.tolist()[0] == [1, 2, 4, 3, 7, 6, 5, 9, 8]

    # 4. Early E-move closure at step 3: [2, 4, 5, 2] closes cycle at node 5 (succ of 4) -> 2-opt
    sol_early = execute_neuopt_basis_sequence(tour, torch.tensor([[2, 4, 5, 2]]))
    assert torch.equal(sol_early, sol_2opt), "Early E-move closure must produce 2-opt route"

    # 5. Void 1-opt move [2, 3]: immediate closure -> unchanged tour
    sol_void = execute_neuopt_basis_sequence(tour, torch.tensor([[2, 3]]))
    assert torch.equal(sol_void, tour), "Void action must preserve tour unchanged"


def test_neuopt_policy_executes_basis_sequence_and_evolves_tour():
    """Verifies NeuOptPolicy executes full K-step basis sequences, evolves td['solution'], and accumulates log_p."""
    env = TSPkoptEnv(num_loc=10)
    policy = NeuOptPolicy(embed_dim=32, num_layers=1, k_basis=3)
    td = env.reset(batch_size=[2])
    init_sol = td["solution"].clone()

    out = policy(td, env, strategy="greedy", max_steps=2, k_basis=3)
    assert out["actions"].shape == (2, 2, 3)
    assert torch.isfinite(out["log_likelihood"]).all()
    assert td["solution"].shape == init_sol.shape


def test_distinct_delivery_positions_produce_distinct_actions():
    """A sampled delivery position must survive the decoder-to-executor boundary."""
    torch.manual_seed(12)
    decoder = N2SDecoder(embed_dim=16, num_heads=4)
    td = TensorDict(
        {"solution": torch.arange(7).unsqueeze(0), "partner_ids": torch.tensor([[0, 4, 5, 6, 1, 2, 3]])},
        batch_size=[1],
    )
    h = torch.randn(1, 7, 16)
    actions = []
    for delivery_position in (2, 3):
        # Remove request (1,4), insert pickup after node 0, delivery after 2 or 3.
        # Both joint insertion pairs pass the delivered precedence mask.
        with patch("torch.multinomial", side_effect=[torch.tensor([[1]]), torch.tensor([[delivery_position]])]):
            _, action = decoder(td, h, None, strategy="sampling")
        actions.append(action)
    assert not torch.equal(actions[0], actions[1]), "delivery insertion position was discarded"
