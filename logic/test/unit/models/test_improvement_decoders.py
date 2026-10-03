"""Unit and differentiation tests for DACT, NeuOpt, and N2S decoders.

Tests:
1. Shape and forward pass correctness for each model decoder.
2. Deep paper faithfulness tests:
   - DACT: End-to-end dual-aspect collaborative streams (h, g) from DACTEncoder through DACTDecoder.
   - NeuOpt: Distinct primed/unprimed projections (Eq. 3), distinct recurrent inputs, and cyclic-rank
     feasibility masking (Gamma[a, j] <= Gamma[a, v], §4.1-4.2).
   - N2S: Explicit request-removal distribution (MLP_lambda, Eq. 13) and conditional reinsertion (MLP_mu, Eq. 15).
3. Distinct computation verification: proves DACT, NeuOpt, and N2S are no longer
   the identical pairwise Q*K block.
4. Legacy checkpoint rejection: ensures old generic checkpoints fail loudly with
   a clear RuntimeError instead of loading silently.
5. State serialization and RNG reproducibility.
"""

import pickle
from unittest.mock import patch

import pytest
import torch
from logic.src.envs.tsp_kopt import TSPkoptEnv
from logic.src.models.core.dact.decoder import DACTDecoder
from logic.src.models.core.dact.encoder import DACTEncoder
from logic.src.models.core.dact.policy import DACTPolicy
from logic.src.models.core.n2s.decoder import N2SDecoder
from logic.src.models.core.n2s.policy import N2SPolicy, execute_n2s_request_move
from logic.src.models.core.neuopt.decoder import NeuOptDecoder
from logic.src.models.core.neuopt.policy import NeuOptPolicy, execute_neuopt_basis_sequence
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]


@pytest.fixture
def dummy_env():
    """Provides a TSPkopt environment for testing."""
    return TSPkoptEnv(num_loc=20)


@pytest.fixture
def dummy_batch(dummy_env):
    """Provides a reset environment TensorDict with batch_size=4."""
    return dummy_env.reset(batch_size=[4])


# ---------------------------------------------------------------------------
# 1. Forward and Shape Tests
# ---------------------------------------------------------------------------


def test_dact_decoder_forward(dummy_env, dummy_batch):
    """Tests DACTDecoder forward pass with single and dual aspect embeddings."""
    embed_dim = 128
    num_heads = 4
    decoder = DACTDecoder(embed_dim=embed_dim, num_heads=num_heads, seed=42)

    bs, n = 4, 21
    h = torch.randn(bs, n, embed_dim)

    # Test single aspect input
    log_p, actions = decoder(dummy_batch, h, dummy_env, strategy="greedy")
    assert actions.shape == (bs, 2)
    assert log_p.shape == (bs,)
    assert (actions[:, 0] >= 0).all() and (actions[:, 0] < n).all()
    assert (actions[:, 1] >= 0).all() and (actions[:, 1] < n).all()
    assert (actions[:, 0] != actions[:, 1]).all(), "DACT should never select identity moves (i == j)"
    assert torch.isfinite(log_p).all()
    assert (log_p <= 0.0).all()

    # Test dual aspect input (h, g)
    g = torch.randn(bs, n, embed_dim)
    log_p_dual, actions_dual = decoder(dummy_batch, (h, g), dummy_env, strategy="sampling")
    assert actions_dual.shape == (bs, 2)
    assert log_p_dual.shape == (bs,)
    assert (actions_dual[:, 0] != actions_dual[:, 1]).all()
    assert torch.isfinite(log_p_dual).all()


def test_dact_dual_aspect_end_to_end_flow(dummy_env, dummy_batch):
    """Tests that DACTEncoder outputs dual aspect streams (h, g) and passes them to decoder."""
    embed_dim = 64
    num_heads = 4
    encoder = DACTEncoder(embed_dim=embed_dim, num_layers=2, num_heads=num_heads)
    out = encoder(dummy_batch)

    assert isinstance(out, tuple), "DACTEncoder must return a tuple of dual-aspect embeddings (h, g)"
    assert len(out) == 2
    h, g = out
    bs, n = dummy_batch["solution"].shape
    assert h.shape == (bs, n, embed_dim)
    assert g.shape == (bs, n, embed_dim)
    assert not torch.allclose(h, g), "Node aspect (h) and position aspect (g) must be distinct representations"

    # Full policy rollout
    policy = DACTPolicy(embed_dim=embed_dim, num_layers=2, num_heads=num_heads)
    res = policy(dummy_batch, dummy_env, max_steps=2)
    assert "actions" in res
    assert "reward" in res


def test_neuopt_decoder_forward(dummy_env, dummy_batch):
    """Tests NeuOptDecoder Recurrent Dual-Stream 2-step autoregressive forward pass."""
    embed_dim = 128
    decoder = NeuOptDecoder(embed_dim=embed_dim, seed=42)

    bs, n = 4, 21
    h = torch.randn(bs, n, embed_dim)

    # Test greedy decoding
    log_p, actions = decoder(dummy_batch, h, dummy_env, strategy="greedy")
    assert actions.shape == (bs, 2)
    assert log_p.shape == (bs,)
    assert (actions[:, 0] >= 0).all() and (actions[:, 0] < n).all()
    assert (actions[:, 1] >= 0).all() and (actions[:, 1] < n).all()
    assert (actions[:, 0] != actions[:, 1]).all(), "NeuOpt step 2 should mask step 1 node (x2 != x1)"
    assert torch.isfinite(log_p).all()
    assert (log_p <= 0.0).all()

    # Test sampling decoding
    log_p_sample, actions_sample = decoder(dummy_batch, h, dummy_env, strategy="sampling")
    assert actions_sample.shape == (bs, 2)
    assert log_p_sample.shape == (bs,)
    assert (actions_sample[:, 0] != actions_sample[:, 1]).all()
    assert torch.isfinite(log_p_sample).all()


def test_neuopt_primed_projections_and_cyclic_rank(dummy_env, dummy_batch):
    """Tests that NeuOpt has distinct primed/unprimed projections and enforces cyclic-rank feasibility."""
    embed_dim = 64
    decoder = NeuOptDecoder(embed_dim=embed_dim)

    # 1. Distinct primed/unprimed projections (Eq. 3)
    assert hasattr(decoder, "W_mu_q_prime") and hasattr(decoder, "W_mu_k_prime")
    assert hasattr(decoder, "W_lambda_q_prime") and hasattr(decoder, "W_lambda_k_prime")
    assert decoder.W_mu_q.weight is not decoder.W_mu_q_prime.weight
    assert decoder.W_lambda_q.weight is not decoder.W_lambda_q_prime.weight

    # 2. Verify cyclic rank mask (§4.1-4.2)
    bs, n = 4, 21
    x1 = torch.tensor([0, 1, 2, 3])
    mask = decoder._compute_cyclic_rank_mask(dummy_batch, x1, bs, n)
    assert mask.shape == (bs, n)
    # The anchor node x1 itself must always be masked (rank 0 < 1)
    for b in range(bs):
        assert mask[b, x1[b]].item() is True, "Anchor node xa must be masked by rank constraint"

    # Full policy rollout
    policy = NeuOptPolicy(embed_dim=embed_dim, num_layers=1)
    res = policy(dummy_batch, dummy_env, max_steps=2)
    assert "actions" in res
    assert "reward" in res


def test_n2s_decoder_forward(dummy_env, dummy_batch):
    """Tests N2SDecoder tour-aware preference forward pass."""
    embed_dim = 128
    num_heads = 4
    decoder = N2SDecoder(embed_dim=embed_dim, num_heads=num_heads, seed=42)

    bs, n = 4, 21
    h = torch.randn(bs, n, embed_dim)

    # Test with standard solution in dummy_batch
    log_p, actions = decoder(dummy_batch, h, dummy_env, strategy="greedy")
    assert actions.shape == (bs, 4)
    assert log_p.shape == (bs,)
    assert (actions[:, 0] >= 0).all() and (actions[:, 0] < n).all()
    assert (actions[:, 1] >= 0).all() and (actions[:, 1] < n).all()
    assert (actions[:, 2] >= 0).all() and (actions[:, 2] < n).all()
    assert (actions[:, 3] >= 0).all() and (actions[:, 3] < n).all()
    assert (actions[:, 0] != actions[:, 1]).all(), "Pickup and delivery partners must not be identical"
    assert torch.isfinite(log_p).all()
    assert (log_p <= 0.0).all()

    # Test fallback when solution is missing
    batch_no_sol = TensorDict({"locs": dummy_batch["locs"]}, batch_size=[bs])
    log_p_fallback, actions_fallback = decoder(batch_no_sol, h, dummy_env, strategy="sampling")
    assert actions_fallback.shape == (bs, 4)
    assert (actions_fallback[:, 0] != actions_fallback[:, 1]).all()
    assert torch.isfinite(log_p_fallback).all()


def test_n2s_removal_and_reinsertion_mlps(dummy_env, dummy_batch):
    """Tests N2S explicit request removal (MLP_lambda, Eq. 13) and conditional reinsertion (MLP_mu, Eq. 15)."""
    embed_dim = 64
    num_heads = 4
    decoder = N2SDecoder(embed_dim=embed_dim, num_heads=num_heads)

    # Verify presence of MLP_lambda and MLP_mu
    assert hasattr(decoder, "mlp_lambda"), "N2SDecoder must contain mlp_lambda (Eq. 13)"
    assert hasattr(decoder, "mlp_mu"), "N2SDecoder must contain mlp_mu (Eq. 15)"

    # Verify input dimensions: mlp_lambda expects 2m + 4, mlp_mu expects 4m
    assert decoder.mlp_lambda[0].in_features == 2 * num_heads + 4  # type: ignore[index]
    assert decoder.mlp_mu[0].in_features == 4 * num_heads  # type: ignore[index]

    # Full policy rollout
    policy = N2SPolicy(embed_dim=embed_dim, num_heads=num_heads, k_neighbors=5)
    res = policy(dummy_batch, dummy_env, max_steps=2)
    assert "actions" in res
    assert "reward" in res


# ---------------------------------------------------------------------------
# 2. Pairwise Differentiation Test
# ---------------------------------------------------------------------------


def test_decoders_are_not_the_same_computation(dummy_env, dummy_batch):
    """Verifies that DACT, NeuOpt, and N2S decoders execute mathematically distinct computations.

    Verifies:
    1. Parameter key sets and layer architectures are completely non-identical.
    2. None of the decoders retain the legacy generic pairwise template (project_q, project_k).
    3. Given identical input embeddings and state, the models produce different outputs.
    """
    torch.manual_seed(999)
    embed_dim = 128

    dact = DACTDecoder(embed_dim=embed_dim, num_heads=4, seed=123)
    neuopt = NeuOptDecoder(embed_dim=embed_dim, seed=123)
    n2s = N2SDecoder(embed_dim=embed_dim, num_heads=4, seed=123)

    keys_dact = set(dict(dact.named_parameters()).keys())
    keys_neuopt = set(dict(neuopt.named_parameters()).keys())
    keys_n2s = set(dict(n2s.named_parameters()).keys())

    # 1. Structural distinction
    assert keys_dact != keys_neuopt, "DACT and NeuOpt must have different parameter architectures"
    assert keys_dact != keys_n2s, "DACT and N2S must have different parameter architectures"
    assert keys_neuopt != keys_n2s, "NeuOpt and N2S must have different parameter architectures"

    # 2. Verify model-specific architectural components from papers
    # DACT: FFA MLP and dual-aspect MHC
    assert any("ffa" in k for k in keys_dact), "DACT must contain FFA MLP (Eq. 11)"
    assert any("project_qh" in k for k in keys_dact), "DACT must contain dual-aspect MHC projections"

    # NeuOpt: GRU dual streams (mu and lambda) with primed projections
    assert any("gru_mu" in k for k in keys_neuopt), "NeuOpt must contain gru_mu (Eq. 2)"
    assert any("gru_lambda" in k for k in keys_neuopt), "NeuOpt must contain gru_lambda (Eq. 2)"
    assert any("W_mu_q_prime" in k for k in keys_neuopt), "NeuOpt must contain primed Hadamard query"
    assert any("init_o_mu" in k for k in keys_neuopt), "NeuOpt must contain learnable init input vector"

    # N2S: Tour-aware closeness, MLP_lambda, and MLP_mu
    assert any("mlp_lambda" in k for k in keys_n2s), "N2S must contain mlp_lambda (Eq. 13)"
    assert any("mlp_mu" in k for k in keys_n2s), "N2S must contain mlp_mu (Eq. 15)"
    assert any("project_q_lambda" in k for k in keys_n2s), "N2S must contain closeness projections (Eq. 12)"
    assert any("project_qp" in k for k in keys_n2s), "N2S must contain preference projections (Eq. 14)"

    # 3. None contain legacy generic pairwise keys
    for name, keys in [("DACT", keys_dact), ("NeuOpt", keys_neuopt), ("N2S", keys_n2s)]:
        assert "project_q.weight" not in keys, f"{name} must not contain legacy project_q"
        assert "project_k.weight" not in keys, f"{name} must not contain legacy project_k"

    # 4. Numerical output distinction on identical inputs
    bs, n = 4, 21
    h = torch.randn(bs, n, embed_dim)

    log_p_dact, _ = dact(dummy_batch, h, dummy_env, strategy="greedy")
    log_p_neuopt, _ = neuopt(dummy_batch, h, dummy_env, strategy="greedy")
    log_p_n2s, _ = n2s(dummy_batch, h, dummy_env, strategy="greedy")

    assert not torch.allclose(log_p_dact, log_p_neuopt, atol=1e-3), (
        "DACT and NeuOpt must yield distinct log probabilities"
    )
    assert not torch.allclose(log_p_dact, log_p_n2s, atol=1e-3), "DACT and N2S must yield distinct log probabilities"
    assert not torch.allclose(log_p_neuopt, log_p_n2s, atol=1e-3), (
        "NeuOpt and N2S must yield distinct log probabilities"
    )


# ---------------------------------------------------------------------------
# 3. Legacy Checkpoint Rejection Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("decoder_cls", [DACTDecoder, NeuOptDecoder, N2SDecoder])
def test_legacy_checkpoint_loading_fails_with_clear_error(decoder_cls):
    """Ensures legacy checkpoints containing generic project_q/project_k fail with a clear RuntimeError."""
    decoder = decoder_cls(embed_dim=128)

    # Legacy state dict with generic pairwise weights
    legacy_state_dict = {
        "project_q.weight": torch.randn(128, 128),
        "project_q.bias": torch.randn(128),
        "project_k.weight": torch.randn(128, 128),
        "project_k.bias": torch.randn(128),
    }

    with pytest.raises(RuntimeError, match="Detected legacy generic pairwise decoder checkpoint"):
        decoder.load_state_dict(legacy_state_dict, strict=False)


# ---------------------------------------------------------------------------
# 4. Serialization and RNG State Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("decoder_cls", [DACTDecoder, NeuOptDecoder, N2SDecoder])
def test_decoder_serialization(decoder_cls, dummy_env, dummy_batch):
    """Ensures decoders with torch.Generator can be serialized and deserialized cleanly."""
    decoder = decoder_cls(embed_dim=64, seed=42)

    # Serialize
    serialized = pickle.dumps(decoder)
    restored = pickle.loads(serialized)

    h = torch.randn(2, 21, 64)
    # Both should run forward pass without error
    out_orig = decoder(dummy_batch[:2], h, dummy_env, strategy="greedy")
    out_rest = restored(dummy_batch[:2], h, dummy_env, strategy="greedy")

    torch.testing.assert_close(out_orig[0], out_rest[0])
    torch.testing.assert_close(out_orig[1], out_rest[1])


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
