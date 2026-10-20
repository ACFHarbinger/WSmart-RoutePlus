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

The shared ``dummy_env`` and ``dummy_batch`` fixtures live in ``conftest.py``; the semantic
action and paper-faithfulness tests (section 5) live in
``test_improvement_decoder_semantics.py``.
"""

import pickle

import pytest
import torch
from logic.src.models.core.dact.decoder import DACTDecoder
from logic.src.models.core.dact.encoder import DACTEncoder
from logic.src.models.core.dact.policy import DACTPolicy
from logic.src.models.core.n2s.decoder import N2SDecoder
from logic.src.models.core.n2s.policy import N2SPolicy
from logic.src.models.core.neuopt.decoder import NeuOptDecoder
from logic.src.models.core.neuopt.policy import NeuOptPolicy
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]


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
