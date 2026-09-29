"""Unit tests verifying unification and parity between AttentionModel and AttentionModelPolicy.

Issue #90 / M-gemini-01:
- Unifies AttentionModel with AttentionModelPolicy
- Verifies bit-level / epsilon-zero output parity over identical weights
- Verifies backward-compatible loading of legacy checkpoints (mapping context_embedder keys
  and filtering deprecated dead projection layers)
- Verifies Lightning policy checkpoint loading
- Verifies strict validation rejecting unexpected unknown keys
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest
import torch
from logic.src.envs.routing.vrpp import VRPPEnv
from logic.src.models.core.attention_model import AttentionModel, AttentionModelPolicy
from logic.src.models.core.moe import MoEAttentionModel
from logic.src.models.core.temporal_attention_model import TemporalAttentionModel
from logic.src.utils.model.loader import load_model

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _build_test_models(seed: int = 42) -> tuple[AttentionModelPolicy, AttentionModel]:
    """Constructs AttentionModelPolicy and AttentionModel with identical weights."""
    torch.manual_seed(seed)
    embed_dim = 64
    hidden_dim = 128
    n_heads = 4
    n_layers = 2

    policy = AttentionModelPolicy(
        env_name="vrpp",
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_encode_layers=n_layers,
        n_heads=n_heads,
    )

    model = AttentionModel(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        problem="vrpp",
        n_encode_layers=n_layers,
        n_heads=n_heads,
    )

    # Compare decoding under the same explicit legacy embedding mode.
    # The canonical default has its own historical projection regression test.
    policy.init_embedding.legacy_depot_projection = True

    # Sync weights exactly
    model.load_state_dict(policy.state_dict(), strict=True)
    policy.eval()
    model.eval()
    return policy, model


def test_m_gemini_01_parameter_keys_and_count():
    """Verify that unified AttentionModel matches AttentionModelPolicy parameter keys."""
    policy, model = _build_test_models()

    policy_keys = set(policy.state_dict().keys())
    model_keys = set(model.state_dict().keys())

    assert policy_keys == model_keys, (
        f"Key mismatch: in policy only: {policy_keys - model_keys}, in model only: {model_keys - policy_keys}"
    )
    # Verify dead context_embedder.project_step_context is eliminated from state_dict
    assert not any("project_step_context" in k and "context_embedder" in k for k in model_keys)


def test_m_gemini_01_weight_level_output_parity_greedy():
    """Verify shared decoding parity with both variants configured for legacy depot projection."""
    policy, model = _build_test_models(seed=123)
    env = VRPPEnv(num_loc=6)
    td = env.reset(batch_size=[3])

    with torch.no_grad():
        out_policy = policy(td.clone(), env, strategy="greedy")
        out_model = model(td.clone(), env, strategy="greedy")

    # Assert exact action match
    assert torch.equal(out_policy["actions"], out_model["actions"]), "Actions diverge between policy and unified model"
    # Assert reward parity
    assert torch.allclose(out_policy["reward"], out_model["reward"], atol=1e-6), "Rewards diverge"
    # Assert log likelihood parity
    assert torch.allclose(out_policy["log_likelihood"], out_model["log_likelihood"], atol=1e-6), (
        "Log likelihoods diverge"
    )
    # Assert cost equals negative reward
    assert torch.allclose(out_model["cost"], -out_model["reward"], atol=1e-6)
    # Assert pi alias matches actions
    assert torch.equal(out_model["pi"], out_model["actions"])


def test_m_gemini_01_load_legacy_checkpoint_with_dead_keys():
    """Verify loading a legacy checkpoint containing context_embedder.* and dead keys."""
    _, model = _build_test_models()
    sd = model.state_dict()

    # Construct legacy checkpoint dict
    legacy_sd: Dict[str, Any] = {}
    for k, v in sd.items():
        if k.startswith("init_embedding.node_embed."):
            legacy_sd["context_embedder.init_embed." + k[len("init_embedding.node_embed.") :]] = v
        elif k.startswith("init_embedding.depot_embed."):
            legacy_sd["context_embedder.init_embed_depot." + k[len("init_embedding.depot_embed.") :]] = v
        else:
            legacy_sd[k] = v

    # Add deprecated dead projection keys that older models used to have
    embed_dim = model.embed_dim
    legacy_sd["context_embedder.project_step_context.weight"] = torch.randn(embed_dim, 3 * embed_dim)
    legacy_sd["context_embedder.project_step_context.bias"] = torch.randn(embed_dim)
    legacy_sd["decoder.project_fixed_context.weight"] = torch.randn(embed_dim, embed_dim)

    # Must load cleanly with strict=True
    res = model.load_state_dict(legacy_sd, strict=True)
    assert len(res.missing_keys) == 0
    assert len(res.unexpected_keys) == 0


def test_m_gemini_01_load_lightning_checkpoint():
    """Verify loading a PyTorch Lightning checkpoint prefixed with policy.*."""
    _, model = _build_test_models()
    sd = model.state_dict()

    pl_sd = {f"policy.{k}": v for k, v in sd.items()}
    res = model.load_state_dict(pl_sd, strict=True)
    assert len(res.missing_keys) == 0
    assert len(res.unexpected_keys) == 0


def test_m_gemini_01_unexpected_keys_rejected():
    """Verify that unexpected unknown keys are rejected by strict loading."""
    _, model = _build_test_models()
    sd = model.state_dict()
    bad_sd = dict(sd)
    bad_sd["decoder.unknown_layer.weight"] = torch.randn(10, 10)

    with pytest.raises((RuntimeError, ValueError)):
        model.load_state_dict(bad_sd, strict=True)


def test_m_gemini_01_loader_roundtrip(tmp_path: Path):
    """Verify saving and loading via central loader.load_model."""
    model_dir = tmp_path / "unified_model"
    model_dir.mkdir()

    embed_dim = 32
    hidden_dim = 64
    hparams = {
        "problem": "vrpp",
        "model": "am",
        "encoder": "gat",
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "n_encode_layers": 1,
        "n_encode_sublayers": 1,
        "n_decode_layers": 1,
        "n_heads": 4,
        "normalization": "batch",
        "tanh_clipping": 10.0,
        "learn_affine": True,
        "track_stats": False,
        "epsilon_alpha": 1e-5,
        "momentum_beta": 0.1,
        "lrnorm_k": 1,
        "gnorm_groups": 1,
        "activation": "relu",
        "af_param": 0.0,
        "af_threshold": 0.0,
        "af_replacement": 0.0,
        "af_nparams": 1,
        "af_urange": 0.0,
        "dropout": 0.0,
        "aggregation": "mean",
        "aggregation_graph": "mean",
    }

    with open(model_dir / "args.json", "w") as f:
        json.dump(hparams, f)

    policy = AttentionModelPolicy(
        env_name="vrpp",
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_encode_layers=1,
        n_heads=4,
    )
    checkpoint_file = model_dir / "epoch-0.pt"
    # Save checkpoint as Lightning state_dict
    torch.save({"state_dict": {f"policy.{k}": v for k, v in policy.state_dict().items()}}, checkpoint_file)

    loaded_model, loaded_args = load_model(str(model_dir))
    assert loaded_model is not None
    assert loaded_args["embed_dim"] == embed_dim
    assert isinstance(loaded_model, AttentionModel)

    # Verify parity between loaded model and original policy
    env = VRPPEnv(num_loc=5)
    td = env.reset(batch_size=[2])
    with torch.no_grad():
        out_p = policy(td.clone(), env, strategy="greedy")
        out_m = loaded_model(td.clone(), env, strategy="greedy")

    assert torch.equal(out_p["actions"], out_m["actions"])
    assert torch.allclose(out_p["reward"], out_m["reward"], atol=1e-6)


def test_m_gemini_01_subclasses():
    """Verify that TemporalAttentionModel and MoEAttentionModel function with the unified base."""
    # Test TemporalAttentionModel
    tam = TemporalAttentionModel(
        embed_dim=32,
        hidden_dim=64,
        problem="vrpp",
        component_factory=None,
        n_encode_layers=1,
        n_heads=4,
        temporal_horizon=3,
    )
    assert isinstance(tam, AttentionModel)
    assert hasattr(tam, "fill_predictor")

    # Test MoEAttentionModel
    moe = MoEAttentionModel(
        embed_dim=32,
        hidden_dim=64,
        problem="vrpp",
        n_encode_layers=1,
        n_heads=4,
        num_experts=2,
        k=1,
    )
    assert isinstance(moe, AttentionModel)
    from logic.src.models.subnets.encoders.moe.encoder import MoEGraphAttentionEncoder

    assert isinstance(moe.encoder, MoEGraphAttentionEncoder)
    assert hasattr(moe.encoder, "layers")


def test_used_model_can_be_copied_for_rollout_baseline():
    """A cached context proxy must follow the copied embedding, not recurse."""
    import copy

    _, model = _build_test_models()
    proxy = model.context_embedder
    cloned = copy.deepcopy(model)

    assert cloned.context_embedder is not proxy
    assert cloned.context_embedder._target is cloned.init_embedding
    assert cloned.context_embedder._target is not model.init_embedding
    before = model.init_embedding.node_embed.weight.detach().clone()
    with torch.no_grad():
        cloned.init_embedding.node_embed.weight.add_(1.0)
    assert torch.equal(model.init_embedding.node_embed.weight, before)


def test_tam_forward_calls_temporal_embedding_override():
    """Verify that TemporalAttentionModel._get_initial_embeddings is called during RL4CO forward."""
    from unittest.mock import patch

    model = TemporalAttentionModel(
        embed_dim=8,
        hidden_dim=16,
        problem="vrpp",
        component_factory=None,
        n_heads=2,
        n_encode_layers=1,
        dropout_rate=0,
    )
    env = VRPPEnv(num_loc=2)
    state = env.reset(batch_size=[1])
    model.eval()
    with patch.object(model, "_get_initial_embeddings", wraps=model._get_initial_embeddings) as hook:
        with torch.no_grad():
            model(state, env, strategy="greedy")
        assert hook.call_count >= 1, f"Expected hook to be called at least once, got {hook.call_count}"


def test_depot_embedding_parity_concatenated_and_separate():
    """Verify bitwise depot and node embedding parity between VRPPContextEmbedder and VRPPInitEmbedding."""
    from logic.src.models.subnets.embeddings.context.vrpp import VRPPContextEmbedder
    from logic.src.models.subnets.embeddings.vrpp import VRPPInitEmbedding

    torch.manual_seed(123)
    old = VRPPContextEmbedder(8, temporal_horizon=0)
    new = VRPPInitEmbedding(8, legacy_depot_projection=True)
    new.node_embed.load_state_dict(old.init_embed.state_dict())
    new.depot_embed.load_state_dict(old.init_embed_depot.state_dict())

    # 1. Concatenated input (e.g. from environment)
    td_cat = {
        "locs": torch.tensor([[[0.0, 0.0], [1.0, 1.0]]]),
        "waste": torch.tensor([[0.0, 1.0]]),
        "depot": torch.zeros(1, 2),
    }
    with torch.no_grad():
        before_cat = old.init_node_embeddings(td_cat)
        after_cat = new(td_cat)
    assert torch.equal(before_cat, after_cat), "Concatenated embedding divergence between old and new"

    # 2. Separate depot input
    td_sep = {
        "locs": torch.tensor([[[1.0, 1.0]]]),
        "waste": torch.tensor([[1.0]]),
        "depot": torch.zeros(1, 2),
    }
    with torch.no_grad():
        before_sep = old.init_node_embeddings(td_sep)
        after_sep = new(td_sep)
    assert torch.equal(before_sep, after_sep), "Separate depot embedding divergence between old and new"


def test_nested_parent_module_checkpoint_loading():
    """Verify that AttentionModel nested inside an outer Module loads legacy checkpoints with strict=True."""
    child = AttentionModel(embed_dim=8, hidden_dim=16, problem="vrpp", n_heads=2, n_encode_layers=1)
    wrapper = torch.nn.Module()
    wrapper.policy = child

    legacy = {
        k.replace("policy.init_embedding.node_embed.", "policy.context_embedder.init_embed.").replace(
            "policy.init_embedding.depot_embed.", "policy.context_embedder.init_embed_depot."
        ): v
        for k, v in wrapper.state_dict().items()
    }
    # Add legacy dead keys
    legacy["policy.context_embedder.project_step_context.weight"] = torch.randn(8, 24)
    legacy["policy.context_embedder.project_step_context.bias"] = torch.randn(8)
    legacy["policy.decoder.project_fixed_context.weight"] = torch.randn(8, 8)

    # Must load without error under strict=True
    wrapper.load_state_dict(legacy, strict=True)


def test_nonzero_temporal_horizon_projection_shape():
    """Verify that temporal_horizon > 0 expands node projection input dimension matching historical embedder."""
    from logic.src.models.subnets.embeddings.context.vrpp import VRPPContextEmbedder

    horizon_model = AttentionModel(
        embed_dim=8, hidden_dim=16, problem="vrpp", n_heads=2, n_encode_layers=1, temporal_horizon=3
    )
    old_shape = tuple(VRPPContextEmbedder(8, temporal_horizon=3).init_embed.weight.shape)
    new_shape = tuple(horizon_model.init_embedding.node_embed.weight.shape)
    assert old_shape == new_shape == (8, 6), f"Shape mismatch: old={old_shape}, new={new_shape}"


@pytest.mark.parametrize("env_name", ["vrpp", "cvrpp", "wcvrp"])
def test_canonical_policy_preserves_historical_depot_projection(env_name):
    """Checkpoint weights retain the canonical depot projection for every shared embedder."""
    policy = AttentionModelPolicy(env_name=env_name, embed_dim=8, hidden_dim=16, n_heads=2, n_encode_layers=1)
    embedding = policy.init_embedding
    td = {
        "locs": torch.tensor([[[0.0, 0.0], [1.0, 1.0]]]),
        "waste": torch.tensor([[0.0, 1.0]]),
        "depot": torch.zeros(1, 2),
    }
    with torch.no_grad():
        embedding.node_embed.bias.fill_(2)
        embedding.depot_embed.bias.fill_(7)
        expected = embedding.node_embed(torch.cat([td["locs"], td["waste"].unsqueeze(-1)], -1))
        expected[:, 0, :] = embedding.depot_embed(td["depot"])
        actual = embedding(td)
    assert not embedding.legacy_depot_projection
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("env_name", ["vrpp", "cvrpp", "wcvrp"])
@pytest.mark.parametrize("horizon", [0, 3])
def test_legacy_model_retains_context_embedding_projection(env_name, horizon):
    """Legacy context modules are the oracle, including actual nonzero temporal features."""
    from logic.src.models.subnets.embeddings.context.vrpp import VRPPContextEmbedder
    from logic.src.models.subnets.embeddings.context.wcvrp import WCVRPContextEmbedder

    old_class = WCVRPContextEmbedder if env_name == "wcvrp" else VRPPContextEmbedder
    old = old_class(8, temporal_horizon=horizon)
    model = AttentionModel(
        embed_dim=8, hidden_dim=16, problem=env_name, n_heads=2, n_encode_layers=1, temporal_horizon=horizon
    )
    model.init_embedding.node_embed.load_state_dict(old.init_embed.state_dict())
    model.init_embedding.depot_embed.load_state_dict(old.init_embed_depot.state_dict())
    for separate in (False, True):
        td = {
            "locs": torch.tensor([[[0.0, 0.0], [1.0, 1.0]]]),
            "waste": torch.tensor([[0.0, 1.0]]),
            "depot": torch.zeros(1, 2),
            "temporal_features": torch.arange(2 * horizon, dtype=torch.float).reshape(1, 2, horizon),
        }
        if separate:
            for key in ("locs", "waste", "temporal_features"):
                td[key] = td[key][:, 1:]
        assert model.init_embedding.legacy_depot_projection
        with torch.no_grad():
            assert torch.equal(model.init_embedding(td), old.init_node_embeddings(td))


def test_legacy_wcvrp_disabled_temporal_features_preserves_projection_width():
    """Disabling temporal inputs still pads the legacy WCVRP projection's feature width."""
    from logic.src.models.subnets.embeddings.context.wcvrp import WCVRPContextEmbedder

    old = WCVRPContextEmbedder(8, temporal_horizon=3)
    model = AttentionModel(
        embed_dim=8, hidden_dim=16, problem="wcvrp", n_heads=2, n_encode_layers=1, temporal_horizon=3
    )
    model.init_embedding.node_embed.load_state_dict(old.init_embed.state_dict())
    model.init_embedding.depot_embed.load_state_dict(old.init_embed_depot.state_dict())
    td = {
        "locs": torch.tensor([[[0.0, 0.0], [1.0, 1.0]]]),
        "waste": torch.tensor([[0.0, 1.0]]),
        "depot": torch.zeros(1, 2),
        "temporal_features": torch.ones(1, 2, 3),
    }
    with torch.no_grad():
        expected = old.init_node_embeddings(td, temporal_features=False)
        actual = model.context_embedder.init_node_embeddings(td, temporal_features=False)
    assert torch.equal(actual, expected)


def test_legacy_factory_preserves_activation_defaults_and_explicit_overrides():
    """The actual factory receives the legacy dataclass defaults, or the caller's overrides."""
    from unittest.mock import patch

    from logic.src.configs.models.activation_function import ActivationConfig
    from logic.src.models.subnets.factories.attention import AttentionComponentFactory

    cases = [
        ({}, ActivationConfig()),
        (
            {
                "activation": "relu",
                "af_param": 2.0,
                "af_threshold": 3.0,
                "af_replacement": 4.0,
                "af_nparams": 2,
                "af_urange": [0.2, 0.4],
            },
            ActivationConfig(
                name="relu", param=2.0, threshold=3.0, replacement_value=4.0, n_params=2, range=[0.2, 0.4]
            ),
        ),
        ({"activation_config": ActivationConfig(name="relu")}, ActivationConfig(name="relu")),
    ]
    for kwargs, expected in cases:
        factory = AttentionComponentFactory()
        with patch.object(factory, "create_encoder", wraps=factory.create_encoder) as encoder:
            model = AttentionModel(
                embed_dim=8,
                hidden_dim=16,
                problem="vrpp",
                n_heads=2,
                n_encode_layers=1,
                component_factory=factory,
                **kwargs,
            )
        assert encoder.call_args.kwargs["activation_config"] == expected
        if not kwargs:
            assert any(isinstance(module, torch.nn.GELU) for module in model.encoder.modules())
    canonical = AttentionModel(embed_dim=8, hidden_dim=16, problem="vrpp", n_heads=2, n_encode_layers=1)
    assert any(isinstance(module, torch.nn.ReLU) for module in canonical.encoder.modules())
