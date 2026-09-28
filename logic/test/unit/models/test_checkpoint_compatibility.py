import json
from pathlib import Path

import pytest
import torch
from logic.src.models.core.attention_model import AttentionModel
from logic.src.models.subnets.decoders.glimpse.decoder import GlimpseDecoder
from logic.src.utils.model.loader import load_model

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_d_gemini_01_glimpse_decoder_has_no_project_fixed_context() -> None:
    """Verify that GlimpseDecoder does not have project_fixed_context attribute

    (D-gemini-01 dead code elimination).
    """
    decoder = GlimpseDecoder(
        embed_dim=64,
        hidden_dim=128,
        problem="vrpp",
        n_heads=4,
    )
    assert not hasattr(decoder, "project_fixed_context"), "GlimpseDecoder still has dead project_fixed_context layer"

    # Verify precompute works with graph_context=None
    dummy_embeddings = torch.randn(2, 10, 64)
    cache = decoder._precompute(dummy_embeddings)
    assert cache.graph_context is None
    # Slicing cache should also preserve graph_context=None without error
    sliced_cache = cache[0]
    assert sliced_cache.graph_context is None


def test_checkpoint_load_legacy_project_fixed_context(tmp_path: Path) -> None:
    """Verify that checkpoints saved with legacy decoder.project_fixed_context.weight

    load cleanly without error into the updated model architecture.
    """
    model_dir = tmp_path / "legacy_model"
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

    # Instantiate a baseline model to produce a valid state_dict
    from logic.src.models.subnets.factories.attention import AttentionComponentFactory
    from logic.src.utils.model.problem_factory import load_problem

    problem = load_problem("vrpp")
    base_model = AttentionModel(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        problem=problem,
        component_factory=AttentionComponentFactory(),
        n_encode_layers=1,
        n_encode_sublayers=1,
        n_decode_layers=1,
        n_heads=4,
    )
    legacy_state_dict = base_model.state_dict()
    # Inject legacy dead projection weights
    legacy_state_dict["decoder.project_fixed_context.weight"] = torch.randn(embed_dim, embed_dim)

    # Save legacy checkpoint
    checkpoint_file = model_dir / "epoch-0.pt"
    torch.save({"model": legacy_state_dict}, checkpoint_file)

    # Load via load_model - should load cleanly despite legacy keys
    loaded_model, loaded_args = load_model(str(model_dir))
    assert loaded_model is not None
    assert loaded_args["embed_dim"] == embed_dim
    assert not hasattr(loaded_model.decoder, "project_fixed_context")

    # If an unexpected key that is NOT an allowed legacy key is present, it must fail
    invalid_state_dict = dict(legacy_state_dict)
    invalid_state_dict["decoder.unexpected_unknown_layer.weight"] = torch.randn(embed_dim, embed_dim)
    invalid_checkpoint_file = model_dir / "epoch-1.pt"
    torch.save({"model": invalid_state_dict}, invalid_checkpoint_file)

    with pytest.raises(ValueError, match="unexpected="):
        load_model(str(invalid_checkpoint_file))
