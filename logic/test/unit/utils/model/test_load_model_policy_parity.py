"""A policy saved by training and reloaded with load_model decodes the same routes."""

import json

import pytest
import torch
from logic.src.configs.models import ActivationConfig, NormalizationConfig
from logic.src.models.core.attention_model.policy import AttentionModelPolicy
from logic.src.pipeline.learning.envs.cptp import CPTPEnv
from logic.src.pipeline.learning.envs.ptp import PTPEnv
from logic.src.utils.model.loader import load_model
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]

_ARGS = {
    "problem": None,
    "model": "am",
    "encoder": "gat",
    "embed_dim": 32,
    "hidden_dim": 64,
    "n_encode_layers": 2,
    "n_encode_sublayers": 1,
    "n_decode_layers": 2,
    "n_heads": 4,
    "normalization": "layer",
    "tanh_clipping": 10.0,
    "learn_affine": True,
    "track_stats": False,
    "epsilon_alpha": 1e-5,
    "momentum_beta": 0.1,
    "lrnorm_k": 1.0,
    "gnorm_groups": 1,
    "activation": "gelu",
    "af_param": 1.0,
    "af_threshold": 6.0,
    "af_replacement": 6.0,
    "af_nparams": 3,
    "af_urange": [0.125, 1 / 3],
    "dropout": 0.0,
    "aggregation": "sum",
    "aggregation_graph": "avg",
}


def _instances(batch: int = 16, num_loc: int = 12) -> TensorDict:
    gen = torch.Generator().manual_seed(3)
    return TensorDict(
        {
            "locs": torch.rand(batch, num_loc, 2, generator=gen),
            "depot": torch.rand(batch, 2, generator=gen),
            "waste": torch.rand(batch, num_loc, generator=gen) * 0.5,
            "capacity": torch.full((batch,), 2.0),
        },
        batch_size=[batch],
    )


@pytest.mark.parametrize(("env_name", "env_class"), [("cptp", CPTPEnv), ("ptp", PTPEnv)])
def test_reloaded_policy_decodes_the_trained_routes(tmp_path, env_name, env_class) -> None:
    torch.manual_seed(11)
    policy = AttentionModelPolicy(
        env_name,
        embed_dim=_ARGS["embed_dim"],
        hidden_dim=_ARGS["hidden_dim"],
        n_encode_layers=_ARGS["n_encode_layers"],
        n_heads=_ARGS["n_heads"],
        norm_config=NormalizationConfig(norm_type="layer"),
        activation_config=ActivationConfig(name="gelu"),
        dropout_rate=0.0,
    ).eval()

    # The layout training writes: a "state_dict" with the policy under "policy." and an args.json.
    torch.save({"state_dict": {f"policy.{k}": v for k, v in policy.state_dict().items()}}, tmp_path / "epoch-1.pt")
    (tmp_path / "args.json").write_text(json.dumps({**_ARGS, "problem": env_name}))
    loaded, _ = load_model(str(tmp_path))

    env = env_class(generator_params={"num_loc": 12}, check_env_specs=False)
    with torch.no_grad():
        trained = policy(env.reset(_instances()), env, strategy="greedy")
        reloaded = loaded(env.reset(_instances()), env, strategy="greedy")

    # Guard against a vacuous pass: the untrained policy must actually leave the depot.
    assert (trained["actions"] > 0).any()
    assert trained["actions"].shape == reloaded["actions"].shape
    assert torch.equal(trained["actions"], reloaded["actions"])
    assert torch.allclose(trained["reward"], reloaded["reward"])
