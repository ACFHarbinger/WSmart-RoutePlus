"""Imitation factories forward the full training setup, and training can start from saved weights."""

from types import SimpleNamespace

import pytest
import torch
from logic.src.configs.policies.vectorized.routing.profitable_nn import ProfitableNNConfig
from logic.src.pipeline.features.train.model_factory.builder import _load_initial_weights

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _cfg(algorithm: str) -> SimpleNamespace:
    rl_algo = SimpleNamespace(
        policy_config=ProfitableNNConfig(),
        loss_fn="nll",
        fill_from="expert",
        expert_cache=None,
        precompute=False,
        il_weight=1.0,
        il_decay=0.9,
        patience=3,
        threshold=0.05,
        decay_step=1,
        epsilon=1e-5,
    )
    return SimpleNamespace(
        task="train",
        train=SimpleNamespace(env=SimpleNamespace(name="cptp")),
        rl=SimpleNamespace(**{algorithm: rl_algo}),
        seed=7,
        device="cpu",
    )


_FACTORY = "logic.src.pipeline.features.train.model_factory"


@pytest.mark.parametrize(
    ("algorithm", "module_name", "create_target", "cls_name"),
    [
        # The imitation factory imports its module on call, so the class is patched at its source.
        (
            "imitation",
            "logic.src.pipeline.learning.core.imitation",
            f"{_FACTORY}.imitation:_create_imitation",
            "ImitationLearning",
        ),
        # Adaptive imitation is not released (Q-12); the release skips this case.
        (
            "adaptive_imitation",
            f"{_FACTORY}.adaptive_imitation",
            f"{_FACTORY}.adaptive_imitation:_create_adaptive_imitation",
            "AdaptiveImitation",
        ),
    ],
)
def test_factory_forwards_the_training_setup(monkeypatch, algorithm, module_name, create_target, cls_name) -> None:
    module = pytest.importorskip(module_name)
    create_module, _, create_name = create_target.partition(":")
    create = getattr(pytest.importorskip(create_module), create_name)
    seen = {}
    monkeypatch.setattr(module, cls_name, lambda **kw: seen.update(kw) or kw)
    cfg = _cfg(algorithm)
    kw = {
        "env": "env",
        "policy": "policy",
        "cfg": cfg,
        "train_time": True,
        "batch_size": 64,
        "policy_config": None,
        "seed": 1,
        "loss_fn": "other",
    }
    create(cfg, "policy", "env", kw)

    # Without cfg and train_time the module trained one batch per epoch, outside time training.
    assert seen["cfg"] is cfg
    assert seen["train_time"] is True
    assert seen["batch_size"] == 64
    assert isinstance(seen["policy_config"], ProfitableNNConfig)
    assert seen["seed"] == 7
    assert seen["loss_fn"] == "nll"
    # Adaptive imitation always decodes the model; only pure imitation has the fill switch.
    assert seen.get("fill_from") == ("expert" if algorithm == "imitation" else None)


def test_initial_weights_load_from_a_final_weights_file(tmp_path) -> None:
    trained = torch.nn.Linear(3, 2)
    path = tmp_path / "epoch-31.pt"
    torch.save({"state_dict": {f"policy.{k}": v for k, v in trained.state_dict().items()}}, path)

    fresh = torch.nn.Linear(3, 2)
    _load_initial_weights(fresh, str(path))

    for key, value in trained.state_dict().items():
        assert torch.equal(fresh.state_dict()[key], value)
