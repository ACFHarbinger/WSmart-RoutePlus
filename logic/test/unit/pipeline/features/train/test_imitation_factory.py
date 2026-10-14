"""Imitation factories forward the full training setup, and training can start from saved weights."""

from types import SimpleNamespace

import pytest
import torch
from logic.src.configs.rl.policies import ProfitableNNConfig
from logic.src.pipeline.features.train.model_factory import imitation as factory
from logic.src.pipeline.features.train.model_factory.builder import _load_initial_weights

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _cfg(algorithm: str) -> SimpleNamespace:
    rl_algo = SimpleNamespace(
        policy_config=ProfitableNNConfig(),
        loss_fn="nll",
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


@pytest.mark.parametrize(
    ("algorithm", "create", "cls_name"),
    [("imitation", factory._create_imitation, "ImitationLearning"), ("adaptive_imitation", factory._create_adaptive_imitation, "AdaptiveImitation")],
)
def test_factory_forwards_the_training_setup(monkeypatch, algorithm, create, cls_name) -> None:
    seen = {}
    monkeypatch.setattr(factory, cls_name, lambda **kw: seen.update(kw) or kw)
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


def test_initial_weights_load_from_a_final_weights_file(tmp_path) -> None:
    trained = torch.nn.Linear(3, 2)
    path = tmp_path / "epoch-31.pt"
    torch.save({"state_dict": {f"policy.{k}": v for k, v in trained.state_dict().items()}}, path)

    fresh = torch.nn.Linear(3, 2)
    _load_initial_weights(fresh, str(path))

    for key, value in trained.state_dict().items():
        assert torch.equal(fresh.state_dict()[key], value)
