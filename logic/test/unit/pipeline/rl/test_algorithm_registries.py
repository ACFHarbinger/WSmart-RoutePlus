"""Every lazy algorithm entry resolves to its class or factory (the registries hold strings)."""

import functools

import pytest
from logic.src.pipeline.features.train.model_factory.registry import _ALGO_REGISTRY, _PPO_FAMILY, get_algorithm_factory
from logic.src.pipeline.rl import RL_ALGORITHM_REGISTRY, get_rl_algorithm

pytestmark = [pytest.mark.unit]


@pytest.mark.parametrize("name", sorted(RL_ALGORITHM_REGISTRY))
def test_rl_algorithm_resolves_to_a_lightning_module(name: str) -> None:
    import pytorch_lightning as pl

    cls = get_rl_algorithm(name)

    assert isinstance(cls, type) and issubclass(cls, pl.LightningModule)


@pytest.mark.parametrize("name", sorted(_ALGO_REGISTRY))
def test_algorithm_factory_resolves(name: str) -> None:
    factory = get_algorithm_factory(name)

    assert callable(factory)
    # The PPO family shares one factory that takes the algorithm name first; gdpo has its own.
    if name in _PPO_FAMILY:
        assert isinstance(factory, functools.partial) and factory.args == (name,)
    else:
        assert not isinstance(factory, functools.partial)


def test_unknown_names() -> None:
    assert get_algorithm_factory("unknown") is None
    with pytest.raises(ValueError, match="Unknown RL algorithm"):
        get_rl_algorithm("unknown")
