"""RL-GD-HH's absolute boundary and iteration budget survive config loading."""

from types import SimpleNamespace

import pytest
from logic.src.configs.policies.other.acceptance_criteria import AcceptanceConfig
from logic.src.configs.policies.rl_gd_hh import RLGDHHConfig
from logic.src.policies.acceptance_criteria.base.factory import AcceptanceCriterionFactory
from omegaconf import OmegaConf


@pytest.mark.parametrize("initial", [100.0, 0.0, -20.0])
@pytest.mark.parametrize(
    "loader",
    [
        lambda x: RLGDHHConfig(**x),
        RLGDHHConfig.from_config,
        lambda x: RLGDHHConfig.from_config(SimpleNamespace(**x)),
        lambda x: RLGDHHConfig.from_config(OmegaConf.create(x)),
    ],
)
def test_absolute_boundary_reaches_configured_target(initial, loader):
    config = loader({"max_iterations": 4, "quality_lb": 10.0, "seed": 17})
    criterion = AcceptanceCriterionFactory.build(config.acceptance_criterion)
    criterion.setup(initial)
    assert config.seed == 17
    assert criterion.water_level == initial
    for _ in range(4):
        criterion.step(initial, initial, True)
    assert criterion.water_level == pytest.approx(10.0)
    assert not hasattr(config.acceptance_criterion, "accept")


def test_legacy_spec_keeps_its_absolute_target_and_solver_budget():
    config = RLGDHHConfig(
        max_iterations=7, acceptance_criterion=AcceptanceConfig(method="great_deluge", params={"quality_lb": -5.0})
    )
    criterion = AcceptanceCriterionFactory.build(config.acceptance_criterion)
    criterion.setup(9.0)
    assert criterion.max_iterations == 7
    for _ in range(7):
        criterion.step(9.0, 9.0, True)
    assert criterion.water_level == pytest.approx(-5.0)


def test_explicit_relative_criterion_is_preserved():
    config = RLGDHHConfig(
        acceptance_criterion=AcceptanceConfig(
            method="gd", params={"target_fitness_multiplier": 1.5, "max_iterations": 2}
        )
    )
    criterion = AcceptanceCriterionFactory.build(config.acceptance_criterion)
    criterion.setup(10.0)
    criterion.step(10.0, 10.0, True)
    assert criterion.water_level == 12.5
