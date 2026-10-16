"""Acceptance prototypes must not retain a solver's search state."""

from logic.src.policies.acceptance_criteria.base.factory import AcceptanceCriterionFactory
from logic.src.policies.acceptance_criteria.boltzmann_metropolis_criterion import (
    BoltzmannAcceptance,
)


def test_build_isolates_temperature_and_seeded_rng_between_solves():
    prototype = BoltzmannAcceptance(initial_temp=10.0, alpha=0.5, seed=42)
    first = AcceptanceCriterionFactory.build(prototype)
    expected_draw = prototype.rng.getstate()
    first.accept(0.0, -1.0)
    first.step(0.0, -1.0, True)

    second = AcceptanceCriterionFactory.build(prototype)
    assert first.T == 5.0
    assert second.T == prototype.T == 10.0
    assert second.rng.getstate() == prototype.rng.getstate() == expected_draw
    assert first is not prototype and second is not prototype and first is not second


def test_build_reads_omegaconf_params():
    from omegaconf import OmegaConf

    criterion = AcceptanceCriterionFactory.build(
        OmegaConf.create({"method": "bmc", "params": {"initial_temp": 7.0, "alpha": 0.8, "seed": 3}})
    )
    assert criterion.T == 7.0
    assert criterion.alpha == 0.8
