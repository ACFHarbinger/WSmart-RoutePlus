"""B-cursor-03: BMC/OI module examples must be maximisation, not a worsening move."""

import pytest
from logic.src.policies.acceptance_criteria import boltzmann_metropolis_criterion as bmc_mod
from logic.src.policies.acceptance_criteria import only_improving as oi_mod
from logic.src.policies.acceptance_criteria.boltzmann_metropolis_criterion import BoltzmannAcceptance
from logic.src.policies.acceptance_criteria.only_improving import OnlyImproving

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_bmc_module_example_is_an_improving_move():
    """The old example claimed accept(100, 98) → True; that is a worsening."""
    doc = bmc_mod.__doc__
    assert doc is not None
    assert "candidate_obj=12.0" in doc
    assert "candidate_obj=98.0" not in doc
    assert "ties are accepted" in doc


def test_oi_module_example_is_an_improving_move():
    doc = oi_mod.__doc__
    assert doc is not None
    assert "candidate_obj=12.0" in doc
    assert "candidate_obj=98.0" not in doc
    assert "ties are rejected" in doc


def test_bmc_accepts_improving_and_ties_rejects_worsening_at_zero_temp():
    improving, _ = BoltzmannAcceptance(initial_temp=1.0, alpha=0.995, seed=0).accept(10.0, 12.0)
    assert improving is True
    tie, _ = BoltzmannAcceptance(initial_temp=1.0, alpha=0.995, seed=0).accept(10.0, 10.0)
    assert tie is True
    cold = BoltzmannAcceptance(initial_temp=1e-12, alpha=0.995, seed=0)
    worsening, _ = cold.accept(100.0, 98.0)
    assert worsening is False


def test_oi_accepts_only_strict_improvement():
    oi = OnlyImproving()
    assert oi.accept(10.0, 12.0)[0] is True
    assert oi.accept(10.0, 10.0)[0] is False
    assert oi.accept(100.0, 98.0)[0] is False
