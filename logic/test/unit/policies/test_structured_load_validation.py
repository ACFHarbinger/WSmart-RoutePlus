"""Invalid values must not escape OmegaConf validation via a dataclass fallback."""

from dataclasses import dataclass

import pytest
from logic.src.policies.routing.base.base_routing_policy import _structured_load
from omegaconf.errors import ValidationError


@dataclass
class SolverConfig:
    max_iterations: int = 10


def test_invalid_known_field_type_fails_in_structured_load():
    with pytest.raises(ValidationError):
        _structured_load(SolverConfig, {"max_iterations": "not-an-integer"})


def test_valid_convertible_value_keeps_omegaconf_semantics():
    cfg = _structured_load(SolverConfig, {"max_iterations": "20"})
    assert cfg.max_iterations == 20
    assert isinstance(cfg.max_iterations, int)
