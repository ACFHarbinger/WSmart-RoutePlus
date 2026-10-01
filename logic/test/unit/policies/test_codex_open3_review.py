"""Review regressions for round-three deliveries.

The two MS-BPC-SP pricing tests of the original review file are left out: that amendment is on hold (#90).
"""

from types import SimpleNamespace

import pytest
from logic.src.pipeline.simulations.checkpoints.persistence import SimulationCheckpoint
from logic.src.policies.route_construction.learning_algorithms.neural_agent.policy_na import NeuralAgentPolicy


@pytest.mark.parametrize(
    "config",
    [
        {"beam_width": 7, "decoding_strategy": "beam_search"},
        {"decoding": {"beam_width": 7, "strategy": "beam_search"}},
    ],
)
def test_na_direct_decoding_config_is_not_discarded(config):
    policy = NeuralAgentPolicy(config)
    assert policy.config.beam_width == 7
    assert policy.config.decoding_strategy == "beam_search"


def test_absolute_checkpoint_base_still_isolates_runs(tmp_path):
    shared = str(tmp_path / "absolute")
    a = SimulationCheckpoint(str(tmp_path / "a"), shared, "policy", 0)
    b = SimulationCheckpoint(str(tmp_path / "b"), shared, "policy", 0)
    a.save_state({"run": "a"}, day=1)
    assert b.load_state() == (None, 0)
    a.save_state({"run": "a"}, day=1, end_simulation=True)
    a.clear()
    assert a.get_checkpoint_file(1, True) != a.get_checkpoint_file(1)
    import os

    assert os.path.isfile(a.get_checkpoint_file(1, True))


def test_registry_scan_checks_fallback_even_when_dataclass_has_default(monkeypatch):
    from logic.src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.params import (
        ARCOParams,
    )
    from logic.test.unit.policies.route_construction import test_registry_constructor_contract as contract

    monkeypatch.setattr(
        ARCOParams, "from_config", classmethod(lambda cls, cfg: SimpleNamespace(constructors=["missing_test_name"]))
    )
    assert any("missing_test_name" in names for _, names in contract._params_pools())
