"""Gate commands must parse as Hydra overrides before expensive execution."""

import hashlib
import logging
import subprocess

import pytest
from hydra.core.override_parser.overrides_parser import OverridesParser
from tools.gates import capture
from tools.gates.cells import GATE_A_CELLS, cell_policy_entry
from tools.gates.run_gate import _hydra_value, cell_dotlist


def test_calibration_reference_requires_exact_amendment(tmp_path, monkeypatch):
    from tools.gates import run_gate

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "-q")
    source = tmp_path / "logic/src/adapter.py"
    source.parent.mkdir(parents=True)
    source.write_text("original\n")
    git("add", ".")
    git("-c", "user.name=Gate Test", "-c", "user.email=gate@example.invalid", "commit", "-qm", "reference")
    monkeypatch.setattr(run_gate, "BASELINE_COMMIT", git("rev-parse", "HEAD"))
    amendment = b"explicit adapter fix\n"
    monkeypatch.setattr(
        run_gate, "BASELINE_AMENDMENTS", {"logic/src/adapter.py": hashlib.sha256(amendment).hexdigest()}
    )
    monkeypatch.chdir(tmp_path)
    assert not run_gate.code_matches_baseline()
    source.write_bytes(amendment)
    assert run_gate.code_matches_baseline()
    source.write_text("different adapter fix\n")
    assert not run_gate.code_matches_baseline()
    source.write_bytes(amendment)
    unexpected = source.with_name("unexpected.py")
    unexpected.write_text("untracked code\n")
    assert not run_gate.code_matches_baseline()
    git("add", str(unexpected))
    assert not run_gate.code_matches_baseline()


def test_every_gate_cell_override_parses_in_hydra():
    parser = OverridesParser.create()
    for cell in GATE_A_CELLS:
        for text in cell_dotlist(cell["constructor"], model_path="/tmp/model [frozen]"):
            parser.parse_override(text)
    value = "/tmp/distance matrix[riomaior].csv"
    assert parser.parse_override("sim.graph.dm_filepath=" + _hydra_value(value)).value() == value


def test_external_harness_uses_the_target_checkout_config_names(tmp_path, monkeypatch):
    policy = tmp_path / "logic/configs/policies/policy_alns.yaml"
    policy.parent.mkdir(parents=True)
    policy.write_text("alns:\n  custom:\n    - max_iterations: 17\n    - profit_aware_operators: false\n")
    monkeypatch.chdir(tmp_path)
    entry = cell_policy_entry("ALNS")["alns"]
    assert entry["max_iterations"] == 17
    assert entry["profit_aware_operators"] is False
    assert "other/ms_last_minute.yaml" in entry["packing"]
    renamed = tmp_path / "logic/configs/policies/other/pk_last_minute.yaml"
    renamed.parent.mkdir(parents=True)
    renamed.write_text("")
    assert "other/pk_last_minute.yaml" in cell_policy_entry("ALNS")["alns"]["packing"]


@pytest.mark.parametrize(
    "converged,pricing_exhausted,timed_out,node_depth,expected",
    [
        (True, True, False, 0, True),
        (False, True, False, 0, False),
        (True, False, False, 0, False),
        (True, True, True, 0, False),
        (True, True, False, 1, False),
    ],
)
def test_baseline_observer_requires_exhausted_root_pricing(
    converged, pricing_exhausted, timed_out, node_depth, expected
):
    capture._PROOF.set(None)

    def column_generation_loop():
        # The real baseline smoke also exercises this against the original CG.
        lagrangian_ub = 12.123456789
        _ = converged, pricing_exhausted, timed_out, node_depth
        record = logging.LogRecord("cg", logging.INFO, __file__, 1, "z_UB=12.1235", (), None)
        capture._RootProofFilter().filter(record)
        return lagrangian_ub

    exact = column_generation_loop()
    if expected:
        assert capture._PROOF.get()["bound"] == exact
    else:
        assert capture._PROOF.get() is None
