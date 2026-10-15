"""Importing the training and evaluation features loads only the paper's neural stack (Q-12).

The release keeps the import closure of its seeds (``logic/release.yaml``), so a module-level
import of an unreleased model or algorithm would put it in the package.
"""

import subprocess
import sys

import pytest

pytestmark = [pytest.mark.unit]

_PROBE = """
import sys
import logic.src.pipeline.features.train, logic.src.pipeline.features.eval
from logic.src.pipeline.features.train.model_factory.registry import get_algorithm_factory
get_algorithm_factory("imitation")
import logic.src.pipeline.rl.core.imitation  # what the imitation factory imports when it runs
print("\\n".join(sorted(m for m in sys.modules if m.startswith("logic.src."))))
"""


def test_only_the_attention_model_reinforce_and_imitation_load() -> None:
    loaded = set(
        subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, check=True).stdout.split()
    )

    models = {m.split(".")[4] for m in loaded if m.startswith("logic.src.models.core.")}
    algorithms = {m.split(".")[5] for m in loaded if m.startswith("logic.src.pipeline.rl.core.")}
    assert models == {"attention_model"}
    assert algorithms == {"reinforce", "imitation", "losses"}
    assert not any(m.startswith(("logic.src.pipeline.rl.hpo", "logic.src.pipeline.rl.meta")) for m in loaded)
    vector = {m.split(".")[4] for m in loaded if m.startswith("logic.src.policies.vector.")}
    assert vector == {"expert_cache", "profitable_nearest_neighbour"}
