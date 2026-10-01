"""
Fixtures for the Simulator pipeline - Checkpoint fixtures.
"""

import os

import pytest
from logic.src.pipeline.simulations.checkpoints.persistence import SimulationCheckpoint


@pytest.fixture
def basic_checkpoint(tmp_path):
    """
    Sets up a real (not mocked) SimulationCheckpoint in a temporary directory.
    Live files are stored under ``output_dir``, so the temporary path never
    touches the repository root.
    """
    output_dir = tmp_path / "test_assets" / "results"
    cp = SimulationCheckpoint(
        output_dir=str(output_dir),
        checkpoint_dir="temp",
        policy="test_policy",
        sample_id=1,
    )

    os.makedirs(cp.checkpoint_dir, exist_ok=True)
    os.makedirs(cp.output_dir, exist_ok=True)

    return cp
