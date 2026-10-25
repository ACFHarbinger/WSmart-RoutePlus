"""Shared fixtures for the improvement-decoder tests.

``dummy_env`` (a 20-node TSP k-opt environment) and ``dummy_batch`` (its reset state,
batch size 4), moved unchanged from ``test_improvement_decoders.py``.
"""

import pytest
from logic.src.envs.tsp_kopt import TSPkoptEnv


@pytest.fixture
def dummy_env():
    """Provides a TSPkopt environment for testing."""
    return TSPkoptEnv(num_loc=20)


@pytest.fixture
def dummy_batch(dummy_env):
    """Provides a reset environment TensorDict with batch_size=4."""
    return dummy_env.reset(batch_size=[4])
