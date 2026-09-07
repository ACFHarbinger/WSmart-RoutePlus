"""
Fixtures for the Simulator pipeline - Base fixtures.
"""

from multiprocessing import Lock, Value

import pytest
from logic.src.configs import Config
from logic.src.configs.envs.graph import GraphConfig
from logic.src.configs.tasks.sim import SimConfig


@pytest.fixture
def wsr_opts(tmp_path):
    """
    Provides a default Config object for simulation tests.
    Tests can modify ``cfg.sim.*`` attributes as needed.
    """
    # Create necessary subdirectories in tmp_path
    (tmp_path / "data" / "simulator").mkdir(parents=True, exist_ok=True)
    results_dir = tmp_path / "assets" / "test_output" / "10_days" / "test_area_2"
    results_dir.mkdir(parents=True, exist_ok=True)

    graph = GraphConfig(
        area="riomaior",
        num_loc=2,
        waste_type="paper",
        vertex_method="mmn",
        distance_method="hsd",
        dm_filepath=None,
        edge_threshold="50",
        edge_method="knn",
    )
    sim = SimConfig(
        policies=["test_policy_gamma1"],
        full_policies=["test_policy_gamma1"],
        data_distribution="gamma",
        problem="vrpp",
        days=10,
        seed=42,
        output_dir="test_output",
        checkpoint_dir="temp",
        checkpoint_days=5,
        n_samples=1,
        resume=False,
        n_vehicles=1,
        graph=graph,
        noise_mean=0.0,
        noise_variance=0.0,
        cache_regular=False,
        no_cuda=False,
        server_run=False,
        env_file="vars.env",
        gplic_file=None,
        symkey_name=None,
        gapik_file=None,
        stats_filepath=None,
        data_dir=None,
        policy_configs={},
    )
    cfg = Config()
    cfg.sim = sim
    return cfg


@pytest.fixture
def mock_lock_counter(mocker):
    """
    Provides mock multiprocessing.Lock and Value objects.
    """
    mock_lock = mocker.MagicMock(spec=Lock())
    mock_counter = mocker.MagicMock(spec=Value("i"))
    mock_counter.value = 0
    # Mock the context manager for the lock
    mock_counter_lock = mocker.MagicMock()
    mock_counter_lock.__enter__.return_value = None
    mock_counter_lock.__exit__.return_value = None
    return mock_lock, mock_counter
