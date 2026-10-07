


import unittest.mock as mock

import logic.src.pipeline.simulations.simulator as simulator
import numpy as np
import pytest
import torch
from logic.src.configs import Config
from logic.src.configs.envs.graph import GraphConfig
from logic.src.configs.tasks.sim import SimConfig
from logic.src.pipeline.simulations.checkpoints.manager import CheckpointError

pytestmark = [pytest.mark.unit, pytest.mark.fast]





def _make_integration_cfg(**overrides):
    """Build a Config for simulator integration tests."""
    graph = GraphConfig(
        area="figueiradafoz",
        num_loc=50,
        waste_type="plastic",
        vertex_method="mmn",
        distance_method="ogd",
        dm_filepath="dummy_path",
        edge_threshold="0.5",
        edge_method="knn",
        n_days=2,
    )
    sim = SimConfig(
        policies=[{"am_gamma1": {"model": {"name": "am"}}}],
        full_policies=["am_gamma1"],
        data_distribution="unif",
        problem="ptp",
        days=2,
        seed=42,
        output_dir="test_out",
        checkpoint_dir="checkpoints",
        checkpoint_days=1,
        n_samples=1,
        resume=False,
        n_vehicles=5,
        graph=graph,
        noise_mean=0.0,
        noise_variance=0.0,
        cache_regular=False,
        no_cuda=False,
        server_run=False,
        env_file="dummy_env",
        gplic_file=None,
        symkey_name="dummy_symkey",
        gapik_file="dummy_gapik",
        stats_filepath=None,
        data_dir=None,
        policy_configs={},
    )
    for k, v in overrides.items():
        if hasattr(sim, k):
            setattr(sim, k, v)
        elif hasattr(sim.graph, k):
            setattr(sim.graph, k, v)
    cfg = Config()
    cfg.sim = sim
    return cfg


class TestSimulatorIntegration:
    @pytest.fixture
    def mock_sim_dependencies(self, mocker):
        deps = {
            "setup_basedata": mocker.patch("logic.src.pipeline.simulations.states.initializing.setup_basedata"),
            "setup_env": mocker.patch("logic.src.pipeline.simulations.states.initializing.setup_env"),
            "setup_models": mocker.patch("logic.src.pipeline.simulations.states.initializing.InitializingState._setup_models"),
            "setup_dist_path_tup": mocker.patch("logic.src.pipeline.simulations.states.initializing.setup_dist_path_tup"),
            "init_new_state": mocker.patch("logic.src.pipeline.simulations.states.initializing.InitializingState._initialize_new_state"),
            "run_day": mocker.patch("logic.src.pipeline.simulations.states.running.run_day"),
            "checkpoint_manager": mocker.patch("logic.src.pipeline.simulations.states.running.checkpoint_manager"),
            "params": mocker.patch("logic.src.pipeline.simulations.repository.load_area_and_waste_type_params"),
        }
        deps["run_day"].side_effect = lambda x: x
        deps["setup_basedata"].return_value = (mock.MagicMock(), mock.MagicMock(), mock.MagicMock())
        deps["params"].return_value = (mock.MagicMock(), mock.MagicMock(), mock.MagicMock(), mock.MagicMock(), mock.MagicMock())

        def mock_init_new_state(ctx, data, coords, depot):
            ctx.new_data = mock.MagicMock()
            ctx.coords = mock.MagicMock()
            ctx.dist_tup = (mock.MagicMock(), mock.MagicMock(), mock.MagicMock(), mock.MagicMock())
            ctx.bins = mock.MagicMock()
            ctx.bins.collected = np.zeros(10)
            ctx.bins.inoverflow = np.zeros(10)
            ctx.bins.ncollections = np.zeros(10)
            ctx.bins.lost = np.zeros(10)
            ctx.bins.travel = 1.0
            ctx.bins.collected_total = 100.0
            ctx.bins.profit = 50.0
            ctx.bins.ndays = 2
            ctx.overflows = 0
            ctx.execution_time = 0.1
            ctx.model_tup = (None, None)
            ctx.output_dict = {}
            ctx.daily_log = {}

        deps["init_new_state"].side_effect = mock_init_new_state

        mock_cm = mock.MagicMock()
        mock_cm.__enter__.return_value = mock.MagicMock()
        deps["checkpoint_manager"].return_value = mock_cm
        return deps

    @pytest.fixture
    def mock_lock_counter(self):
        return mock.MagicMock(), mock.MagicMock()

    @pytest.fixture
    def mock_torch_device(self):
        return torch.device("cpu")

    @pytest.mark.integration
    def test_single_simulation_happy_path_am(self, mock_sim_dependencies, mock_lock_counter, mock_torch_device):
        cfg = _make_integration_cfg()
        simulator._lock, simulator._counter = mock_lock_counter

        result = simulator.single_simulation(
            cfg, mock_torch_device, indices=None, sample_id=0, pol_id=0, model_weights_path="weights", n_cores=1
        )
        assert "none_am_none" in result
        assert result["success"] is True
        assert mock_sim_dependencies["run_day"].call_count == 2

    @pytest.mark.integration
    def test_single_simulation_resume(self, mock_sim_dependencies, mock_lock_counter, mock_torch_device):
        cfg = _make_integration_cfg(resume=True)
        simulator._lock, simulator._counter = mock_lock_counter

        with mock.patch("logic.src.pipeline.simulations.states.initializing.InitializingState._load_checkpoint_if_needed") as mock_load:
            mock_dist_tup = (mock.MagicMock(), mock.MagicMock(), mock.MagicMock(), mock.MagicMock())
            mock_bins = mock.MagicMock()
            mock_bins.collected = np.zeros(10)
            mock_bins.inoverflow = np.zeros(10)
            mock_bins.ncollections = np.zeros(10)
            mock_bins.lost = np.zeros(10)
            mock_bins.travel = 1.0
            mock_bins.collected_total = 100.0
            mock_bins.profit = 50.0
            mock_bins.ndays = 2
            mock_load.return_value = ((mock.MagicMock(), mock.MagicMock(), mock_dist_tup, None, mock_bins, (None, None), [], 0, 0, {}, 0.0), 1)

            result = simulator.single_simulation(
                cfg, mock_torch_device, indices=None, sample_id=0, pol_id=0, model_weights_path="weights", n_cores=1
            )
            assert result["success"] is True

    @pytest.mark.integration
    def test_single_simulation_checkpoint_error(self, mock_sim_dependencies, mock_lock_counter, mock_torch_device):
        cfg = _make_integration_cfg()
        simulator._lock, simulator._counter = mock_lock_counter

        error_res = {"error": "Test failure", "policy": "none_am_none", "success": False}
        mock_sim_dependencies["checkpoint_manager"].side_effect = CheckpointError(error_res)

        result = simulator.single_simulation(
            cfg, mock_torch_device, indices=None, sample_id=0, pol_id=0, model_weights_path="weights", n_cores=1
        )
        assert result["success"] is False

    @pytest.mark.integration
    def test_sequential_simulations_multi_sample(self, mock_sim_dependencies, mock_lock_counter, mock_torch_device):
        cfg = _make_integration_cfg(n_samples=2)
        lock, counter = mock_lock_counter
        simulator._lock, simulator._counter = lock, counter

        results, results_std, failed = simulator.sequential_simulations(
            cfg, mock_torch_device, [None, None], [[0], [1]], "weights", lock
        )

        assert "none_am_none" in results
        assert len(failed) == 0
