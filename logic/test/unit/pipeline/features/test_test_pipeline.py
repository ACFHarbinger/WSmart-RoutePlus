

from unittest.mock import MagicMock, patch

import pytest
import torch
from logic.src.configs import Config
from logic.src.configs.envs.graph import GraphConfig
from logic.src.configs.tasks.sim import SimConfig
from logic.src.pipeline.features.test import run_simulator_test as run_sim_test_func
from logic.src.pipeline.features.test import simulator_testing

pytestmark = [pytest.mark.unit, pytest.mark.fast]





def _make_pipeline_cfg(**overrides):
    """Build a Config object for pipeline test fixtures."""
    graph = GraphConfig(
        area="mixrmbac",
        num_loc=20,
        waste_type="glass",
    )
    sim = SimConfig(
        policies=["policy1"],
        full_policies=["policy1"],
        data_distribution="test_dist",
        days=5,
        seed=1234,
        output_dir="test_output",
        checkpoint_dir="checkpoints",
        n_samples=2,
        resume=False,
        cpu_cores=1,
        graph=graph,
        run_name="test_run",
    )
    for k, v in overrides.items():
        if hasattr(sim, k):
            setattr(sim, k, v)
        elif hasattr(sim.graph, k):
            setattr(sim.graph, k, v)
    cfg = Config()
    cfg.sim = sim
    return cfg


class TestPipelineFeaturesTest:
    @pytest.fixture
    def cfg(self):
        return _make_pipeline_cfg()

    @patch("logic.src.pipeline.simulations.repository.filesystem.udef.MAP_DEPOTS", {"riomaior": "CTEASO"})
    @patch("logic.src.pipeline.features.test.orchestrator.udef")
    @patch("logic.src.pipeline.features.test.orchestrator.load_indices")
    @patch("logic.src.pipeline.features.test.orchestrator.parallel_runner.Pool")
    @patch("logic.src.pipeline.features.test.orchestrator.results_handler.output_stats")
    @patch("logic.src.pipeline.features.test.orchestrator.send_final_output_to_gui")
    @patch("logic.src.pipeline.features.test.orchestrator.display_log_metrics")
    def test_simulator_testing_parallel(
        self,
        mock_display,
        mock_send,
        mock_out_stats,
        mock_pool,
        mock_load,
        mock_udef,
        cfg,
    ):
        mock_udef.ROOT_DIR = "/tmp/test"
        mock_udef.LOCK_TIMEOUT = 100
        mock_udef.PBAR_WAIT_TIME = 0.1

        cfg.sim.cpu_cores = 2
        cfg.sim.graph.n_samples = 2

        mock_load.return_value = [0, 1]

        pool_instance = MagicMock()
        mock_pool.return_value = pool_instance

        mock_metrics = [0.0, 100.0, 5.0, 0.0, 50.0, 2.0, 25.0, 75.0, 1.0, 10.0]
        task1 = MagicMock()
        task1.ready.return_value = True
        task1.get.return_value = {"success": True, "policy1": mock_metrics}

        pool_instance.apply_async.return_value = task1

        device = torch.device("cpu")
        data_size = 10

        with patch("multiprocessing.Manager") as mock_manager:
            mock_dict = MagicMock()
            mock_list = MagicMock()
            mock_manager.return_value.dict.return_value = mock_dict
            mock_manager.return_value.list.return_value = mock_list

            def side_effect_apply(func, args, callback):
                callback({"success": True, "policy1": mock_metrics})
                return task1

            pool_instance.apply_async.side_effect = side_effect_apply

            mock_dict.items.return_value = [("policy1", [mock_metrics])]

            simulator_testing(cfg, data_size, device)

            assert pool_instance.apply_async.called

    @patch("logic.src.pipeline.features.test.engine.wst.init")
    @patch("logic.src.pipeline.features.test.engine.load_simulator_data")
    @patch("logic.src.pipeline.features.test.engine.simulator_testing")
    @patch("logic.src.pipeline.features.test.engine.os.makedirs")
    @patch("logic.src.pipeline.features.test.engine.expand_policy_configs")
    def test_run_simulator_test(self, mock_expand, mock_makedirs, mock_sim_test, mock_load_data, mock_wst_init, cfg):
        mock_load_data.return_value = ([1] * 10, None)

        run_sim_test_func(cfg)

        assert mock_sim_test.called
        assert mock_makedirs.called

    @patch("logic.src.pipeline.features.test.engine.wst.init")
    @patch("logic.src.pipeline.features.test.engine.load_simulator_data", side_effect=Exception("Fail"))
    @patch("logic.src.pipeline.features.test.engine.simulator_testing")
    @patch("logic.src.pipeline.features.test.engine.os.makedirs")
    @patch("logic.src.pipeline.features.test.engine.expand_policy_configs")
    def test_run_simulator_test_fallback(self, mock_expand, mock_makedirs, mock_sim_test, mock_load_data, mock_wst_init, cfg):
        cfg.sim.graph.area = "mixrmbac"
        cfg.sim.graph.num_loc = 20

        run_sim_test_func(cfg)

        # Should use fallback data_size logic for mixrmbac with size 20
        assert mock_sim_test.call_args[0][1] == 20
