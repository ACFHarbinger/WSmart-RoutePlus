"""Unit tests for the HPO simulation handler and objective functions."""

import os
from typing import Any
from unittest.mock import MagicMock, patch

import optuna
import pytest
from logic.src.configs import Config
from logic.src.constants import SIM_METRICS
from logic.src.pipeline.simulations.hpo.hpo_handler import (
    HPOSimulationHandler,
    _extract_metric,
    _metric_direction,
    _select_pareto_representative,
    objective,
    run_hpo_sim,
    worker,
)
from omegaconf import OmegaConf

pytestmark = [pytest.mark.unit, pytest.mark.slow]





class TestHPOHandlerUtilities:
    @pytest.mark.unit
    def test_metric_direction(self) -> None:
        assert _metric_direction("overflows") == "minimize"
        assert _metric_direction("kg_lost") == "minimize"
        assert _metric_direction("length") == "minimize"
        assert _metric_direction("profit") == "maximize"
        assert _metric_direction("ncol") == "maximize"

    @pytest.mark.unit
    def test_extract_metric(self) -> None:
        # If log is empty or None
        assert _extract_metric({}, "profit") == float("-inf")
        assert _extract_metric({}, "overflows") == float("inf")

        # Unknown metric returns default "profit" value if profit exists in SIM_METRICS
        # profit index is 7, so it gets log["alns"][7] which is 1.0
        assert _extract_metric({"alns": [1.0] * 20}, "unknown_metric") == 1.0

        # Correct metric retrieval
        # SIM_METRICS is a list, find index of profit
        profit_idx = SIM_METRICS.index("profit")
        overflows_idx = SIM_METRICS.index("overflows")

        log = {"alns": [0.0] * len(SIM_METRICS)}
        log["alns"][profit_idx] = 123.45
        log["alns"][overflows_idx] = 2.0

        assert _extract_metric(log, "profit") == 123.45
        assert _extract_metric(log, "overflows") == 2.0


class TestHPOSimulationHandler:
    @pytest.fixture
    def mock_cfg(self) -> Any:
        cfg = Config()
        cfg.seed = 42
        cfg.hpo_sim.method = "tpe"
        cfg.hpo_sim.search_space = {"param_a": {"type": "float", "low": 0.0, "high": 1.0}}
        cfg.hpo_sim.policy_name = "alns"
        return OmegaConf.structured(cfg)

    @pytest.mark.unit
    @patch("optuna.create_study")
    def test_handler_init_and_methods(self, mock_create_study: Any, mock_cfg: Any, tmp_path: Any) -> None:
        db_file = tmp_path / "hpo_test.db"
        storage_url = f"sqlite:///{db_file}"

        mock_study = MagicMock()
        mock_create_study.return_value = mock_study

        handler = HPOSimulationHandler(
            cfg=mock_cfg,
            study_name="test_study",
            storage_url=storage_url,
            directions=["maximize"],
            metric_names=["profit"],
            max_budget=10,
        )

        assert handler.study_name == "test_study"
        assert handler.storage_url == storage_url
        assert os.path.exists(db_file)

        # test get_objective
        lock = MagicMock()
        obj_fn = handler.get_objective(lock=lock, data_size=50)
        assert callable(obj_fn)

        # test log_pareto_front with best_trial
        handler.study.best_trials = []
        best_trial = MagicMock()
        best_trial.value = 100.0
        best_trial.params = {"param_a": 0.5}
        handler.study.best_trial = best_trial
        handler.log_pareto_front()

        # test log_pareto_front with multi-objective best_trials
        best_trial_mo = MagicMock()
        best_trial_mo.number = 1
        best_trial_mo.values = [100.0, 5.0]
        handler.study.best_trials = [best_trial_mo]
        handler.log_pareto_front()

        # test run_fanova_analysis with too few trials
        m_trial = MagicMock()
        m_trial.state = optuna.trial.TrialState.COMPLETE
        handler.study.trials = [m_trial] * 10
        with patch("logic.src.pipeline.simulations.hpo.hpo_handler.logger") as mock_logger:
            handler.run_fanova_analysis()
            mock_logger.info.assert_any_call("fANOVA requires ~30 trials (found 10). Skipping.")

        # test run_fanova_analysis with enough trials
        trial_mock = MagicMock()
        trial_mock.state = optuna.trial.TrialState.COMPLETE
        trial_mock.values = [100.0]
        trial_mock.params = {"param_a": 0.5}
        handler.study.trials = [trial_mock] * 35

        with patch("optuna.importance.get_param_importances", return_value={"param_a": 0.8}):
            handler.run_fanova_analysis()


class TestHPOObjectiveAndWorker:
    @pytest.fixture
    def mock_cfg(self) -> Any:
        cfg = Config()
        cfg.seed = 42
        cfg.hpo_sim.method = "tpe"
        cfg.hpo_sim.search_space = {"param_a": {"type": "float", "low": 0.0, "high": 1.0}}
        cfg.hpo_sim.policy_name = "alns"
        cfg.hpo_sim.graph.num_loc = 50
        cfg.hpo_sim.graph.area = "test_area"
        cfg.hpo_sim.graph.waste_type = "residual"
        cfg.hpo_sim.graph.n_days = 5
        cfg.hpo_sim.graph.n_samples = 2
        cfg.sim.graph.num_loc = 50
        cfg.sim.graph.area = "test_area"
        cfg.sim.graph.waste_type = "residual"
        cfg.sim.graph.n_days = 5
        cfg.sim.graph.n_samples = 2
        cfg.sim.full_policies = []
        return OmegaConf.structured(cfg)

    @pytest.mark.unit
    @patch("logic.src.pipeline.simulations.hpo.hpo_objective.load_indices")
    @patch("logic.src.pipeline.simulations.hpo.hpo_objective.sequential_simulations")
    @patch("logic.src.pipeline.simulations.hpo.hpo_objective.expand_policy_configs")
    def test_objective_single_objective(
        self,
        mock_expand: Any,
        mock_seq_sim: Any,
        mock_load_indices: Any,
        mock_cfg: Any,
    ) -> None:
        trial = MagicMock()
        trial.suggest_float.return_value = 0.5
        trial.should_prune.return_value = False

        # Set up load_indices mock
        mock_load_indices.return_value = [0, 1]

        # Set up sequential_simulations mock
        profit_idx = SIM_METRICS.index("profit")
        dummy_log_metrics = [0.0] * len(SIM_METRICS)
        dummy_log_metrics[profit_idx] = 999.0
        mock_seq_sim.return_value = ({"alns": dummy_log_metrics}, None, None)

        lock = MagicMock()
        val = objective(trial, mock_cfg, data_size=100, lock=lock)

        assert val == 999.0
        trial.suggest_float.assert_called_with("param_a", 0.0, 1.0, step=None, log=False)
        mock_seq_sim.assert_called_once()

        # Test iterative callback execution
        called_args, called_kwargs = mock_seq_sim.call_args
        callback = called_kwargs.get("callback")
        assert callback is not None
        # Execute callback
        callback(day=1, cum_metrics={"profit": 100.0}, s_id=0)
        trial.report.assert_called_with(100.0, step=1)

        # Test trial pruning in callback
        trial.should_prune.return_value = True
        with pytest.raises(optuna.TrialPruned):
            callback(day=2, cum_metrics={"profit": 120.0}, s_id=0)

    @pytest.mark.unit
    @patch("logic.src.pipeline.simulations.hpo.hpo_objective.load_indices")
    @patch("logic.src.pipeline.simulations.hpo.hpo_objective.sequential_simulations")
    def test_objective_multi_objective(
        self,
        mock_seq_sim: Any,
        mock_load_indices: Any,
        mock_cfg: Any,
    ) -> None:
        trial = MagicMock()
        trial.suggest_float.return_value = 0.5

        # Set up multiple metrics
        mock_cfg.hpo_sim.metrics = ["profit", "overflows"]

        mock_load_indices.return_value = [0, 1]

        profit_idx = SIM_METRICS.index("profit")
        overflows_idx = SIM_METRICS.index("overflows")
        dummy_log_metrics = [0.0] * len(SIM_METRICS)
        dummy_log_metrics[profit_idx] = 500.0
        dummy_log_metrics[overflows_idx] = 1.0
        mock_seq_sim.return_value = ({"alns": dummy_log_metrics}, None, None)

        lock = MagicMock()
        val = objective(trial, mock_cfg, data_size=100, lock=lock)

        assert val == (500.0, 1.0)

    @pytest.mark.unit
    @patch("optuna.load_study")
    def test_worker(self, mock_load_study: Any, mock_cfg: Any) -> None:
        mock_study = MagicMock()
        mock_load_study.return_value = mock_study

        yaml_cfg = OmegaConf.to_yaml(mock_cfg)
        lock = MagicMock()

        worker(
            study_name="test_study",
            storage_url="sqlite:///dummy.db",
            base_cfg_yaml=yaml_cfg,
            data_size=50,
            n_trials=5,
            lock=lock,
        )

        mock_load_study.assert_called_once_with(study_name="test_study", storage="sqlite:///dummy.db")
        mock_study.optimize.assert_called_once()


class TestHPOSimOrchestrator:
    @pytest.mark.unit
    def test_select_pareto_representative(self) -> None:
        trial_1 = MagicMock()
        trial_1.values = [10.0, 5.0]
        trial_2 = MagicMock()
        trial_2.values = [20.0, 1.0]

        # Case 1: Empty list
        assert _select_pareto_representative([], ["maximize", "minimize"]) is None

        # Case 2: Single item
        assert _select_pareto_representative([trial_1], ["maximize", "minimize"]) == trial_1

        # Case 3: Multiple items
        rep = _select_pareto_representative([trial_1, trial_2], ["maximize", "minimize"])
        assert rep in (trial_1, trial_2)

    @pytest.mark.unit
    @patch("logic.src.pipeline.simulations.hpo.hpo_handler.HPOSimulationHandler")
    @patch("logic.src.pipeline.simulations.hpo.hpo_handler.worker")
    @patch("logic.src.pipeline.simulations.hpo.hpo_handler.set_repository_from_path")
    @patch("logic.src.pipeline.simulations.hpo.hpo_handler.load_simulator_data")
    @patch("logic.src.tracking.init")
    @patch("logic.src.tracking.get_active_run")
    def test_run_hpo_sim_single_worker(
        self,
        mock_get_active_run: Any,
        mock_wst_init: Any,
        mock_load_sim_data: Any,
        mock_set_repo: Any,
        mock_worker: Any,
        mock_handler_cls: Any,
    ) -> None:
        cfg = Config()
        cfg.seed = 42
        cfg.hpo_sim.policy_name = "alns"
        cfg.hpo_sim.n_trials = 10
        cfg.hpo_sim.num_workers = 1
        cfg.hpo_sim.method = "tpe"
        cfg.hpo_sim.search_space = {"param_a": {"type": "float", "low": 0.0, "high": 1.0}}
        omega_cfg = OmegaConf.structured(cfg)

        mock_load_sim_data.return_value = ([0] * 50, None)
        mock_set_repo.return_value = False

        # Set up active run mock
        mock_run = MagicMock()
        mock_get_active_run.return_value = mock_run

        # Set up handler instance mock
        mock_handler = MagicMock()
        mock_handler_cls.return_value = mock_handler
        mock_handler.study.best_trial.value = 100.0
        mock_handler.study.best_trial.params = {"param_a": 0.5}

        val = run_hpo_sim(omega_cfg)

        assert val == 100.0
        mock_wst_init.assert_called_once()
        mock_worker.assert_called_once()
        mock_run.log_params.assert_called_once_with({"hpo/best/param_a": 0.5})

    @pytest.mark.unit
    @patch("logic.src.pipeline.simulations.hpo.hpo_handler.HPOSimulationHandler")
    @patch("multiprocessing.Process")
    @patch("logic.src.pipeline.simulations.hpo.hpo_handler.set_repository_from_path")
    @patch("logic.src.pipeline.simulations.hpo.hpo_handler.load_simulator_data")
    @patch("logic.src.tracking.init")
    @patch("logic.src.tracking.get_active_run")
    def test_run_hpo_sim_multi_workers(
        self,
        mock_get_active_run: Any,
        mock_wst_init: Any,
        mock_load_sim_data: Any,
        mock_set_repo: Any,
        mock_process: Any,
        mock_handler_cls: Any,
    ) -> None:
        cfg = Config()
        cfg.seed = 42
        cfg.hpo_sim.policy_name = "alns"
        cfg.hpo_sim.n_trials = 10
        cfg.hpo_sim.num_workers = 2
        cfg.hpo_sim.method = "tpe"
        cfg.hpo_sim.search_space = {"param_a": {"type": "float", "low": 0.0, "high": 1.0}}
        omega_cfg = OmegaConf.structured(cfg)

        mock_load_sim_data.return_value = ([0] * 50, None)
        mock_set_repo.return_value = False

        # Set up active run mock
        mock_run = MagicMock()
        mock_get_active_run.return_value = mock_run

        # Set up handler instance mock
        mock_handler = MagicMock()
        mock_handler_cls.return_value = mock_handler
        mock_handler.study.best_trial.value = 100.0
        mock_handler.study.best_trial.params = {"param_a": 0.5}

        # Process mock
        proc_mock = MagicMock()
        mock_process.return_value = proc_mock

        val = run_hpo_sim(omega_cfg)

        assert val == 100.0
        assert mock_process.call_count == 2
        assert proc_mock.start.call_count == 2
        assert proc_mock.join.call_count == 2
