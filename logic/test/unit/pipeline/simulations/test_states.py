"""Unit tests for Simulation State Machine (states.py)."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from logic.src.configs import Config
from logic.src.configs.envs.graph import GraphConfig
from logic.src.configs.tasks.sim import SimConfig
from logic.src.pipeline.simulations.checkpoints import CheckpointError
from logic.src.pipeline.simulations.states import FinishingState, InitializingState, RunningState, SimulationContext

pytestmark = [pytest.mark.unit, pytest.mark.slow]





def _make_test_cfg(**overrides):
    """Build a Config object with sensible test defaults."""
    graph = GraphConfig(
        area="mixrmbac",
        num_loc=50,
        waste_type="glass",
        vertex_method="coord",
        distance_method="haversine",
        dm_filepath=None,
        edge_threshold="0.5",
        edge_method="knn",
    )
    sim = SimConfig(
        policies=["am_dirichlet", "vrpp_gurobi_dirichlet"],
        full_policies=["am_dirichlet", "vrpp_gurobi_dirichlet"],
        data_distribution="dirichlet",
        problem="ptp",
        days=2,
        seed=1234,
        output_dir="test_out",
        checkpoint_dir="checkpoints",
        checkpoint_days=1,
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
    # Apply any overrides
    for k, v in overrides.items():
        if hasattr(sim, k):
            setattr(sim, k, v)
        elif hasattr(sim.graph, k):
            setattr(sim.graph, k, v)

    cfg = Config()
    cfg.sim = sim
    return cfg


@pytest.fixture
def mock_cfg():
    return _make_test_cfg()


@pytest.fixture
def ctx_vars():
    return {"lock": MagicMock(), "counter": MagicMock(), "overall_progress": MagicMock(), "tqdm_pos": 0}


class TestSimulationContext:
    def test_context_init(self, mock_cfg, ctx_vars):
        device = torch.device("cpu")
        ctx = SimulationContext(
            cfg=mock_cfg,
            device=device,
            indices=[0, 1, 2],
            sample_id=0,
            pol_id=0,
            model_weights_path="weights",
            variables_dict=ctx_vars,
        )
        assert ctx.pol_name == "none_am_dirichlet_none"
        assert isinstance(ctx.current_state, InitializingState)

    @patch("logic.src.pipeline.simulations.states.InitializingState.handle")
    def test_run_transitions(self, mock_handle, mock_cfg, ctx_vars):
        ctx = SimulationContext(mock_cfg, torch.device("cpu"), [0], 0, 0, "w", ctx_vars)

        def side_effect(c):
            c.transition_to(None)

        mock_handle.side_effect = side_effect
        result = ctx.run()
        assert result is None
        assert ctx.current_state is None


class TestInitializingState:
    @patch("logic.src.pipeline.simulations.states.initializing.setup_basedata")
    @patch("logic.src.pipeline.simulations.states.initializing.process_data")
    @patch("logic.src.pipeline.simulations.states.initializing.setup_dist_path_tup")
    @patch("logic.src.pipeline.simulations.states.initializing.Bins")
    @patch("logic.src.pipeline.simulations.states.initializing.SimulationCheckpoint")
    @patch("logic.src.pipeline.simulations.states.initializing.setup_model")
    @patch("logic.src.pipeline.simulations.states.initializing.load_config")
    @patch("logic.src.pipeline.simulations.states.initializing.os.path.exists")
    @patch("logic.src.pipeline.simulations.states.initializing.os.makedirs")
    @patch("logic.src.pipeline.simulations.states.initializing.setup_hrl_manager")
    @patch("logic.src.pipeline.simulations.states.initializing.process_model_data")
    def test_initializing_handle_am(
        self,
        mock_model_data,
        mock_hrl,
        mock_makedirs,
        mock_exists,
        mock_load_cfg,
        mock_setup_model,
        mock_checkpoint,
        mock_bins,
        mock_dist,
        mock_proc,
        mock_base,
        mock_cfg,
        ctx_vars,
        tmp_path,
    ):
        mock_cfg.sim.output_dir = str(tmp_path)
        ctx = SimulationContext(mock_cfg, torch.device("cpu"), [0], 0, 0, "w", ctx_vars)

        mock_exists.return_value = False
        mock_base.return_value = (MagicMock(), MagicMock(), MagicMock())
        mock_proc.return_value = (MagicMock(), MagicMock())
        mock_dist.return_value = ((MagicMock(), MagicMock(), MagicMock(), MagicMock()), MagicMock())
        mock_setup_model.return_value = (MagicMock(), {})
        mock_model_data.return_value = (MagicMock(), MagicMock())

        # Mock the local import of load_area_and_waste_type_params
        with patch(
            "logic.src.pipeline.simulations.repository.load_area_and_waste_type_params",
            return_value=(100.0, None, None, None, None),
        ):
            state = InitializingState()
            state.handle(ctx)

        assert isinstance(ctx.current_state, RunningState)
        mock_makedirs.assert_called()

    @patch("logic.src.pipeline.simulations.states.initializing.setup_env")
    @patch("logic.src.pipeline.simulations.states.initializing.setup_basedata")
    @patch("logic.src.pipeline.simulations.states.initializing.process_data")
    @patch("logic.src.pipeline.simulations.states.initializing.setup_dist_path_tup")
    @patch("logic.src.pipeline.simulations.states.initializing.Bins")
    @patch("logic.src.pipeline.simulations.states.initializing.SimulationCheckpoint")
    @patch("logic.src.pipeline.simulations.repository.load_area_and_waste_type_params")
    @patch("logic.src.pipeline.simulations.states.initializing.os.path.exists")
    @patch("logic.src.pipeline.simulations.states.initializing.os.makedirs")
    def test_initializing_handle_vrpp(
        self,
        mock_makedirs,
        mock_exists,
        mock_area_params,
        mock_checkpoint,
        mock_bins,
        mock_dist,
        mock_proc,
        mock_base,
        mock_setup_env,
        ctx_vars,
        tmp_path,
    ):
        cfg = _make_test_cfg(
            policies=["vrpp_gurobi_dirichlet"],
            full_policies=["vrpp_gurobi_dirichlet"],
        )
        cfg.sim.output_dir = str(tmp_path)

        # Policy config needs model name for vrpp detection
        ctx = SimulationContext(cfg, torch.device("cpu"), [0], 0, 0, "w", ctx_vars)

        mock_exists.return_value = False
        mock_base.return_value = (MagicMock(), MagicMock(), MagicMock())
        mock_proc.return_value = (MagicMock(), MagicMock())
        mock_dist.return_value = ((MagicMock(), MagicMock(), MagicMock(), MagicMock()), MagicMock())
        mock_area_params.return_value = (100.0, None, None, None, None)

        state = InitializingState()
        state.handle(ctx)

        assert isinstance(ctx.current_state, RunningState)
        assert ctx.pol_name == "none_vrpp_gurobi_dirichlet_none"

    @patch("logic.src.pipeline.simulations.states.initializing.setup_basedata")
    @patch("logic.src.pipeline.simulations.states.initializing.Bins")
    @patch("logic.src.pipeline.simulations.states.initializing.SimulationCheckpoint")
    @patch("logic.src.pipeline.simulations.states.initializing.os.path.exists")
    @patch("logic.src.pipeline.simulations.states.initializing.os.makedirs")
    def test_initializing_handle_resume(
        self, mock_makedirs, mock_exists, mock_checkpoint, mock_bins, mock_base, ctx_vars, tmp_path
    ):
        cfg = _make_test_cfg(
            policies=["regular_dirichlet"],
            full_policies=["regular_dirichlet"],
            resume=True,
        )
        cfg.sim.output_dir = str(tmp_path)
        ctx = SimulationContext(cfg, torch.device("cpu"), [0], 0, 0, "w", ctx_vars)

        mock_base.return_value = (MagicMock(), MagicMock(), MagicMock())
        mock_checkpoint_inst = MagicMock()
        mock_checkpoint.return_value = mock_checkpoint_inst

        saved_state = (
            MagicMock(),
            MagicMock(),
            (MagicMock(), MagicMock(), MagicMock(), MagicMock()),
            None,
            MagicMock(),
            (None, None),
            [],
            0,
            0,
            {},
            0.0,
        )
        mock_checkpoint_inst.load_state.return_value = (saved_state, 1)

        with patch(
            "logic.src.pipeline.simulations.repository.load_area_and_waste_type_params",
            return_value=(100.0, None, None, None, None),
        ):
            state = InitializingState()
            state.handle(ctx)

        assert ctx.start_day == 2
        assert isinstance(ctx.current_state, RunningState)


class TestRunningState:
    @patch("logic.src.pipeline.simulations.states.running.run_day")
    @patch("logic.src.pipeline.simulations.states.running.checkpoint_manager")
    def test_running_handle(self, mock_cp_manager, mock_run_day, mock_cfg, ctx_vars):
        ctx = SimulationContext(mock_cfg, torch.device("cpu"), [0], 0, 0, "w", ctx_vars)
        ctx.bins = MagicMock()
        distance_matrix = MagicMock()
        ctx.dist_tup = (distance_matrix, MagicMock(), MagicMock(), MagicMock())
        ctx.checkpoint = MagicMock()
        ctx.start_day = 1
        ctx.daily_log = {"profit": []}

        mock_hook = MagicMock()
        mock_cp_manager.return_value.__enter__.return_value = mock_hook

        mock_day_ctx = MagicMock()
        mock_day_ctx.new_data = MagicMock()
        mock_day_ctx.coords = MagicMock()
        mock_day_ctx.bins = ctx.bins
        mock_day_ctx.overflows = 0
        mock_day_ctx.daily_log = {"profit": 10.0}
        mock_day_ctx.output_dict = {}
        mock_day_ctx.cached = []
        mock_run_day.return_value = mock_day_ctx

        state = RunningState()
        ctx.transition_to(state)
        state.handle(ctx)

        assert isinstance(ctx.current_state, FinishingState)
        assert mock_run_day.call_count == mock_cfg.sim.graph.n_days

    @patch("logic.src.pipeline.simulations.states.running.checkpoint_manager")
    def test_running_handle_checkpoint_error(self, mock_cp_manager, mock_cfg, ctx_vars):
        ctx = SimulationContext(mock_cfg, torch.device("cpu"), [0], 0, 0, "w", ctx_vars)
        ctx.checkpoint = MagicMock()

        error_res = {"error": "Test failure", "policy": [0], "success": False}
        mock_cp_manager.side_effect = CheckpointError(error_res)

        state = RunningState()
        ctx.transition_to(state)
        state.handle(ctx)

        assert ctx.result == error_res
        assert ctx.current_state is None


class TestFinishingState:
    @patch("logic.src.pipeline.simulations.states.finishing.update_policy_log_section")
    @patch("logic.src.pipeline.simulations.states.finishing.save_matrix_to_excel")
    def test_finishing_handle(self, mock_excel, mock_log_json, mock_cfg, ctx_vars):
        ctx = SimulationContext(mock_cfg, torch.device("cpu"), [0], 0, 0, "w", ctx_vars)
        ctx.bins = MagicMock()
        ctx.bins.inoverflow = [1, 0]
        ctx.bins.collected = [1, 0]
        ctx.bins.ncollections = [1, 0]
        ctx.bins.lost = [0, 0]
        ctx.bins.travel = 100.0
        ctx.bins.profit = 50.0
        ctx.bins.ndays = 2
        ctx.bins.get_fill_history.return_value = []
        ctx.daily_log = MagicMock()
        ctx.daily_log.values.return_value = []
        ctx.tic = 0.0

        state = FinishingState()
        ctx.transition_to(state)
        state.handle(ctx)

        assert ctx.current_state is None
        assert ctx.result is not None
        assert ctx.result["success"] is True
        mock_log_json.assert_called()
        mock_excel.assert_called()
