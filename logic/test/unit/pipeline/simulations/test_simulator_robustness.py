"""Regression tests for simulator resume, failure, and naming bugs (issue 80)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from logic.src.configs import Config
from logic.src.configs.envs.graph import GraphConfig
from logic.src.configs.tasks.sim import SimConfig
from logic.src.pipeline.features.test.orchestrator.parallel_runner import execute_and_monitor_tasks
from logic.src.pipeline.features.test.orchestrator.results_handler import aggregate_final_results
from logic.src.pipeline.simulations.checkpoints import CheckpointError
from logic.src.pipeline.simulations.day_context import get_full_policy_name, policy_result_key
from logic.src.pipeline.simulations.simulator import sequential_simulations
from logic.src.pipeline.simulations.states.running import RunningState, pad_model_ls
from logic.test.unit.pipeline.simulations.test_states import _make_test_cfg

pytestmark = [pytest.mark.unit, pytest.mark.fast]


@pytest.fixture
def ctx_vars():
    """Shared resources the running state reads off the context."""
    return {"lock": MagicMock(), "counter": MagicMock(), "overall_progress": MagicMock(), "tqdm_pos": 0}


def _orch_cfg(**overrides) -> Config:
    """Minimal orchestrator config. Defaults match the existing feature tests."""
    graph = GraphConfig(area="mixrmbac", num_loc=20, waste_type="glass")
    sim = SimConfig(
        policies=["alns"],
        full_policies=["alns"],
        data_distribution="unif",
        days=1,
        seed=42,
        output_dir="test_out",
        n_samples=1,
        resume=False,
        cpu_cores=1,
        graph=graph,
    )
    for key, value in overrides.items():
        if hasattr(sim, key):
            setattr(sim, key, value)
        elif hasattr(sim.graph, key):
            setattr(sim.graph, key, value)
    cfg = Config()
    cfg.sim = sim
    return cfg


def test_pad_model_ls_is_a_three_tuple():
    """NA unpacks three slots. A missing model must not be a 1- or 2-tuple."""
    assert pad_model_ls(None) == (None, None, None)
    assert pad_model_ls((1, 2)) == (1, 2, None)
    assert pad_model_ls((1, 2, 3, 4)) == (1, 2, 3, 4)


@patch("logic.src.pipeline.simulations.states.running.run_day")
@patch("logic.src.pipeline.simulations.states.running.checkpoint_manager")
def test_resume_clock_adds_stored_elapsed(mock_cp_manager, mock_run_day, ctx_vars):
    """A resumed run's next elapsed time is stored time plus new work."""
    from logic.src.pipeline.simulations.states import FinishingState, SimulationContext

    cfg = _make_test_cfg()
    cfg.sim.graph.n_days = 1
    ctx = SimulationContext(cfg, torch.device("cpu"), [0], 0, 0, "w", ctx_vars)
    ctx.bins = MagicMock()
    ctx.dist_tup = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
    ctx.checkpoint = MagicMock()
    ctx.start_day = 1
    ctx.daily_log = {"profit": []}
    ctx.run_time = 12.5
    ctx.model_tup = None

    clock = {"t": 1000.0}
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

    def _advance(_day_ctx):
        clock["t"] = 1003.0
        return mock_day_ctx

    mock_run_day.side_effect = _advance

    with patch("logic.src.pipeline.simulations.states.running.time.perf_counter", lambda: clock["t"]):
        RunningState().handle(ctx)

    assert isinstance(ctx.current_state, FinishingState)
    elapsed = mock_hook.after_day.call_args_list[0][0][0]
    assert elapsed == pytest.approx(15.5)
    assert elapsed > 0
    handed = mock_run_day.call_args[0][0].model_ls
    assert len(handed) >= 3
    assert handed[:3] == (None, None, None)


def test_expanded_id_keeps_alns_constructor():
    """ms_regular_alns_ri_none names the ALNS constructor, not an empty slot."""
    name = get_full_policy_name(
        "ms_regular_alns_ri_none",
        {"mandatory_selection": "regular", "route_improvement": "none"},
    )
    assert name.split(" + ")[1] == "ALNS"


def test_unexpanded_display_name_stays_the_full_id():
    """Ids without an ms/ri token keep the previous display string."""
    name = get_full_policy_name("lookahead_alns_bmc_fast_tsp", {})
    assert "LOOKAHEAD_ALNS_BMC_FAST_TSP" in name


def test_stats_file_horizon_keeps_opening_row_out_of_the_deposits(basic_bins):
    """Row 0 is opening stock. Days 1..n deposit rows 1..n, once each.

    Rows 10, 20, 30. Opening 10. Day 1 adds 20 (level 30). Day 2 adds 30
    (level 60). The next day is past the sample and must not wrap onto row 0.
    """
    n_days = 2
    n_bins = basic_bins.n
    rows = np.zeros((n_days + 1, n_bins), dtype=float)
    rows[0] = 10.0
    rows[1] = 20.0
    rows[2] = 30.0
    basic_bins.start_with_fill = True
    basic_bins.waste_fills = rows
    basic_bins.noisy_waste_fills = rows.copy()
    basic_bins.real_c = rows[0].copy()
    basic_bins.c = rows[0].copy()

    for day in range(1, n_days + 1):
        basic_bins.load_filling(day)

    deposits = np.stack(basic_bins.history, axis=0)
    assert np.array_equal(deposits, rows[1:])
    assert np.array_equal(basic_bins.real_c, rows.sum(axis=0))
    assert np.array_equal(basic_bins.c, rows.sum(axis=0))
    assert basic_bins.real_c[0] == pytest.approx(60.0)
    with pytest.raises(IndexError, match="opening level"):
        basic_bins.load_filling(n_days + 1)


def test_failed_policy_is_not_an_all_zero_mean():
    """No successful sample means the policy is absent, not a zero vector."""
    cfg = _orch_cfg()
    cfg.sim.graph.n_samples = 2
    slug = policy_result_key("alns", cfg.sim)
    log_tmp = {slug: []}

    log, log_std = aggregate_final_results(log_tmp, cfg, lock=None)

    assert slug not in log
    assert log_std is not None
    assert slug not in log_std


def test_partial_samples_average_only_successes():
    """A failed sibling sample does not pull the mean toward zero."""
    cfg = _orch_cfg()
    cfg.sim.graph.n_samples = 2
    slug = policy_result_key("alns", cfg.sim)
    log_tmp = {slug: [[2.0, 4.0]]}

    log, _log_std = aggregate_final_results(log_tmp, cfg, lock=None)

    assert log[slug] == [2.0, 4.0]


@patch("logic.src.pipeline.simulations.simulator.SimulationContext")
def test_sequential_checkpoint_error_is_recorded(mock_context, tmp_path):
    """A CheckpointError is a failed sample, not a skipped one."""
    cfg = _orch_cfg()
    cfg.tracking.no_progress_bar = True
    instance = mock_context.return_value
    instance.run.side_effect = CheckpointError({"error": "disk", "policy": "alns"})

    with patch("logic.src.pipeline.simulations.simulator.ROOT_DIR", str(tmp_path)):
        log, _log_std, failed = sequential_simulations(cfg, torch.device("cpu"), [None], [[0]], "weights", lock=None)

    assert len(failed) == 1
    assert failed[0]["sample_id"] == 0
    assert failed[0]["error"] == "disk"
    assert not any(isinstance(vals, list) and vals and all(v == 0 for v in vals) for vals in log.values())


@patch("logic.src.pipeline.simulations.simulator.SimulationContext")
def test_sequential_unsuccessful_result_is_recorded(mock_context, tmp_path):
    """A day loop that stores an error result still reaches failed_log."""
    cfg = _orch_cfg()
    cfg.tracking.no_progress_bar = True
    mock_context.return_value.run.return_value = {
        "success": False,
        "policy": "alns",
        "error": "day failed",
    }

    with patch("logic.src.pipeline.simulations.simulator.ROOT_DIR", str(tmp_path)):
        _log, _log_std, failed = sequential_simulations(cfg, torch.device("cpu"), [None], [[0]], "weights", lock=None)

    assert failed[0]["sample_id"] == 0
    assert failed[0]["error"] == "day failed"


def test_parallel_failure_keeps_sample_id():
    """The failure record still carries the sample id after it is popped."""
    cfg = _orch_cfg()
    cfg.tracking.no_progress_bar = True

    class _Pool:
        def apply_async(self, fn, args=None, callback=None):
            callback({"success": False, "sample_id": 4, "policy": "alns", "error": "boom"})
            task = MagicMock()
            task.ready.return_value = True
            task.get.return_value = None
            return task

    class _Manager:
        def dict(self):
            return {}

        def list(self):
            return []

    with (
        patch("logic.src.pipeline.features.test.orchestrator.parallel_runner.monitor_tasks_until_complete"),
        patch("logic.src.pipeline.features.test.orchestrator.parallel_runner.collect_all_task_results"),
    ):
        _log, _log_std, failed = execute_and_monitor_tasks(
            _Pool(),
            cfg,
            torch.device("cpu"),
            [(None, 4, 0)],
            "weights",
            1,
            MagicMock(),
            _Manager(),
            None,
            {},
        )

    assert failed[0]["sample_id"] == 4
    assert failed[0]["sample"] == 4


def test_resume_filters_by_display_slug():
    """Resume looks up the same slug the log file uses, and drops finished samples."""
    from logic.src.pipeline.features.test import simulator_testing

    cfg = _orch_cfg(n_samples=2, resume=True, cpu_cores=1)
    cfg.sim.graph.n_samples = 2
    slug = policy_result_key("alns", cfg.sim)

    def _finished(**_kwargs):
        return [{slug: [0]}]

    with (
        patch("logic.src.pipeline.features.test.orchestrator.sequential_simulations") as mock_seq,
        patch("logic.src.pipeline.features.test.orchestrator.runs_per_policy", side_effect=_finished),
        patch("logic.src.pipeline.features.test.orchestrator.send_final_output_to_gui"),
        patch("logic.src.pipeline.features.test.orchestrator.display_log_metrics"),
    ):
        mock_seq.return_value = ({slug: [1.0]}, None, [])
        simulator_testing(cfg, 20, torch.device("cpu"))

    sample_idx_ls = mock_seq.call_args[0][3]
    assert sample_idx_ls == [[1]]


def test_any_failed_sample_exits_nonzero():
    """A recorded failure is a non-zero process status."""
    from logic.src.pipeline.features.test import simulator_testing

    cfg = _orch_cfg()
    with (
        patch("logic.src.pipeline.features.test.orchestrator.sequential_simulations") as mock_seq,
        patch("logic.src.pipeline.features.test.orchestrator.send_final_output_to_gui"),
        patch("logic.src.pipeline.features.test.orchestrator.display_log_metrics"),
    ):
        mock_seq.return_value = ({}, None, [{"policy": "alns", "sample": 0, "error": "boom"}])
        with pytest.raises(SystemExit) as excinfo:
            simulator_testing(cfg, 20, torch.device("cpu"))

    assert excinfo.value.code != 0


def test_simulator_requests_one_extra_row_with_a_stats_file():
    """The real setup: with a stats file, the waste dataset must hold n_days + 1 rows (row 0 = opening stock)."""
    from types import SimpleNamespace

    from logic.src.pipeline.simulations.states import initializing as init_mod

    for stats, expected in ((None, 3), ("stats.csv", 4)):
        sim = SimpleNamespace(
            data_distribution="gamma1",
            stats_filepath=stats,
            graph=GraphConfig(n_days=3, num_loc=5, area="riomaior", waste_type="plastic", n_samples=1),
            noise_mean=0.0,
            noise_variance=0.0,
            seed=42,
        )
        ctx = SimpleNamespace(cfg=SimpleNamespace(sim=sim), data_dir="unused", sample_id=0, coords=None)
        with patch.object(init_mod, "Bins") as bins_cls:
            init_mod.InitializingState()._initialize_bins(ctx)
        assert bins_cls.call_args.kwargs["n_days"] == expected


def test_generated_stats_horizon_runs_every_day_without_replaying_row_zero(tmp_path, mock_bins_params_loader):
    """A generated sample with n_days + 1 rows runs the whole horizon; each day deposits its own row once."""
    from logic.src.pipeline.simulations.bins import Bins

    n_days, n_bins = 4, 6
    bins = Bins(n=n_bins, data_dir=str(tmp_path), sample_dist="gamma", area="riomaior", waste_type="paper",
                n_days=n_days + 1, n_samples=1, seed=7)
    bins.set_gamma_distribution(option=0)
    bins.start_with_fill = True
    bins.set_sample_waste(0)
    opening = np.asarray(bins.waste_fills[0], dtype=float).copy()
    assert np.allclose(bins.real_c, opening)
    for day in range(1, n_days + 1):
        bins.load_filling(day)
    deposits = np.stack(bins.history, axis=0)
    assert deposits.shape[0] == n_days
    assert np.allclose(deposits, np.asarray(bins.waste_fills[1 : n_days + 1], dtype=float))
    with pytest.raises(IndexError, match="opening level"):
        bins.load_filling(n_days + 1)
