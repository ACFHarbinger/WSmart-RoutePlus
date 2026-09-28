"""Regression tests for the scenario-tree gate and the single mean writer (issue 82)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from logic.src.pipeline.simulations.actions.route_construction import RouteConstructionAction
from logic.src.pipeline.simulations.simulator import display_log_metrics, sequential_simulations
from logic.src.pipeline.simulations.states import FinishingState, SimulationContext
from logic.test.unit.pipeline.simulations.test_simulator_robustness import _orch_cfg
from logic.test.unit.pipeline.simulations.test_states import _make_test_cfg

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _route_context(adapter_flag):
    """Minimal day context that reaches adapter.execute."""
    bins = MagicMock()
    bins.c = np.zeros(2)
    bins.means = np.zeros(2)
    bins.std = np.zeros(2)
    adapter = MagicMock()
    if adapter_flag is not None:
        adapter.uses_scenario_tree = adapter_flag
    adapter.execute.return_value = ([0, 1, 0], 0.0, 0.0, None, None)
    context = {
        "full_policy": "alns",
        "config": {"policy": {"type": "alns"}},
        "mandatory": [1],
        "policy_seed": 1,
        "bins": bins,
        "day": 1,
        "sample_id": 0,
        "distance_matrix": np.zeros((3, 3)),
    }
    return adapter, context


def _run_route(adapter, context):
    generator = MagicMock()
    with (
        patch("logic.src.policies.route_construction.base.factory.RouteConstructorFactory.ensure_registered"),
        patch(
            "logic.src.policies.route_construction.base.factory.RouteConstructorFactory.get_adapter",
            return_value=adapter,
        ),
        patch(
            "logic.src.policies.route_construction.base.registry.RouteConstructorRegistry.list_route_constructors",
            return_value=["alns"],
        ),
        patch(
            "logic.src.pipeline.simulations.bins.prediction.ScenarioGenerator",
            return_value=generator,
        ),
        patch("logic.src.tracking.logging.modules.policy_viz_emit.PolicyVizStreamSession") as viz,
    ):
        viz.return_value.__enter__.return_value = None
        viz.return_value.__exit__.return_value = False
        RouteConstructionAction().execute(context)
    return generator


def test_scenario_tree_is_built_by_default():
    """An adapter that does not opt out still receives a scenario tree."""
    adapter, context = _route_context(None)
    generator = _run_route(adapter, context)
    generator.generate.assert_called_once()
    assert context["scenario_tree"] is generator.generate.return_value


def test_scenario_tree_skips_when_adapter_opts_out():
    """uses_scenario_tree False does not call ScenarioGenerator.generate."""
    adapter, context = _route_context(False)
    generator = _run_route(adapter, context)
    generator.generate.assert_not_called()
    assert context["scenario_tree"] is None
    adapter.execute.assert_called_once()


def test_finishing_writes_samples_and_daily_only(ctx_vars):
    """Mean and std are not written from the per-sample finishing state."""
    cfg = _make_test_cfg()
    ctx = SimulationContext(cfg, torch.device("cpu"), [0], 0, 0, "w", ctx_vars)
    ctx.bins = MagicMock()
    ctx.bins.inoverflow = [1, 0]
    ctx.bins.collected = [1, 0]
    ctx.bins.ncollections = [1, 0]
    ctx.bins.lost = [0, 0]
    ctx.bins.travel = 100.0
    ctx.bins.profit = 50.0
    ctx.bins.ndays = 2
    ctx.bins.get_fill_history.return_value = []
    ctx.daily_log = {"time": [1.0], "profit": [1.0]}
    ctx.tic = 0.0
    sections = []

    def _capture(_path, section, *_args, **_kwargs):
        sections.append(section)

    with (
        patch(
            "logic.src.pipeline.simulations.states.finishing.update_policy_log_section",
            side_effect=_capture,
        ),
        patch("logic.src.pipeline.simulations.states.finishing.save_matrix_to_excel"),
    ):
        FinishingState().handle(ctx)

    assert sections == ["samples", "daily"]


@pytest.fixture
def ctx_vars():
    """Shared resources the finishing state reads off the context."""
    return {"lock": MagicMock(), "counter": MagicMock(), "overall_progress": MagicMock(), "tqdm_pos": 0}


@patch("logic.src.pipeline.simulations.simulator.SimulationContext")
def test_sequential_loop_does_not_write_mean(mock_context, tmp_path):
    """The sequential loop keeps the in-memory mean and does not write it."""
    cfg = _orch_cfg()
    cfg.sim.graph.n_samples = 2
    cfg.tracking.no_progress_bar = True
    slug_holder = {}

    def _run():
        # The context resolves the slug before run(); mirror a successful sample.
        from logic.src.pipeline.simulations.day_context import policy_result_key

        slug = policy_result_key("alns", cfg.sim)
        slug_holder["slug"] = slug
        return {slug: [1.0, 2.0], "success": True}

    mock_context.return_value.run.side_effect = _run

    with (
        patch("logic.src.pipeline.simulations.simulator.ROOT_DIR", str(tmp_path)),
        patch("logic.src.pipeline.simulations.simulator.update_policy_log_section") as mock_write,
    ):
        log, log_std, failed = sequential_simulations(
            cfg, torch.device("cpu"), [None, None], [[0, 1]], "weights", lock=None
        )

    mock_write.assert_not_called()
    assert failed == []
    assert log[slug_holder["slug"]] == [1.0, 2.0]
    assert log_std[slug_holder["slug"]] == [0.0, 0.0]


def test_display_log_metrics_is_the_mean_writer(tmp_path):
    """Aggregated mean and std are written once, from display_log_metrics."""
    sections = []

    def _capture(_path, section, *_args, **_kwargs):
        sections.append(section)

    with (
        patch("logic.src.pipeline.simulations.simulator.ROOT_DIR", str(tmp_path)),
        patch(
            "logic.src.pipeline.simulations.simulator.update_policy_log_section",
            side_effect=_capture,
        ),
        patch("logic.src.pipeline.simulations.simulator.display_simulation_summary_table"),
    ):
        display_log_metrics(
            "test_out",
            20,
            2,
            1,
            "mixrmbac",
            ["alns"],
            {"alns": [1.0, 2.0]},
            {"alns": [0.0, 0.0]},
        )

    assert sections == ["mean", "std"]
