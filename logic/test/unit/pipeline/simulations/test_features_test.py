"""Tests for pipeline/features/test.py."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from logic.src.configs import Config
from logic.src.configs.envs.graph import GraphConfig
from logic.src.configs.tasks.sim import SimConfig
from logic.src.pipeline.features.test import simulator_testing

pytestmark = [pytest.mark.unit, pytest.mark.fast]



def _make_sim_cfg(**overrides):
    """Build a minimal Config for testing orchestrator."""
    graph = GraphConfig(
        area="mixrmbac",
        num_loc=20,
        waste_type="glass",
    )
    sim = SimConfig(
        policies=["policy1"],
        full_policies=["policy1"],
        data_distribution="unif",
        days=1,
        seed=42,
        output_dir="test_out",
        n_samples=1,
        resume=False,
        cpu_cores=1,
        graph=graph,
    )
    for k, v in overrides.items():
        if hasattr(sim, k):
            setattr(sim, k, v)
        elif hasattr(sim.graph, k):
            setattr(sim.graph, k, v)
    cfg = Config()
    cfg.sim = sim
    return cfg


def test_simulator_testing_single_core():
    """Verify simulation testing orchestration on a single core."""
    cfg = _make_sim_cfg()
    device = torch.device("cpu")

    with patch("logic.src.pipeline.features.test.orchestrator.sequential_simulations") as mock_seq, patch(
        "logic.src.pipeline.features.test.orchestrator.send_final_output_to_gui"
    ), patch("logic.src.pipeline.features.test.orchestrator.display_log_metrics"):
        mock_seq.return_value = ({"policy1": [0.5]}, None, [])

        simulator_testing(cfg, 20, device)

        assert mock_seq.called


def test_simulator_testing_multi_core():
    """Verify multi-core simulation testing flow with process pool mocking."""
    cfg = _make_sim_cfg(cpu_cores=2, n_samples=2)
    cfg.sim.graph.n_samples = 2
    device = torch.device("cpu")

    with patch("multiprocessing.Manager") as mock_manager, patch(
        "logic.src.pipeline.features.test.orchestrator.parallel_runner.Pool"
    ) as mock_pool_cls, patch("logic.src.pipeline.features.test.orchestrator.send_final_output_to_gui"), patch(
        "logic.src.pipeline.features.test.orchestrator.display_log_metrics"
    ), patch("logic.src.pipeline.features.test.orchestrator.output_stats"):
        mock_instance = mock_manager.return_value

        from multiprocessing.managers import DictProxy



        mock_dict_proxy = MagicMock(spec=DictProxy)
        mock_dict_proxy.items.return_value = [("policy1", [[1.0], [2.0]])]
        mock_instance.dict.return_value = mock_dict_proxy
        mock_instance.list.return_value = []

        mock_pool = mock_pool_cls.return_value
        mock_task = MagicMock()
        mock_task.ready.return_value = True
        mock_pool.apply_async.return_value = mock_task

        with patch(
            "logic.src.pipeline.features.test.orchestrator.isinstance",
            side_effect=lambda x, y: True if x is mock_dict_proxy and y is DictProxy else isinstance(x, y),
        ):
            simulator_testing(cfg, 20, device)

        assert mock_pool.apply_async.called
