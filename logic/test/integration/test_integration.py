"""
Integration tests for the WSmart-Route pipeline (Training & Simulation).
Modernized for the PyTorch Lightning architecture.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from logic.src.configs import Config
from logic.src.configs.envs.graph import GraphConfig
from logic.src.envs.problems import VRPP
from logic.src.pipeline.features.train import run_training
from logic.src.pipeline.simulations.simulator import sequential_simulations
from logic.src.policies.route_construction.base.registry import RouteConstructorRegistry
from omegaconf import OmegaConf

pytestmark = [pytest.mark.integration, pytest.mark.slow]





class TestIntegrationTraining:
    """Integration tests for Training workflows."""

    @pytest.mark.parametrize("problem_name", ["vrpp", "cvrpp"])
    def test_run_training_integration(self, problem_name, tmp_path):
        """Test standard training orchestration with real entry point (mocked trainer)."""
        cfg = Config()
        cfg.device = "cpu"
        cfg.train.env.name = problem_name
        if not cfg.train.env.curriculum_graphs:
            cfg.train.env.curriculum_graphs.append(GraphConfig())
        cfg.train.env.curriculum_graphs[0].n_days = 1
        cfg.tracking.log_dir = str(tmp_path / "logs")
        cfg.train.final_model_path = str(tmp_path / "final.pt")
        # Safety net: reduce data size in case mock fails
        cfg.train.env.curriculum_graphs[0].n_samples = 10

        # We mock WSTrainer to avoid actual training in unit tests,
        # but check if run_training flow completes.
        # Must patch where it is IMPORTED in engine.py
        with patch("logic.src.pipeline.features.train.engine.WSTrainer") as mock_trainer_cls, patch(
            "logic.src.pipeline.features.train.create_model"
        ):
            mock_trainer = mock_trainer_cls.return_value
            mock_trainer.callback_metrics = {"val/reward": torch.tensor(0.1)}

            reward = run_training(cfg)
            assert isinstance(reward, float)
            mock_trainer.fit.assert_called_once()


class TestIntegrationSimulation:
    """Integration tests for Simulation workflows."""

    def _run_sim(self, opts):
        """Helper to run simulation."""
        # Mock indices/samples as required by sequential_simulations signature
        indices_ls = [list(range(opts["size"]))]
        sample_idx_ls = [[0] for _ in opts["policies"]]
        lock = MagicMock()

        cfg = OmegaConf.create({
            "device": opts["device"],
            "sim": {**opts, "full_policies": opts["policies"], "config_path": None, "noise_mean": 0.0, "noise_variance": 1.0, "policy_configs": {}, "run_name": None, "output_dir": "output", "data_distribution": "gamma1", "graph": {**opts, "area": "Rio Maior", "num_loc": opts["size"], "size": opts["size"], "n_days": opts.get("days", 1), "n_samples": opts.get("n_samples", 1)}},
            "tracking": {"no_progress_bar": True, "log_file": None, "log_dir": "logs", "log_level": "INFO"},
            "env": {"name": "vrpp", "graph_size": opts["size"]},
            "model": {},
            "train": {"train_time": False}
        })
        return sequential_simulations(cfg, opts["device"], indices_ls, sample_idx_ls, opts["model_path"], lock) # type: ignore[call-arg]

    @pytest.mark.unit
    def test_sim_sequential_basic(self, sim_opts):
        """Test basic sequential simulation run."""
        # Ensure we have some policies to test
        sim_opts["policies"] = ["policy_regular_emp"]
        MockPolicy = MagicMock()
        RouteConstructorRegistry.register("policy_regular_emp")(MockPolicy)
        RouteConstructorRegistry.register("none_regular_none")(MockPolicy)
        instance = MockPolicy.return_value
        instance.execute.return_value = ([0, 1, 2, 0], 10.0, None)
        try:
            log, log_std, failed = self._run_sim(sim_opts)
            assert not failed
            assert any("regular" in k for k in log.keys())
        finally:
            if "policy_regular_emp" in RouteConstructorRegistry._registry:
                del RouteConstructorRegistry._registry["policy_regular_emp"]
            if "none_regular_none" in RouteConstructorRegistry._registry:
                del RouteConstructorRegistry._registry["none_regular_none"]

    @pytest.mark.unit
    @patch("logic.src.pipeline.simulations.states.initializing.setup_model")
    def test_sim_policy_neural_mock(self, mock_setup, sim_opts):
        """Test Neural policy integration in simulation."""
        # Register a Mock class that accepts any arguments (like config)
        MockPolicy = MagicMock()
        RouteConstructorRegistry.register("meanstd0.84_am_emp")(MockPolicy)
        RouteConstructorRegistry.register("none_meanstd0.84_am_none")(MockPolicy)

        instance = MockPolicy.return_value
        instance.execute.return_value = ([0, 1, 2, 0], 10.0, None)

        try:
            mock_setup.return_value = (MagicMock(), {})
            # mock_exec is useless if we don't use NeuralPolicy class.

            sim_opts["policies"] = ["meanstd0.84_am_emp"]
            log, _, failed = self._run_sim(sim_opts)
            assert not failed
            assert any("am" in k for k in log.keys())
        finally:
            if "meanstd0.84_am_emp" in RouteConstructorRegistry._registry:
                del RouteConstructorRegistry._registry["meanstd0.84_am_emp"]
            if "none_meanstd0.84_am_none" in RouteConstructorRegistry._registry:
                del RouteConstructorRegistry._registry["none_meanstd0.84_am_none"]


class TestIntegrationProblems:
    """Tests for problem physics and state transitions (End-to-End)."""

    def test_vrpp_physics_flow(self):
        """Verify VRPP physics behaves correctly in an end-to-end state update."""
        batch = {
            "loc": torch.rand(1, 10, 2),
            "depot": torch.rand(1, 2),
            "waste": torch.rand(1, 10),
            "max_waste": torch.ones(1, 10),
        }
        state = VRPP.make_state(batch)
        assert state.td["i"] == 0

        # Take an action
        action = torch.tensor([1])
        next_state = state.update(action)
        assert next_state.td["i"] == 1
        assert next_state.td["visited"][:, 1].all()
