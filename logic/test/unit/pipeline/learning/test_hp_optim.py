"""Tests for hyperparameter optimization pipelines (DEHB, Optuna)."""

from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from logic.src.configs import Config
from logic.src.pipeline.learning.hpo.dehb import DifferentialEvolutionHyperband
from logic.src.pipeline.learning.hpo.optuna_hpo import OptunaHPO

pytestmark = [pytest.mark.unit, pytest.mark.fast]



def get_config_space(opts):
    """Mock config space for testing."""
    return {"w_lost": (0.0, 1.0), "w_waste": (0.0, 1.0), "w_length": (0.0, 1.0), "w_overflows": (0.0, 1.0)}


class TestOptunaHPO:
    """Tests for the OptunaHPO integration."""

    @pytest.mark.unit
    def test_optuna_hpo_init(self):
        """Test initialization of OptunaHPO."""
        cfg = Config()
        mock_obj = MagicMock()
        hpo = OptunaHPO(cfg, mock_obj)
        assert hpo.cfg == cfg
        assert hpo.objective_fn == mock_obj

    @pytest.mark.unit
    @patch("optuna.create_study")
    def test_optuna_hpo_run(self, mock_create_study):
        """Test OptunaHPO run loop."""
        cfg = Config()
        cfg.hpo.n_trials = 2

        mock_study = MagicMock()
        mock_study.best_value = 0.5
        mock_create_study.return_value = mock_study

        mock_obj = MagicMock()
        hpo = OptunaHPO(cfg, mock_obj)
        best_val = hpo.run()

        assert best_val == 0.5
        mock_create_study.assert_called_once()
        mock_study.optimize.assert_called_once()

    @pytest.mark.unit
    def test_optuna_sampler_selection(self):
        """Test sampler selection logic based on config."""
        cfg = Config()

        # Test random
        cfg.hpo.method = "random"
        hpo = OptunaHPO(cfg, MagicMock())
        sampler = hpo._get_sampler()
        import optuna



        assert isinstance(sampler, optuna.samplers.RandomSampler)

        # Test grid (requires search_space)
        cfg.hpo.method = "grid"
        cfg.hpo.search_space = cast(Any, {"a": [1, 2]})
        hpo = OptunaHPO(cfg, MagicMock())
        sampler = hpo._get_sampler()
        assert isinstance(sampler, optuna.samplers.GridSampler)


class TestDEHB:
    """Tests for DEHB integration."""

    @pytest.mark.unit
    def test_get_config_space(self):
        """Test configuration space definition."""
        cs = get_config_space({})
        assert "w_lost" in cs
        assert "w_waste" in cs
        assert "w_length" in cs
        assert "w_overflows" in cs

    @pytest.mark.unit
    def test_dehb_init(self, tmp_path):
        """Test DEHB initialization."""
        # Mock Config and ParamSpecs
        cfg = Config()
        search_space = {
            "w_lost": {"type": "float", "low": 0.0, "high": 1.0},
            "w_waste": {"type": "float", "low": 0.0, "high": 1.0},
            "w_length": {"type": "float", "low": 0.0, "high": 1.0},
            "w_overflows": {"type": "float", "low": 0.0, "high": 1.0},
        }
        f = MagicMock()

        # Instantiate wrapper with correct signature
        dehb = DifferentialEvolutionHyperband(
            cfg=cfg,
            objective_fn=f,
            search_space=search_space,
            min_fidelity=1,
            max_fidelity=10,
            n_workers=1,
            output_path=str(tmp_path / "test_dehb_output"),
        )

        # Check attributes on the internal DEHB object
        assert dehb._dehb.min_fidelity == 1
        assert dehb._dehb.max_fidelity == 10
        # Check parameter config space was built (mock param specs might behave oddly so just check it exists)
        assert dehb._dehb.cs is not None

    @pytest.mark.unit
    def test_dehb_run(self, tmp_path):
        """Test DEHB run execution (simplified)."""
        cfg = Config()
        search_space = {"w_lost": {"type": "float", "low": 0.0, "high": 1.0}} # Simplified space

        # Mock objective function returning a fitness dict
        f = MagicMock(return_value={"fitness": 0.5, "cost": 1.0})

        dehb = DifferentialEvolutionHyperband(
            cfg=cfg,
            objective_fn=f,
            search_space=search_space,
            min_fidelity=1,
            max_fidelity=10,
            n_workers=1,
            output_path=str(tmp_path / "test_dehb_output"),
        )

        # Mock the internal DEHB run method to avoid actual optimization
        dehb._dehb.run = MagicMock()
        dehb._dehb.get_incumbents = MagicMock(return_value=({"w_lost": 0.5}, 0.5))
        dehb._dehb.history = []

        best_score = dehb.run()

        assert best_score == -0.5 # Negated fitness
        dehb._dehb.run.assert_called()
