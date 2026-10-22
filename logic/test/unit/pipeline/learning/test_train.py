"""Tests for the modernized PyTorch Lightning training pipeline."""

from typing import Any, Dict, cast
from unittest.mock import MagicMock, patch

import pytest
import torch
from logic.src.configs import Config
from logic.src.pipeline.features.train import create_model, run_training
from logic.src.pipeline.learning.common import WSTrainer

pytestmark = [pytest.mark.unit, pytest.mark.slow]



class TestTrainingOrchestration:
    """Tests for train.py orchestration logic."""

    @pytest.mark.unit
    def test_create_model_reinforce(self):
        """Verify create_model creates a REINFORCE LightningModule."""
        cfg = Config()
        cfg.device = "cpu"
        cfg.rl.algorithm = "reinforce"
        cfg.train.policy.model.name = "am"

        with patch("logic.src.pipeline.features.train.get_env"), patch(
            "logic.src.pipeline.features.train.AttentionModelPolicy"
        ):
            model = create_model(cfg)
            from logic.src.pipeline.learning.core.reinforce import REINFORCE

            assert isinstance(model, REINFORCE)

    @pytest.mark.unit
    def test_create_model_ppo(self):
        """Verify create_model creates a PPO LightningModule."""
        cfg = Config()
        cfg.device = "cpu"
        cfg.rl.algorithm = "ppo"
        cfg.train.policy.model.name = "am"

        with patch("logic.src.pipeline.features.train.get_env"), patch(
            "logic.src.pipeline.features.train.AttentionModelPolicy"
        ), patch("logic.src.models.common.critic_network.policy.create_critic_from_actor"):
            model = create_model(cfg)
            from logic.src.pipeline.learning.core.ppo import PPO

            assert isinstance(model, PPO)

    @pytest.mark.unit
    @patch("logic.src.pipeline.features.train.engine.WSTrainer")
    @patch("logic.src.pipeline.features.train.engine.create_model")
    def test_run_training_flow(self, mock_create, mock_trainer_cls):
        """Verify run_training initializes model and calls trainer.fit."""
        from omegaconf import OmegaConf
        cfg = OmegaConf.create(cast(Any, Config()))
        cfg.device = "cpu"
        # Reduce data size to prevent hanging
        if not cfg.train.env.curriculum_graphs:
            from logic.src.configs.learning.envs.graph import GraphConfig
            cfg.train.env.curriculum_graphs.append(OmegaConf.create(cast(Any, GraphConfig())))
        cfg.train.env.curriculum_graphs[0].n_samples = 10

        mock_model = MagicMock()
        mock_create.return_value = mock_model

        mock_trainer = mock_trainer_cls.return_value
        mock_trainer.callback_metrics = {"val/reward": torch.tensor(0.5)}

        reward = run_training(cfg) # pyrefly: ignore [bad-argument-type]

        assert reward == 0.5
        mock_create.assert_called_once()
        mock_trainer.fit.assert_called_once_with(mock_model, ckpt_path=None)

    @pytest.mark.unit
    def test_create_model_complex_algos(self):
        """Test creation of models for various algorithms (SAPO, SymNCO, POMO)."""
        from omegaconf import OmegaConf
        for algo in ["sapo", "symnco", "pomo"]:
            cfg = Config()
            # Convert to DictConfig to allow overriding with mocks if needed,
            # or just use plain objects.
            cfg_dict = cast(Dict[str, Any], OmegaConf.to_container(OmegaConf.create(cast(Any, cfg)), resolve=True))
            cfg_dict["device"] = "cpu"
            cfg_dict["rl"]["algorithm"] = algo
            cfg_dict["train"]["policy"]["model"]["name"] = "am"

            # Setup specific sub-configs if needed
            if algo == "symnco":
                cfg_dict["rl"]["symnco"] = {"alpha": 0.2, "beta": 1.0}
            if algo == "pomo":
                cfg_dict["rl"]["pomo"] = {
                    "num_augment": 8,
                    "augment_fn": "dihedral8",
                    "num_starts": 50,
                    "mandatory_starts_only": False,
                }

            # Create a DictConfig from the modified dict
            cfg = OmegaConf.create(cfg_dict)

            with patch("logic.src.pipeline.features.train.get_env"), \
                 patch("logic.src.pipeline.features.train.AttentionModelPolicy"), \
                 patch("logic.src.models.common.critic_network.policy.create_critic_from_actor"):

                model = create_model(cast(Any, cfg))
                assert model is not None
                assert model.__class__.__name__.lower() == algo

    @pytest.mark.unit
    def test_run_hpo_optuna(self):
        """Verify run_hpo calls OptunaHPO runner."""
        cfg = Config()
        cfg.device = "cpu"
        cfg.hpo.method = "tpe"

        with patch("logic.src.pipeline.features.train.hpo.OptunaHPO") as mock_optuna_cls:
            mock_runner = mock_optuna_cls.return_value
            mock_runner.run.return_value = 0.75

            from logic.src.pipeline.features.train import run_hpo
            reward = run_hpo(cfg)

            assert reward == 0.75
            mock_optuna_cls.assert_called_once()


class TestWSTrainer:
    """Tests for the custom WSTrainer class."""

    @pytest.mark.unit
    def test_wstrainer_init_defaults(self):
        """Verify WSTrainer sets up default callbacks and loggers."""
        with patch("logic.src.pipeline.learning.common.WSTrainer._create_default_logger"):
            trainer = WSTrainer(max_epochs=5, accelerator="cpu")
            assert trainer.max_epochs == 5
            # Check for ModelCheckpoint and RichProgressBar in callbacks
            callback_types = [type(c) for c in cast(Any, trainer).callbacks]
            from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar

            assert ModelCheckpoint in callback_types
            assert RichProgressBar in callback_types

    @pytest.mark.unit
    def test_wstrainer_custom_logger(self):
        """Verify WSTrainer accepts a custom logger."""
        from pytorch_lightning.loggers import Logger

        custom_logger = MagicMock(spec=Logger)
        trainer = WSTrainer(logger=custom_logger, accelerator="cpu")
        assert trainer.logger == custom_logger


class TestLitModuleInterfaces:
    """Tests for RL4COLitModule and algorithm implementations."""

    @pytest.mark.unit
    def test_reinforce_calculate_loss(self):
        """Simple test for REINFORCE loss calculation logic."""
        from logic.src.pipeline.learning.core.reinforce import REINFORCE
        from tensordict import TensorDict



        # Mock policy and baseline
        policy = MagicMock()
        env = MagicMock()

        model = REINFORCE(policy=policy, env=env)
        mock_baseline = MagicMock(spec=torch.nn.Module)
        model.baseline = mock_baseline
        model.baseline.eval.return_value = torch.tensor([1.0])

        td = TensorDict({}, batch_size=[1])
        out = {"reward": torch.tensor([2.0]), "log_likelihood": torch.tensor([-0.5], requires_grad=True)}

        loss = model.calculate_loss(td, out, 0, env=env)
        # advantage = 2 - 1 = 1.0 (after normalization might be different but here we check it runs)
        assert loss is not None
        assert loss.requires_grad
