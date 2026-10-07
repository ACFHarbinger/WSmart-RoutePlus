"""Tests for GFACS model."""

from unittest.mock import MagicMock

import pytest
import torch
from logic.src.models.core.gfacs import GFACS
from logic.src.models.core.gfacs.policy import GFACSPolicy
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]





class TestGFACS:
    """Tests for GFACS Model and Policy."""

    def setup_method(self):
        self.batch_size = 2
        self.num_nodes = 50
        self.env = MagicMock()
        self.env.name = "ptp"
        self.env.curriculum_graphs[0].num_loc = self.num_nodes

        # Mock env.reset
        td = TensorDict({
            "locs": torch.rand(self.batch_size, self.num_nodes, 2),
            "waste": torch.rand(self.batch_size, self.num_nodes),
            "depot": torch.rand(self.batch_size, 2),
            "current_node": torch.zeros(self.batch_size, dtype=torch.long),
            "mask": torch.ones(self.batch_size, self.num_nodes, dtype=torch.bool),
            "done": torch.zeros(self.batch_size, dtype=torch.bool),
        }, batch_size=[self.batch_size])
        self.env.reset.return_value = td

        # Mock env.get_reward
        self.env.get_reward.side_effect = lambda td, actions: -torch.ones(actions.shape[0])

        # Mock env.step to return results matching input batch size
        self.step_count = 0
        def mock_step(td):
            self.step_count += 1
            done = torch.ones(td.batch_size, dtype=torch.bool) if self.step_count >= self.num_nodes else torch.zeros(td.batch_size, dtype=torch.bool)
            return {"next": TensorDict({
                "done": done,
                "mask": torch.ones(td.batch_size[0], self.num_nodes, dtype=torch.bool),
            }, batch_size=td.batch_size)}

        self.env.step.side_effect = mock_step

        # Mock ACO
        self.mock_aco = MagicMock()
        self.mock_aco.run.return_value = (
            torch.zeros(self.batch_size, self.num_nodes, dtype=torch.long),
            torch.zeros(self.num_nodes, self.batch_size) # iter_rewards[iter] -> [batch]
        )
        self.mock_aco.local_search.return_value = (
            torch.zeros(self.batch_size * 2, self.num_nodes, dtype=torch.long),
            torch.zeros(self.batch_size * 2)
        )
        self.aco_class = MagicMock(return_value=self.mock_aco)

        self.policy = GFACSPolicy(
            embed_dim=32,
            env_name="ptp",
            num_encoder_layers=1,
            n_ants=2,
            train_with_local_search=True,
            aco_class=self.aco_class,
        )

        self.model = GFACS(
            env=self.env,
            policy=self.policy,
            train_with_local_search=True,
        )

    def test_policy_forward(self):
        """Verify GFACSPolicy forward flow."""
        td = self.env.reset()
        out = self.policy(td, self.env)

        assert "actions" in out
        assert "reward" in out
        assert "log_likelihood" in out
        assert "logZ" in out

        # GFACS uses n_ants dimension
        assert out["actions"].dim() == 3  # [batch, n_ants, n_nodes]
        assert out["reward"].dim() == 2   # [batch, n_ants]

    def test_model_forward_training(self):
        """Verify GFACS model forward flow for training."""
        td = self.env.reset()
        out = self.model(td, self.env, phase="train")

        assert "actions" in out
        assert "reward" in out
        assert "log_likelihood" in out

    def test_calculate_loss(self):
        """Verify Trajectory Balance loss calculation."""
        td = self.env.reset()
        policy_out = self.policy(td, self.env)

        # Add required mock methods for env
        self.env.get_reward.return_value = torch.rand(self.batch_size, 2)

        loss = self.model.calculate_loss(td, td, policy_out)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
