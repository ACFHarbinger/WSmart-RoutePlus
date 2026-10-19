

from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch
from logic.src.pipeline.learning.meta.hrl import HRLModule
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]





class TestHRLModule:
    @pytest.fixture
    def manager(self):
        m = MagicMock()
        # Mock select_action return
        # mask_action, gate_action, manager_value
        m.select_action.return_value = (
            torch.zeros(2, 10),  # mask_action (2 batches, 10 nodes)
            torch.tensor([[1], [0]]),  # gate_action (1=dispatch, 0=wait)
            torch.tensor([0.1, 0.2]),  # value
        )

        # Mock storage lists
        m.rewards = []
        m.states_static = []
        m.states_dynamic = []
        m.states_global = []
        m.actions_mask = []
        m.actions_gate = []
        m.log_probs_mask = []
        m.log_probs_gate = []
        m.values = []
        m.target_masks = []

        # Mock forward function for PPO phase
        m.return_value = (
            torch.randn(2, 10, 2),  # mask_logits
            torch.randn(2, 2),  # gate_logits
            torch.randn(2, 1),  # values
        )

        return m

    @pytest.fixture
    def worker(self):
        w = MagicMock()
        w.return_value = {"reward": torch.tensor([10.0])}
        return w

    @pytest.fixture
    def env(self):
        e = MagicMock()
        e.reset.return_value = TensorDict(
            {
                "locs": torch.randn(2, 10, 2),
                "waste": torch.rand(2, 10),
                "visited": torch.zeros(2, 10, dtype=torch.bool),
            },
            batch_size=2,
        )
        return e

    @pytest.fixture
    def hrl_module(self, manager, worker, env):
        # Patch device property on class because it's read-only on instance
        with patch("logic.src.pipeline.learning.meta.hrl.HRLModule.device", new_callable=PropertyMock) as mock_device:
            mock_device.return_value = torch.device("cpu")
            module = HRLModule(manager=manager, worker=worker, env=env, ppo_epochs=1)
            # Mock Lightning manual optimization
            module.manual_backward = MagicMock()
            module.optimizers = MagicMock(return_value=MagicMock())
            module.clip_gradients = MagicMock()

            # Mock logger
            module.log = MagicMock()
            yield module

    def test_training_step(self, hrl_module):
        batch = TensorDict({}, batch_size=2)

        # Pre-fill manager memory to simulate "old" data for PPO update
        # because the test step here runs collection AND PPO update back-to-back.
        # Collection step fills memory once. PPO uses it.
        # We need select_action to implicitly NOT CRASH if it appends to lists.
        # The mock setup in fixture initializes empty lists.
        # select_action mock does NOT append, so memory remains empty unless we fill it or make mock have side effects.
        # But HRL reads from self.manager.rewards which is appended to IN training_step.
        # The PPO reading (old_states_*) happens AFTER collection.
        # So we can just rely on the fact that training_step reads what it just wrote?
        # NO, PPO reads `self.manager.states_static` etc. which `select_action` is supposed to fill.
        # Since we mocked `select_action`, those lists won't be filled by the call.
        # So we PRE-FILL them to simulate what select_action would have done.

        hrl_module.manager.rewards = []  # Will be appended by training_step logic (Collection phase)
        hrl_module.manager.states_static = [torch.randn(2, 10, 2)]
        hrl_module.manager.states_dynamic = [torch.randn(2, 10, 1)]
        hrl_module.manager.states_global = [torch.randn(2, 2)]
        hrl_module.manager.actions_mandatory = [torch.zeros(2, 10)]
        hrl_module.manager.actions_gate = [torch.zeros(2, 1)]
        hrl_module.manager.log_probs_mandatory = [torch.zeros(2)]
        hrl_module.manager.log_probs_gate = [torch.zeros(2)]
        hrl_module.manager.values = [torch.zeros(2, 1)]
        hrl_module.manager.target_mandatory = None

        hrl_module.training_step(batch, 0)

        # 1. Collection Phase Verification
        hrl_module.env.reset.assert_called()
        hrl_module.manager.select_action.assert_called()
        hrl_module.worker.assert_called()  # Called for dispatch==1
        assert len(hrl_module.manager.rewards) == 1  # Appended result

        # 2. PPO Optimization Verification
        # Should have called backward
        hrl_module.manual_backward.assert_called()
        opt = hrl_module.optimizers()
        opt.step.assert_called()

        # Memory cleared?
        hrl_module.manager.clear_memory.assert_called()
