

from unittest.mock import MagicMock, patch

import pytest
import torch
from logic.src.pipeline.learning.core.adaptive_imitation import AdaptiveImitation
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]





class TestAdaptiveImitation:
    @pytest.fixture
    def expert_policy(self):
        policy = MagicMock()
        # Mock forward return
        policy.return_value = {"actions": torch.tensor([[0, 1]]), "reward": torch.tensor([2.0])}
        return policy

    @pytest.fixture
    def mock_env(self):
        env = MagicMock()
        env.reset.return_value = TensorDict({}, batch_size=2)
        return env

    @pytest.fixture
    def adaptive_imitation_module(self, expert_policy, mock_env):
        # We need a policy mock because REINFORCE uses it
        dummy_policy = MagicMock()
        dummy_policy.return_value = {"log_likelihood": torch.tensor([-1.0, -0.5]), "actions": torch.tensor([[0, 1]])}

        with patch("logic.src.pipeline.learning.common.base.RL4COLitModule._init_baseline"), \
             patch("logic.src.pipeline.learning.core.adaptive_imitation.AdaptiveImitation._create_expert_policy", return_value=expert_policy):
            module = AdaptiveImitation(
                MagicMock(),
                "test_env",
                expert_policy=expert_policy,
                policy=dummy_policy,
                env=mock_env,
                il_weight=1.0,
                il_decay=0.5,
                patience=2,
                baseline="rollout",
            )
            # Manually set baseline to a Mock since we skipped init
            module.baseline = MagicMock()
            # Ensure no epoch_callback to avoid val_dataset access
            del module.baseline.epoch_callback

        return module

    def test_initialization(self, expert_policy, mock_env):
        dummy_policy = MagicMock()

        # Patch _init_baseline and _create_expert_policy
        with patch("logic.src.pipeline.learning.common.base.RL4COLitModule._init_baseline"), \
             patch("logic.src.pipeline.learning.core.adaptive_imitation.AdaptiveImitation._create_expert_policy"):
            module = AdaptiveImitation(
                MagicMock(), "test_env", expert_policy=expert_policy, env=mock_env, policy=dummy_policy, il_weight=0.8
            )

        assert module.il_weight == 0.8
        assert module.current_il_weight == 0.8

        # Test valid hparams (primitives)
        with patch("logic.src.pipeline.learning.common.base.RL4COLitModule._init_baseline"), \
             patch("logic.src.pipeline.learning.core.adaptive_imitation.AdaptiveImitation._create_expert_policy"):
            module = AdaptiveImitation(
                MagicMock(), "test_env", expert_policy=expert_policy, env=mock_env, policy=dummy_policy, valid_arg=1
            )
        assert "valid_arg" in module.hparams

        # Test sanitization (complex object)
        complex_obj = MagicMock()
        with patch("logic.src.pipeline.learning.common.base.RL4COLitModule._init_baseline"), \
             patch("logic.src.pipeline.learning.core.adaptive_imitation.AdaptiveImitation._create_expert_policy"):
            module = AdaptiveImitation(
                MagicMock(),
                "test_env",
                expert_policy=expert_policy,
                env=mock_env,
                policy=dummy_policy,
                complex_arg=complex_obj,
            )
        assert "complex_arg" not in module.hparams
        assert "il_weight" in module.hparams

    def test_calculate_loss(self, adaptive_imitation_module):
        td = TensorDict({"a": torch.randn(2)}, batch_size=2)
        out = {"log_likelihood": torch.tensor([-0.1, -0.2]), "reward": torch.tensor([1.0, 1.0])}

        # Patching REINFORCE.calculate_loss to allow us to check the combined logic independent of REINFORCE internals
        with patch("logic.src.pipeline.learning.core.reinforce.REINFORCE.calculate_loss", return_value=torch.tensor(0.5)):
            loss = adaptive_imitation_module.calculate_loss(td, out, 0)

            # RL loss (0.5) + IL weight (1.0) * IL loss
            # IL loss is -mean(log_likelihood from policy with expert actions)
            # policy returns log_likelihood=[-1.0, -0.5], mean is -0.75
            # IL loss is -(-0.75) = 0.75
            # Total = 0.5 + 1.0 * 0.75 = 1.25

            assert isinstance(loss, torch.Tensor)
            assert torch.isclose(loss, torch.tensor(1.25))

            adaptive_imitation_module.expert_policy.assert_called_once()

            # Verify policy call args manually to avoid RuntimeError with tensors
            args, kwargs = adaptive_imitation_module.policy.call_args
            # The policy call uses the tensor dict returned by env.reset(td), which is a new object
            assert args[0] is not td  # It's a new dict from reset
            assert args[1] is adaptive_imitation_module.env  # env
            assert torch.equal(kwargs["actions"], torch.tensor([[0, 1]]))

    def test_annealing_schedule_decay(self, adaptive_imitation_module):
        adaptive_imitation_module.trainer = MagicMock()
        # Mock no improvement
        adaptive_imitation_module.trainer.callback_metrics = {"val/reward": torch.tensor(1.0)}
        adaptive_imitation_module.best_reward = 2.0
        adaptive_imitation_module.wait = 0
        adaptive_imitation_module.patience = 2

        adaptive_imitation_module.on_train_epoch_end()
        assert adaptive_imitation_module.wait == 1
        assert adaptive_imitation_module.current_il_weight == 0.5  # 1.0 * 0.5

    def test_annealing_schedule_reheat(self, adaptive_imitation_module):
        adaptive_imitation_module.trainer = MagicMock()
        adaptive_imitation_module.trainer.callback_metrics = {"val/reward": torch.tensor(1.0)}
        adaptive_imitation_module.best_reward = 2.0

        # Already waiting 1, patience 2. Next step triggers reset.
        adaptive_imitation_module.patience = 2
        adaptive_imitation_module.wait = 1

        adaptive_imitation_module.on_train_epoch_end()  # wait becomes 2 -> reset
        assert adaptive_imitation_module.wait == 0
        assert adaptive_imitation_module.current_il_weight == 1.0  # Reset to initial (1.0)

    def test_annealing_schedule_improvement(self, adaptive_imitation_module):
        adaptive_imitation_module.trainer = MagicMock()
        adaptive_imitation_module.current_il_weight = 0.5

        # Improvement
        adaptive_imitation_module.trainer.callback_metrics = {"val/reward": torch.tensor(3.0)}
        adaptive_imitation_module.best_reward = 2.0
        adaptive_imitation_module.wait = 1

        adaptive_imitation_module.on_train_epoch_end()
        assert adaptive_imitation_module.best_reward == 3.0
        assert adaptive_imitation_module.wait == 0
        assert adaptive_imitation_module.current_il_weight == 0.25  # Decayed normally
