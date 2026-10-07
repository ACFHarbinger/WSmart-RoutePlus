"""
Meta-Reinforcement Learning (Meta-RL) module.

This module implements bi-level optimization where a meta-learner
adjusts environment reward weights to optimize the training progress of an inner RL agent.

Attributes:
    MetaRLModule: Meta-Reinforcement Learning module.

Example:
    >>> from logic.src.pipeline.rl.meta import MetaRLModule
    >>> meta_rl_module = MetaRLModule(agent="dummy_agent")
    >>> meta_rl_module
    <logic.src.pipeline.rl.meta.module.MetaRLModule object at 0x...>
"""

from typing import Any

import pytorch_lightning as pl
import torch
from tensordict import TensorDict

from logic.src.models.meta.weight_adjustment_rnn import WeightAdjustmentRNN
from logic.src.pipeline.rl.meta.hypernet_strategy import HyperNetworkStrategy
from logic.src.pipeline.rl.meta.registry import get_meta_strategy
from logic.src.pipeline.rl.meta.td_learning import CostWeightManager


class MetaRLModule(pl.LightningModule):
    """
    Lightning module for Meta-Reinforcement Learning.

    Wraps an RL agent (inner loop) and a meta-model (outer loop).

    Attributes:
        agent (Any): Description of agent.
        meta_strategy (Any): Description of meta_strategy.
        initial_weights (dict[str, float]): Description of initial_weights.
        automatic_optimization (bool): Whether automatic optimization is enabled.
    """

    def __init__(
        self,
        agent: Any,
        strategy: str = "rnn",
        meta_lr: float = 1e-3,
        history_length: int = 10,
        hidden_size: int = 64,
        **kwargs,
    ):
        """Initialize MetaRLModule.

        Args:
            agent: The RL agent to wrap (inner loop).
            strategy (str): The meta-learning strategy to use. Options: "rnn", "tdl", "hypernet".
            meta_lr (float): Learning rate for the meta-learner.
            history_length (int): Number of past steps to consider for the meta-learner.
            hidden_size (int): Size of the hidden layer in the meta-learner.
            kwargs: Additional keyword arguments for the meta-learner.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["agent", "env", "kwargs", "generator"])
        self.agent = agent

        # Initial weights configuration
        initial_weights = {
            "collection": 10.0,  # Default collection weight
            "cost": 1.0,
        }

        # Strategy selection
        strategy_kwargs = {
            "model_class": WeightAdjustmentRNN,
            "initial_weights": initial_weights,
            "history_length": history_length,
            "hidden_size": hidden_size,
            "lr": meta_lr,
            "device": str(self.device),
            "meta_optimizer": "adam",
            "problem": self.agent.env,  # Pass env as problem proxy
            **kwargs,
        }

        # Add strategy specific defaults
        if strategy == "tdl":
            strategy_kwargs["model_class"] = CostWeightManager
        elif strategy == "hypernet":
            strategy_kwargs["model_class"] = HyperNetworkStrategy

        self.meta_strategy = get_meta_strategy(strategy, **strategy_kwargs)

        # Disable automatic optimization for the meta-learner to control outer loop updates
        self.automatic_optimization = False

    def training_step(self, batch: TensorDict, batch_idx: int):
        """Step for Meta-RL bi-level optimization.

        Args:
            batch: Batch of data containing trajectories from the inner RL agent.
            batch_idx: Batch index.

        Returns:
            Total loss for the batch.
        """
        # 1. Inner Loop: Train the Agent
        loss = self.agent.training_step(batch, batch_idx)

        # Log inner agent loss
        if loss is not None:
            self.log("meta/inner_loss", loss)

        # 2. Outer Loop: Update Meta-Learner
        if hasattr(self.agent, "last_out"):
            out = self.agent.last_out
            reward = out["reward"].mean().item()

            # Pack metrics for strategy
            collection_val = out.get("collection", torch.tensor(0.0)).mean().item()
            cost_val = out.get("cost", torch.tensor(0.0)).mean().item()
            metrics = [collection_val, cost_val]

            # Feedback to meta-strategy
            self.meta_strategy.feedback(reward, metrics)

            # Log meta-level metrics
            self.log("meta/reward", reward)
            self.log("meta/collection", collection_val)
            self.log("meta/cost", cost_val)

            # 3. Adapt weights for next step
            new_weights = self.meta_strategy.propose_weights()

            # Update environment attributes (Duck Typing)
            for name, val in new_weights.items():
                attr_name = f"{name}_reward" if name == "collection" else f"{name}_weight"
                if hasattr(self.agent.env, attr_name):
                    setattr(self.agent.env, attr_name, val)

                self.log(f"meta/weight_{name}", val)

        return loss

    def configure_optimizers(self):
        """Configure optimizers for both loops.

        Returns:
            Dict[str, torch.optim.Optimizer]: Dictionary of optimizers.
        """
        return self.agent.configure_optimizers()

    def validation_step(self, batch: TensorDict, batch_idx: int):
        """Validation step for meta RL.

        Args:
            batch: Batch of data containing trajectories from the inner RL agent.
            batch_idx: Batch index.

        Returns:
            Total loss for the batch.
        """
        return self.agent.validation_step(batch, batch_idx)

    def test_step(self, batch: TensorDict, batch_idx: int):
        """Test step for meta RL.

        Args:
            batch: Batch of data containing trajectories from the inner RL agent.
            batch_idx: Batch index.

        Returns:
            Total loss for the batch.
        """
        return self.agent.test_step(batch, batch_idx)
