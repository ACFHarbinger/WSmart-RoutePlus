"""
Neural Network-Based Meta-Learning for Reward Weight Optimization.

Attributes:
    RewardWeightOptimizer: Meta-learner for reward weight optimization.

Example:
    >>> from logic.src.pipeline.learning.meta.weight_optimizer import RewardWeightOptimizer
    >>> optimizer = RewardWeightOptimizer(initial_weights={"weight1": 0.5, "weight2": 0.5})
    >>> optimizer.propose_weights()
    {'weight1': 0.5, 'weight2': 0.5}
"""

from typing import Any, Dict, List, Optional, Tuple, Union, cast

import torch
from torch import nn

from logic.src.pipeline.learning.meta.weight_strategy import WeightAdjustmentStrategy

try:
    from logic.src.tracking.core.run import get_active_run
except ImportError:
    get_active_run = None  # type: ignore[assignment]


class RewardWeightOptimizer(WeightAdjustmentStrategy):
    """
    Manager for the meta-learning process of optimizing reward weights
    based on historical performance data. Uses an RNN to propose updates.

    Attributes:
        device: Device to run the meta-learning on.
        weight_names: Names of the weights.
        num_weights: Number of weights.
        history_length: Length of the history to use for training.
        meta_batch_size: Batch size for training.
        current_weights: Current weights.
        min_weights: Minimum values for the weights.
        max_weights: Maximum values for the weights.
        num_perf_metrics: Number of performance metrics.
        meta_model: PyTorch model for meta-learning.
        optimizer: Optimizer for the meta-model.
        weight_history: History of weights.
        performance_history: History of performance metrics.
        reward_history: History of rewards.
        meta_step: Step count for meta-learning.

    """

    def __init__(
        self,
        model_class: nn.Module,
        initial_weights: Dict[str, float],
        history_length: int = 10,
        hidden_size: int = 64,
        lr: float = 0.001,
        device: str = "cpu",
        meta_batch_size: int = 8,
        min_weights: Optional[List[float]] = None,
        max_weights: Optional[List[float]] = None,
        meta_optimizer: str = "adam",
        **kwargs,
    ):
        """Initialize AdaptiveWeightOptimizer.

        Args:
            model_class: PyTorch model class for meta-learning.
            initial_weights: Initial weights.
            history_length: Length of the history to use for training.
            hidden_size: Hidden layer size for the meta-model.
            lr: Learning rate for the meta-model.
            device: Device to run the meta-learning on.
            meta_batch_size: Batch size for training.
            min_weights: Minimum values for the weights.
            max_weights: Maximum values for the weights.
            meta_optimizer: Optimizer for the meta-model.
            kwargs: Additional keyword arguments.
        """
        self.device = torch.device(device)
        self.weight_names = list(initial_weights.keys())
        self.num_weights = len(self.weight_names)
        self.history_length = history_length
        self.meta_batch_size = meta_batch_size

        # Initialize current weights
        self.current_weights = torch.tensor(list(initial_weights.values()), dtype=torch.float32, device=self.device)

        # Set weight constraints
        self.min_weights = torch.tensor(
            min_weights if min_weights else [0.0] * self.num_weights,
            dtype=torch.float32,
            device=self.device,
        )
        self.max_weights = torch.tensor(
            max_weights if max_weights else [float("inf")] * self.num_weights,
            dtype=torch.float32,
            device=self.device,
        )

        # Initialize feature extraction parameters
        self.num_perf_metrics = len(self.weight_names) + 1  # overflows, length, waste, total_reward

        # Define input size: current weights + performance metrics
        input_size = self.num_weights + self.num_perf_metrics

        # Initialize DNN meta-learner
        self.meta_model = model_class(input_size=input_size, hidden_size=hidden_size, output_size=self.num_weights).to(
            self.device
        )

        # Initialize optimizer for the meta-learner
        optimizer_params = [{"params": self.meta_model.parameters(), "lr": lr}]
        self.optimizer = {
            "adam": torch.optim.Adam(optimizer_params),
            "adamw": torch.optim.AdamW(optimizer_params),
            "rmsprop": torch.optim.RMSprop(optimizer_params),
        }.get(meta_optimizer.lower(), torch.optim.Adam(optimizer_params))

        # Initialize histories for weights and performance
        self.weight_history: List[torch.Tensor] = []
        self.performance_history: List[torch.Tensor] = []
        self.reward_history: List[float] = []

        # Keep track of step for logging
        self.meta_step = 0

    def propose_weights(self, context: Optional[Any] = None) -> dict[str, float]:
        """Propose weights based on context.

        Args:
            context: Context information for weight proposal.

        Returns:
            dict[str, float]: Proposed weights for the next epoch.
        """
        self.update_weights_internal()
        return self.get_current_weights()

    def feedback(
        self,
        reward: float,
        metrics: Dict[str, float],
        day: Optional[Any] = None,
        step: Optional[Any] = None,
    ):
        """Provide feedback to weight optimizer.

        Args:
            reward: Reward for the last epoch.
            metrics: Metrics for the last epoch.
            day: Day of the last epoch.
            step: Step of the last epoch.
        """
        self.update_histories(metrics, reward)
        self.meta_learning_step()

    def update_histories(self, performance_metrics, reward):
        """Update performance histories.

        Args:
            performance_metrics: Description of performance_metrics.
            reward: Description of reward.
        """
        self.weight_history.append(self.current_weights.clone().detach().cpu())
        p_tensor = torch.as_tensor(performance_metrics, dtype=torch.float32).cpu()
        self.performance_history.append(p_tensor)
        self.reward_history.append(reward)

        if len(self.weight_history) > self.history_length:
            self.weight_history.pop(0)
            self.performance_history.pop(0)
            self.reward_history.pop(0)

    def prepare_meta_learning_batch(self) -> Tuple[Optional[torch.Tensor], Union[int, torch.Tensor]]:
        """Prepare batch for meta learning.

        Returns:
            Tuple[Optional[torch.Tensor], Union[int, torch.Tensor]]: Batch of features and targets.
        """
        if len(self.weight_history) < 2:
            return None, 0

        seq_len = min(self.history_length, len(self.weight_history) - 1)
        features = []
        targets = []

        for start_idx in range(len(self.weight_history) - seq_len):
            seq_features = []
            for i in range(start_idx, start_idx + seq_len):
                w_hist = self.weight_history[i]
                p_hist = self.performance_history[i]
                if p_hist.dim() > 1:
                    p_hist = p_hist.view(-1)
                combined = torch.cat([w_hist, p_hist])
                seq_features.append(combined)

            target = self.reward_history[start_idx + seq_len]
            features.append(torch.stack(seq_features))
            targets.append(target)

            if len(features) >= self.meta_batch_size:
                break

        if features:
            features = torch.stack(features).to(self.device)  # type: ignore[assignment]
            targets = torch.tensor(targets, dtype=torch.float32).to(self.device)  # type: ignore[assignment]
            return features, targets  # type: ignore[return-value]  # stacked feature and target tensors

        return None, 0

    def meta_learning_step(self) -> Optional[float]:
        """Execute meta learning step.

        Returns:
            float: Loss for the meta learning step.
        """
        features, targets = self.prepare_meta_learning_batch()
        if features is None:
            return None

        self.meta_model.train()
        self.optimizer.zero_grad()
        weight_adjustments, _ = self.meta_model(features)

        pred_weights = features[:, -1, : self.num_weights] + weight_adjustments
        pred_weights = torch.clamp(pred_weights, self.min_weights, self.max_weights)

        weights_variance = torch.var(pred_weights, dim=0, unbiased=False).sum()
        reward_pred = -cast(torch.Tensor, targets)  # targets are the reward tensor after the empty check

        alpha = 0.8
        loss = alpha * reward_pred.mean() + (1 - alpha) * weights_variance

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.meta_model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.meta_step += 1

        # Log to WSTracker
        run = get_active_run() if get_active_run is not None else None
        if run is not None:
            run.log_metric("meta/optimizer_loss", loss.item(), step=self.meta_step)
            run.log_metric("meta/optimizer_step", self.meta_step, step=self.meta_step)

        return loss.item()

    def recommend_weights(self) -> torch.Tensor:
        """Recommend weights based on learning.

        Returns:
            torch.Tensor: Recommended weights.
        """
        if len(self.weight_history) < self.history_length:
            return self.current_weights

        self.meta_model.eval()
        with torch.no_grad():
            seq_features = []
            for i in range(-self.history_length, 0):
                w_hist = self.weight_history[i]
                p_hist = self.performance_history[i]
                if p_hist.dim() > 1:
                    p_hist = p_hist.view(-1)
                combined = torch.cat([w_hist, p_hist]).to(self.device)
                seq_features.append(combined)

            features = torch.stack(seq_features).unsqueeze(0).to(self.device)
            weight_adjustments, _ = self.meta_model(features)
            new_weights = self.current_weights + weight_adjustments.squeeze(0)
            new_weights = torch.clamp(new_weights, self.min_weights, self.max_weights)
            return new_weights

    def update_weights_internal(self, force_update: bool = False) -> bool:
        """Update weights internally.

        Args:
            force_update: Whether to force the update.

        Returns:
            bool: True if the weights were updated, False otherwise.
        """
        if len(self.weight_history) < self.history_length and not force_update:
            return False
        self.current_weights = self.recommend_weights()
        return True

    def get_current_weights(self) -> dict[str, float]:
        """Get current weights.

        Returns:
            dict[str, float]: Current weights.
        """
        return {name: self.current_weights[i].item() for i, name in enumerate(self.weight_names)}
