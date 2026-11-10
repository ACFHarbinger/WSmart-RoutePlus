"""
Training health monitoring callback for PyTorch Lightning (§A.4).

Detects gradient explosions, reward stagnation, and entropy collapse during
RL training and emits structured warnings for the Studio Training Monitor.

Attributes:
    TrainingHealthCallback: Lightning callback for automated health guardrails.

Example:
    >>> from logic.src.pipeline.callbacks.pytorch.training_health import TrainingHealthCallback
    >>> trainer = L.Trainer(callbacks=[TrainingHealthCallback()])
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Optional

import pytorch_lightning as pl
from loguru import logger
from pytorch_lightning.callbacks import Callback

from logic.src.tracking.logging.modules.training_health_emit import emit_training_health_alert


class TrainingHealthCallback(Callback):
    """
    Automated training instability detection for the Lightning RL pipeline.

    Raises structured warnings when:
    - Gradient norm exceeds ``max_grad_norm_threshold`` (default 100)
    - Reward moving average stagnates for ``stagnation_epochs`` (default 50)
    - Policy entropy drops below ``min_entropy_threshold`` (default 0.01)

    Alerts are logged via loguru and emitted as ``TRAINING_HEALTH_START:`` stdout
    markers for the WSmart-Route Studio Training Monitor.
    """

    def __init__(
        self,
        max_grad_norm_threshold: float = 100.0,
        min_entropy_threshold: float = 0.01,
        stagnation_epochs: int = 50,
        stagnation_epsilon: float = 1e-3,
        reward_key: str = "train/reward",
        entropy_key: str = "train/entropy",
        grad_norm_key: str = "train/grad_norm",
        alert_cooldown_epochs: int = 5,
    ) -> None:
        """
        Initialize health thresholds and tracking state.

        Args:
            max_grad_norm_threshold: Alert when gradient norm exceeds this value.
            min_entropy_threshold: Alert when entropy drops below this value.
            stagnation_epochs: Consecutive stagnant epochs before alerting.
            stagnation_epsilon: Minimum relative reward change to count as progress.
            reward_key: Lightning metric key for reward tracking.
            entropy_key: Lightning metric key for entropy tracking.
            grad_norm_key: Lightning metric key for gradient norm tracking.
            alert_cooldown_epochs: Minimum epochs between repeat alerts of same code.
        """
        super().__init__()
        self.max_grad_norm_threshold = max_grad_norm_threshold
        self.min_entropy_threshold = min_entropy_threshold
        self.stagnation_epochs = stagnation_epochs
        self.stagnation_epsilon = stagnation_epsilon
        self.reward_key = reward_key
        self.entropy_key = entropy_key
        self.grad_norm_key = grad_norm_key
        self.alert_cooldown_epochs = alert_cooldown_epochs

        self._reward_history: Deque[float] = deque(maxlen=stagnation_epochs + 1)
        self._stagnant_epochs: int = 0
        self._last_alert_epoch: Dict[str, int] = {}
        self._log_path: Optional[str] = None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Resolve JSONL path under the Lightning log directory."""
        # trainer.log_dir broadcasts from rank 0 under DDP, so every rank must read it.
        log_dir = getattr(trainer, "log_dir", None)
        if not trainer.is_global_zero:
            return
        if log_dir:
            self._log_path = f"{log_dir}/training_health.jsonl"

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Check per-step gradient norm and entropy after each training batch."""
        if not trainer.is_global_zero:
            return

        metrics = trainer.callback_metrics
        grad_norm = self._metric_value(metrics, self.grad_norm_key)
        if grad_norm is not None and grad_norm > self.max_grad_norm_threshold:
            self._raise_alert(
                trainer,
                code="grad_norm_explosion",
                severity="critical",
                details={
                    "grad_norm": grad_norm,
                    "threshold": self.max_grad_norm_threshold,
                },
            )

        entropy = self._metric_value(metrics, self.entropy_key)
        if entropy is not None and entropy < self.min_entropy_threshold:
            self._raise_alert(
                trainer,
                code="entropy_collapse",
                severity="warning",
                details={
                    "entropy": entropy,
                    "threshold": self.min_entropy_threshold,
                },
            )

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Check reward stagnation at epoch boundaries."""
        if not trainer.is_global_zero:
            return

        reward = self._metric_value(trainer.callback_metrics, self.reward_key)
        if reward is None:
            reward = self._metric_value(trainer.callback_metrics, "val/reward")
        if reward is None:
            return

        self._reward_history.append(reward)
        if len(self._reward_history) < 2:
            return

        prev = self._reward_history[-2]
        delta = abs(reward - prev)
        denom = max(abs(prev), 1e-8)
        relative_change = delta / denom

        if relative_change < self.stagnation_epsilon:
            self._stagnant_epochs += 1
        else:
            self._stagnant_epochs = 0

        if self._stagnant_epochs >= self.stagnation_epochs:
            self._raise_alert(
                trainer,
                code="reward_stagnation",
                severity="warning",
                details={
                    "reward": reward,
                    "stagnant_epochs": self._stagnant_epochs,
                    "epsilon": self.stagnation_epsilon,
                    "history": list(self._reward_history),
                },
            )
            self._stagnant_epochs = 0

    def _raise_alert(
        self,
        trainer: pl.Trainer,
        code: str,
        severity: str,
        details: Dict[str, Any],
    ) -> None:
        """Emit alert when cooldown permits."""
        epoch = trainer.current_epoch
        last = self._last_alert_epoch.get(code, -self.alert_cooldown_epochs)
        if epoch - last < self.alert_cooldown_epochs:
            return

        self._last_alert_epoch[code] = epoch
        step = trainer.global_step
        msg = f"[{severity.upper()}] {code} at epoch {epoch + 1}, step {step}: {details}"
        logger.warning(msg)

        emit_training_health_alert(
            code=code,
            severity=severity,
            epoch=epoch,
            step=step,
            details=details,
            log_path=self._log_path,
        )

    @staticmethod
    def _metric_value(metrics: Dict[str, Any], key: str) -> Optional[float]:
        """Extract a float metric from Lightning callback metrics."""
        if key not in metrics:
            return None
        val = metrics[key]
        if hasattr(val, "item"):
            val = val.item()
        try:
            return float(val)
        except (TypeError, ValueError):
            return None
