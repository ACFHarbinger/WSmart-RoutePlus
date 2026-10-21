"""
Main RL4COLitModule class assembling all mixins.

Attributes:
    RL4COLitModule: Base PyTorch Lightning module for RL training.

Example:
    None
"""

from __future__ import annotations

import json
import os
from abc import ABC
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from logic.src.pipeline.learning.common.baselines import WarmupBaseline, get_baseline
from logic.src.pipeline.learning.common.epoch import apply_time_step, prepare_epoch, regenerate_dataset
from logic.src.pipeline.learning.common.pbrs_wrapper import PBRSShaper
from logic.src.tracking.logging.pylogger import get_pylogger

from .data import DataMixin, _cfg_get, _task_env_cfg
from .optimization import OptimizationMixin
from .steps import StepMixin

if TYPE_CHECKING:
    from logic.src.configs import Config
    from logic.src.interfaces.env import IEnv
    from logic.src.interfaces.policy import IPolicy
    from logic.src.policies.packing.vectorized import VectorizedSelector

logger = get_pylogger(__name__)


class RL4COLitModule(DataMixin, OptimizationMixin, StepMixin, pl.LightningModule, ABC):
    """
    Base PyTorch Lightning module for RL training.

    This module handles:
    - Metric logging
    Attributes:
        None
    """

    cfg: Optional[Config] = None
    baseline: Any

    def __init__(
        self,
        env: IEnv,
        policy: IPolicy,
        baseline: Optional[str] = "rollout",
        optimizer: str = "adam",
        optimizer_kwargs: Optional[dict] = None,
        lr_scheduler: Optional[str] = None,
        lr_scheduler_kwargs: Optional[dict] = None,
        val_dataset_path: Optional[str] = None,
        batch_size: int = 256,
        num_workers: int = 4,
        persistent_workers: bool = True,
        pin_memory: bool = False,
        mandatory_selector: Optional[VectorizedSelector] = None,
        **kwargs,
    ):
        """
        Initialize the RL4COLitModule.

        Args:
            env: The RL environment for the problem.
            policy: The neural network policy.
            baseline: Type of baseline for variance reduction ('rollout', 'exponential', 'critic', etc.).
            optimizer: Optimizer name ('adam' or 'adamw').
            optimizer_kwargs: Keyword arguments for the optimizer.
            lr_scheduler: Learning rate scheduler name ('cosine', 'step', or None).
            lr_scheduler_kwargs: Keyword arguments for the scheduler.
            val_dataset_path: Optional path to a pre-saved validation dataset.
            train_dataset_path: Optional path to a pre-saved training dataset.
            batch_size: Batch size for training and validation.
            num_workers: Number of data loading workers.
            persistent_workers: If True, the data loader will not shutdown the worker processes after a dataset has been consumed once.
            pin_memory: If True, the data loader will copy Tensors into device pinned memory before returning them.
            mandatory_selector: Optional vectorized selector for mandatory bin selection.
            kwargs: Additional keyword arguments.
        """
        pl.LightningModule.__init__(self)
        DataMixin.__init__(self)
        OptimizationMixin.__init__(self)
        StepMixin.__init__(self)

        # Explicitly save hyperparameters to handle MRO/introspection issues
        params_to_save = {
            k: v
            for k, v in locals().items()
            if k not in ["self", "__class__", "env", "policy", "mandatory_selector", "generator"]
        }
        # Avoid shadowing self.baseline (the object) with baseline (the string)
        if "baseline" in params_to_save:
            params_to_save["baseline_type"] = params_to_save.pop("baseline")
        self.save_hyperparameters(params_to_save)

        self.env = env
        self.policy = policy
        self.baseline_type = baseline
        self.train_dataset: Optional[Any] = None
        self.mandatory_selector = mandatory_selector

        # Config initialization
        self.cfg = kwargs.get("cfg")
        if isinstance(self.cfg, dict):
            self.cfg = OmegaConf.create(self.cfg)

        self.val_dataset_path = val_dataset_path
        self.train_dataset_path = kwargs.get("train_dataset_path")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory

        # Time-based training parameters
        self.train_time = kwargs.get("train_time", False)
        self._epoch_actions: List[torch.Tensor] = []

        # Optimizer params
        self.optimizer_name = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {"lr": 1e-4}
        self.lr_scheduler_name = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs or {}

        # Initialize baseline
        self._init_baseline()

        # --- Potential-Based Reward Shaping (PBRS) ----------------------------
        # Build a PBRSShaper when cfg.rl.use_pbrs is True. The shaper uses
        # rl.gamma as the discount factor (same field used by other algorithms).
        # self._pbrs is checked by StepMixin.shared_step(); when None, the code
        # path is a no-op so there is zero overhead without PBRS enabled.
        self._pbrs: Optional[PBRSShaper] = self._build_pbrs_shaper()
        # -----------------------------------------------------------------------

    def save_weights(self, path: str):
        """
        Save model weights and hyperparameters.

        Args:
            path: Path to save the weights to.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save weights and hparams to .pt file
        torch.save(
            {
                "state_dict": self.state_dict(),
                "hparams": self.hparams,
            },
            path,
        )

        # Save hparams to sidecar args.json
        args_path = os.path.join(os.path.dirname(path), "args.json")

        # Convert hparams to a serializable dict
        hparams_dict: Dict[str, Any] = {}
        for k, v in self.hparams.items():
            if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
                hparams_dict[k] = v
            else:
                hparams_dict[k] = str(v)

        try:
            with open(args_path, "w") as f:
                json.dump(hparams_dict, f, indent=4)
            logger.info(f"Saved sidecar args.json to {args_path}")
        except (OSError, TypeError, ValueError) as e:
            logger.warning(f"Could not save sidecar args.json: {e}")

        # Save full Hydra config if available
        if hasattr(self, "cfg") and self.cfg is not None:
            config_path = os.path.join(os.path.dirname(path), "config.yaml")
            try:
                OmegaConf.save(config=self.cfg, f=config_path)
            except (OSError, ImportError, TypeError, ValueError) as e:
                logger.warning(f"Could not save config.yaml: {e}")

    def _init_baseline(self):
        """Initialize baseline for advantage estimation."""
        if self.baseline_type is None:
            self.baseline_type = "rollout"

        # cfg.rl (exp_beta, bl_warmup_epochs, ...) reaches __init__ through **kwargs, which
        # save_hyperparameters nests under hparams["kwargs"]; lift it next to the named params.
        bl_kwargs = {k: v for k, v in self.hparams.items() if k != "kwargs"}
        bl_kwargs.update(self.hparams.get("kwargs", {}) or {})
        baseline = get_baseline(self.baseline_type, self.policy, **bl_kwargs)  # type: ignore[arg-type]

        # Handle warmup
        warmup_epochs = int(bl_kwargs.get("bl_warmup_epochs", 0) or 0)
        if warmup_epochs > 0:
            baseline = WarmupBaseline(baseline, warmup_epochs)

        self.baseline = baseline

    def _build_pbrs_shaper(self) -> Optional[PBRSShaper]:
        """Build a :class:`PBRSShaper` from the active configuration.

        Reads ``cfg.rl.use_pbrs``, ``cfg.rl.gamma``, ``cfg.rl.pbrs_shaping_weight``,
        and ``cfg.rl.pbrs_potential`` (all with safe defaults).  Returns ``None``
        when PBRS is disabled so :meth:`shared_step` incurs zero overhead.

        Returns:
            PBRSShaper if ``cfg.rl.use_pbrs`` is True, else None.
        """
        cfg = getattr(self, "cfg", None)
        rl_cfg = getattr(cfg, "rl", None)

        use_pbrs: bool = bool(getattr(rl_cfg, "use_pbrs", False))
        if not use_pbrs:
            return None

        # Pull γ from rl.gamma (same field consumed by DRALNS, PPO, etc.)
        gamma: float = float(getattr(rl_cfg, "gamma", 1.0))
        shaping_weight: float = float(getattr(rl_cfg, "pbrs_shaping_weight", 1.0))
        potential_key: str = str(getattr(rl_cfg, "pbrs_potential", "ptp"))

        # env_name is used to look up the registered potential function
        env_name: str = getattr(self.env, "name", potential_key)

        shaper = PBRSShaper(
            gamma=gamma,
            env_name=env_name,
            shaping_weight=shaping_weight,
        )
        logger.info(f"PBRS enabled — env={env_name}, γ={gamma}, weight={shaping_weight}, potential='{potential_key}'")
        return shaper

    def _time_training_td(self) -> Optional[Any]:
        """The unwrapped training TensorDict that time training mutates, if any."""
        ds = self.train_dataset
        if ds is not None and hasattr(self.baseline, "unwrap_dataset"):
            ds = self.baseline.unwrap_dataset(ds)
        td = getattr(ds, "data", None)
        return td if td is not None and hasattr(td, "keys") else None

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Store the carried bin fill, so a resumed time-training run continues from it."""
        td = self._time_training_td() if getattr(self, "train_time", False) else None
        if td is None:
            return
        key = "waste" if "waste" in td.keys() else "fill_level"
        if key in td.keys() and td[key].dim() == 2:
            checkpoint["time_training_fill"] = {"key": key, "fill": td[key].detach().cpu().clone()}

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Keep the stored bin fill until the training dataset exists (see on_train_epoch_start)."""
        self._resume_fill = checkpoint.get("time_training_fill")
        if getattr(self, "train_time", False) and self._resume_fill is None:
            logger.warning(
                "Checkpoint has no time-training fill (saved before this was stored): "
                "the bins restart from the initial accumulation, on the resumed day."
            )

    def _restore_time_training_fill(self) -> None:
        """Put the fill stored in a resumed checkpoint back into the training dataset, once."""
        saved = getattr(self, "_resume_fill", None)
        self._resume_fill = None
        td = self._time_training_td()
        if saved is None or td is None:
            return
        key, fill = saved["key"], saved["fill"]
        if key not in td.keys() or tuple(td.batch_size) != tuple(fill.shape[:1]):
            logger.warning("Stored time-training fill does not match the dataset; not restored.")
            return
        if td[key].dim() == 3 and f"_{key}_full" not in td.keys():
            td.set(f"_{key}_full", td[key])
        td.set(key, fill.to(td.device))
        logger.info(f"Restored the time-training fill from the checkpoint. Mean fill: {fill.mean():.3f}")

    def _time_training_day(self) -> int:
        """Zero-based simulated day of this epoch: the training graph's start_day plus the epoch.

        A stage with start_day=16 trains on days 17.. of the dataset (e.g. RL after 16 days of imitation).
        """
        _stage_graph = getattr(_task_env_cfg(self.cfg), "graph", None)
        return int(_cfg_get(_stage_graph, "start_day", 0) or 0) + self.current_epoch

    def on_train_epoch_start(self) -> None:
        """Prepare dataset for the new epoch (e.g. wrap with baseline)."""
        assert self.train_dataset is not None
        self._restore_time_training_fill()
        self.train_dataset = prepare_epoch(
            self.policy,  # type: ignore[arg-type]
            self.env,
            self.baseline,
            self.train_dataset,
            self._time_training_day(),
            phase="train",
            precompute_baseline=False,
        )

    def on_train_epoch_end(self):
        """Update baseline and regenerate dataset."""
        if hasattr(self.baseline, "epoch_callback"):
            # RolloutBaseline owns its promotion pool; validation stays reporting-only.
            self.baseline.epoch_callback(
                self.policy,
                self.current_epoch,
                val_dataset=self.val_dataset,
                env=self.env,
            )

        # Apply time-step updates (waste generation, collection reset)
        if getattr(self, "train_time", False) and self.train_dataset is not None:
            # Reconstruct dataset reference in case it was wrapped
            ds_ref = self.train_dataset
            if hasattr(self.baseline, "unwrap_dataset"):
                ds_ref = self.baseline.unwrap_dataset(ds_ref)

            _stage_graph = getattr(_task_env_cfg(self.cfg), "graph", None)
            apply_time_step(
                dataset=ds_ref,
                epoch_actions=self._epoch_actions,
                day=self._time_training_day(),
                env=self.env,
                refresh_fraction=float(_cfg_get(_stage_graph, "state_refresh_fraction", 0.0) or 0.0),
                refresh_max_days=int(_cfg_get(_stage_graph, "init_accumulation_days", 1) or 1),
            )
            self._epoch_actions.clear()

            # Log current day properties
            td = ds_ref.data if hasattr(ds_ref, "data") else ds_ref
            if isinstance(td, dict) or hasattr(td, "get"):
                key = "waste" if "waste" in td.keys() else "fill_level"
                mean_fill = td[key].mean() if key in td.keys() else 0.0
                self.log("train/current_day", float(self._time_training_day() + 1), sync_dist=True)
                self.log("train/mean_fill", mean_fill, sync_dist=True)

        # Regenerate training dataset for next epoch if configured
        if (
            self.hparams.get("regenerate_per_epoch", False)
            and self.trainer.max_epochs is not None
            and self.current_epoch < self.trainer.max_epochs - 1
            and hasattr(self.env, "generator")
        ):
            _graph = getattr(_task_env_cfg(self.cfg), "graph", None)
            n_samples = int(getattr(_graph, "n_samples", 1) or 1)
            new_dataset = regenerate_dataset(self.env, n_samples)
            if new_dataset is not None:
                self.train_dataset = new_dataset

            # Log generation
            if self.local_rank == 0:
                logger.info(f"Regenerated training dataset for epoch {self.current_epoch + 1}")
