"""
Training/Validation step logic for RL4COLitModule.

Attributes:
    StepMixin: Mixin for training, validation, and test steps.

Example:
    None
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Union, cast

import torch
from tensordict import TensorDict

from logic.src.constants.metrics import METRIC_MAPPING
from logic.src.interfaces import ITraversable
from logic.src.pipeline.learning.common.pbrs_wrapper import PBRSShaper
from logic.src.tracking.logging.pylogger import get_pylogger
from logic.src.utils.functions.rl import ensure_tensordict

if TYPE_CHECKING:
    from logic.src.interfaces.env import IEnv
    from logic.src.interfaces.policy import IPolicy
    from logic.src.policies.packing import VectorizedSelector

logger = get_pylogger(__name__)


class StepMixin:
    """Mixin for training, validation, and test steps.

    Attributes:
        env: Environment for data generation.
        policy: Policy for data generation.
        baseline: Type of baseline for variance reduction ('rollout', 'exponential', 'critic', etc.).
        device: Device for data generation.
        mandatory_selector: Optional vectorized selector for mandatory bin selection.
        _current_baseline_val: Current baseline value.
        last_out: Last output from the policy.
    """

    def __init__(self) -> None:
        """Initialize Class.

        Args:
            None.
        """
        # Type hints
        self.env: IEnv
        self.policy: IPolicy
        self.baseline: Any
        self.device: torch.device
        self.mandatory_selector: Optional[VectorizedSelector]
        self._current_baseline_val: Any = None
        self.last_out: Any = None
        self.trainer: Any  # provided by LightningModule at runtime

    def _apply_packing(self, td: TensorDict) -> TensorDict:  # noqa: C901
        """
        Apply packing to determine which bins must be collected.

        Args:
            td: TensorDict with problem instance data.

        Returns:
            TensorDict with 'mandatory' mask added (always 2D: [B, num_loc+1]).
        """
        if self.mandatory_selector is None:
            return td

        # Get fill levels from the TensorDict
        fill_levels = None
        for key in ["waste", "fill_level"]:
            if key in td.keys():
                fill_levels = td[key]
                break

        if fill_levels is None:
            logger.warning("No fill levels found in TensorDict for packing")
            return td

        selector_kwargs: dict = {}

        # Handle 3D temporal waste [B, n_days, num_loc]:
        # extract the current day's slice and derive accumulation rates from
        # consecutive-day differences so lookahead selectors work out-of-the-box.
        if fill_levels.dim() == 3:
            waste_3d = fill_levels
            n_days_avail = waste_3d.shape[1]

            current_day_t = td.get("current_day", None)
            day_idx = int(current_day_t.reshape(-1)[0].item()) if current_day_t is not None else 0
            day_idx = min(day_idx, n_days_avail - 1)

            # Compute mean positive day-over-day accumulation rate when no
            # explicit rate key is present in the TensorDict.
            if "accumulation_rate" not in td.keys() and n_days_avail > 1:
                deltas = (waste_3d[:, 1:, :] - waste_3d[:, :-1, :]).clamp(min=0.0)
                selector_kwargs["accumulation_rates"] = deltas.mean(dim=1)  # [B, num_loc]

            fill_levels = waste_3d[:, day_idx, :]  # [B, num_loc]
        elif fill_levels.dim() == 1:
            fill_levels = fill_levels.unsqueeze(0)

        # Explicit keys in the TensorDict take priority over computed values.
        if "accumulation_rate" in td.keys():
            selector_kwargs["accumulation_rates"] = td["accumulation_rate"]
        if "std_deviation" in td.keys():
            selector_kwargs["std_deviations"] = td["std_deviation"]
        # LookaheadSelector expects current_collection_day, not current_day.
        if "current_day" in td.keys():
            selector_kwargs["current_collection_day"] = td["current_day"]

        if "locs" in td.keys():
            selector_kwargs["locs"] = td["locs"]
        elif "loc" in td.keys():
            selector_kwargs["locs"] = td["loc"]

        if "waste_history" in td.keys():
            selector_kwargs["waste_history"] = td["waste_history"]
        elif "fill_history" in td.keys():
            selector_kwargs["waste_history"] = td["fill_history"]

        # Vectorized selectors treat column 0 as the depot. The TensorDict waste is
        # customer-only [B, num_loc], so prepend a depot column (fill 0) to the fill
        # levels and per-node statistics before selecting.
        num_loc = getattr(self.env, "num_loc", None)
        if num_loc is None and hasattr(self.env, "generator"):
            num_loc = getattr(self.env.generator, "num_loc", None)
        if num_loc is not None and fill_levels.shape[-1] == num_loc:

            def _with_depot(t: torch.Tensor) -> torch.Tensor:
                return torch.cat([torch.zeros_like(t[..., :1]), t], dim=-1)

            fill_levels = _with_depot(fill_levels)
            for key in ("accumulation_rates", "std_deviations"):
                val = selector_kwargs.get(key)
                if torch.is_tensor(val) and val.shape[-1] == num_loc:
                    selector_kwargs[key] = _with_depot(val)

        mandatory_mask = self.mandatory_selector.select(fill_levels, **selector_kwargs)

        # Collapse any extra dimensions — output must be 2D [B, num_loc].
        while mandatory_mask.dim() > 2:
            mandatory_mask = mandatory_mask.any(dim=1)

        # The mask already has the depot column when it was prepended above.
        if num_loc is not None and mandatory_mask.shape[-1] == num_loc:
            depot_col = torch.zeros(mandatory_mask.shape[0], 1, dtype=torch.bool, device=mandatory_mask.device)
            mandatory_mask = torch.cat([depot_col, mandatory_mask], dim=-1)

        td["mandatory"] = mandatory_mask
        return td

    @abstractmethod
    def calculate_loss(
        self,
        td: TensorDict,
        out: dict,
        batch_idx: int,
        env: Optional[IEnv] = None,
    ) -> torch.Tensor:
        """
        Compute RL loss.

        Args:
            td: TensorDict with problem instance data.
            out: Output dictionary from the policy.
            batch_idx: Batch index.
            env: The RL environment.

        Returns:
            Loss tensor.
        """
        raise NotImplementedError

    def shared_step(  # noqa: C901
        self,
        batch: Union[TensorDict, Dict[str, Any]],
        batch_idx: int,
        phase: str,
        env: Optional[Any] = None,
    ) -> dict:
        """
        Common step for train/val/test.

        PBRS integration (training phase only):
            After ``env.reset()``, Φ(s₀) is recorded by the :class:`PBRSShaper`.
            After the policy rollout, the shaping bonus
            ``F = gamma · Φ(s_final) − Φ(s₀)`` is applied to ``out["reward"]``.
            The **base reward** is always what gets logged as ``{phase}/reward``
            to prevent metric corruption; shaped reward is used only for the
            loss calculation.
            During validation and test, PBRS is **not** applied so that
            evaluation metrics remain comparable across runs.

        Args:
            batch: TensorDict batch.
            batch_idx: Batch index.
            phase: One of "train", "val", "test".
            env: Optional env override; falls back to self.env when None.
                 Used by validation_step to select the per-eval-graph env.

        Returns:
            Output dictionary with loss, reward, etc.
        """
        _env = env if env is not None else self.env

        # Unwrap batch if it's from a baseline dataset
        batch, baseline_val = self.baseline.unwrap_batch(batch)

        # Move to device (crucial when pin_memory=False)
        if hasattr(batch, "to"):
            batch = cast(Any, batch).to(self.device)
        else:
            batch_obj: object = batch
            if isinstance(batch_obj, ITraversable):
                batch = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in batch_obj.items()}

        if baseline_val is not None:
            baseline_val = cast(Any, baseline_val).to(self.device)
        self._current_baseline_val = baseline_val

        # env.reset expects data on the environment's device.
        td = ensure_tensordict(batch, self.device)

        # Apply mandatory selector if configured
        if self.mandatory_selector is not None:
            td = self._apply_packing(td)

        td = _env.reset(td)

        # --- PBRS: record Φ(s₀) immediately after reset ----------------------
        # self._pbrs is set by the training engine when cfg.rl.use_pbrs=True.
        # It is intentionally absent (None) during val/test.
        _pbrs: Optional[PBRSShaper] = getattr(self, "_pbrs", None)
        if _pbrs is not None and phase == "train":
            _pbrs.record_initial(td)
        # ---------------------------------------------------------------------

        # Run policy
        out = self._rollout(td, _env, phase)
        # Imitation in fill_from=expert mode returns the expert's solve instead of a model rollout.
        model_rollout = out.get("source") != "expert"

        # Get updated td from rollout (if available)
        final_td = out.get("td", td)

        # --- PBRS: apply shaping bonus to training reward --------------------
        # Anti-pattern guard: we log R_base to {phase}/reward (unchanged) so
        # that learning curves are not artificially inflated.  The shaped
        # reward is only used inside calculate_loss (for advantage computation).
        if _pbrs is not None and phase == "train" and model_rollout:
            base_reward = out["reward"].detach().clone()
            shaped_reward, shaping_reward = _pbrs.apply(out["reward"], final_td)
            # Store decomposed signals in the output dict for downstream logging
            out["reward_base"] = base_reward
            out["reward_shaping"] = shaping_reward
            # Replace reward with shaped version for loss calculation
            out["reward"] = shaped_reward
            # Log shaping diagnostics
            self.log(  # type: ignore
                "train/reward_base",
                base_reward.mean(),
                sync_dist=True,
                batch_size=base_reward.shape[0],
            )
            self.log(  # type: ignore
                "train/reward_shaping",
                shaping_reward.mean(),
                sync_dist=True,
                batch_size=shaping_reward.shape[0],
            )
        # ---------------------------------------------------------------------

        # Compute loss for training
        if phase == "train":
            out["loss"] = self.calculate_loss(td, out, batch_idx, env=_env)

        # Log reward.  When PBRS is active we log R_base (not the shaped value)
        # to keep the metric comparable with non-PBRS runs and prevent leakage
        # of the shaping signal into evaluation dashboards.
        reward_for_log = out.get("reward_base", out["reward"]) if phase == "train" else out["reward"]
        batch_size = reward_for_log.shape[0]
        if not model_rollout:
            # The expert's reward is logged by the loss as train/expert_reward; there is no model
            # rollout to report. Its actions still drive time training below.
            self.last_out = out
            if phase == "train" and getattr(self, "train_time", False):
                self._epoch_actions.append(out["actions"].detach().cpu())  # type: ignore[attr-defined]
            return out
        reward_mean = reward_for_log.mean()
        self.log(  # type: ignore
            f"{phase}/reward",
            reward_mean,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        # Log granular metrics from td if available (standardized)
        # Prioritize reward_* keys as they are typically populated at the end of rollout
        for log_key, td_keys in METRIC_MAPPING.items():
            val = None
            for k in td_keys:
                if k in final_td.keys():
                    val = final_td[k]
                    break

            if val is not None:
                # Handle negative cost/overflow convention
                if (log_key in ["cost", "overflows", "initial_overflows"]) and val.mean() < 0:
                    val = -val

                self.log(  # type: ignore
                    f"{phase}/{log_key}",
                    val.mean(),
                    sync_dist=True,
                    batch_size=batch_size,
                )

        # Store for meta-learning or logging access
        self.last_out = out

        # Log policy output diagnostics
        if "log_likelihood" in out:
            self.log(  # type: ignore
                f"{phase}/log_likelihood",
                out["log_likelihood"].mean(),
                sync_dist=True,
                batch_size=batch_size,
            )
        if "entropy" in out:
            self.log(  # type: ignore
                f"{phase}/entropy",
                out["entropy"].mean(),
                sync_dist=True,
                batch_size=batch_size,
            )

        # Time-based training: accumulate actions for epoch-end update
        if phase == "train" and getattr(self, "train_time", False) and "actions" in out:
            # Type ignore is safe as we checked getattr in __init__
            self._epoch_actions.append(out["actions"].detach().cpu())  # type: ignore[attr-defined]

        return out

    def _rollout(self, td: TensorDict, env: Any, phase: str) -> dict:
        """Decode the policy: sampled in training, greedy in validation and test.

        Args:
            td: Reset environment state.
            env: Environment to decode in.
            phase: One of "train", "val", "test".

        Returns:
            Policy output with ``reward`` and ``actions``, optionally ``td``,
            ``log_likelihood`` and ``entropy``.
        """
        return self.policy(td, env, strategy="sampling" if phase == "train" else "greedy")

    def training_step(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Execute a single training step.

        Args:
            args: Positional arguments (batch, batch_idx).
            kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The computed loss.
        """
        batch: Any = args[0] if args else kwargs["batch"]
        batch_idx: int = args[1] if len(args) > 1 else kwargs.get("batch_idx", 0)

        # 1. Unwrap batch if it was wrapped by baseline (e.g. RolloutBaseline)
        if hasattr(self.baseline, "unwrap_batch"):
            td, baseline_val = self.baseline.unwrap_batch(batch)
        else:
            td, baseline_val = batch, None

        # 2. Run shared step
        out = self.shared_step(td, batch_idx, phase="train")

        # 3. Calculate loss with baseline_val if available
        self._current_baseline_val = baseline_val

        return out["loss"]

    def validation_step(self, *args: Any, **kwargs: Any) -> dict:
        """
        Execute a single validation step.

        Supports multiple validation dataloaders (from eval_graphs): the
        dataloader_idx selects the matching env from self.eval_envs.

        Args:
            args: Positional arguments (batch, batch_idx[, dataloader_idx]).
            kwargs: Additional keyword arguments.

        Returns:
            dict: Output dictionary with metrics.
        """
        batch: Any = args[0] if args else kwargs["batch"]
        batch_idx: int = args[1] if len(args) > 1 else kwargs.get("batch_idx", 0)
        dataloader_idx: int = args[2] if len(args) > 2 else kwargs.get("dataloader_idx", 0)

        eval_envs = getattr(self, "eval_envs", None) or []
        env = eval_envs[dataloader_idx] if eval_envs and dataloader_idx < len(eval_envs) else None
        return self.shared_step(batch, batch_idx, phase="val", env=env)

    def on_validation_epoch_end(self) -> None:
        """
        Aggregate rewards across multiple validation dataloaders (graphs)
        to provide a single 'val/reward' metric for checkpoint monitoring.
        """
        # Fetch per-dataloader rewards from the callback_metrics
        # Lightning stores them as 'val/reward/dataloader_idx_N'
        rewards = []
        metrics = self.trainer.callback_metrics  # type: ignore[attr-defined]
        for k, v in metrics.items():
            if k.startswith("val/reward/dataloader_idx_"):
                rewards.append(v)

        if rewards:
            # Simple average of the means of each graph size
            # This ensures 'val/reward' is available for ModelCheckpoint(monitor='val/reward')
            avg_reward = torch.stack(rewards).mean()
            self.log("val/reward", avg_reward, sync_dist=True, prog_bar=True)  # type: ignore[attr-defined]

    def test_step(self, *args: Any, **kwargs: Any) -> dict:
        """
        Execute a single test step.

        Args:
            args: Positional arguments (batch, batch_idx).
            kwargs: Additional keyword arguments.

        Returns:
            dict: Output dictionary with metrics.
        """
        batch: Any = args[0] if args else kwargs["batch"]
        batch_idx: int = args[1] if len(args) > 1 else kwargs.get("batch_idx", 0)
        return self.shared_step(batch, batch_idx, phase="test")
