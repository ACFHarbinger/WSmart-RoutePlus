"""
Greedy rollout baseline with significance-based updates.

Attributes:
    RolloutBaseline: Greedy rollout baseline with significance-based updates.

Example:
    >>> from logic.src.pipeline.rl.common.baselines import RolloutBaseline
    >>> baseline = RolloutBaseline()
    >>> baseline.eval()
    tensor(0.0)
"""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, Optional, cast

import numpy as np
import torch
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader, Dataset

from logic.src.constants.routing import DEFAULT_ROLLOUT_BATCH_SIZE
from logic.src.data.datasets import BaselineDataset, TensorDictDataset, tensordict_collate_fn
from logic.src.tracking.logging.pylogger import get_pylogger
from logic.src.utils.data.rl_utils import safe_td_copy
from logic.src.utils.functions.rl import ensure_tensordict

from .base import Baseline

logger = get_pylogger(__name__)


class RolloutBaseline(Baseline):
    """
    Greedy rollout baseline with significance-based updates.

    Uses greedy decoding with a frozen policy copy as baseline.
    The baseline policy is updated only if the current policy outperforms it
    significantly according to a T-test.

    Attributes:
        baseline_policy: Copy of the policy used as baseline.
        update_every: Frequency of baseline updates.
        bl_alpha: Significance level for T-test.
    """

    def __init__(
        self,
        policy: Optional[nn.Module] = None,
        update_every: int = 1,
        bl_alpha: float = 0.05,
        **kwargs,
    ):
        """
        Initialize RolloutBaseline.



        Args:
            policy: Policy to use as initial baseline (will be copied).
            update_every: Update baseline every N epochs.
            bl_alpha: Significance level for T-test to decide on updates.
            kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.update_every = update_every
        self.bl_alpha = bl_alpha

        self.comparison_dataset = None
        # The environment is runtime state, not a trainable child module.
        object.__setattr__(self, "comparison_env", None)
        self.comparison_seed = 0
        self.comparison_generation = 0
        self.comparison_size = 0
        self._comparison_restored = False
        self.baseline_policy = None
        if policy is not None:
            self.setup(policy)

    def setup(self, policy: nn.Module):
        """Copy policy for baseline.

        Args:
            policy: Policy to use as initial baseline (will be copied).
        """
        self.baseline_policy = copy.deepcopy(policy)  # type: ignore[assignment]
        if self.baseline_policy is not None:
            self.baseline_policy.eval()  # type: ignore[misc]
            for param in self.baseline_policy.parameters():
                param.requires_grad = False

    def configure_comparison(self, env: Any, sample_size: int, seed: int = 0) -> None:
        """Create a private pool using the training environment's distribution.

        Reporting validation graphs and their datasets are never used for
        promotion. The pool size stays fixed across accepted replacements.
        """
        if sample_size < 2:
            raise ValueError("Rollout comparison requires at least two instances")
        if self.comparison_env is not None:
            # A second Trainer.fit must not reset a pool retained after rejection
            # or generated after a promotion. Construct a new baseline for a
            # different training distribution, rather than silently reusing it.
            return
        object.__setattr__(self, "comparison_env", copy.deepcopy(env))
        if not self._comparison_restored:
            self.comparison_size = sample_size
            self.comparison_seed = int(seed)
            self.comparison_generation = 0
        self.comparison_dataset = self._generate_comparison(self.comparison_generation)

    def get_extra_state(self) -> Dict[str, int]:
        """Save the deterministic pool identity without storing all instances."""
        return {"seed": self.comparison_seed, "generation": self.comparison_generation, "size": self.comparison_size}

    def set_extra_state(self, state: Dict[str, int]) -> None:
        """Recreate the saved pool lazily after the environment is configured."""
        if state.get("size", 0) >= 2:
            self.comparison_seed = state["seed"]
            self.comparison_generation = state["generation"]
            self.comparison_size = state["size"]
            self.comparison_dataset = None
            self._comparison_restored = True

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Checkpoints predating the private pool contain only policy weights.
        state_dict.setdefault(prefix + "_extra_state", {})
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _generate_comparison(self, generation: int) -> Dataset:
        """Generate reproducibly without consuming the training RNG streams."""
        generator = copy.deepcopy(self.comparison_env.generator)
        seed = self.comparison_seed + generation
        if hasattr(generator, "rng"):
            generator.rng = np.random.default_rng(seed)
        if isinstance(getattr(generator, "generator", None), torch.Generator):
            device = getattr(generator.generator, "device", "cpu")
            generator.generator = torch.Generator(device=device).manual_seed(seed)
        # Some generators use global RNGs in addition to their private streams.
        py_state, np_state = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                random.seed(seed)
                np.random.seed(seed % (2**32))
                data = generator(batch_size=self.comparison_size)
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
        return TensorDictDataset(data)

    def train(self, mode: bool = True) -> RolloutBaseline:
        """Override train to keep baseline_policy in eval mode.

        Note:
            This triggers a PyTorch Lightning warning about modules being in
            eval mode at the start of training. This is INTENTIONAL for
            RolloutBaseline as the baseline policy must remain frozen and
            greedy to provide a stable reference for advantage estimation.

        Args:
            mode: Whether to set training mode (True) or evaluation mode (False).

        Returns:
            RolloutBaseline: Self.
        """
        super().train(mode)
        if self.baseline_policy is not None:
            self.baseline_policy.eval()
        return self

    def _rollout(self, policy: nn.Module, td_or_dataset: Any, env: Optional[Any] = None) -> torch.Tensor:
        """Run greedy rollout on a batch or dataset.

        Args:
            policy: Policy to use for rollout.
            td_or_dataset: TensorDict or Dataset to evaluate.
            env: Environment instance.

        Returns:
            Rewards from the rollout.
        """
        if isinstance(td_or_dataset, Dataset):
            return self._rollout_dataset(policy, td_or_dataset, env)
        return self._rollout_batch(policy, td_or_dataset, env)

    @staticmethod
    def _invoke_greedy(policy: nn.Module, td: Any, env: Optional[Any]) -> Dict[str, Any]:
        """Invoke either policy API, preserving the legacy reward tuple contract."""
        if hasattr(policy, "set_strategy"):
            cast(Any, policy).set_strategy("greedy")
            result = cast(Any, policy)(td)
            return {"reward": result[0]} if isinstance(result, tuple) else result
        if env is None:
            raise ValueError("Environment (env) is required for RolloutBaseline evaluation")
        return cast(Any, policy)(td, env, strategy="greedy")

    def _rollout_dataset(self, policy: nn.Module, dataset: Any, env: Optional[Any] = None) -> torch.Tensor:
        """Run greedy rollout on a complete dataset.

        Args:
            policy: Policy to use for rollout.
            dataset: Dataset to evaluate.
            env: Environment instance.

        Returns:
            Rewards from the rollout.
        """
        if env is None:
            raise ValueError("Environment (env) is required for RolloutBaseline evaluation")

        # Determine strict batch size from environment
        batch_size = DEFAULT_ROLLOUT_BATCH_SIZE  # Default from constants
        if hasattr(env, "batch_size") and len(env.batch_size) > 0:
            batch_size = int(env.batch_size[0])

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=tensordict_collate_fn,
            num_workers=0,
        )
        rewards = []
        policy.eval()
        with torch.no_grad():
            for batch in loader:
                # Get device from policy
                device = next(policy.parameters()).device
                td_data = ensure_tensordict(batch, device)

                try:
                    if hasattr(env, "reset"):
                        td_data = env.reset(td_data)
                except (OSError, ValueError, KeyError) as e:
                    logger.warning(f"Environment reset failed: {e}")
                    raise e

                # Padding logic for last batch
                real_size = td_data.batch_size[0]
                if real_size != batch_size:
                    pad_size = batch_size - real_size
                    # Repeat the first element to pad safely
                    padding = td_data[0].expand(pad_size)
                    padding_safe = safe_td_copy(padding)
                    td_data = torch.cat([td_data, padding_safe], 0)

                out = self._invoke_greedy(policy, td_data, env)

                # Unpad rewards
                if real_size != batch_size:
                    out["reward"] = out["reward"][:real_size]

                rewards.append(out["reward"].cpu())

        if len(rewards) == 0:
            return torch.tensor([], device="cpu")
        return torch.cat(rewards)

    def _rollout_batch(self, policy: nn.Module, td: Any, env: Optional[Any] = None) -> torch.Tensor:
        """Run greedy rollout on a single batch.

        Args:
            policy: Policy to use for rollout.
            td: TensorDict to evaluate.
            env: Environment instance.

        Returns:
            Rewards from the rollout.
        """
        # Note: deepcopy can be expensive, but ensure_tensordict helps standardize
        device = next(policy.parameters()).device
        td_data = ensure_tensordict(td, device)
        td_copy = copy.deepcopy(td_data)

        policy.eval()
        with torch.no_grad():
            out = self._invoke_greedy(policy, td_copy, env)

        return out["reward"]

    def wrap_dataset(
        self,
        dataset: Any,
        policy: Optional[nn.Module] = None,
        env: Optional[Any] = None,
    ) -> Any:
        """Wrap the dataset with rollout baseline values.

        Args:
            dataset: Dataset to wrap.
            policy: Policy to use for rollout.
            env: Environment instance.

        Returns:
            Dataset with rollout baseline values.
        """
        # Compatibility: handle positional arguments if called as (policy, dataset, env)
        if isinstance(dataset, nn.Module):
            # Probably called as wrap_dataset(policy, dataset, env)
            policy_arg = dataset
            dataset_arg = policy  # Actually the 2nd arg
            env_arg = env
            dataset = dataset_arg
            policy = policy_arg
            env = env_arg
            dataset = dataset_arg

        if env is None:
            # We can't actually rollout without env, so return original
            return dataset

        # Baseline values must come from the frozen baseline policy; the live policy
        # passed by prepare_epoch is only a fallback before setup() has run.
        p = self.baseline_policy if self.baseline_policy is not None else policy
        if p is None:
            return dataset

        print("Evaluating baseline on dataset...")
        bl_vals = self._rollout(p, dataset, env)
        return BaselineDataset(dataset, bl_vals.view(-1, 1))

    def eval(self, td: Any, reward: torch.Tensor, env: Optional[Any] = None) -> torch.Tensor:  # type: ignore[override]
        """
        Compute baseline value.

        Args:
            td: TensorDict to evaluate.
            reward: Reward tensor.
            env: Environment instance.

        Returns:
            Baseline value.
        """
        # If we have a baseline policy, run it to get the baseline value
        if self.baseline_policy is not None:
            # Note: This is computationally expensive if done every step
            # Ideally use wrap_dataset/unwrap_batch flow
            with torch.no_grad():
                td = ensure_tensordict(td, next(iter(self.baseline_policy.parameters())).device)
                return self._rollout(self.baseline_policy, td, env)

        return torch.zeros_like(reward)

    def epoch_callback(
        self,
        policy: nn.Module,
        epoch: int,
        val_dataset: Optional[Any] = None,
        env: Optional[Any] = None,
    ):
        """Update baseline policy if current policy improves significantly.

        Args:
            policy: Policy to evaluate.
            epoch: Epoch number.
            val_dataset: Dataset for evaluation.
            env: Environment instance.
        """
        # val_dataset/env remain accepted for API compatibility; reporting
        # validation must never become the promotion pool implicitly.
        if (epoch + 1) % self.update_every != 0:
            return
        if self.comparison_dataset is None and self.comparison_env is not None and self._comparison_restored:
            self.comparison_dataset = self._generate_comparison(self.comparison_generation)
        if self.comparison_dataset is None or self.comparison_env is None or self.baseline_policy is None:
            logger.warning("Skipping rollout promotion: comparison pool or baseline is not initialized")
            return
        if hasattr(self.comparison_env, "to"):
            object.__setattr__(self, "comparison_env", self.comparison_env.to(next(policy.parameters()).device))
        was_training = policy.training
        strategies = {key: getattr(policy, key) for key in ("strategy", "_strategy") if hasattr(policy, key)}
        try:
            candidate_vals = self._rollout(policy, self.comparison_dataset, self.comparison_env).reshape(-1)
            baseline_vals = self._rollout(self.baseline_policy, self.comparison_dataset, self.comparison_env).reshape(
                -1
            )
        finally:
            policy.train(was_training)
            for key, value in strategies.items():
                setattr(policy, key, value)
        if (
            candidate_vals.shape != baseline_vals.shape
            or candidate_vals.numel() != self.comparison_size
            or not torch.isfinite(candidate_vals).all()
            or not torch.isfinite(baseline_vals).all()
        ):
            logger.warning("Skipping rollout promotion: invalid paired rewards")
            return
        candidate_mean, baseline_mean = candidate_vals.mean().item(), baseline_vals.mean().item()
        if candidate_mean <= baseline_mean:
            return
        _, p_val = stats.ttest_rel(candidate_vals.cpu().numpy(), baseline_vals.cpu().numpy())
        if np.isfinite(p_val) and p_val / 2 < self.bl_alpha:
            # Prepare first: failed generation must not partially promote policy.
            next_generation = self.comparison_generation + 1
            dataset = self._generate_comparison(next_generation)
            self.setup(policy)
            self.comparison_dataset = dataset
            self.comparison_generation = next_generation
            logger.info(f"Update baseline: {baseline_mean:.4f} -> {candidate_mean:.4f} (p={p_val / 2:.4f})")
