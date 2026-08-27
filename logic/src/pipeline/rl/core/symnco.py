"""
SymNCO algorithm implementation.

Reference:
    Kim, M., Park, J., & Park, J. (2022).
    Sym-NCO: Leveraging Symmetricity for Neural Combinatorial Optimization.
    Advances in Neural Information Processing Systems, 35, 1936-1949.

Attributes:
    SymNCO: SymNCO algorithm.

Example:
    >>> from logic.src.pipeline.rl.core import SymNCO
    >>> from logic.src.envs import COEnv
    >>> from logic.src.models import COPolicy
    >>> env = COEnv()
    >>> agent = COPolicy(env)
    >>> symnco = SymNCO(env, agent)
    >>> symnco
    SymNCO(env=<COEnv>, policy=<COPolicy>, baseline='rollout', actor_optimizer='adam', actor_lr=0.0001, critic_optimizer='adam', critic_lr=0.001, entropy_coef=0.01, value_loss_coef=0.5, normalize_advantage=True, enable_checkpointing=True, num_starts=10, num_augment=8, augmentation='dihedral', alpha=0.2, beta=1.0)
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from logic.src.pipeline.rl.core.pomo import POMO
from logic.src.utils.tasks.losses import (
    invariance_loss,
    problem_symmetricity_loss,
    solution_symmetricity_loss,
)


class SymNCO(POMO):
    """
    SymNCO algorithm: REINFORCE with problem/solution symmetricity and invariance losses.

    Includes support for:
    - Data Augmentation (Dihedral/Symmetric)
    - Multi-start decoding
    - Shared baseline across starts AND augmentations
    - Consistency losses

    Reference:
        Kim, M., Park, J., Kim, J., & Park, J. (2022).
        Sym-NCO: Leveraging Symmetricity for Neural Combinatorial Optimization.
        NeurIPS 2022. arXiv:2205.13209
        https://arxiv.org/abs/2205.13209

    Attributes:
        alpha: Weight for the invariance loss.
        beta: Weight for the solution symmetricity loss.
    """

    def __init__(
        self,
        alpha: float = 0.2,  # weight for invariance loss
        beta: float = 1.0,  # weight for solution symmetricity loss
        **kwargs,
    ):
        """
        Initialize SymNCO module.

        Args:
            alpha: Weight for the invariance loss (default: 0.2).
            beta: Weight for the solution symmetricity loss (default: 1.0).
            kwargs: Additional arguments to pass to the parent class (POMO).
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

    def shared_step(
        self,
        batch: TensorDict,
        batch_idx: int,
        phase: str,
    ) -> dict:
        """
        SymNCO shared step with symmetricity losses.

        Args:
            batch: Input tensor dictionary containing state information.
            batch_idx: Index of the current batch.
            phase: Phase of the training (train, val, or test).

        Returns:
            Dictionary containing training losses and metrics.
        """
        td = self.env.reset(batch)
        bs = td.batch_size[0]

        # Determine number of starts
        n_start = self.num_starts
        if n_start is None:
            n_start = self.env.get_num_starts(td) if hasattr(self.env, "get_num_starts") else td["locs"].shape[1]

        # Augmentation
        n_aug = self.num_augment
        if self.augmentation is not None:
            td = self.augmentation(td)

        # Run policy (must return init_embeds or proj_embeddings)
        out = self.policy(
            td,
            self.env,
            strategy="sampling" if phase == "train" else "greedy",
            num_starts=n_start,
        )

        # reward: [batch, n_aug, n_start]
        reward = out["reward"].view(bs, n_aug, n_start)

        # Initialize metrics for logging
        loss_ps = torch.tensor(0.0, device=reward.device)
        loss_ss = torch.tensor(0.0, device=reward.device)
        loss_inv = torch.tensor(0.0, device=reward.device)
        loss = torch.tensor(0.0, device=reward.device)
        ll = torch.tensor(0.0, device=reward.device)
        if phase == "train":
            # log_likelihood: [batch, n_aug, n_start]
            ll = out["log_likelihood"].view(bs, n_aug, n_start)

            # 1. Problem symmetricity loss (consistency across augmentations)
            loss_ps = problem_symmetricity_loss(reward, ll, dim=1)

            # 2. Solution symmetricity loss (consistency across starts)
            # Baseline is mean across starts for each augmentation
            loss_ss = solution_symmetricity_loss(reward, ll, dim=-1)

            # 3. Invariance loss (invariant representation across augmentations)
            if "proj_embeddings" in out:
                loss_inv = invariance_loss(out["proj_embeddings"], n_aug)
            else:
                loss_inv = torch.zeros((), device=reward.device)

            loss = loss_ps + self.beta * loss_ss + self.alpha * loss_inv

            if self.entropy_weight > 0 and "entropy" in out:
                loss = loss - self.entropy_weight * out["entropy"].mean()

            out["loss"] = loss
            out["loss_ps"] = loss_ps
            out["loss_ss"] = loss_ss
            out["loss_inv"] = loss_inv

            # Update metrics
            best_reward, _ = reward.view(bs, -1).max(dim=-1)
            out["reward"] = best_reward
        else:
            # During val/test, we take the best across starts and augments
            max_reward_per_aug, _ = reward.max(dim=-1)
            best_reward, _ = max_reward_per_aug.max(dim=-1)
            out["reward"] = best_reward

        # Log metrics
        self.log(f"{phase}/reward", out["reward"].mean(), prog_bar=True, sync_dist=True)
        if phase == "train":
            self.log("train/loss", loss, sync_dist=True)
            self.log("train/loss_ps", loss_ps, sync_dist=True)
            self.log("train/loss_ss", loss_ss, sync_dist=True)
            self.log("train/log_likelihood", ll.mean(), sync_dist=True)
            if "proj_embeddings" in out:
                self.log("train/loss_inv", loss_inv, sync_dist=True)
            if "entropy" in out:
                self.log("train/entropy", out["entropy"].mean(), sync_dist=True)

        return out
