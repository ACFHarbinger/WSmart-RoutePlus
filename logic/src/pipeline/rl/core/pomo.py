"""
POMO (Policy Optimization with Multiple Optima) implementation.

Reference:
    Kwon, Y. D., Choo, J., Kim, B., Yoon, I., Gwon, Y., & Min, S. (2020).
    POMO: Policy Optimization with Multiple Optima for Neural Combinatorial Optimization.
    Advances in Neural Information Processing Systems, 33, 21188-21198.

Attributes:
    POMO: POMO algorithm: REINFORCE with shared baseline and multi-start decoding.

Example:
    >>> from logic.src.pipeline.rl.core import POMO
    >>> from logic.src.pipeline.rl.envs import COEnv
    >>> from logic.src.models import COPolicy
    >>> env = COEnv()
    >>> agent = COPolicy(env)
    >>> pomo = POMO(env, agent)
    >>> pomo
    POMO(env=<COEnv>, policy=<COPolicy>, baseline='rollout', actor_optimizer='adam', actor_lr=0.0001, critic_optimizer='adam', critic_lr=0.001, entropy_coef=0.01, value_loss_coef=0.5, normalize_advantage=True, enable_checkpointing=True)
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import torch
from tensordict import TensorDict

from logic.src.data.processor.transforms import StateAugmentation
from logic.src.pipeline.rl.core.reinforce import REINFORCE


class POMO(REINFORCE):
    """
    POMO algorithm: REINFORCE with shared baseline and multi-start decoding.

    Includes support for:
    - Data Augmentation (Dihedral/Symmetric)
    - Multi-start decoding
    - Shared baseline across starts

    Reference:
        Kwon, Y. D., Choo, J., Kim, B., Yoon, I., Gwon, Y., & Min, S. (2020).
        POMO: Policy Optimization with Multiple Optima for Reinforcement Learning.
        NeurIPS 2020. arXiv:2010.16011
        https://arxiv.org/abs/2010.16011

    Attributes:
        policy: The policy network used to generate actions.
        env: The environment used to generate rewards.
        num_augment: The number of augmentations used.
        augment_fn: The augmentation function used.
        first_aug_identity: Whether the first augmentation is identity.
        num_starts: The number of starts used.
    """

    def __init__(
        self,
        num_augment: int = 8,
        augment_fn: Union[str, Callable] = "dihedral8",
        first_aug_identity: bool = True,
        num_starts: Optional[int] = None,
        mandatory_starts_only: bool = False,
        **kwargs,
    ):
        """
        Initialize POMO module.

        Args:
            num_augment: Number of augmentations to use (default: 8).
            augment_fn: Function to apply augmentations. Can be a string ('dihedral8') or a callable.
            first_aug_identity: Whether to apply identity augmentation first.
            num_starts: Number of starts to use for multi-start decoding. If None, uses the number of nodes.
            mandatory_starts_only: If True, restrict starts to mandatory nodes only (requires
                'mandatory' BoolTensor in TensorDict). Falls back to normal behaviour when no
                mandatory field is present.
            kwargs: Additional arguments to pass to the parent class.
        """
        # POMO generally uses a shared baseline of rewards across starts
        # We set baseline to 'no' because we handle the shared baseline logic in calculate_loss
        kwargs["baseline"] = "none"
        super().__init__(**kwargs)

        self.num_augment = num_augment
        self.num_starts = num_starts
        self.augment_fn = augment_fn
        self.mandatory_starts_only = mandatory_starts_only

        self.augmentation: Optional[StateAugmentation]
        if self.num_augment > 1:
            self.augmentation = StateAugmentation(
                num_augment=num_augment,
                augment_fn=augment_fn,
                first_aug_identity=first_aug_identity,
            )
        else:
            self.augmentation = None

    def _resolve_starts(
        self,
        td: TensorDict,
    ):
        """Determine n_start and optional forced start-node indices.

        When ``mandatory_starts_only`` is True and a ``mandatory`` BoolTensor
        is present in *td*, each POMO rollout is pinned to a distinct mandatory
        node.  The j-th copy of instance i starts from the j-th mandatory node
        of that instance (0-indexed).  Instances with fewer mandatory nodes than
        the batch-maximum have their last mandatory index repeated as padding.

        Returns:
            Tuple of:
            - n_start (int): number of rollout starts to use.
            - start_nodes (Optional[torch.Tensor]): shape ``[batch * n_start]``
              with the forced first-action index per trajectory, or ``None`` when
              normal POMO start-node selection should be used.
        """
        if self.mandatory_starts_only:
            mandatory = td.get("mandatory", None)
            if mandatory is not None and mandatory.any():
                # mandatory: [batch, n_nodes] bool
                mandatory_counts = mandatory.sum(dim=-1)  # [batch]
                n_start = int(mandatory_counts.max().item())
                if n_start > 0:
                    rows = []
                    for i in range(td.batch_size[0]):
                        idx = mandatory[i].nonzero(as_tuple=False).squeeze(-1)
                        if idx.numel() == 0:
                            # Safety: no mandatory nodes — use node 1 (skip depot at 0)
                            idx = torch.tensor([1], device=mandatory.device)
                        if idx.numel() < n_start:
                            pad = idx[-1:].expand(n_start - idx.numel())
                            idx = torch.cat([idx, pad])
                        rows.append(idx[:n_start])
                    # [batch, n_start] -> [batch * n_start] (matches batchify ordering)
                    start_nodes = torch.stack(rows, dim=0).reshape(-1)
                    return n_start, start_nodes

        # Fallback: standard POMO start resolution
        n_start = self.num_starts
        if n_start is None:
            n_start = self.env.get_num_starts(td) if hasattr(self.env, "get_num_starts") else td["locs"].shape[1]
        return n_start, None

    def shared_step(
        self,
        batch: TensorDict,
        batch_idx: int,
        phase: str,
    ) -> dict:
        """
        POMO shared step with augmentation and multi-start.

        Args:
            batch: Batch of states from the dataset.
            batch_idx: Index of the batch.
            phase: Phase of the training (train, val, or test).

        Returns:
            Dictionary containing loss and other metrics.
        """
        td = self.env.reset(batch)
        bs = td.batch_size[0]

        # Determine number of starts and optional forced start nodes
        n_start, start_nodes = self._resolve_starts(td)

        # Augmentation during val/test (usually not during training in basic POMO,
        # but RL4CO allows it. We'll follow RL4CO: training = no aug, unless specified)
        n_aug = self.num_augment if phase != "train" else 1
        if phase != "train" and self.augmentation is not None:
            td = self.augmentation(td)

        # Run policy with multi-start
        out = self.policy(
            td,
            self.env,
            strategy="sampling" if phase == "train" else "greedy",
            num_starts=n_start,
            start_nodes=start_nodes,
        )

        # Reshape rewards and log_probs if we have multiple starts/augments
        # out['reward'] is [batch * n_aug * n_start]
        # We want to unbatchify it to [batch, n_aug, n_start]

        # Note: reward is [batch_size] in normal REINFORCE
        # Here it is expanded.
        reward = out["reward"].view(bs, n_aug, n_start)

        if phase == "train":
            # Shared baseline calculation: advantage = reward - mean(reward across starts)
            # shape: [batch, 1, n_start]
            log_likelihood = out["log_likelihood"].view(bs, 1, n_start)

            # Shared baseline for each instance in batch
            baseline_val = reward.mean(dim=-1, keepdim=True)  # [batch, 1, 1]
            advantage = reward - baseline_val  # [batch, 1, n_start]

            # Standard RL4CO normalization
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            # Loss = -mean(advantage * log_p)
            loss = -(advantage.detach() * log_likelihood).mean()

            if self.entropy_weight > 0 and "entropy" in out:
                loss = loss - self.entropy_weight * out["entropy"].mean()

            out["loss"] = loss

            # Update metrics with the best start reward for logging
            best_reward, _ = reward.max(dim=-1)  # [batch, 1]
            out["reward"] = best_reward.squeeze(-1)  # [batch]

            # Log training diagnostics
            self.log(f"{phase}/loss", loss, sync_dist=True)
            self.log(f"{phase}/advantage", advantage.mean(), sync_dist=True)
            self.log(f"{phase}/baseline", baseline_val.mean(), sync_dist=True)
            self.log(f"{phase}/log_likelihood", log_likelihood.mean(), sync_dist=True)
            if "entropy" in out:
                self.log(f"{phase}/entropy", out["entropy"].mean(), sync_dist=True)
        else:
            # During val/test, we take the best across starts and augments
            # reward is [batch, n_aug, n_start]
            max_reward_per_aug, _ = reward.max(dim=-1)  # [batch, n_aug]
            best_reward, _ = max_reward_per_aug.max(dim=-1)  # [batch]
            out["reward"] = best_reward

        # Log metrics
        self.log(f"{phase}/reward", out["reward"].mean(), prog_bar=True, sync_dist=True)

        return out
