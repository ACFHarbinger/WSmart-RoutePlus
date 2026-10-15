"""
Imitation Learning algorithm implementation.
Trains a policy to mimic an expert solver (e.g., HGS, Local Search).

Attributes:
    ImitationLearning: Imitation Learning algorithm implementation.

Example:
    >>> from logic.src.pipeline.rl.core import ImitationLearning
    >>> from logic.src.envs import COEnv
    >>> from logic.src.models import COPolicy
    >>> env = COEnv()
    >>> agent = COPolicy(env)
    >>> imitation_learning = ImitationLearning(env, agent)
    >>> imitation_learning
    ImitationLearning(env=<COEnv>, policy=<COPolicy>, baseline='none', actor_optimizer='adam', actor_lr=0.0001, critic_optimizer='adam', critic_lr=0.001, entropy_coef=0.01, value_loss_coef=0.5, normalize_advantage=True, enable_checkpointing=True)
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
from tensordict import TensorDict

from logic.src.configs.rl.policies import (
    ACOConfig,
    ProfitableNNConfig,
    RLSConfig,
    VectorALNSConfig,
    VectorHGSALNSConfig,
    VectorHGSConfig,
    VectorILSConfig,
)
from logic.src.pipeline.rl.common.base import RL4COLitModule
from logic.src.pipeline.rl.core.losses import (
    js_divergence_loss,
    kl_divergence_loss,
    nll_loss,
    reverse_kl_divergence_loss,
    weighted_nll_loss,
)
from logic.src.policies.vector.alns import VectorizedALNS
from logic.src.policies.vector.ant_colony_system import VectorizedACOPolicy
from logic.src.policies.vector.hgs import VectorizedHGS
from logic.src.policies.vector.hgs_alns import VectorizedHGSALNS
from logic.src.policies.vector.iterated_local_search import IteratedLocalSearchPolicy
from logic.src.policies.vector.profitable_nearest_neighbour import ProfitableNearestNeighbourPolicy
from logic.src.policies.vector.random_local_search import RandomLocalSearchPolicy
from logic.src.utils.data.rl_utils import safe_td_copy


class ImitationLearning(RL4COLitModule):
    """
    Imitation Learning / Supervised Learning.

    Minimizes the difference between the policy's predicted action probabilities
    and the actions taken by an expert solver (e.g., HGS, Local Search).

    Features:
    - On-the-fly expert data generation or pre-computed dataset support.
    - Configurable Loss (NLL, Weighted NLL, KL, JS).
    - Support for multiple expert types.

    Attributes:
        expert_policy: The expert policy used to generate expert actions.
        expert_name: The name of the expert policy.
        loss_fn: The loss function used to compute the loss.
        loss_fn_name: The name of the loss function.
    """

    def __init__(
        self,
        policy_config: Any,  # Expert policy configuration (HGSConfig, ALNSConfig, etc.)
        env_name: str,
        loss_fn: str = "nll",
        seed: int = 42,
        device: str = "cpu",
        fill_from: str = "model",
        **kwargs,
    ):
        """
        Initialize ImitationLearning module.

        Args:
            policy_config: Expert policy configuration (HGSConfig, ALNSConfig, etc.).
            env_name: Environment name for the policy.
            loss_fn: Loss function to use ('nll', 'weighted_nll', 'kl', 'reverse_kl', 'js').
            seed: Random seed for reproducibility.
            device: Device to run the policy on.
            fill_from: Whose routes empty the bins between time-training days: "model" (the
                model's sampled rollout, which trained the paper models) or "expert" (the
                expert's tours; the model rollout is then skipped).
            kwargs: Additional arguments to pass to the parent class.
        """
        if fill_from not in ("model", "expert"):
            raise ValueError(f"fill_from must be 'model' or 'expert', not {fill_from!r}")
        # Baseline is not used in IL, but we keep the structure
        kwargs["baseline"] = "none"

        # Exclude non-serializable objects from hyperparameters
        self.save_hyperparameters(ignore=["policy_config", "env", "policy", "kwargs", "generator"])

        super().__init__(**kwargs)

        # Create expert policy from config
        self.expert_policy = self._create_expert_policy(policy_config, env_name, seed, device)
        self.expert_name = type(policy_config).__name__.replace("Config", "").lower()

        # Map loss functions
        self._loss_map = {
            "nll": nll_loss,
            "weighted_nll": weighted_nll_loss,
            "kl": kl_divergence_loss,
            "reverse_kl": reverse_kl_divergence_loss,
            "js": js_divergence_loss,
        }
        self.loss_fn_name = loss_fn
        self.loss_fn = self._loss_map.get(loss_fn, nll_loss)
        self.fill_from = fill_from

    def _create_expert_policy(self, policy_config: Any, env_name: str, seed: int, device: str) -> Any:
        """Create expert policy from configuration.

        Args:
            policy_config: Expert policy configuration (HGSConfig, ALNSConfig, etc.).
            env_name: Environment name for the policy.
            seed: Random seed for reproducibility.
            device: Device to run the policy on.

        Returns:
            Initialized expert policy instance.
        """
        # Map config types to policy classes
        config_to_policy_map = {
            VectorHGSConfig: VectorizedHGS,
            VectorALNSConfig: VectorizedALNS,
            VectorHGSALNSConfig: VectorizedHGSALNS,
            RLSConfig: RandomLocalSearchPolicy,
            VectorILSConfig: IteratedLocalSearchPolicy,
            ACOConfig: VectorizedACOPolicy,
            ProfitableNNConfig: ProfitableNearestNeighbourPolicy,
        }

        # Get the policy class for this config type
        policy_cls = config_to_policy_map.get(type(policy_config))
        if policy_cls is None:
            raise ValueError(
                f"Unknown policy config type: {type(policy_config)}. "
                f"Supported types: {list(config_to_policy_map.keys())}"
            )

        # Convert config to dict and add env_name
        config_dict = asdict(policy_config)
        config_dict["env_name"] = env_name
        config_dict["device"] = device
        config_dict["seed"] = seed

        # Create and return the policy
        return policy_cls(**config_dict)

    def _rollout(self, td: TensorDict, env: Any, phase: str) -> dict:
        """Skip the model rollout in training when the expert's routes empty the bins.

        With ``fill_from="expert"`` the training step returns the expert's solve, marked with
        ``source="expert"``: its actions drive time training and the loss reuses it. The
        model's sampled decode, which only fed logging and the fill update, is not run, so the
        sampling random numbers it drew are no longer drawn.

        Args:
            td: Reset environment state.
            env: Environment to decode in.
            phase: One of "train", "val", "test".

        Returns:
            Policy or expert output.
        """
        if phase != "train" or self.fill_from != "expert":
            return super()._rollout(td, env, phase)
        out = dict(self._expert_solve(td))
        out["source"] = "expert"
        return out

    def _expert_solve(self, td: TensorDict) -> dict:
        """Expert actions and reward for the reset state ``td``."""
        if self.expert_policy is None:
            raise ValueError("expert_policy must be provided for ImitationLearning.")
        with torch.no_grad():
            return self.expert_policy(td, self.env)

    def calculate_loss(
        self,
        td: TensorDict,
        out: dict,
        batch_idx: int,
        env: Any = None,
    ) -> torch.Tensor:
        """
        Compute Imitation Learning loss.

        Args:
            td: Current environment state.
            out: Policy output dictionary (not used for expert selection).
            batch_idx: Batch index.
            env: Environment instance.

        Returns:
            Scalar loss tensor.
        """
        # The policy's own rollout is only logged here. Detaching it frees its graph before the
        # teacher-forced pass, which otherwise holds both graphs at once (out of memory at 350 bins).
        for key in ("log_likelihood", "entropy"):
            if isinstance(out.get(key), torch.Tensor):
                out[key] = out[key].detach()

        # 1. Generate Expert Solutions (already done by the step when fill_from="expert")
        expert_out = out if out.get("source") == "expert" else self._expert_solve(td)
        expert_actions = expert_out["actions"]

        # 2. Clone state and reset to ensure clean state for teacher forcing
        td_clone = safe_td_copy(td)
        td_clone = self.env.reset(td_clone)

        # 3. Teacher Forcing: Get policy log probabilities for expert actions
        teacher_out = self.policy(td_clone, self.env, actions=expert_actions)
        log_likelihood = teacher_out["log_likelihood"]

        # 4. Compute Loss using injected loss function
        if self.loss_fn_name == "weighted_nll":
            expert_reward = expert_out.get("reward", torch.ones_like(log_likelihood))
            loss = weighted_nll_loss(log_likelihood, weights=expert_reward, reduction="mean")
        elif self.loss_fn_name in ["kl", "reverse_kl", "js"]:
            # Distillation requires target distribution (logits or log_probs)
            target_log_probs = expert_out.get("log_probs", None)
            if target_log_probs is None:
                print(
                    f"[ImitationLearning] Warning: {self.loss_fn_name} requires target log_probs. Falling back to NLL."
                )
                loss = nll_loss(log_likelihood, reduction="mean")
            else:
                loss = self._loss_map[self.loss_fn_name](log_likelihood, target_log_probs, reduction="mean")  # type: ignore[operator]
        else:
            loss = nll_loss(log_likelihood, reduction="mean")

        # Log metrics
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/log_likelihood", log_likelihood.mean())
        if hasattr(expert_out, "get") or isinstance(expert_out, dict):
            er = expert_out.get("reward", None)
            if er is not None:
                self.log("train/expert_reward", er.mean())
            ec = expert_out.get("cost", None)
            if ec is not None:
                self.log("train/expert_cost", ec.mean())

        return loss
