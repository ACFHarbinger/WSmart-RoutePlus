"""Transductive (Test-time Adaptation) model base module.

This module provides the abstract foundation for transductive methods, such as
Active Search and EAS, which perform instance-specific weight adaptation or
stochastic search during the inference phase to refine solution quality.

Attributes:
    TransductiveModel: Base class for transductive search methods.

Example:
    >>> policy = AttentionModelPolicy(...)
    >>> eas = TransductiveModel(policy, n_search_steps=100)
    >>> out = eas(td, env)
"""

from __future__ import annotations

from abc import ABC
from typing import Any, Dict, List, Optional

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase


class TransductiveModel(nn.Module, ABC):
    """Base class for transductive search methods.

    Transductive models perform weight optimization or stochastic search at
    inference time for each problem instance. This allows the model to escape
    local minima of the pre-trained policy by adapting to the specific features
    of the current instance.

    Attributes:
        model: The underlying constructive or improvement policy.
        optimizer_kwargs: Parameters for the search optimizer.
        n_search_steps: Number of optimization iterations per instance.
        kwargs: Additional model configuration parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_search_steps: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize the TransductiveModel.

        Args:
            model: The base policy model to be adapted.
            optimizer_kwargs: Optimizer settings (e.g., learning rate).
            n_search_steps: Number of test-time optimization iterations.
            kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.model = model
        self.optimizer_kwargs = optimizer_kwargs or {"lr": 1e-4}
        self.n_search_steps = n_search_steps
        self.kwargs = kwargs

    def _setup_optimizer(self, params: Any) -> torch.optim.Optimizer:
        """Initialize the optimizer used for the test-time search phase.

        Args:
            params: Iterable of parameters to optimize.

        Returns:
            torch.optim.Optimizer: An initialized Adam optimizer.
        """
        return torch.optim.Adam(params, **self.optimizer_kwargs)

    def _get_search_params(self) -> Any:
        """Define the subset of parameters to optimize during the search.

        Subclasses (like EAS) can override this to optimize only specific
        adapter layers or embeddings rather than the full model.

        Returns:
            Any: An iterable of torch.nn.Parameter objects.
        """
        return self.model.parameters()

    def forward(
        self,
        td: TensorDict,
        env: Optional[RL4COEnvBase] = None,
        strategy: str = "greedy",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute the transductive search loop on the provided instances.

        Clones the original model state, performs the specified number of
        optimization steps using sampling-based REINFORCE, and restores the
        original state before returning the best found solution.

        Args:
            td: TensorDict containing the problem instance(s).
            env: Environment managing problem rules and reward logic.
            strategy: Evaluation strategy (usually "greedy" or "sampling").
            kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: Best results found during search including:
                - reward (torch.Tensor): Highest reward achieved.
                - actions (torch.Tensor): Corresponding action sequence.
                - search_history (List[Dict]): Logging of the search metrics.

        Raises:
            ValueError: If the model forward pass does not return rewards or probs.
        """
        # Save original state to restore after search
        original_state = {k: v.cpu().detach().clone() for k, v in self.model.state_dict().items()}

        optimizer = self._setup_optimizer(self._get_search_params())

        # Track best solution found during search
        best_reward: Optional[torch.Tensor] = None
        best_actions: Optional[torch.Tensor] = None

        search_history: List[Dict[str, float]] = []
        for _ in range(self.n_search_steps):
            optimizer.zero_grad()

            # Forward pass on the instances (sampling for exploration)
            out = self.model(td, env, strategy="sampling", return_pi=True, **kwargs)

            # Handle variant return types from dynamic AttentionModels
            if isinstance(out, tuple):
                cost = out[0]
                log_p = out[1]
                actions = out[3]
                reward = -cost
            else:
                reward = out.get("reward")
                log_p = out.get("log_p")
                actions = out.get("actions")

                if reward is None and "cost" in out:
                    reward = -out["cost"]

            if reward is None or log_p is None:
                raise ValueError("Reward or log_p not found in model output")

            # Update best-so-far state
            if best_reward is None:
                best_reward = reward.detach().clone()
                best_actions = actions.detach().clone() if actions is not None else None
            else:
                better = reward > best_reward
                best_reward = torch.where(better, reward.detach(), best_reward)
                if best_actions is not None and actions is not None:
                    # Sync sequence lengths (padding) if necessary
                    if actions.size(1) != best_actions.size(1):
                        max_len = max(actions.size(1), best_actions.size(1))
                        if actions.size(1) < max_len:
                            padding = torch.zeros(
                                actions.size(0),
                                max_len - actions.size(1),
                                dtype=actions.dtype,
                                device=actions.device,
                            )
                            actions = torch.cat([actions, padding], dim=1)
                        if best_actions.size(1) < max_len:
                            padding = torch.zeros(
                                best_actions.size(0),
                                max_len - best_actions.size(1),
                                dtype=best_actions.dtype,
                                device=best_actions.device,
                            )
                            best_actions = torch.cat([best_actions, padding], dim=1)

                    better_expanded = better.view(-1, 1).expand_as(best_actions)
                    best_actions = torch.where(better_expanded, actions.detach(), best_actions)

            # Optimization Step (REINFORCE with moving baseline)
            advantage = reward - reward.mean()
            loss = -(advantage * log_p).mean()

            loss.backward()
            optimizer.step()

            search_history.append(
                {
                    "mean_reward": reward.mean().item(),
                    "best_reward": best_reward.mean().item(),
                }
            )

        # Build final output package
        final_out = {
            "reward": best_reward,
            "actions": best_actions,
            "search_history": search_history,
        }

        # Restore original global weights
        self.model.load_state_dict(original_state)

        return final_out
