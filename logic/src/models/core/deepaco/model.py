"""DeepACO Model for ant-based neural optimization.

This module provides the `DeepACO` wrapper (Ye et al. 2023), which combines
Ant Colony Optimization (ACO) with neural heuristic learning. It uses a
REINFORCE-based training scheme to optimize a neural pheromone generator.

Attributes:
    DeepACO: Primary training wrapper for neural ACO policies.

Example:
    >>> from logic.src.models.core.deepaco.model import DeepACO
    >>> model = DeepACO(n_ants=20, alpha=1.0, beta=2.0)
    >>> out = model(td, env)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase

from .policy import DeepACOPolicy


class DeepACO(nn.Module):
    """DeepACO training wrapper for REINFORCE.

    Assembles the `DeepACOPolicy` and implements the REINFORCE algorithm with
    flexible baseline subtraction to stabilize the pheromone generator training.

    Attributes:
        policy (DeepACOPolicy): The underlying ACO-neural hybrid policy.
        baseline_type (str): RL baseline strategy ('rollout', 'exponential').
        _baseline_val (Optional[Any]): Rolling state for exponential baseline.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_encoder_layers: int = 3,
        num_heads: int = 8,
        n_ants: int = 20,
        n_iterations: int = 1,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.1,
        use_local_search: bool = True,
        baseline: str = "rollout",
        env_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the DeepACO model.

        Args:
            embed_dim: Dimensionality of the node features.
            num_encoder_layers: Number of Transformer encoder layers.
            num_heads: Number of attention heads.
            n_ants: Number of parallel ants in the colony.
            n_iterations: Number of ACO pheromone update cycles.
            alpha: Pheromone importance exponent.
            beta: Heuristic visibility importance exponent.
            rho: Pheromone evaporation rate.
            use_local_search: Whether to apply refinement after construction.
            baseline: RL baseline type ("rollout", "exponential", "none").
            env_name: Name of the environment identifier.
            kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.policy = DeepACOPolicy(
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            n_ants=n_ants,
            n_iterations=n_iterations,
            alpha=alpha,
            beta=beta,
            rho=rho,
            use_local_search=use_local_search,
            env_name=env_name,
            **kwargs,
        )
        self.baseline_type = baseline
        self._baseline_val: Optional[Any] = None

    def forward(
        self,
        td: TensorDict,
        env: RL4COEnvBase,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Calculates construction output and computes REINFORCE loss.

        Args:
            td: TensorDict containing problem instance data.
            env: Environment managing the problem physics.
            kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: Policy outputs including rewards, actions, and `loss`.
        """
        out = self.policy(td, env, **kwargs)

        reward = out["reward"]
        log_likelihood = out["log_likelihood"]

        # 1. Update/compute baseline for variant reduction
        if self.baseline_type == "exponential":
            if self._baseline_val is None:
                self._baseline_val = reward.mean().detach()
            else:
                self._baseline_val = 0.8 * self._baseline_val + 0.2 * reward.mean().detach()
            baseline = self._baseline_val
        elif self.baseline_type == "rollout":
            baseline = reward.mean()
        else:
            baseline = 0.0

        # 2. Compute Advantage and REINFORCE loss
        advantage = reward - baseline
        loss = -(advantage.detach() * log_likelihood).mean()

        out["loss"] = loss
        out["baseline"] = baseline
        return out

    def set_strategy(self, strategy: str, **kwargs: Any) -> None:
        """Configures constructive decoding strategy.

        Args:
            strategy: Identifier for the decoding strategy (e.g., "greedy").
            kwargs: Additional parameters for the decoding strategy.
        """
        self.policy.set_strategy(strategy, **kwargs)

    def eval(self) -> DeepACO:
        """Switches the model to evaluation mode.

        Returns:
            DeepACO: The model instance in eval state.
        """
        super().eval()
        return self
