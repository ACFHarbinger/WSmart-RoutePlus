"""Neural Heuristic Hybrid: Constructive-Refinement Policy.

This module implements a hybrid execution flow where a neural network generates
an initial solution (constructive phase), and a classic meta-heuristic
(refinement phase) iterates upon it to reach local optimality.

Attributes:
    NeuralHeuristicHybrid: Unified policy for construction and refinement.

Example:
    >>> hybrid = NeuralHeuristicHybrid(neural_model, hgs_solver)
    >>> out = hybrid(td, env)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from tensordict import TensorDict

from logic.src.models.common.autoregressive.policy import AutoregressivePolicy
from logic.src.pipeline.learning.envs.base.base import RL4COEnvBase

if TYPE_CHECKING:
    from logic.src.policies.vectorized.routing.alns import VectorizedALNS
    from logic.src.policies.vectorized.routing.hgs import VectorizedHGS


class NeuralHeuristicHybrid(AutoregressivePolicy):
    """Hybrid policy combining deep learning construction and OR-based refinement.

    Attributes:
        neural_policy (AutoregressivePolicy): The model used for Stage 1.
        heuristic_policy (Union[VectorizedALNS, VectorizedHGS]): Classic solver
            used for Stage 2 refinement.
    """

    def __init__(
        self,
        neural_policy: AutoregressivePolicy,
        heuristic_policy: Union[VectorizedALNS, VectorizedHGS],
        **kwargs: Any,
    ) -> None:
        """Initializes the hybrid policy.

        Args:
            neural_policy: Autoregressive neural model for initial construction.
            heuristic_policy: Classical OR solver for refinement.
            kwargs: Additional keyword arguments.
        """
        super().__init__(env_name=neural_policy.env_name, **kwargs)
        self.neural_policy = neural_policy
        self.heuristic_policy = heuristic_policy

    def forward(
        self,
        td: TensorDict,
        env: Optional[RL4COEnvBase] = None,
        strategy: str = "greedy",
        num_starts: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Solves problem instances using the hybrid execution flow.

        Args:
            td: TensorDict containing instance data.
            env: The problem environment.
            strategy: Construction strategy ("greedy" or "sampling").
            num_starts: Number of starts for the neural model.
            kwargs: Additional keyword arguments passed to both policies.

        Returns:
            Dict[str, Any]: Results dictionary from the heuristic refinement phase.
        """
        # Phase 1: Contextual Neural Construction
        assert env is not None, "Environment must be provided for hybrid solving."
        neural_out = self.neural_policy(td, env, strategy=strategy, **kwargs)

        # Phase 2: Heuristic Refinement
        # The heuristic solver starts from the neural-generated sequences
        heuristic_out = self.heuristic_policy(td, env, initial_solution=neural_out["actions"], **kwargs)

        return heuristic_out
