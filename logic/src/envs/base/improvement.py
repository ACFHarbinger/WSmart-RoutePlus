"""
Improvement-based environment base class.

Attributes:
    ImprovementEnvBase: Base class for improvement-based metaheuristics.

Example:
    >>> from logic.src.envs.base.improvement import ImprovementEnvBase
    >>> class MyEnv(ImprovementEnvBase):
    ...     def __init__(self):
    ...         self.batch_size = 32
    >>> env = MyEnv()
    >>> env.batch_size
    torch.Size([32])
"""

from __future__ import annotations

from abc import abstractmethod

import torch
from tensordict import TensorDict

from .base import RL4COEnvBase


class ImprovementEnvBase(RL4COEnvBase):
    """
    Base environment for improvement-based methods.

    Improvement methods start with an initial solution and iteratively
    improve it through local search operations.

    Attributes:
        name: Name of the environment.
    """

    name: str = "improvement_base"

    @abstractmethod
    def _get_initial_solution(self, td: TensorDict) -> torch.Tensor:
        """
        Generate initial solution for improvement.

        Args:
            td: TensorDict containing the state.

        Returns:
            torch.Tensor: The initial solution for the given state.
        """
        raise NotImplementedError

    def _reset_instance(self, td: TensorDict) -> TensorDict:
        """
        Reset with initial solution generation.

        Args:
            td: TensorDict containing the state.

        Returns:
            TensorDict: The reset environment.
        """
        # Generate initial solution
        td["solution"] = self._get_initial_solution(td)

        return td
