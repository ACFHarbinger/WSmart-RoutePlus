"""
RL4CO Base Environment.

Attributes:
    RL4COEnvBase: Base class for RL4CO environments.

Example:
    None
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import torch
from tensordict import TensorDictBase
from torchrl.envs import EnvBase

from .batch import BatchMixin
from .ops import OpsMixin


class RL4COEnvBase(BatchMixin, OpsMixin, EnvBase, ABC):
    """
    Base environment class for combinatorial optimization problems.

    This class provides a unified interface for problem environments following
    the torchrl/RL4CO architecture pattern. All problem-specific environments should
    inherit from this class.

    Attributes:
        name: Unique identifier for the environment type.
        generator: Optional data generator for creating problem instances.
        device: Device to place tensors on (cpu/cuda).
    """

    name: str = "base"
    node_dim: int = 3  # Default dimension for most routing problems (x, y, waste)

    @property
    def dim(self) -> int:
        """Alias for node_dim, expected by some models.

        Returns:
            int: Node dimension.
        """
        return self.node_dim

    def __init__(
        self,
        generator: Optional[Any] = None,
        generator_params: Optional[dict] = None,
        device: Union[str, torch.device] = "cpu",
        batch_size: Optional[Union[torch.Size, int]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the base environment.

        Args:
            generator: Data generator instance for creating problem instances.
            generator_params: Parameters to pass to the generator if not provided.
            device: Device to place tensors on.
            batch_size: Batch size for the environment.
            kwargs: Additional keyword arguments.
        """
        # Default batch_size to empty Size if not provided
        if batch_size is None:
            batch_size = torch.Size([])
        else:
            if isinstance(batch_size, int):
                batch_size = torch.Size([batch_size])
            batch_size = torch.Size(batch_size)  # Ensure it's a torch.Size object

        # Filter kwargs for EnvBase
        env_base_kwargs = {
            "device": device,
            "batch_size": batch_size,
        }
        for k in list(kwargs.keys()):
            if k in ["run_type_checks", "allow_done_after_reset"]:
                env_base_kwargs[k] = kwargs.pop(k)

        super().__init__(**env_base_kwargs)

        # Manually set check_env_specs AFTER super init to avoid TypeError in 0.3.1
        self.check_env_specs = False

        # Manually set check_env_specs if provided (bypass strict checks in reset)
        self.check_env_specs = kwargs.get("check_env_specs", False)

        self.generator = generator
        self.generator_params = generator_params or {}
        self._kwargs = kwargs

    def render(self, td: Any, **kwargs: Any) -> Any:
        """
        Render the current state (optional).

        Args:
            td: TensorDict containing the state.
            kwargs: Additional keyword arguments.

        Returns:
            Any: The rendered state.
        """
        raise NotImplementedError(f"Rendering not implemented for {self.name}")

    def __repr__(self) -> str:
        """String representation of the environment.

        Returns:
            str: String representation of the environment.
        """
        return f"{self.__class__.__name__}(name={self.name}, device={self.device})"

    @abstractmethod
    def _reset_instance(self, td: TensorDictBase) -> TensorDictBase:
        """
        Reset the environment for a specific problem instance.

        Args:
            td: TensorDict containing the state.

        Returns:
            TensorDictBase: The reset environment.
        """
        pass

    @abstractmethod
    def _step(self, td: TensorDictBase) -> TensorDictBase:
        """Perform a transition in the environment.

        Args:
            td: TensorDict containing the state.

        Returns:
            TensorDictBase: The environment after the transition.
        """
        pass

    @abstractmethod
    def _get_reward(self, td: TensorDictBase, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the reward for a specific action sequence.

        Args:
            td: TensorDict containing the state.
            actions: Optional tensor of actions.

        Returns:
            torch.Tensor: The reward for the given action sequence.
        """
        pass

    @abstractmethod
    def _get_action_mask(self, td: TensorDictBase) -> torch.Tensor:
        """
        Return the valid action mask for the current state.

        Args:
            td: TensorDict containing the state.

        Returns:
            torch.Tensor: The action mask for the given state.
        """
        pass
