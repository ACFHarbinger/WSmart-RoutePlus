"""
Reinforcement Learning (RL) Environment interface module.

Attributes:
    IEnv: Protocol for RL Environments

Example:
    >>> from logic.src.interfaces.env import IEnv
    >>> class MyEnv(IEnv):
    ...     name: str = "my_env"
    ...     @property
    ...     def device(self) -> torch.device:
    ...         return torch.device("cpu")
    ...     generator: Optional[Any] = None
    ...     def reset(self, tensordict: Optional[TensorDictBase] = None, **kwargs: Any) -> TensorDictBase:
    ...         return TensorDict({}, {})
    ...     def step(self, tensordict: TensorDictBase) -> TensorDictBase:
    ...         return TensorDict({}, {})
    ...     def get_reward(self, tensordict: TensorDictBase, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
    ...         return torch.tensor(0.0)
    ...     def get_action_mask(self, tensordict: TensorDictBase) -> torch.Tensor:
    ...         return torch.tensor(True)
    ...
    >>> env = MyEnv()
    >>> env.reset()
    TensorDict(batch_size=torch.Size([]), contents={})
"""

from typing import Any, Optional, Protocol

import torch
from tensordict import TensorDictBase


class IEnv(Protocol):
    """
    Protocol for RL Environments.
    Matches RL4COEnvBase interface.

    Attributes:
        name: Name of the environment
        device: Device to place tensors on
        generator: Generator for the environment
    """

    name: str
    graph: Any

    @property
    def device(self) -> torch.device:
        """
        Device to place tensors on.

        Returns:
            torch.device: Device to place tensors on
        """
        ...

    generator: Optional[Any] = None

    def reset(self, tensordict: Optional[TensorDictBase] = None, **kwargs: Any) -> TensorDictBase:
        """
        Reset environment.

        Args:
            tensordict: TensorDict containing the input data
            kwargs: Additional keyword arguments

        Returns:
            TensorDictBase: TensorDict containing the output data
        """
        ...

    def step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """
        Step environment.

        Args:
            tensordict: TensorDict containing the input data

        Returns:
            TensorDictBase: TensorDict containing the output data
        """
        ...

    def get_reward(self, tensordict: TensorDictBase, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate reward.

        Args:
            tensordict: TensorDict containing the input data
            actions: Actions to calculate reward for

        Returns:
            torch.Tensor: Reward for the given actions
        """
        ...
