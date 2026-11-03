"""
Reinforcement Learning (RL) Policy protocol module.

Attributes:
    IPolicy: Protocol for RL Policies

Example:
    >>> from logic.src.interfaces.policy import IPolicy
    >>> class MyPolicy(IPolicy):
    ...     def forward(self, td: TensorDict, env: Optional[Any] = None, strategy: str = "sampling", num_starts: int = 1, **kwargs: Any) -> Union[TensorDict, Dict[str, Any]]:
    ...         return {}
    ...     def __call__(self, *args, **kwargs) -> Any:
    ...         return {}
    ...
    >>> policy = MyPolicy()
    >>> policy.forward(TensorDict({}, {}))
    {}
"""

from typing import Any, Dict, Iterator, Optional, Protocol, Union

import torch
from tensordict import TensorDict


class IPolicy(Protocol):
    """
    Protocol for RL Policies.
    Matches the interface expected by RL4COLitModule and StepwisePPO.

    Attributes:
        encoder: Encoder module
        decoder: Decoder module
    """

    encoder: Optional[torch.nn.Module]
    decoder: Optional[torch.nn.Module]

    def forward(
        self,
        td: TensorDict,
        env: Optional[Any] = None,
        strategy: str = "sampling",
        num_starts: int = 1,
        **kwargs: Any,
    ) -> Union[TensorDict, Dict[str, Any]]:
        """
        Forward pass to generate solutions.

        Args:
            td: TensorDict containing the input data
            env: Environment
            strategy: Sampling strategy
            num_starts: Number of starts
            kwargs: Additional keyword arguments

        Returns:
            Union[TensorDict, Dict[str, Any]]: TensorDict or dict containing the output data
        """
        ...

    def __call__(self, *args, **kwargs) -> Any:
        """call  .

        Args:
            args: Description of args.
            kwargs: Description of kwargs.
        """
        ...

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        """Iterate over the trainable parameters (the ``torch.nn.Module`` method).

        Args:
            recurse: If True, include the parameters of submodules.
        """
        ...
