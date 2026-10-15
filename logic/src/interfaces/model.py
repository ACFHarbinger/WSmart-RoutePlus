"""
Neural Model interface module.

Attributes:
    IModel: Protocol for Neural Models

Example:
    >>> from logic.src.interfaces.model import IModel
    >>> class MyModel(IModel):
    ...     def forward(self, td: TensorDict, **kwargs: Any) -> Any:
    ...         return {}
    ...     def to(self, device: torch.device) -> "IModel":
    ...         return self
    ...     def eval(self) -> "IModel":
    ...         return self
    ...     def train(self) -> "IModel":
    ...         return self
    ...
    >>> model = MyModel()
    >>> model.forward(TensorDict({}, {}))
    {}
"""

from typing import Any, Protocol, runtime_checkable

import torch
from tensordict import TensorDict


@runtime_checkable
class IModel(Protocol):
    """
    Protocol for Neural Models.

    Attributes:
        None: No attributes
    """

    def forward(self, td: TensorDict, **kwargs: Any) -> Any:
        """Forward.

        Args:
            td: Description of td.
            kwargs: Description of kwargs.

        Returns:
            Any: Description of return value.
        """
        ...

    def to(self, device: torch.device) -> "IModel":
        """To.

        Args:
            device: Description of device.

        Returns:
            IModel: Description of return value.
        """
        ...

    def eval(self) -> "IModel":
        """Eval.

        Returns:
            IModel: Description of return value.
        """
        ...

    def train(self) -> "IModel":
        """Train.

        Returns:
            IModel: Description of return value.
        """
        ...
