"""
Protocol for map-style datasets that know their length.

``torch.utils.data.Dataset`` declares ``__getitem__`` only, so a wrapper that needs ``len(dataset)`` cannot type its
wrapped dataset as ``Dataset``. This protocol declares both members.

Attributes:
    ISizedDataset: Protocol for datasets with ``__len__`` and ``__getitem__``.

Example:
    >>> from logic.src.interfaces.sized_dataset import ISizedDataset
    >>> def n_samples(dataset: ISizedDataset) -> int:
    ...     return len(dataset)
"""

from typing import Any, Protocol


class ISizedDataset(Protocol):
    """Map-style dataset with a length."""

    def __len__(self) -> int:
        """Return the number of samples."""
        ...

    def __getitem__(self, index: int) -> Any:
        """Return the sample at ``index``.

        Args:
            index: Index of the sample.
        """
        ...
