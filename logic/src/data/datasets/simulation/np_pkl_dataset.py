"""
NumPy pickle dataset for legacy simulation data stored in .pkl files.

Attributes:
    NumpyPickleDataset: Dataset wrapping legacy pickle simulation data.

Example:
    >>> from logic.src.data.datasets import NumpyPickleDataset
    >>> dataset = NumpyPickleDataset(data)
    >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=tensordict_collate_fn)
    >>> for batch in dataloader:
    ...     print(batch)
    ...     break
"""

import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .sim_dataset import SimulationDataset


class NumpyPickleDataset(SimulationDataset):
    """
    Dataset wrapping legacy pickle simulation data.

    The old format is a list of tuples per sample, where each tuple is either:
        - (depot, locs, waste, noisy_waste, max_waste) (legacy stochastic datasets)
        - (depot, locs, waste, max_waste) for standard problems

    This class normalises access so that __getitem__ always returns a dict
    with the same keys as NumpyDictDataset.

    Attributes:
        data: List of tuples containing the dataset.
    """

    def __init__(self, data: List[Tuple[Any, ...]]):
        """Initialize the NumPy pickle dataset.

        Args:
            data: Description of data.
        """
        self._data = data

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            Description of return value.
        """
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """Return the sample at the given index.

        Args:
            index: Description of index.

        Returns:
            Description of return value.
        """
        sample = self._data[index]
        if len(sample) == 5:
            depot, locs, waste, noisy_waste, max_waste = sample
        elif len(sample) == 4:
            depot, locs, waste, max_waste = sample
            noisy_waste = waste
        else:
            raise ValueError(f"Unexpected sample length: {len(sample)}")
        return {
            "depot": np.asarray(depot),
            "locs": np.asarray(locs),
            "waste": np.asarray(waste),
            "noisy_waste": np.asarray(noisy_waste),
            "max_waste": np.asarray(max_waste),
        }

    @staticmethod
    def load(path: str, area: Optional[str] = None, waste_type: Optional[str] = None) -> "NumpyPickleDataset":
        """Load a NumpyPickleDataset from a .pkl file.

        Args:
            path: Path to the .pkl file.
            area: Ignored.
            waste_type: Ignored.

        Returns:
            A NumpyPickleDataset instance.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301
        return NumpyPickleDataset(data)
