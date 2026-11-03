"""
Dataset for precomputed baseline values.

Attributes:
    BaselineDataset: Dataset wrapping baseline values for training.

Example:
    >>> from logic.src.data.datasets import BaselineDataset
    >>> dataset = BaselineDataset(dataset, baseline)
    >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=tensordict_collate_fn)
    >>> for batch in dataloader:
    ...     print(batch)
    ...     break
"""

import torch
from torch.utils.data import Dataset

from logic.src.interfaces.sized_dataset import ISizedDataset


class BaselineDataset(Dataset):
    """
    Dataset wrapping baseline values for training.

    Attributes:
        dataset: The underlying dataset.
        baseline: Tensor of precomputed baseline values.
    """

    def __init__(self, dataset: ISizedDataset, baseline: torch.Tensor):
        """
        Initialize the BaselineDataset.

        Args:
            dataset: The underlying dataset.
            baseline: Tensor of precomputed baseline values.
        """
        super().__init__()
        self.dataset = dataset
        self.baseline = baseline
        assert len(self.dataset) == len(self.baseline)

    def __getitem__(self, index: int) -> dict:
        """
        Retrieve a sample with its associated baseline value.

        Args:
            index: Index of the sample.

        Returns:
            dict: Dictionary with 'data' and 'baseline' keys.
        """
        return {"data": self.dataset[index], "baseline": self.baseline[index]}

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.dataset)
