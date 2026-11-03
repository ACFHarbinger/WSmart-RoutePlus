"""
Dataset with extra key/value pairs.

Attributes:
    ExtraKeyDataset: Dataset that includes an extra key/value pair (e.g., for baseline rewards).

Example:
    >>> from logic.src.data.datasets import ExtraKeyDataset
    >>> dataset = ExtraKeyDataset(dataset, extra)
    >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=tensordict_collate_fn)
    >>> for batch in dataloader:
    ...     print(batch)
    ...     break
"""

import torch
from torch.utils.data import Dataset

from logic.src.interfaces.sized_dataset import ISizedDataset


class ExtraKeyDataset(Dataset):
    """
    Dataset that includes an extra key/value pair (e.g., for baseline rewards).

    Attributes:
        dataset: The underlying dataset.
        extra: Dictionary of extra tensors to include (e.g., {'baseline': ...}).
    """

    def __init__(self, dataset: ISizedDataset, extra: dict[str, torch.Tensor]):
        """
        Initialize the ExtraKeyDataset.

        Args:
            dataset: The underlying dataset.
            extra: Dictionary of extra tensors to include (e.g., {'baseline': ...}).
        """
        super().__init__()
        self.dataset = dataset
        self.extra = extra
        # Validate lengths
        for k, v in extra.items():
            assert len(dataset) == len(v), (
                f"Length mismatch for key {k}: {len(dataset)} vs {len(v)}"
            )

    def __getitem__(self, index: int) -> dict:
        """
        Retrieve a sample with extra keys.

        Args:
            index: Index of the sample.

        Returns:
            dict: Dictionary with 'data' and extra keys.
        """
        item = {"data": self.dataset[index]}
        for k, v in self.extra.items():
            item[k] = v[index]
        return item

    def __len__(self) -> int:
        """
        Return the number of samples.

        Returns:
            int: Number of samples.
        """
        return len(self.dataset)
