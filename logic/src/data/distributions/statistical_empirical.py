"""
Empirical sampling distributions.

Attributes:
    Empirical(grid, area, indices, data_path, dataset):
        Empirical dataset sampling.

Example:
    dist = Empirical(data_path="data/test.pkl")
    data = dist._sample_tensor(torch.Size((10, 50)))
"""

from __future__ import annotations

import os
import pickle
from typing import TYPE_CHECKING, Any, Optional, Tuple, cast

import numpy as np
import pandas
import torch

if TYPE_CHECKING:
    from logic.submodules.wsmart_bin_analysis import GridBase

from .base import BaseDistribution


class Empirical(BaseDistribution):
    """Sampling from an empirical dataset (e.g. file or Bins object).

    Attributes:
        grid (Optional[GridBase]): The grid object.
        dataset (Optional[Any]): The dataset object.
        data_path (Optional[str]): The path to the dataset.
    """

    def __init__(
        self,
        grid: Optional[GridBase] = None,
        area: Optional[str] = None,
        indices: Optional[np.ndarray] = None,
        data_path: Optional[str] = None,
        dataset: Optional[Any] = None,
    ):
        """Initialize Class.



        Args:
            grid: Description of grid.
            area: Description of area.
            indices: Description of indices.
            data_path: Description of data_path.
            dataset: Description of dataset.
        """
        self.grid = grid
        self.dataset = dataset
        self.data_path = data_path
        if grid is None and data_path is not None and indices is not None and os.path.isdir(data_path):
            from logic.src.utils.data.loader import load_grid_base

            self.grid = load_grid_base(indices, area, data_path)
        if data_path is not None and os.path.isfile(data_path):
            if data_path.endswith(".pkl"):
                with open(data_path, "rb") as f:
                    self.dataset = pickle.load(f)
            elif data_path.endswith(".csv"):
                self.dataset = pandas.read_csv(data_path)
            elif data_path.endswith(".xlsx"):
                self.dataset = pandas.read_excel(data_path)
            elif data_path.endswith(".npz"):
                self.dataset = np.load(data_path)
            else:
                raise ValueError("Data path must be a directory or a pkl/csv/xlsx/npz file.")
        assert self.grid is not None or self.dataset is not None

    def _sample_tensor(self, size: Tuple[int, ...], generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Sample from empirical dataset.

        Args:
            size: Sampling shape. First dimension is assumed to be batch size.
            generator: Description of generator.

        Returns:
            torch.Tensor: Sampled values
        """
        if generator is None:
            generator = torch.Generator()
        batch_size = size[0]
        # Draw a fresh seed from the generator on every call. Seeding from initial_seed() gave
        # the same sample each time, so every day of a multi-day dataset held identical waste.
        draw_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator, device=generator.device).item())
        if self.dataset is not None:
            if isinstance(self.dataset, pandas.DataFrame):
                vals = self.dataset.sample(n=batch_size, random_state=draw_seed).values
                vals_tensor = torch.tensor(vals).float() / 100.0
            elif isinstance(self.dataset, np.ndarray):
                vals = self.dataset[np.random.choice(len(self.dataset), batch_size)]
                vals_tensor = torch.from_numpy(vals).float() / 100.0
            elif isinstance(self.dataset, dict):
                vals = self.dataset[np.random.choice(len(self.dataset), batch_size)]
                vals_tensor = torch.tensor(vals).float() / 100.0
            elif isinstance(self.dataset, torch.Tensor):
                vals_tensor = self.dataset[np.random.choice(len(self.dataset), batch_size)]
            else:
                raise ValueError("Dataset must be a pandas DataFrame, numpy array, dictionary, or torch Tensor.")
        elif self.grid is not None:
            vals = self.grid.sample(n_samples=batch_size, rng=np.random.default_rng(draw_seed))
            vals_tensor = torch.from_numpy(vals).float() / 100.0
        else:
            raise ValueError("No grid or dataset found.")
        return torch.clip(vals_tensor, 0, 1)

    def _sample_array(self, size: Tuple[int, ...], rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample from empirical dataset.

        Args:
            size: Sampling shape. First dimension is assumed to be batch size.
            rng: Optional numpy default_rng for reproducibility.

        Returns:
            np.ndarray: Sampled values
        """
        if rng is None:
            rng = cast(np.random.Generator, np.random)

        n_samples = size[0]
        if self.dataset is not None:
            if isinstance(self.dataset, pandas.DataFrame):
                sampled = self.dataset.sample(n=n_samples, random_state=rng).values
            elif isinstance(self.dataset, (np.ndarray, dict)):
                indices = rng.choice(len(self.dataset), n_samples)
                sampled = self.dataset[indices]
            elif isinstance(self.dataset, torch.Tensor):
                indices = rng.choice(len(self.dataset), n_samples)
                sampled = self.dataset[indices].cpu().numpy()
            else:
                raise ValueError("Dataset must be a pandas DataFrame, numpy array, dictionary, or torch Tensor.")
        elif self.grid is not None:
            sampled = self.grid.sample(n_samples=n_samples, rng=rng)
        else:
            raise ValueError("No grid or dataset found.")
        return np.clip(sampled, 0, 100) / 100.0
