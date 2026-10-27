"""
HTML simulation dataset class for loading simulation data directly from WSmart+ Route HTML dashboards.

This module provides the ``HtmlSimulationDataset``, which parses an HTML file or URL,
projects bin waste fill levels based on current fill and daily accumulation rates,
and structures the data to conform to the ``SimulationDataset`` interface.
"""

from typing import Any, Dict, Optional, cast

import numpy as np

from logic.src.constants import MAX_CAPACITY_PERCENT
from logic.src.data.datasets.simulation.sim_dataset import SimulationDataset
from logic.src.data.datasets.web.dashboard_crawler import extract_dataframe


class HtmlSimulationDataset(SimulationDataset):
    """
    Dataset wrapping simulation data extracted from a WSmart+ Route HTML dashboard file or URL.

    Attributes:
        _sample: Dictionary containing the dataset.
    """

    def __init__(self, sample: Dict[str, Any]):
        """Initialize the HTML simulation dataset.

        Args:
            sample: The parsed and formatted sample dictionary.
        """
        self._sample = sample

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            int: Always 1.
        """
        return 1

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return the sample at the given index.

        Args:
            index: Sample index.

        Returns:
            Dict[str, Any]: The sample dictionary.

        Raises:
            IndexError: If index is not 0.
        """
        if index != 0:
            raise IndexError("HtmlSimulationDataset only contains one sample.")
        return self._sample

    @classmethod
    def load(
        cls,
        path: str,
        area: Optional[str] = None,
        waste_type: Optional[str] = None,
        n_days: int = 31,
        n_bins: Optional[int] = None,
    ) -> "HtmlSimulationDataset":
        """Load a HtmlSimulationDataset from an HTML dashboard file or URL.

        Args:
            path: Path to the HTML file or a URL.
            area: Optional geographic area for coordinate fallback.
            waste_type: Optional waste type for coordinate fallback.
            n_days: Number of days to simulate waste accumulation.
            n_bins: Optional number of bins to sample.

        Returns:
            HtmlSimulationDataset: An initialized dataset instance.
        """
        df = extract_dataframe(path)

        if n_bins is not None:
            if n_bins > len(df):
                raise ValueError(f"n_bins={n_bins} exceeds available locations ({len(df)}).")
            df = df.sample(n=n_bins, random_state=42).reset_index(drop=True)

        # Fallback to repository coordinates for depot
        from logic.src.constants import ROOT_DIR
        from logic.src.pipeline.simulations.repository.filesystem import FileSystemRepository

        repo = FileSystemRepository(ROOT_DIR)
        try:
            depot_df = repo.get_depot(area or "Rio Maior")
            depot_coords = np.array([depot_df["Lat"].iloc[0], depot_df["Lng"].iloc[0]], dtype=np.float64)
            depot_id = depot_df["ID"].iloc[0]
        except Exception:
            # Fallback values if repository lookup fails
            depot_coords = np.array([39.188769, -9.148471], dtype=np.float64)
            depot_id = 0

        locs = df[["Lat", "Lng"]].values.astype(np.float64)
        node_ids = df["ID"].tolist()

        # Project waste fill levels for each day over n_days
        # Formula: fill_d = np.clip(Fill_Pct + Acum_Rate_Pct * d, 0.0, MAX_CAPACITY_PERCENT)
        fill_pct = cast(np.ndarray, df["Fill_Pct"].values).astype(np.float64)
        acum_rate_pct = cast(np.ndarray, df["Acum_Rate_Pct"].values).astype(np.float64)

        waste_list = []
        for d in range(n_days):
            day_waste = np.clip(fill_pct + acum_rate_pct * d, 0.0, MAX_CAPACITY_PERCENT)
            waste_list.append(day_waste)

        waste = np.vstack(waste_list)  # (n_days, n_bins)
        max_waste = np.full(len(df), MAX_CAPACITY_PERCENT, dtype=np.float64)

        sample = {
            "depot": depot_coords,
            "depot_id": depot_id,
            "locs": locs,
            "node_ids": node_ids,
            "waste": waste,
            "noisy_waste": waste.copy(),  # default noisy waste is clean waste
            "max_waste": max_waste,
        }

        return cls(sample)
