"""
Pandas Excel dataset for simulation data stored in .xlsx files.

Expected sheet format (one sheet per sample, named 'Sample0'..'SampleN-1'):
    - Row 0: depot (fill values are 0)
    - Rows 1..N: bins
    - Columns: 'Lat', 'Lng', 'Day 0', 'Day 1', ..., 'Day D-1'
    - Optional: 'Max Fill' (per-bin max capacity; defaults to MAX_CAPACITY_PERCENT)
    - Optional: 'Noisy Day 0', 'Noisy Day 1', ..., 'Noisy Day D-1'

If a single sheet is present, it is treated as a single sample regardless
of its name.

Attributes:
    PandasExcelDataset: Dataset wrapping simulation data loaded from an Excel workbook.

Example:
    >>> from logic.src.data.datasets import PandasExcelDataset
    >>> dataset = PandasExcelDataset(data)
    >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=tensordict_collate_fn)
    >>> for batch in dataloader:
    ...     print(batch)
    ...     break
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from logic.src.constants import MAX_CAPACITY_PERCENT

from .sim_dataset import SimulationDataset


class PandasExcelDataset(SimulationDataset):
    """
    Dataset wrapping simulation data loaded from an Excel workbook.

    Each sheet represents one sample. The first row in each sheet is the
    depot and the remaining rows are bins. Coordinates are in 'Lat'/'Lng'
    columns and daily fill values are in 'Day X' columns.

    Attributes:
        samples: List of samples.
    """

    def __init__(self, samples: List[Dict[str, np.ndarray]]):
        """Initialize the Pandas Excel dataset.

        Args:
            samples: Description of samples.
        """
        self._samples = samples

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            Description of return value.
        """
        return len(self._samples)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """Return the sample at the given index.

        Args:
            index: Description of index.

        Returns:
            Description of return value.
        """
        return self._samples[index]

    @staticmethod
    def load(path: str, area: Optional[str] = None, waste_type: Optional[str] = None) -> "PandasExcelDataset":
        """Load a PandasExcelDataset from an .xlsx file.

        Args:
            path: Path to the .xlsx file.
            area: Optional geographic area for coordinate fallback.
            waste_type: Optional waste type for coordinate fallback.

        Returns:
            A PandasExcelDataset instance.
        """
        sheets = pd.read_excel(path, sheet_name=None)
        samples = [PandasExcelDataset._parse_sheet(df, area=area, waste_type=waste_type) for df in sheets.values()]
        return PandasExcelDataset(samples)

    @staticmethod
    def _parse_sheet(
        df: pd.DataFrame, area: Optional[str] = None, waste_type: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """Parse a single sheet into a sample dict.

        Args:
            df: Dataframe to parse.
            area: Optional area for coordinate fallback.
            waste_type: Optional waste type for coordinate fallback.

        Returns:
            A dictionary representing the sample.
        """
        # Fallback to repository coordinates if Lat/Lng are missing
        if "Lat" not in df.columns and area is not None:
            from logic.src.constants import ROOT_DIR
            from logic.src.pipeline.simulations.repository.filesystem import FileSystemRepository

            repo = FileSystemRepository(ROOT_DIR)
            depot_df = repo.get_depot(area)

            if "ID" in df.columns:
                # Bins are N-1 rows (excluding depot)
                _, bins_coords = repo.get_simulator_data(len(df) - 1, area=area, waste_type=waste_type)
                full_coords = pd.concat(
                    [depot_df[["ID", "Lat", "Lng"]], bins_coords[["ID", "Lat", "Lng"]]],
                    axis=0,
                ).drop_duplicates(subset=["ID"])
                df = df.merge(full_coords, on="ID", how="left")
            else:
                # Row-based fallback
                df.loc[df.index[0], "Lat"] = depot_df["Lat"].iloc[0]
                df.loc[df.index[0], "Lng"] = depot_df["Lng"].iloc[0]
                if len(df) > 1:
                    _, bins_coords = repo.get_simulator_data(len(df) - 1, area=area, waste_type=waste_type)
                    n_bins = min(len(df) - 1, len(bins_coords))
                    df.loc[df.index[1 : n_bins + 1], "Lat"] = bins_coords["Lat"].values[:n_bins]
                    df.loc[df.index[1 : n_bins + 1], "Lng"] = bins_coords["Lng"].values[:n_bins]

        depot = df.iloc[0]
        bins_df = df.iloc[1:]

        depot_coords = np.array([depot["Lat"], depot["Lng"]], dtype=np.float64)
        locs = np.asarray(bins_df[["Lat", "Lng"]].values, dtype=np.float64)

        day_cols = sorted(
            [c for c in df.columns if isinstance(c, str) and c.startswith("Day ")],
            key=lambda c: int(c.split(" ", 1)[1]),
        )
        noisy_cols = sorted(
            [c for c in df.columns if isinstance(c, str) and c.startswith("Noisy Day ")],
            key=lambda c: int(c.split(" ", 2)[2]),
        )

        # waste shape: (n_days, n_bins) — depot row excluded
        waste = np.asarray(bins_df[day_cols].values, dtype=np.float64).T
        noisy_waste = np.asarray(bins_df[noisy_cols].values, dtype=np.float64).T if noisy_cols else waste

        if "Max Fill" in df.columns:
            max_waste = np.asarray(bins_df["Max Fill"].values, dtype=np.float64)
        else:
            max_waste = np.full(len(bins_df), MAX_CAPACITY_PERCENT, dtype=np.float64)

        return {
            "depot": depot_coords,
            "locs": locs,
            "waste": waste,
            "noisy_waste": noisy_waste,
            "max_waste": max_waste,
        }

    def save(self, path: str) -> None:
        """Save the dataset to an .xlsx file.

        Args:
            path: Description of path.
        """
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for i, sample in enumerate(self._samples):
                df = self._sample_to_dataframe(sample)
                df.to_excel(writer, sheet_name=f"Sample{i}", index=False)

    @staticmethod
    def _sample_to_dataframe(sample: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Convert a sample dict back into a DataFrame.

        Args:
            sample: Description of sample.

        Returns:
            Description of return value.
        """
        depot = sample["depot"]
        locs = sample["locs"]
        waste = sample["waste"]
        noisy_waste = sample["noisy_waste"]
        max_waste = sample["max_waste"]
        n_days = waste.shape[0]
        n_bins = locs.shape[0]

        day_cols = [f"Day {d}" for d in range(n_days)]

        # Depot row: coords + zero fills
        depot_row: Dict[str, float] = {"Lat": depot[0], "Lng": depot[1]}
        depot_row.update({col: 0.0 for col in day_cols})

        # Bin rows: coords + fill values
        rows = [depot_row]
        for b in range(n_bins):
            row: Dict[str, float] = {"Lat": locs[b, 0], "Lng": locs[b, 1]}
            row.update({day_cols[d]: waste[d, b] for d in range(n_days)})
            rows.append(row)

        # Max Fill column (write if not all default)
        max_arr = np.asarray(max_waste)
        has_custom_max = max_arr.ndim > 0 and not np.all(max_arr == MAX_CAPACITY_PERCENT)
        if has_custom_max:
            rows[0]["Max Fill"] = 0.0
            for b in range(n_bins):
                rows[b + 1]["Max Fill"] = float(max_arr[b]) if max_arr.ndim > 0 else float(max_arr)

        # Noisy columns (write only if different from waste)
        has_noisy = not np.array_equal(waste, noisy_waste)
        if has_noisy:
            noisy_cols = [f"Noisy Day {d}" for d in range(n_days)]
            rows[0].update({col: 0.0 for col in noisy_cols})
            for b in range(n_bins):
                rows[b + 1].update({noisy_cols[d]: noisy_waste[d, b] for d in range(n_days)})

        return pd.DataFrame(rows)
