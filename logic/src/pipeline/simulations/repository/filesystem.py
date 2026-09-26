"""
File-based implementation of SimulationRepository.

This module provides the FileSystemRepository class, which loads simulation
data from local CSV, Excel, and JSON files.

Attributes:
    FileSystemRepository: Repository sourcing data from the local filesystem.

Example:
    >>> # repo = FileSystemRepository(data_root_dir=".")
    >>> # depot = repo.get_depot("Rio Maior")
"""

import contextlib
import json
import os
from typing import Any, List, Optional, Tuple, cast

import numpy as np
import pandas as pd

import logic.src.constants as udef

from .base import SimulationRepository

try:
    from logic.src.tracking.core.run import get_active_run
except ImportError:
    get_active_run = None  # type: ignore[assignment]


class FileSystemRepository(SimulationRepository):
    """
    File-based implementation of SimulationRepository.

    Loads data from CSV, Excel, and JSON files stored in the project's
    data/simulator directory. Supports multiple geographic areas
    and waste types with area-specific file naming conventions.

    Attributes:
        default_data_dir: The default path for data resolution.
    """

    def __init__(self, data_root_dir):
        """
        Initialize the repository.

        Args:
            data_root_dir: Root directory path for data resolution.
        """
        self.default_data_dir = os.path.join(data_root_dir, "data", "simulator")

    def _get_data_dir(self, override_dir: Optional[str] = None) -> str:
        """get data dir.

        Args:
            override_dir: Optional path to override the default data directory.

        Returns:
            The resolved data directory path.
        """
        return override_dir if override_dir else self.default_data_dir

    def get_indices(
        self, filename: Any, n_samples: int, n_nodes: int, data_size: int, lock: Optional[Any] = None
    ) -> List[List[int]]:
        """
        Implementation of get_indices that persists generated indices to JSON.

        Args:
            filename: Path to the index file within bins_selection.
            n_samples: Number of samples to generate/load.
            n_nodes: Number of nodes per sample.
            data_size: Total size of the source data.
            lock: Optional file lock.

        Returns:
            A list of lists containing bin indices for each sample.
        """
        _indices_source = "file"
        graphs_file_path = os.path.join(self.default_data_dir, "bins_selection", filename)
        if os.path.isfile(graphs_file_path):
            if lock is not None:
                lock.acquire(timeout=udef.LOCK_TIMEOUT)
            try:
                with open(graphs_file_path) as fp:
                    indices = json.load(fp)
            finally:
                if lock is not None:
                    lock.release()
                if len(indices) == 1 and n_samples > 1:
                    indices *= n_samples
        else:
            _indices_source = "generated"
            df = pd.Series(range(data_size))
            indices = []
            for _ in range(n_samples):
                data = df.sample(n=n_nodes).to_list()
                data.sort()
                while len(indices) > 0 and data in indices:
                    data = df.sample(n=n_nodes).to_list()
                    data.sort()
                indices.append(data)

            if lock is not None:
                lock.acquire(timeout=udef.LOCK_TIMEOUT)
            try:
                os.makedirs(os.path.dirname(graphs_file_path), exist_ok=True)
                with open(graphs_file_path, "w") as fp:
                    fp.write("[\n")
                    fp.write(",\n".join(json.dumps(idx) for idx in indices))
                    fp.write("\n]")
            finally:
                if lock is not None:
                    lock.release()

        with contextlib.suppress(Exception):
            run = get_active_run() if get_active_run is not None else None
            if run is not None:
                run.log_params({"data.indices_source": _indices_source})
                run.log_metric("data/n_index_groups", float(len(indices)))

        return indices

    def get_depot(self, area: Any, data_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Implementation of get_depot that reads from Facilities.csv.

        Args:
            area: Name of the geographic area.
            data_dir: Optional path to the data directory.

        Returns:
            Pandas DataFrame containing depot coordinates and initial state.
        """
        src_area = area.translate(str.maketrans("", "", "-_ ")).lower()
        d_dir = self._get_data_dir(data_dir)
        facilities = pd.read_csv(os.path.join(d_dir, "coordinates", "Facilities.csv"))
        depot_df = (
            facilities[facilities["Sigla"] == udef.MAP_DEPOTS[src_area]].loc[:, ["Lat", "Lng"]].reset_index(drop=True)
        )
        depot_df.insert(0, "ID", [0])
        new_cols = pd.DataFrame({"Stock": [0], "Accum_Rate": [0]})
        return pd.concat([depot_df, new_cols], axis=1)

    def get_simulator_data(
        self,
        number_of_bins: int,
        area: str = "Rio Maior",
        waste_type: Optional[str] = None,
        lock: Optional[Any] = None,
        data_dir: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Implementation of get_simulator_data that handles area-specific file logic.

        Args:
            number_of_bins: Number of bins to load.
            area: Name of the geographic area.
            waste_type: Type of waste.
            lock: Optional file lock.
            data_dir: Optional path to the data directory.

        Returns:
            Tuple containing (stats_df, coordinates_df).
        """
        d_dir = self._get_data_dir(data_dir)
        src_area = area.translate(str.maketrans("", "", "-_ ")).lower()
        wtype = udef.WASTE_TYPES[waste_type] if waste_type and waste_type in udef.WASTE_TYPES else None

        if lock is not None:
            lock.acquire(timeout=udef.LOCK_TIMEOUT)
        try:
            if src_area == "mixrmbac":
                data, bins_coordinates = self._get_mixrmbac_data(d_dir, number_of_bins, src_area)
            elif src_area == "riomaior":
                data, bins_coordinates = self._get_riomaior_data(d_dir, number_of_bins, src_area, wtype)
            elif src_area == "figueiradafoz":
                data, bins_coordinates = self._get_figueiradafoz_data(d_dir, number_of_bins, src_area, wtype)
            else:
                assert src_area == "both", f"Invalid area: {src_area}"
                data, bins_coordinates = self._get_both_areas_data(d_dir, number_of_bins, src_area)

            # Final filtering/sorting common to all
            data = data[data["ID"].isin(bins_coordinates["ID"])]
            bins_coordinates = bins_coordinates[bins_coordinates["ID"].isin(data["ID"])]

            data = data.sort_values(by="ID").reset_index(drop=True)
            bins_coordinates = bins_coordinates.sort_values(by="ID").reset_index(drop=True)

            with contextlib.suppress(Exception):
                run = get_active_run() if get_active_run is not None else None
                if run is not None:
                    run.log_dataset_event(
                        "mutate",
                        metadata={
                            "event": "filter_mismatched_bins",
                            "variable_name": "base_data",
                            "n_bins": len(data),
                            "source_file": "repository/filesystem.py",
                            "source_line": 132,
                        },
                    )

            return data, bins_coordinates
        finally:
            if lock is not None:
                lock.release()

    def _get_mixrmbac_data(self, d_dir: str, number_of_bins: int, src_area: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get data for mixrmbac area.

        Args:
            d_dir: Root data directory.
            number_of_bins: Target bin count.
            src_area: Normalized area name.

        Returns:
            Tuple containing (stats_df, coordinates_df).
        """
        if number_of_bins <= 20:
            data = pd.read_excel(os.path.join(d_dir, "bins_waste", "StockAndAccumulationRate - small.xlsx"))
            bins_coordinates = pd.read_excel(os.path.join(d_dir, "coordinates", "Coordinates - small.xlsx"))
        elif number_of_bins <= 50:
            data = pd.read_excel(os.path.join(d_dir, "bins_waste", "StockAndAccumulationRate - 50bins.xlsx"))
            bins_coordinates = pd.read_excel(os.path.join(d_dir, "coordinates", "Coordinates - 50bins.xlsx"))
        else:
            assert number_of_bins <= 225, f"Number of bins for area {src_area} must be <= 225"
            data = pd.read_excel(os.path.join(d_dir, "bins_waste", "StockAndAccumulationRate.xlsx"))
            bins_coordinates = pd.read_excel(os.path.join(d_dir, "coordinates", "Coordinates.xlsx"))
        return data, bins_coordinates

    def _get_riomaior_data(
        self, d_dir: str, number_of_bins: int, src_area: str, wtype: Optional[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get data for Rio Maior area.

        Args:
            d_dir: Root data directory.
            number_of_bins: Target bin count.
            src_area: Normalized area name.
            wtype: Waste type filter string.

        Returns:
            Tuple containing (stats_df, coordinates_df).
        """
        assert number_of_bins <= 317, f"Number of bins for area {src_area} must be <= 317"
        if number_of_bins == 104:
            df = pd.read_csv(os.path.join(d_dir, "bins_waste", "Rio_Maior_Sensores_2021_2024_cleaned_104.csv"))
            df["Data da leitura"] = pd.to_datetime(df["Data da leitura"], format="%Y-%m-%d")
            data = df.pivot_table(index="Data da leitura", columns="idcontentor", values="Enchimento", aggfunc="mean")
            data = data.round()
            data.columns = data.columns.astype(int)
            coords_tmp = pd.read_csv(os.path.join(d_dir, "coordinates", "coordinates104.csv"))
            coords_tmp = coords_tmp.rename(columns={"Lon": "Lng", "Longitude": "Lng", "Latitude": "Lat"})
        else:
            data = self._preprocess_county_date(
                pd.read_csv(os.path.join(d_dir, "bins_waste", f"old_out_crude_rate[{src_area}].csv"))
            )
            coords_tmp = pd.read_csv(os.path.join(d_dir, "coordinates", f"old_out_info[{src_area}].csv"))
            coords_tmp = coords_tmp.rename(columns={"Latitude": "Lat", "Longitude": "Lng"})
            if wtype:
                _n_before = len(coords_tmp)
                coords_tmp = coords_tmp[coords_tmp["Tipo de Residuos"] == wtype]
                with contextlib.suppress(Exception):
                    run = get_active_run() if get_active_run is not None else None
                    if run is not None:
                        run.log_params(
                            {
                                "data.waste_filter": wtype,
                                "data.bins_before_waste_filter": _n_before,
                                "data.bins_after_waste_filter": len(coords_tmp),
                            }
                        )

        bins_coordinates = coords_tmp[["ID", "Lat", "Lng"]]
        data = self._preprocess_county_data(data)
        return data, bins_coordinates

    def _get_figueiradafoz_data(
        self, d_dir: str, number_of_bins: int, src_area: str, wtype: Optional[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get data for Figueira da Foz area.

        Args:
            d_dir: Root data directory.
            number_of_bins: Target bin count.
            src_area: Normalized area name.
            wtype: Waste type filter string.

        Returns:
            Tuple containing (stats_df, coordinates_df).
        """
        data = self._preprocess_county_date(
            pd.read_csv(os.path.join(d_dir, "bins_waste", "out_rate_crude[figdafoz].csv"))
        )
        assert number_of_bins <= 1094, f"Number of bins for area {src_area} must be <= 1094"
        coords_tmp = pd.read_csv(os.path.join(d_dir, "coordinates", "out_info[figdafoz].csv"))
        coords_tmp = coords_tmp.rename(columns={"Latitude": "Lat", "Longitude": "Lng"})
        # Coordinates are stored without decimal points (e.g. 401484222222222 instead of
        # 40.1484222222222). Recover the decimal value by counting digits: a lat in the
        # 40.xxx range has 2 integer digits, so divide by 10^(total_digits - 2); a lng
        # in the -8.xxx range has 1 integer digit, so divide by 10^(total_digits - 1).
        def _decode_coord(s: pd.Series, integer_digits: int) -> pd.Series:
            v = s.astype(float)
            abs_v = v.abs().replace(0, 1)
            total_digits = np.floor(np.log10(abs_v)) + 1
            scale = (total_digits - integer_digits).clip(lower=0)
            return v / (10.0 ** scale)

        coords_tmp["Lat"] = _decode_coord(coords_tmp["Lat"], 2)
        coords_tmp["Lng"] = _decode_coord(coords_tmp["Lng"], 1)
        if wtype:
            _n_before = len(coords_tmp)
            coords_tmp = coords_tmp[coords_tmp["description"] == wtype]
            with contextlib.suppress(Exception):
                run = get_active_run() if get_active_run is not None else None
                if run is not None:
                    run.log_params(
                        {
                            "data.waste_filter": wtype,
                            "data.bins_before_waste_filter": _n_before,
                            "data.bins_after_waste_filter": len(coords_tmp),
                        }
                    )
        bins_coordinates = coords_tmp[["ID", "Lat", "Lng"]]
        data = self._preprocess_county_data(data)
        return data, bins_coordinates

    def _get_both_areas_data(self, d_dir: str, number_of_bins: int, src_area: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get data for both areas combined.

        Args:
            d_dir: Root data directory.
            number_of_bins: Target bin count.
            src_area: Normalized area name.

        Returns:
            Tuple containing (stats_df, coordinates_df).
        """
        wsrs_data = pd.read_excel(os.path.join(d_dir, "bins_waste", "StockAndAccumulationRate.xlsx"))
        wsba_data = self._preprocess_county_data(
            self._preprocess_county_date(
                pd.read_csv(os.path.join(d_dir, "bins_waste", f"old_out_crude_rate[{src_area}].csv"))
            )
        )
        if number_of_bins <= 57:
            coords_tmp = pd.read_csv(os.path.join(d_dir, "coordinates", "intersection.csv"))
            bins_coordinates = coords_tmp.drop("ID317", axis=1).rename(columns={"ID225": "ID"})
            data = wsrs_data[wsrs_data["ID"].isin(bins_coordinates["ID"])]
        elif number_of_bins <= 371:
            coords_tmp = pd.read_csv(os.path.join(d_dir, "coordinates", "merged.csv"))
            data = wsrs_data[wsrs_data["ID"].isin(coords_tmp["ID225"])]
            data_tmp = wsba_data[wsba_data["ID"].isin(coords_tmp.loc[coords_tmp["ID225"].isna(), "ID317"])]
            data = pd.concat([data, data_tmp], axis=0)
            bins_coordinates = pd.DataFrame(
                {
                    "ID": coords_tmp["ID225"].fillna(coords_tmp["ID317"]),
                    "Lat": coords_tmp["Lat"],
                    "Lng": coords_tmp["Lng"],
                }
            )
        elif number_of_bins <= 485:
            coords_tmp = pd.read_csv(os.path.join(d_dir, "coordinates", "union.csv"))
            data = wsba_data[wsba_data["ID"].isin(coords_tmp["ID317"])]
            data_tmp = wsrs_data[wsrs_data["ID"].isin(coords_tmp.loc[coords_tmp["ID317"].isna(), "ID225"])]
            data = pd.concat([data, data_tmp], axis=0)
            bins_coordinates = pd.DataFrame(
                {
                    "ID": coords_tmp["ID317"].fillna(coords_tmp["ID225"]),
                    "Lat": coords_tmp["Lat"],
                    "Lng": coords_tmp["Lng"],
                }
            )
        else:
            assert number_of_bins <= 542, f"Number of bins for {src_area} must be <= 542"
            bins_coordinates = pd.read_csv(os.path.join(d_dir, "coordinates", f"old_out_info[{src_area}].csv"))
            bins_coordinates = bins_coordinates.rename(columns={"Latitude": "Lat", "Longitude": "Lng"})
            bins_coordinates = bins_coordinates[["ID", "Lat", "Lng"]]
            coords_tmp = pd.read_excel(os.path.join(d_dir, "Coordinates.xlsx"))

            data = wsba_data[wsba_data["ID"].isin(bins_coordinates["ID"])]
            data_tmp = wsrs_data[wsrs_data["ID"].isin(coords_tmp["ID"])]

            # Change ID since bin with same ID (but diff coords) exists in out_INFO.csv
            coords_tmp.loc[coords_tmp["ID"] == 1610, "ID"] = coords_tmp.iloc[-1]["ID"] + 1
            data_tmp.loc[data_tmp["ID"] == 1610, "ID"] = data_tmp.iloc[-1]["ID"] + 1
            bins_coordinates = pd.concat([bins_coordinates, coords_tmp])
            data = pd.concat([data, data_tmp])

        return data, bins_coordinates

    def _preprocess_county_date(self, data: pd.DataFrame, date_str: str = "Date") -> pd.DataFrame:
        """Preprocess county date.

        Args:
            data: Raw DataFrame with date column.
            date_str: Name of the column containing dates.

        Returns:
            DataFrame indexed by date with integer column names.
        """
        data[date_str] = pd.to_datetime(data[date_str], format="%Y-%m-%d")
        data = data.set_index(date_str)
        data = data.round()
        data.columns = data.columns.astype(int)
        return data

    def _preprocess_county_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess county data for simulation.

        Args:
            data: Pivoted DataFrame with bin levels.

        Returns:
            DataFrame containing normalized Stock and Accum_Rate for each bin.
        """

        def __get_stock(col):
            """Helper to extract initial stock from time series column."""
            positive_values = col[col >= 1e-32].dropna()
            if not positive_values.empty:
                return positive_values.iloc[0]
            return 0

        accum_rate = data.clip(lower=0).fillna(0).mean()
        stock = cast(pd.Series, data.apply(__get_stock, axis=0))
        new_data = pd.DataFrame({"ID": data.columns})
        new_data["Stock"] = new_data["ID"].map(stock)
        new_data["Accum_Rate"] = new_data["ID"].map(accum_rate)

        # Capture raw distribution stats before min-max normalisation
        _raw_stock_min = float(new_data["Stock"].min())
        _raw_stock_max = float(new_data["Stock"].max())
        _raw_rate_min = float(new_data["Accum_Rate"].min())
        _raw_rate_max = float(new_data["Accum_Rate"].max())

        new_data[["Stock", "Accum_Rate"]] = (
            new_data[["Stock", "Accum_Rate"]] - new_data[["Stock", "Accum_Rate"]].min()
        ) / (new_data[["Stock", "Accum_Rate"]].max() - new_data[["Stock", "Accum_Rate"]].min())

        with contextlib.suppress(Exception):
            run = get_active_run() if get_active_run is not None else None
            if run is not None:
                run.log_params(
                    {
                        "data.raw_stock_min": _raw_stock_min,
                        "data.raw_stock_max": _raw_stock_max,
                        "data.raw_accum_rate_min": _raw_rate_min,
                        "data.raw_accum_rate_max": _raw_rate_max,
                        "data.n_bins_preprocessed": len(new_data),
                    }
                )
                run.log_dataset_event(
                    "mutate",
                    metadata={
                        "event": "min_max_normalisation",
                        "variable_name": "base_data",
                        "n_bins_preprocessed": len(new_data),
                        "source_file": "repository/filesystem.py",
                        "source_line": 299,
                    },
                )

        return new_data
