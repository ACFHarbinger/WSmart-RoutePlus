"""loader.py module.

Attributes:
    collate_fn: Collate function for PyTorch DataLoader.
    load_focus_coords: Loads coordinates and depot information from simulator data.
    check_extension: Ensures filename has the specified extension.
    save_dataset: Saves a dataset using pickle.
    save_td_dataset: Saves a TensorDict dataset.
    load_td_dataset: Loads a TensorDict dataset.
    load_dataset: Loads a dataset from a pickle file.
    save_simulation_dataset: Saves a simulation dataset as a compressed .npz file.
    load_simulation_dataset: Loads a simulation dataset from a .npz file.
    load_grid_base: Loads grid base data from simulation files.

Example:
    >>> import loader
    >>> loader.load_dataset("path/to/dataset.pkl")
"""

from __future__ import annotations

import contextlib
import json
import os
import pickle
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from tensordict import TensorDict

# --- Function Definitions (Top-level for circular import safety) ---


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for PyTorch DataLoader.
    Filters out None values from samples.

    Args:
        batch: List of samples.

    Returns:
        Collated batch.
    """
    batch = [{key: val for key, val in sample.items() if val is not None} for sample in batch if sample is not None]

    # Empty lists can break collate
    if len(batch) == 0:
        return {}
    return torch.utils.data.dataloader.default_collate(batch)


def load_focus_coords(
    graph_size: int,
    method: Optional[str],
    area: str,
    waste_type: str,
    focus_graph: str,
    focus_size: int = 1,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], List[int], pd.Series]:
    """
    Loads coordinates and depot information from simulator data.

    Args:
        graph_size: Size of the graph.
        method: Method for vertex coordinate processing (e.g., 'mmn', '').
        area: Geographic area name.
        waste_type: Type of waste.
        focus_graph: Path to focus graph file.
        focus_size: Replication factor for focus graph. Defaults to 1.

    Returns:
        Tuple of (depot, locations, minmax array, index).
    """
    from logic.src.utils.functions.path import get_path_until_string

    focus_graph_dir = get_path_until_string(focus_graph, "simulator")
    if focus_graph_dir is None:
        raise ValueError(f"Could not find 'simulator' in path {focus_graph}")

    from logic.src.pipeline.simulations.repository import load_depot, load_simulator_data

    depot = load_depot(focus_graph_dir, area)
    data, coords = load_simulator_data(focus_graph_dir, graph_size, area, waste_type)
    with open(focus_graph) as js:
        idx = json.load(js)

    from logic.src.data.processor import process_coordinates, process_data

    _, coords = process_data(data, coords, depot, idx[0])
    depot, loc = process_coordinates(coords, method)
    if focus_size > 0:
        lat_minmax = (coords["Lat"].min(), coords["Lat"].max())
        lng_minmax = (coords["Lng"].min(), coords["Lng"].max())
        mm_arr = np.array([lng_minmax, lat_minmax])
        ret_val = (
            np.tile(depot, (focus_size, 1)),
            np.tile(loc, (focus_size, 1, 1)),
            mm_arr.T,
            idx,
            coords["ID"],
        )
    else:
        ret_val = (depot, loc, None, idx, coords["ID"])  # type: ignore
    return ret_val  # type: ignore


def check_extension(filename: str, extension: str = ".pkl") -> str:
    """
    Ensures filename has the specified extension.

    Args:
        filename: Input filename.
        extension: Desired extension (e.g., '.pkl', '.td', '.pt').

    Returns:
        Filename with the specified extension.
    """
    if os.path.splitext(filename)[1] != extension:
        return filename + extension
    return filename


def save_dataset(dataset: Any, filename: str) -> None:
    """
    Saves a dataset using pickle.

    Args:
        dataset: The data to save.
        filename: Target filename.

    Raises:
        Exception: If directory creation fails.
    """
    filedir = os.path.split(filename)[0]
    if filedir and not os.path.isdir(filedir):
        try:
            os.makedirs(filedir, exist_ok=True)
        except Exception as e:
            raise Exception("directories to save datasets do not exist and could not be created") from e

    with open(filename, "wb") as f:
        pickle.dump(dataset, f)


def save_td_dataset(td: TensorDict, filename: str) -> None:
    """
    Saves a TensorDict dataset.

    Args:
        td: The TensorDict to save.
        filename: Target filename.
    """
    filedir = os.path.split(filename)[0]
    if filedir and not os.path.isdir(filedir):
        os.makedirs(filedir, exist_ok=True)

    torch.save(td, check_extension(filename, ".td"))


def load_td_dataset(filename: str, device: str = "cpu") -> TensorDict:
    """
    Loads a TensorDict dataset.

    Args:
        filename: The filename.
        device: Device to load onto.

    Returns:
        The loaded TensorDict.
    """
    return torch.load(check_extension(filename, ".td"), map_location=device, weights_only=False)


def load_dataset(filename: str) -> Any:
    """
    Loads a dataset from a pickle file.

    Args:
        filename: The filename.

    Returns:
        The loaded dataset.
    """
    with open(check_extension(filename, ".pkl"), "rb") as f:
        return pickle.load(f)


def save_simulation_dataset(dataset: Dict[str, np.ndarray], filename: str) -> None:
    """
    Saves a simulation dataset as a compressed .npz file.

    Args:
        dataset: Dict of named numpy arrays (e.g. 'waste', 'noisy_waste', 'depot', 'locs', 'max_waste').
        filename: Target filename (extension will be set to .npz).
    """
    filedir = os.path.split(filename)[0]
    if filedir and not os.path.isdir(filedir):
        try:
            os.makedirs(filedir, exist_ok=True)
        except Exception as e:
            raise Exception("directories to save datasets do not exist and could not be created") from e

    print(f"[INFO] Saving compressed dataset to '{filename}'... (This may take a minute for large datasets)")
    np.savez_compressed(check_extension(filename, ".npz"), **dataset)


def load_simulation_dataset(filename: str) -> Dict[str, np.ndarray]:
    """
    Loads a simulation dataset from a .npz file.

    Args:
        filename: The filename.

    Returns:
        Dict of named numpy arrays.
    """
    data = np.load(check_extension(filename, ".npz"))
    return dict(data)


def load_grid_base(
    indices: Union[np.ndarray, List[int]],
    area: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> GridBase:
    """
    Loads grid base data from simulation files.

    Args:
        indices: List of indices to load.
        area: Geographic area name.
        data_dir: Directory containing the simulation data.

    Returns:
        GridBase object.
    """
    from logic.src.constants.paths import ROOT_DIR
    from logic.src.constants.waste import COUNTY_ALIASES
    from logic.src.pipeline.simulations.wsmart_bin_analysis import GridBase

    if data_dir is None:
        data_dir = os.path.join(ROOT_DIR, "data", "simulator")

    src_area = area.translate(str.maketrans("", "", "-_ ")).lower() if area is not None else ""
    src_area = COUNTY_ALIASES.get(src_area, src_area)

    waste_csv = f"out_rate_crude[{src_area}].csv"
    info_csv = f"out_info[{src_area}].csv"

    # Handle nested list of indices (list of lists)
    if isinstance(indices, list) and len(indices) > 0 and isinstance(indices[0], list):
        indices = indices[0]

    # Read info file to map indices to IDs
    info_path = os.path.join(data_dir, "coordinates", info_csv)
    if os.path.exists(info_path):
        info_df = pd.read_csv(info_path)
        # Heuristic: if all indices are within bounds of info_df, treat them as indices
        try:
            if all(isinstance(i, (int, np.integer)) and 0 <= i < len(info_df) for i in indices):
                real_ids = info_df.iloc[indices]["ID"].tolist()
            else:
                real_ids = [str(i) for i in indices]
        except (TypeError, IndexError, ValueError):
            # Fallback if indices are not list-like or other issues
            real_ids = [str(i) for i in indices]
    else:
        # Fallback if no info file, treat indices as IDs
        real_ids = [str(i) for i in indices]

    # Check ID type in waste csv to ensure correct matching
    waste_path = os.path.join(data_dir, "bins_waste", waste_csv)
    if os.path.exists(waste_path):
        waste_header = pd.read_csv(waste_path, nrows=0).columns
        # If waste header uses strings (e.g. "192"), coerce real_ids to strings
        if pd.api.types.is_string_dtype(waste_header):
            real_ids = [str(i) for i in real_ids]
        # Conversely, if waste header is numeric but real_ids are strings, coerce back
        elif pd.api.types.is_numeric_dtype(waste_header):
            with contextlib.suppress(ValueError, TypeError):
                real_ids = [int(float(i)) for i in real_ids]

    return GridBase(
        real_ids,
        data_dir,
        rate_type="crude",
        names=[waste_csv, info_csv, None],
        same_file=True,
    )


# --- Logic Strategy Imports (Below function definitions) ---

if TYPE_CHECKING:
    from logic.src.pipeline.simulations.wsmart_bin_analysis import GridBase
