"""Repository Pattern for Simulation data access.

Attributes:
    _REPOSITORY: Singleton repository instance.

Example:
    None
"""

import contextlib
import os
from typing import TYPE_CHECKING, Optional, Union

import torch

from logic.src.constants import DATASET_EXTENSIONS, ROOT_DIR

if TYPE_CHECKING:
    from logic.src.data.datasets import NumpyDictDataset, PandasCsvDataset, PandasExcelDataset


from .base import SimulationRepository
from .dataset import DatasetRepository
from .filesystem import FileSystemRepository

try:
    from logic.src.tracking.core.run import get_active_run
    from logic.src.tracking.integrations.data import RuntimeDataTracker
except ImportError:
    get_active_run = None  # type: ignore[assignment,misc]
    RuntimeDataTracker = None  # type: ignore[assignment,misc]

# Singleton repository instance
_REPOSITORY = None


def set_repository(
    repo: Union[DatasetRepository, FileSystemRepository],
) -> None:
    """
    Replace the singleton repository instance.

    Call this before simulation initialization to switch between
    filesystem-based and npz-based data loading.

    Args:
        repo: The repository instance to set.
    """
    global _REPOSITORY
    _REPOSITORY = repo
    with contextlib.suppress(Exception):
        run = get_active_run() if get_active_run is not None else None
        if run is not None:
            run.log_params({"sim.repository_type": type(repo).__name__})


def set_repository_from_path(
    path: str,
    root_dir: Optional[Union[str, os.PathLike]] = None,
    area: Optional[str] = None,
    waste_type: Optional[str] = None,
) -> bool:
    """Load a dataset file (or directory) and set it as the active repository.

    Detects the dataset format from the file extension, loads it using
    the matching dataset class, wraps it in a ``DatasetRepository``,
    and calls ``set_repository()``.  If *path* points to a directory,
    a ``FileSystemRepository`` is created instead.

    Args:
        path: Path to a dataset file (``.npz``, ``.xlsx``, ``.csv``) or
              a data directory.  Can be relative (resolved against
              *root_dir*) or absolute.
        root_dir: Base directory for resolving relative paths.
                  Defaults to ``ROOT_DIR``.
        area: Optional geographic area name for coordinate fallback.
        waste_type: Optional waste type for coordinate fallback.

    Returns:
        ``True`` if the repository was successfully set, ``False`` if the
        path does not exist or has an unsupported extension.
    """
    if root_dir is None:
        root_dir = str(ROOT_DIR)

    abs_path = os.path.join(root_dir, path) if not os.path.isabs(path) else path

    # Directory → FileSystemRepository
    if os.path.isdir(abs_path):
        set_repository(FileSystemRepository(abs_path))
        return True

    # File → DatasetRepository
    if not os.path.isfile(abs_path):
        return False

    ext = os.path.splitext(abs_path)[1].lower()
    if ext not in DATASET_EXTENSIONS:
        return False

    from logic.src.data.datasets import NumpyDictDataset, PandasCsvDataset, PandasExcelDataset

    loader_map = {
        ".npz": NumpyDictDataset,
        ".xlsx": PandasExcelDataset,
        ".csv": PandasCsvDataset,
    }

    dataset = loader_map[ext].load(abs_path, area=area, waste_type=waste_type)  # type: ignore[attr-defined]
    set_repository(DatasetRepository(dataset))
    return True


def load_indices(filename, n_samples, n_nodes, data_size, lock=None):
    """
    Convenience wrapper to load indices from the singleton repository.

    Args:
        filename: Path to the index file.
        n_samples: Number of samples.
        n_nodes: Number of nodes.
        data_size: Size of the dataset.
        lock: Optional file lock.

    Returns:
        List of bin indices for each sample.
    """
    assert _REPOSITORY is not None, "Repository not initialized. Call set_repository() first."
    indices = _REPOSITORY.get_indices(filename, n_samples, n_nodes, data_size, lock)
    with contextlib.suppress(Exception):
        run = get_active_run() if get_active_run is not None else None
        if run is not None:
            run.log_params(
                {
                    "data.indices_file": str(filename),
                    "data.n_samples": n_samples,
                    "data.n_nodes": n_nodes,
                    "data.data_size": data_size,
                }
            )
            run.log_dataset_event(
                "load",
                file_path=str(filename),
                shape=(n_samples,),
                metadata={
                    "n_nodes": n_nodes,
                    "data_size": data_size,
                    "variable_name": "indices",
                    "source_file": "repository/__init__.py",
                    "source_line": 53,
                },
            )
    return indices


def load_depot(data_dir, area="Rio Maior"):
    """
    Convenience wrapper to load depot coords from the singleton repository.

    Args:
        data_dir: Path to the data directory.
        area: Name of the geographic area.

    Returns:
        Pandas DataFrame with depot coordinates.
    """
    assert _REPOSITORY is not None, "Repository not initialized. Call set_repository() first."
    depot_df = _REPOSITORY.get_depot(area, data_dir=data_dir)
    with contextlib.suppress(Exception):
        run = get_active_run() if get_active_run is not None else None
        if run is not None:
            lat = float(depot_df["Lat"].iloc[0])
            lng = float(depot_df["Lng"].iloc[0])
            run.log_params(
                {
                    "data.depot_area": str(area),
                    "data.depot_lat": lat,
                    "data.depot_lng": lng,
                }
            )
            run.log_dataset_event(
                "load",
                shape=depot_df.shape,
                metadata={
                    "event": "depot_load",
                    "area": str(area),
                    "variable_name": "depot_df",
                    "source_file": "repository/__init__.py",
                    "source_line": 82,
                },
            )
    return depot_df


def load_simulator_data(data_dir, number_of_bins, area="Rio Maior", waste_type=None, lock=None):
    """
    Convenience wrapper to load simulator data from the singleton repository.

    Args:
        data_dir: Path to the data directory.
        number_of_bins: Number of bins.
        area: Name of the geographic area.
        waste_type: Type of waste.
        lock: Optional file lock.

    Returns:
        Description of return value.
    """
    assert _REPOSITORY is not None, "Repository not initialized. Call set_repository() first."
    data, bins_coordinates = _REPOSITORY.get_simulator_data(number_of_bins, area, waste_type, lock, data_dir=data_dir)
    with contextlib.suppress(Exception):
        run = get_active_run() if get_active_run is not None else None
        if run is not None:
            n_bins = len(data)
            run.log_params(
                {
                    "data.area": str(area),
                    "data.n_bins_requested": number_of_bins,
                    "data.n_bins_loaded": n_bins,
                    "data.waste_type": str(waste_type) if waste_type else "all",
                }
            )
            run.log_dataset_event(
                "load",
                shape=data.shape,
                metadata={
                    "event": "simulator_data_load",
                    "area": str(area),
                    "waste_type": str(waste_type) if waste_type else "all",
                    "variable_name": "data",
                    "source_file": "repository/__init__.py",
                    "source_line": 113,
                },
            )
            stock_tensor = torch.as_tensor(data["Stock"].values, dtype=torch.float32)
            rate_tensor = torch.as_tensor(data["Accum_Rate"].values, dtype=torch.float32)
            RuntimeDataTracker(run).on_load(
                {"stock": stock_tensor, "accum_rate": rate_tensor},
                shape=stock_tensor.shape,
                metadata={"area": str(area), "waste_type": str(waste_type) if waste_type else "all"},
                log_event=False,
            )
    return data, bins_coordinates


def load_area_and_waste_type_params(area, waste_type):
    """
    Convenience wrapper to load area params.

    Args:
        area: Geographic area name.
        waste_type: Type of waste.

    Returns:
        Dictionary containing area parameters.
    """
    params = SimulationRepository.get_area_params(area, waste_type)
    with contextlib.suppress(Exception):
        run = get_active_run() if get_active_run is not None else None
        if run is not None:
            run.log_params(
                {
                    "data.area_params_area": str(area),
                    "data.area_params_waste_type": str(waste_type) if waste_type else "all",
                }
            )
    return params


def load_temporal_params(**kwargs):
    """
    Convenience wrapper to load the temporal (time-budget) resource params.

    Returns:
        Tuple of (shift_hours, speed_or_time_matrix, service_time_h). See
        SimulationRepository.get_temporal_params.
    """
    params = SimulationRepository.get_temporal_params(**kwargs)
    with contextlib.suppress(Exception):
        run = get_active_run() if get_active_run is not None else None
        if run is not None:
            run.log_params(
                {
                    "data.temporal_shift_hours": params[0],
                    "data.temporal_avg_speed_kmh": kwargs.get("avg_speed_kmh", 35.0),
                    "data.temporal_matrix_path": str(kwargs.get("tm_filepath") or "uniform_speed"),
                    "data.temporal_service_time_h": params[2],
                }
            )
    return params


__all__ = [
    "SimulationRepository",
    "FileSystemRepository",
    "DatasetRepository",
    "set_repository",
    "set_repository_from_path",
    "load_indices",
    "load_depot",
    "load_simulator_data",
    "load_area_and_waste_type_params",
    "load_temporal_params",
]
