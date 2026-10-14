"""Travel-time matrices in hours, aligned to simulation coordinates."""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from logic.src.constants import ROOT_DIR


def _node_id(value: object) -> str:
    """Normalize numeric IDs and dashboard labels such as '663 - 661'."""
    label = str(value).strip().split(" - ")[0]
    try:
        return str(int(float(label))) if float(label).is_integer() else label
    except ValueError:
        return label


def compute_time_matrix(
    coords: pd.DataFrame,
    distance_matrix: np.ndarray,
    tm_filepath: Optional[Union[str, Path]] = None,
    avg_speed_kmh: float = 35.0,
    time_unit: str = "seconds",
) -> np.ndarray:
    """Create uniform-speed times or read an ID-labelled CSV matrix.

    Distances are kilometres; the returned directed matrix is always hours.
    Bare filenames resolve under ``data/simulator/time_matrix``. CSVs contain
    row and column IDs; dashboard ``ID - secondary ID`` labels use the first
    ID. Both axes are independently reordered to match ``coords``. Bin-only
    files use distance/speed for depot legs (the simulator depot has ID 0).
    Missing customer IDs are errors, never positional substitutions.
    """
    distances = np.asarray(distance_matrix, dtype=float)
    size = len(coords)
    if distances.shape != (size, size):
        raise ValueError("Distance matrix must match the coordinate count")
    if not np.isfinite(avg_speed_kmh) or avg_speed_kmh <= 0:
        raise ValueError("avg_speed_kmh must be finite and positive")
    if not np.isfinite(distances).all() or (distances < 0).any():
        raise ValueError("Distances must be finite and nonnegative")
    times = distances / avg_speed_kmh
    if tm_filepath is None:
        return times

    factors = {"seconds": 3600.0, "minutes": 60.0, "hours": 1.0}
    if time_unit not in factors:
        raise ValueError("time_unit must be seconds, minutes, or hours")
    path = Path(tm_filepath)
    if path.parent == Path("."):
        path = Path(ROOT_DIR) / "data/simulator/time_matrix" / path
    frame = pd.read_csv(path, index_col=0)
    frame.index = [_node_id(value) for value in frame.index]
    frame.columns = [_node_id(value) for value in frame.columns]
    if not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError("Time matrix IDs must be unique")
    if frame.shape[0] != frame.shape[1] or set(frame.index) != set(frame.columns):
        raise ValueError("Time matrix must be square with matching row and column IDs")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Travel times must be finite and nonnegative")
    if not np.allclose(np.diag(frame.loc[frame.index, frame.index]), 0.0):
        raise ValueError("Time matrix diagonal must be zero")
    requested = [_node_id(value) for value in coords["ID"]]
    if len(set(requested)) != size:
        raise ValueError("Coordinate IDs must be unique")
    missing = set(requested) - set(frame.index) - {"0"}
    if missing:
        raise ValueError(f"Time matrix missing bin IDs: {sorted(missing)}")
    selected = [i for i, node_id in enumerate(requested) if node_id in frame.index]
    labels = [requested[i] for i in selected]
    times[np.ix_(selected, selected)] = frame.loc[labels, labels].to_numpy(dtype=float) / factors[time_unit]
    return times


__all__ = ["compute_time_matrix"]
