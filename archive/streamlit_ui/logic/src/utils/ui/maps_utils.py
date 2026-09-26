"""
Shared utilities and constants for map rendering.

Attributes:
    get_map_center: Calculate the center point of a tour.
    load_distance_matrix: Load the distance matrix for a given problem instance.

Example:
    >>> from logic.src.utils.ui.maps_utils import get_map_center, load_distance_matrix
    >>> get_map_center([{"lat": 1, "lng": 2}, {"lat": 3, "lng": 4}])
    (2.0, 3.0)
    >>> load_distance_matrix("riomaior")
    DataFrame containing the distance matrix for riomaior.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def get_map_center(tour: List[Dict[str, Any]]) -> Tuple[float, float]:
    """
    Calculate the center point of a tour.

    Args:
        tour: List of tour points with lat/lng.

    Returns:
        (latitude, longitude) tuple for center.
    """
    lats = [p.get("lat", 0) for p in tour if "lat" in p]
    lngs = [p.get("lng", 0) for p in tour if "lng" in p]

    if not lats or not lngs:
        return (39.33, -8.94)  # Default to Portugal area

    return (sum(lats) / len(lats), sum(lngs) / len(lngs))


def load_distance_matrix(instance_name: str = "riomaior") -> Optional[pd.DataFrame]:
    """
    Load the distance matrix for a given problem instance.

    Args:
        instance_name: Name of the instance (e.g., 'riomaior').

    Returns:
        DataFrame containing the distance matrix, or None if not found.
    """
    # Try to find a matching file in data/simulator/distance_matrix
    # Common pattern seems to be gmaps_distmat_plastic[{instance_name}].csv
    base_path = Path("data/simulator/distance_matrix")
    if not base_path.exists():
        return None

    # Search for files containing the instance name
    candidates = list(base_path.glob(f"*{instance_name}*.csv"))

    if not candidates:
        return None

    # Prefer the 'plastic' one if multiple, or just take the first
    # Example: gmaps_distmat_plastic[riomaior].csv
    selected_file = candidates[0]
    for cand in candidates:
        if f"plastic[{instance_name}]" in cand.name:
            selected_file = cand
            break

    try:
        # Load matrix, assuming first row/col are headers/indices if it's a named matrix
        # Based on file inspection, it might be a raw matrix or have headers.
        # Usually these matrices are square.
        df = pd.read_csv(selected_file, header=None)
        return df
    except Exception:
        return None
