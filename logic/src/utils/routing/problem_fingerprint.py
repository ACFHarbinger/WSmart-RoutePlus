"""Stable identity of a routing construction instance for paired certificates."""

import hashlib
import json
from typing import Dict, Iterable, Optional

import numpy as np


def problem_fingerprint(
    distances: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    revenue: float,
    cost: float,
    mandatory: Optional[Iterable[int]],
    vehicle_limit: Optional[int],
) -> str:
    """Hash the ordered matrix, objective rates and construction constraints."""
    matrix = np.ascontiguousarray(distances, dtype="<f8")
    metadata = {
        "shape": list(matrix.shape),
        "wastes": [[int(key), float(value)] for key, value in sorted(wastes.items())],
        "capacity": float(capacity),
        "revenue": float(revenue),
        "cost": float(cost),
        "mandatory": sorted(int(node) for node in (mandatory if mandatory is not None else [])),
        "vehicle_limit": int(vehicle_limit) if vehicle_limit is not None and int(vehicle_limit) > 0 else None,
    }
    digest = hashlib.sha256(matrix.tobytes())
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()
