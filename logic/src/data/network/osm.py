"""
Strategy for computing distances using OpenStreetMap via OSRM.

Attributes:
    OSMStrategy: Strategy for computing road distances via the OSRM Table API.

Example:
    from logic.src.data.network.osm import OSMStrategy
    strategy = OSMStrategy()
    strategy.calculate(pd.DataFrame({"Lng": [-8.81, -8.82], "Lat": [40.15, 40.16]}))
"""

from typing import Any

import numpy as np
import pandas as pd
import requests  # type: ignore[import-untyped]  # requests has no stubs

from .base import DistanceStrategy


class OSMStrategy(DistanceStrategy):
    """Strategy for computing road distances via the OSRM Table API (no key required).

    Batches coordinates into chunks of FREE_SIZE // 2 sources and FREE_SIZE // 2
    destinations per request, keeping the total coordinate count within the
    FREE_SIZE limit imposed by the public OSRM demo server.

    Attributes:
        None
    """

    def calculate(self, coords: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """
        Computes road distances using the OSRM Table API.

        Args:
            coords: DataFrame with coordinates (must contain 'Lat', 'Lng' columns).
            kwargs: Additional arguments (unused, accepted for interface compatibility).

        Returns:
            np.ndarray: Distance matrix in kilometres.
        """
        _OSRM_URL = "http://router.project-osrm.org/table/v1/driving"
        FREE_SIZE = 100  # max coordinates per OSRM table request

        half = FREE_SIZE // 2
        size = len(coords)
        distance_matrix = np.zeros((size, size))

        # OSRM expects lng,lat order (opposite of lat,lng)
        lnglat = coords[["Lng", "Lat"]].values.tolist()
        for i in range(0, size, half):
            src_slice = list(range(i, min(i + half, size)))
            for j in range(0, size, half):
                dst_slice = list(range(j, min(j + half, size)))

                # Deduplicate so diagonal blocks (src == dst) don't exceed the limit
                all_idx = list(dict.fromkeys(src_slice + dst_slice))
                local = {g: l for l, g in enumerate(all_idx)}

                coords_str = ";".join(f"{lnglat[k][0]},{lnglat[k][1]}" for k in all_idx)
                src_param = ";".join(str(local[k]) for k in src_slice)
                dst_param = ";".join(str(local[k]) for k in dst_slice)

                resp = requests.get(
                    f"{_OSRM_URL}/{coords_str}",
                    params={"sources": src_param, "destinations": dst_param, "annotations": "distance"},
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()

                for r, si in enumerate(src_slice):
                    for c, dj in enumerate(dst_slice):
                        val = data["distances"][r][c]
                        if val is not None:
                            distance_matrix[si, dj] = val / 1000  # metres → km

        return distance_matrix
