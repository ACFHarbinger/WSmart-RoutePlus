"""
Strategy for computing distances using Google Maps API.

Attributes:
    GoogleMapsStrategy: Strategy for computing distances using Google Maps API.

Example:
    >>> from logic.src.data.network.google import GoogleMapsStrategy
    >>> strategy = GoogleMapsStrategy()
    >>> strategy.calculate(pd.DataFrame({"Lng": [0, 1], "Lat": [0, 1]}))
    array([[0.        , 111.31949079],
           [111.31949079, 0.        ]])
"""

import os
from typing import Any, Optional, cast

import googlemaps
import numpy as np
import pandas as pd
from dotenv import dotenv_values

from logic.src.constants import ROOT_DIR

try:
    from logic.src.utils.security import decrypt_file_data, load_key
    _SECURITY_AVAILABLE = True
except ImportError:
    _SECURITY_AVAILABLE = False

from .base import DistanceStrategy


class GoogleMapsStrategy(DistanceStrategy):
    """Strategy for computing distances using Google Maps API.

    Attributes:
        None
    """

    def calculate(self, coords: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """
        Computes distances using the Google Maps Distance Matrix API.

        Args:
            coords: DataFrame with coordinates (must contain 'Lat', 'Lng' columns).
            kwargs: Additional arguments for the distance strategy.

        Returns:
            Distance matrix as a NumPy array.
        """
        assert self._eval_kwarg("env_filename", kwargs)
        env_path = os.path.join(ROOT_DIR, "env", kwargs["env_filename"])
        config = dotenv_values(env_path)
        api_key = config.get("GOOGLE_API_KEY", "")

        if api_key == "" and self._eval_kwarg("symkey_name", kwargs):
            assert self._eval_kwarg("gapik_file", kwargs)
            if not _SECURITY_AVAILABLE:
                raise RuntimeError(
                    "symkey_name was provided but the security module is not available. "
                    "Install it or provide GOOGLE_API_KEY directly in the .env file."
                )
            sym_key = load_key(kwargs["symkey_name"], kwargs["env_filename"])
            api_key = cast(Optional[str], decrypt_file_data(sym_key, kwargs["gapik_file"]))  # decrypted key text
        elif api_key == "" and self._eval_kwarg("gapik_file", kwargs):
            with open(kwargs["gapik_file"], "r") as gapik_file:
                api_key = gapik_file.read()

        # Free tier (no key): 10×10 batches (100 element limit per request).
        # Paid tier (key present): 25×25 batches (625 element limit per request).
        FREE_SIZE = 10
        PAID_SIZE = 25
        batch_size = FREE_SIZE if not api_key else PAID_SIZE

        gmaps = googlemaps.Client(key=api_key) if api_key else googlemaps.Client()
        size = len(coords)
        distance_matrix = np.zeros((size, size))
        src = dst = coords[["Lat", "Lng"]].values.tolist()

        for id_i in range(0, size, batch_size):
            for id_j in range(0, size, batch_size):
                origins = src[id_i : id_i + batch_size]
                dests = dst[id_j : id_j + batch_size]
                response = cast(Any, gmaps).distance_matrix(origins, dests, mode="driving", units="metric")
                for row_id, row in enumerate(response["rows"]):
                    for col_id, elem in enumerate(row["elements"]):
                        if "distance" in elem:
                            distance_matrix[id_i + row_id, id_j + col_id] = elem["distance"]["value"] / 1000

        return distance_matrix
