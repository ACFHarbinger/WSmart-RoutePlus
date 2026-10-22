"""Classical policies get sub-problem coordinates by node index, not by real bin ID."""

from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest
from logic.src.policies.routing.base import base_routing_policy
from logic.src.policies.routing.base.base_routing_policy import BaseRoutingPolicy

pytestmark = [pytest.mark.unit, pytest.mark.fast]


class _CoordProbe(BaseRoutingPolicy):
    """Records the coordinates the template passes to the solver and visits every node."""

    seen: Dict[str, Any] = {}

    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        mandatory_nodes: List[int],
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        _CoordProbe.seen = {"x": kwargs["x_coords"], "y": kwargs["y_coords"], "subset": kwargs["subset_indices"]}
        return [list(range(1, len(sub_dist_matrix)))], 0.0, 0.0


def _coords():
    # The simulator's layout: depot (ID 0) in row 0, then bins sorted by their real IDs.
    ids = [0, 192, 211, 530, 22056]
    return pd.DataFrame({"ID": ids, "Lat": [40.0, 41.0, 42.0, 43.0, 44.0], "Lng": [-8.0, -7.0, -6.0, -5.0, -4.0]})


def test_real_bin_ids_still_yield_coordinates(monkeypatch):
    monkeypatch.setattr(
        base_routing_policy, "load_area_and_waste_type_params", lambda *args: (500.0, 0.2, 10.0, 1.0, 2.5)
    )
    bins = SimpleNamespace(n=4, c=np.array([50.0, 60.0, 70.0, 80.0]), means=np.zeros(4), std=np.zeros(4))
    _CoordProbe().execute(
        mandatory=[2, 4],
        bins=bins,
        distance_matrix=np.ones((5, 5)) - np.eye(5),
        coords=_coords(),
        config={},
    )
    seen = _CoordProbe.seen
    assert seen["x"] is not None and seen["y"] is not None
    subset = list(seen["subset"])
    coords = _coords()
    np.testing.assert_allclose(seen["x"], coords["Lng"].to_numpy()[subset])
    np.testing.assert_allclose(seen["y"], coords["Lat"].to_numpy()[subset])
