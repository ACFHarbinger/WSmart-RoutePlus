"""ProblemContext.from_kwargs reads node g's coordinates from row g and the day from the day context."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from logic.src.interfaces.context.problem_context import ProblemContext

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _kwargs(**extra):
    n = 3
    dm = np.ones((n + 1, n + 1))
    np.fill_diagonal(dm, 0.0)
    bins = SimpleNamespace(c=np.array([10.0, 20.0, 30.0]), means=np.zeros(3), std=np.zeros(3), max_fill=100.0)
    # Real bin IDs, as in the simulator's coordinates: none of them equals its node index.
    coords = pd.DataFrame({"ID": [0, 192, 211, 22056], "Lng": [-8.9, -8.1, -8.2, -8.3], "Lat": [39.3, 39.1, 39.2, 39.4]})
    base = {"distance_matrix": dm, "bins": bins, "coords": coords}
    base.update(extra)
    return base


def test_locations_follow_the_row_order_not_the_id_column() -> None:
    problem = ProblemContext.from_kwargs(_kwargs(), 100.0, 1.0, 1.0)
    np.testing.assert_allclose(problem.locations, [[-8.9, 39.3], [-8.1, 39.1], [-8.2, 39.2], [-8.3, 39.4]])


@pytest.mark.parametrize("extra, expected", [({}, 0), ({"day": 4}, 4), ({"day_index": 2, "day": 4}, 2)])
def test_day_index_comes_from_the_day_context(extra, expected) -> None:
    assert ProblemContext.from_kwargs(_kwargs(**extra), 100.0, 1.0, 1.0).day_index == expected
