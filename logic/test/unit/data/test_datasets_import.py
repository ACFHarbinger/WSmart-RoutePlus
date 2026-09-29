"""Core dataset imports must not require optional dashboard extras."""

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_core_datasets_import_without_beautifulsoup4() -> None:
    """Simulator bins load datasets without the HTML-dashboard crawler extra."""
    from logic.src.data.datasets import (
        GenerativeDataset,
        NumpyDictDataset,
        PandasCsvDataset,
        SimulationDataset,
    )

    assert SimulationDataset is not None
    assert GenerativeDataset is not None
    assert NumpyDictDataset is not None
    assert PandasCsvDataset is not None


def test_haversine_import_does_not_require_geopy() -> None:
    """Coordinate formatting uses haversine without the geo extra."""
    from logic.src.data.network import haversine_distance

    assert haversine_distance(0.0, 0.0, 0.0, 0.0) == 0.0


def test_html_dataset_is_still_reachable() -> None:
    """The dashboard loader stays on the public datasets namespace."""
    from logic.src.data.datasets import HtmlSimulationDataset

    assert HtmlSimulationDataset.__name__ == "HtmlSimulationDataset"
