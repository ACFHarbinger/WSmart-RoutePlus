"""Unit tests for wsmart_bin_analysis export modules (extract and transform)."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from logic.submodules.wsmart_bin_analysis.export.container import TAG
from logic.submodules.wsmart_bin_analysis.export.extract import (
    pre_process_data,
)
from logic.submodules.wsmart_bin_analysis.export.transform import (
    filter_containers,
    get_overall_sensors_statistics,
    pre_process_container_metrics,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]


@pytest.mark.unit
@pytest.mark.fast
def test_pre_process_data():
    df_fill = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "date": ["2021-01-01 10:00:00", "2021-01-02 10:00:00", "2021-01-01 10:00:00", "2021-01-02 10:00:00"],
            "fill": [20, 80, 10, 90],
        }
    )
    df_collect = pd.DataFrame(
        {
            "id": [1, 2],
            "date": ["2021-01-02 08:00:00", "2021-01-02 08:00:00"],
            "meta": ["A", "B"],
        }
    )

    fill, collect, info = pre_process_data(
        df_fill,
        df_collect,
        id_header_fill="id",
        date_header_fill="date",
        date_format_fill="%Y-%m-%d %H:%M:%S",
        fill_header_fill="fill",
        id_header_collect="id",
        date_header_collect="date",
        date_format_collect="%Y-%m-%d %H:%M:%S",
        start_date="01/01/2020",
        end_date="01/01/2025",
    )
    assert isinstance(fill, pd.DataFrame)
    assert isinstance(collect, pd.DataFrame)
    assert isinstance(info, pd.DataFrame)


@pytest.mark.unit
@pytest.mark.fast
def test_get_overall_sensors_statistics():
    mock_container = MagicMock()
    mock_container.get_collection_quantities.return_value = (np.array([1.0]), np.array([0.9]))
    containers_dict = {1: mock_container}

    dict_dist, dict_spear = get_overall_sensors_statistics(containers_dict)
    assert 1 in dict_dist
    assert 1 in dict_spear
    assert dict_dist[1][0] == 1.0


@pytest.mark.unit
@pytest.mark.fast
def test_filter_containers():
    c1 = MagicMock()
    c1.tag = TAG.OK
    c2 = MagicMock()
    c2.tag = TAG.INSIDE_BOX
    containers = {1: c1, 2: c2}

    res = filter_containers(containers)
    assert 1 in res
    assert 2 not in res


@pytest.mark.unit
@pytest.mark.fast
def test_pre_process_container_metrics():
    c1 = MagicMock()
    containers = {1: c1}
    pre_process_container_metrics(containers, calc_spearman=True)
    assert c1.calc_max_min_mean.called
    assert c1.calc_avg_dist_metric.called
    assert c1.calc_spearman.called
