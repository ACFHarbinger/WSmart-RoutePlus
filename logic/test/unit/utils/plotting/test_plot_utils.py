"""Unit tests for plotting utilities."""

from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from logic.src.utils.plotting import (
    discrete_cmap,
    draw_graph,
    plot_attention_maps_wrapper,
    plot_linechart,
    plot_vehicle_routes,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]


@patch("logic.src.utils.plotting.routes.plt")
@patch("logic.src.utils.plotting.routes.nx")
def test_draw_graph(mock_nx, mock_plt):
    """Test graph drawing calls."""
    dm = np.zeros((3, 3))
    draw_graph(dm)
    assert mock_nx.from_numpy_array.called


@patch("logic.src.utils.plotting.charts.plt")
def test_plot_linechart_simple(mock_plt):
    """Test generic line chart plotting."""
    log = np.random.randn(5, 6)  # 2D log (single policy)
    plot_func = MagicMock()
    plot_linechart("out.png", log, plot_func, ["pol1"])
    assert plot_func.called
    assert mock_plt.savefig.called


@patch("logic.src.utils.plotting.charts.plt")
def test_plot_linechart_pareto(mock_plt):
    """Test Pareto front calculation and plotting."""
    # Data: (x, y) where x is col 0, y is col 5
    # Let's make 3 points: (1, 10), (2, 20), (3, 5)
    log = np.zeros((3, 6))
    log[0, 0] = 1
    log[0, 5] = 10
    log[1, 0] = 2
    log[1, 5] = 20
    log[2, 0] = 3
    log[2, 5] = 5

    # (1, 10) is not dominated by (2, 20) because 1 < 2.
    # (2, 20) dominates (1, 10)? No, x should be minimized, y maximized.
    # point1 (1, 10), point2 (2, 20).
    # other[0] <= point[0] and other[1] >= point[1]
    # for point2: other (1, 10). 1 <= 2 and 10 >= 20? No.
    # for point1: other (2, 20). 2 <= 1? No.
    # So both (1, 10) and (2, 20) are Pareto optimal.
    # (3, 5) is dominated by (1, 10) because 1 <= 3 and 10 >= 5.

    res = plot_linechart("out.png", log, MagicMock(), ["pol1"], pareto_front=True)
    assert res is not None
    # res is list of dominants: [1, 1, 0]
    assert res[0] == [1, 1, 0]


def test_discrete_cmap():
    """Test colormap discretization."""
    cmap = discrete_cmap(5, "viridis")
    assert cmap.N == 5


@patch("logic.src.utils.plotting.routes.plt.cm.get_cmap", side_effect=plt.cm.get_cmap)
@patch("logic.src.utils.plotting.routes.plt.figure")
@patch("logic.src.utils.plotting.routes.PatchCollection")
def test_plot_vehicle_routes(mock_pc, mock_fig, mock_get_cmap):
    """Test VRP route visualization."""
    data = {
        "depot": torch.tensor([0.5, 0.5]),
        "loc": torch.tensor([[0.1, 0.1], [0.2, 0.2]]),
        "waste": torch.tensor([0.1, 0.2]),
    }
    # Route: 0 -> 1 -> 2 -> 0 represented as [0, 1, 2, 0]
    route = torch.tensor([0, 1, 2, 0])
    ax = MagicMock()
    plot_vehicle_routes(data, route, ax, visualize_waste=True)
    assert ax.quiver.called
    assert mock_pc.called


@patch("logic.src.utils.plotting.attention.plt")
@patch("logic.src.utils.plotting.attention.sns")
def test_plot_attention_maps(mock_sns, mock_plt, tmp_path):
    """Test attention map wrapper."""
    # attention_weights shape: [layers, heads, batch, size, size]
    weights = torch.randn(1, 1, 1, 5, 5)
    attn_dict = {"model": [{"attention_weights": weights}]}
    exec_func = MagicMock()

    plot_attention_maps_wrapper(
        str(tmp_path),  # home_dir
        30,  # ndays
        50,  # nbins
        "out",  # output_dir
        "area",  # area
        attention_dict=attn_dict,
        model_name="model",
        execution_function=exec_func,
    )

    assert mock_sns.heatmap.called
    assert exec_func.called
