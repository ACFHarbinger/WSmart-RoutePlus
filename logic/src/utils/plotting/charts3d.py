"""3D chart plotting utilities for tracking metrics.

This module provides a high-level function for generating 3D charts from
multi-policy metric data. It mirrors the API of :func:`plot_linechart` so
callers can swap between 2D and 3D representations with minimal changes.

Attributes:
    plot_3dchart: The primary entry point for generating 3D metric charts.

Example:
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from logic.src.utils.plotting import plot_3dchart
    >>> data = np.random.rand(2, 10, 3)  # 2 policies, 10 points, (x, y, z)
    >>> plot_3dchart(
    ...     "metrics_3d.png",
    ...     data,
    ...     lambda ax, x, y, z, **kw: ax.plot(x, y, z, **kw),
    ...     ["policy_a", "policy_b"],
    ...     x_label="Day", y_label="KG", z_label="Profit",
    ... )
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers the '3d' projection


def plot_3dchart(
    output_dest: str,
    graph_log: np.ndarray,
    plot_func: Callable[..., Any],
    policies: List[str],
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    z_label: Optional[str] = None,
    title: Optional[str] = None,
    fsave: bool = True,
    scale: str = "linear",
    x_values: Optional[Union[List[float], np.ndarray]] = None,
    linestyles: Optional[List[str]] = None,
    markers: Optional[List[str]] = None,
    annotate: bool = True,
) -> None:
    """Plot a generic 3D chart, handling multiple policies.

    ``plot_func`` is called as ``plot_func(ax, x, y, z, **kwargs)`` for each
    policy series, where ``ax`` is the ``Axes3D`` instance created internally.
    A typical value is ``lambda ax, x, y, z, **kw: ax.plot(x, y, z, **kw)``.

    Args:
        output_dest (str): File path to save the plot.
        graph_log (np.ndarray): Array of shape ``(n_policies, n_points, 3)``
            where the last dimension encodes ``[x, y, z]`` per point. When
            ``x_values`` is provided the first column is ignored for x.
        plot_func (callable): Axes-aware plotting callable with signature
            ``(ax, x, y, z, **kwargs)``.
        policies (list): Policy names used for the legend; must align with the
            first axis of ``graph_log``.
        x_label (str, optional): Label for the x-axis.
        y_label (str, optional): Label for the y-axis.
        z_label (str, optional): Label for the z-axis.
        title (str, optional): Chart title.
        fsave (bool, optional): Whether to save the figure to ``output_dest``.
            Defaults to True.
        scale (str, optional): Axis scale applied to x and y (``'linear'`` or
            ``'log'``). ``Axes3D`` does not expose ``set_zscale``; the z-axis
            is always rendered on a linear scale. Defaults to ``"linear"``.
        x_values (list or ndarray, optional): Explicit x-coordinates shared
            across all policies. When supplied the first column of each
            ``graph_log[i]`` slice is ignored.
        linestyles (list, optional): Cycle of matplotlib linestyle strings.
        markers (list, optional): Cycle of matplotlib marker strings.
        annotate (bool, optional): Whether to mark the terminal point of each
            series with a hollow circle. Defaults to True.
    """
    fig = plt.figure(figsize=(10, 8), dpi=200)
    ax = fig.add_subplot(111, projection="3d")

    if title is not None:
        ax.set_title(title)

    _plot_3d_series(ax, plot_func, graph_log, policies, x_values, linestyles, markers)

    if annotate:
        _annotate_3d_plot(ax, graph_log, x_values)

    _set_3d_plot_attributes(ax, scale, x_label, y_label, z_label, policies)

    if fsave:
        _save_3d_plot(output_dest, x_values)
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close()


def _plot_3d_series(
    ax: "Axes3D",
    plot_func: Callable[..., Any],
    graph_log: np.ndarray,
    policies: List[str],
    x_values: Optional[Union[List[float], np.ndarray]],
    linestyles: Optional[List[str]],
    markers: Optional[List[str]],
) -> None:
    """Iterate over policies and dispatch each series to ``plot_func``.

    Args:
        ax: The 3D axes to draw on.
        plot_func: Callable with signature ``(ax, x, y, z, **kwargs)``.
        graph_log: Shape ``(n_policies, n_points, 3)``.
        policies: Legend labels aligned with the first axis.
        x_values: Override for x-coordinates; None uses column 0 of each row.
        linestyles: Linestyle cycle.
        markers: Marker cycle.
    """
    for i, policy_data in enumerate(graph_log):
        xs = policy_data[:, 0] if x_values is None else np.asarray(x_values)
        ys = policy_data[:, 1]
        zs = policy_data[:, 2]

        kwargs: dict = {"label": policies[i]}
        if linestyles is not None:
            kwargs["linestyle"] = linestyles[i % len(linestyles)]
        if markers is not None:
            kwargs["marker"] = markers[i % len(markers)]

        plot_func(ax, xs, ys, zs, **kwargs)


def _annotate_3d_plot(
    ax: "Axes3D",
    graph_log: np.ndarray,
    x_values: Optional[Union[List[float], np.ndarray]],
) -> None:
    """Mark the terminal point of each policy series with a hollow circle.

    Args:
        ax: The 3D axes to annotate.
        graph_log: Shape ``(n_policies, n_points, 3)``.
        x_values: Override x-coordinates; None uses column 0 of each row.
    """
    for policy_data in graph_log:
        x_end = policy_data[-1, 0] if x_values is None else np.asarray(x_values)[-1]
        y_end = policy_data[-1, 1]
        z_end = policy_data[-1, 2]
        ax.scatter(
            [x_end],
            [y_end],
            [z_end],
            s=200,
            marker="o",
            facecolors="none",
            edgecolors="black",
            linewidths=1,
            zorder=10,
        )


def _set_3d_plot_attributes(
    ax: "Axes3D",
    scale: str,
    x_label: Optional[str],
    y_label: Optional[str],
    z_label: Optional[str],
    policies: List[str],
) -> None:
    """Apply axis labels, scale, and legend to a 3D axes object.

    ``Axes3D`` supports ``set_xscale`` and ``set_yscale`` but not
    ``set_zscale``; the z-axis is always linear regardless of ``scale``.

    Args:
        ax: The 3D axes to configure.
        scale: ``'linear'`` or ``'log'`` (applied to x and y only).
        x_label: Text for the x-axis label.
        y_label: Text for the y-axis label.
        z_label: Text for the z-axis label.
        policies: Policy names forwarded to ``ax.legend``.
    """
    if scale != "linear":
        ax.set_xscale(scale)  # pyrefly: ignore [not-callable]
        ax.set_yscale(scale)  # pyrefly: ignore [not-callable]
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if z_label is not None:
        ax.set_zlabel(z_label)
    ax.legend(policies)


def _save_3d_plot(output_dest: str, x_values: Optional[Union[List[float], np.ndarray]]) -> None:
    """Save the current figure with parameters matching ``_save_plot`` in charts.py.

    Args:
        output_dest: File path where the plot will be saved.
        x_values: When None, ``bbox_inches="tight"`` is applied.
    """
    if x_values is not None:
        plt.savefig(output_dest)
    else:
        plt.savefig(output_dest, bbox_inches="tight")
