"""
Web-based dataset loaders and crawlers for WSmart Route+.

This module provides tools for extracting simulation and routing data from
WSmart Route+ web/HTML dashboards, including the ``HtmlSimulationDataset``
for on-the-fly dashboard-to-simulation conversion.

Attributes:
    HtmlSimulationDataset: Dataset loader from HTML dashboards.
    extract_dataframe: Low-level parser for HTML table extraction.
    to_csv: Utility to export dashboard to CSV.
    to_excel: Utility to export dashboard to Excel.
    to_simulation_data: Utility to convert dashboard data to standard frames.

Example:
    >>> from logic.src.data.datasets.web import HtmlSimulationDataset
    >>> dataset = HtmlSimulationDataset.load(
    ...     "path/to/dashboard.html",
    ...     area="candaval",
    ...     waste_type="residuo",
    ...     n_days=31
    ... )
"""

from logic.src.data.datasets.web.dashboard_crawler import (
    extract_dataframe,
    to_csv,
    to_excel,
    to_simulation_data,
)
from logic.src.data.datasets.web.html_sim_dataset import HtmlSimulationDataset

__all__ = [
    "extract_dataframe",
    "to_csv",
    "to_excel",
    "to_simulation_data",
    "HtmlSimulationDataset",
]
