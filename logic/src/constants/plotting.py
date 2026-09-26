"""
Matplotlib plotting style constants.

Only used for post-simulation result visualisation (plotting/output utilities).
Not required for the simulator or training pipeline.

Attributes:
    MARKERS: Marker shapes for scatter plots and line plots
    PLOT_COLOURS: Plot color palette (matplotlib color names)
    LINESTYLES: Line styles for multi-series plots
"""

from typing import List, Tuple, Union

# Marker shapes for scatter plots and line plots
# Cycles through markers for multiple series on same plot.
MARKERS: List[str] = ["P", "s", "^", "8", "*"]

# Plot color palette (matplotlib color names)
PLOT_COLOURS: List[str] = [
    "red",
    "blue",
    "green",
    "yellow",
    "magenta",
    "cyan",
    "black",
]

# Line styles for multi-series plots
LINESTYLES: List[Union[str, Tuple[int, Tuple[int, ...]]]] = [
    "dotted",
    "dashed",
    "dashdot",
    (0, (3, 5, 1, 5, 1, 5)),
    "solid",
]
