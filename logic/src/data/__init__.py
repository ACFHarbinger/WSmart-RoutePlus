"""
Data generation and management module for WSmart-Route.

This package contains tools for creating synthetic VRP instances, including
builders for various problem types (PTP, MVPTP, TCMVPTP) and generation scripts.

Attributes:
    generate_datasets: Generate datasets for various problem types.

Example:
    >>> from logic.src.data import generate_datasets
    >>> generate_datasets()
"""

from typing import Any

# Imported lazily: data.generators.datasets imports the simulator repository, which
# imports the dataset classes from this package. Eager import made that a cycle in
# spawned simulation workers (they hung at start-up).


def __getattr__(name: str) -> Any:
    if name == "generate_datasets":
        from .generators.datasets import generate_datasets

        return generate_datasets
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "generate_datasets",
]
