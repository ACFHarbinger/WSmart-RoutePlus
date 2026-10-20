"""Shared policy utilities package.

This package provides shared algorithmic components used across different
routing policies, including split algorithms for transforming giant tours
into multiple routes and utilities for route reconstruction.

Attributes:
    vectorized_linear_split: Linear time split algorithm.
    vectorized_split_limited: Limited capacity split algorithm.
    reconstruct_routes: Reconstructs tour sequences from split indices.
    reconstruct_limited: Reconstruction utility for limited-capacity scenarios.

Example:
    None
"""

from __future__ import annotations

from .limited import vectorized_split_limited
from .linear import vectorized_linear_split
from .reconstruction import reconstruct_limited, reconstruct_routes

__all__ = [
    "vectorized_linear_split",
    "vectorized_split_limited",
    "reconstruct_routes",
    "reconstruct_limited",
]
