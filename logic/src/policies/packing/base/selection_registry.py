"""Packing registry alias for the shared ``PACKING`` instance."""

from logic.src.policies.packing._manifest import load as _load_manifest
from logic.src.utils.registry import PACKING

_load_manifest()

PackingRegistry = PACKING

__all__ = ["PackingRegistry"]
