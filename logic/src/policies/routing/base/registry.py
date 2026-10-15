"""Routing registry alias for the shared ``ROUTING`` instance."""

from logic.src.policies.routing._manifest import load as _load_manifest
from logic.src.utils.registry import ROUTING

_load_manifest()

RoutingAlgorithmRegistry = ROUTING

__all__ = ["RoutingAlgorithmRegistry"]
