"""Route-improvement registry alias for the shared ``IMPROVEMENT`` instance."""

from logic.src.policies.route_improvement._manifest import load as _load_manifest
from logic.src.utils.registry import IMPROVEMENT

_load_manifest()

RouteImproverRegistry = IMPROVEMENT

__all__ = ["RouteImproverRegistry"]
