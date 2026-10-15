"""Joint-policy registry alias for the shared ``JOINT`` instance."""

from logic.src.policies.packing_and_routing._manifest import load as _load_manifest
from logic.src.utils.registry import JOINT

_load_manifest()

JointPolicyRegistry = JOINT

__all__ = ["JointPolicyRegistry"]
