"""Acceptance-criterion registry alias for the shared ``ACCEPTANCE`` instance."""

from logic.src.policies.acceptance_criteria._manifest import load as _load_manifest
from logic.src.utils.registry import ACCEPTANCE

_load_manifest()

AcceptanceCriterionRegistry = ACCEPTANCE

__all__ = ["AcceptanceCriterionRegistry"]
