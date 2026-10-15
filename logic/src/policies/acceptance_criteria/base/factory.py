"""Factory for acceptance criteria. Construction is ``ACCEPTANCE.create``."""

import sys
from typing import Any, Dict, Optional, Union

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion
from logic.src.utils.registry import ACCEPTANCE

from .registry import AcceptanceCriterionRegistry


# Method names the configs use besides the registry keys.
_ALIASES = {"only_improving": "oi"}


class AcceptanceCriterionFactory:
    """Create an acceptance criterion by registry name."""

    @classmethod
    def ensure_registered(cls) -> None:
        return None

    @classmethod
    def create(
        cls,
        name: str,
        config: Optional[Union[Dict[str, Any], Any]] = None,
        **kwargs: Any,
    ) -> IAcceptanceCriterion:
        if AcceptanceCriterionRegistry.get(name) is None:
            raise ValueError(f"Unknown acceptance criterion: {name}")
        inst = ACCEPTANCE.create(name, cfg=config, **kwargs)
        consumed: Dict[str, Any] = {"name": name}
        recorder = sys.modules.get("logic.src.pipeline.simulations.actions.base")
        if recorder is not None:
            try:
                recorder._record_live_params("consumer_acceptance", consumed)
            except Exception:
                pass
        return inst

    @classmethod
    def build(cls, spec: Any) -> IAcceptanceCriterion:
        """Return a criterion for ``spec``, building a fresh one from a config (P-11).

        Args:
            spec: A criterion instance (used as is), an ``AcceptanceConfig``-like object
                with ``method``/``params``, a ``{"method", "params"}`` mapping, or None
                (only-improving).

        Returns:
            The acceptance criterion.
        """
        if spec is None:
            return cls.create("oi")
        if hasattr(spec, "accept"):
            return spec
        if isinstance(spec, dict):
            method, params = spec.get("method", "oi"), spec.get("params")
        else:
            method, params = spec.method, getattr(spec, "params", None)
        return cls.create(_ALIASES.get(method, method), config=params)
