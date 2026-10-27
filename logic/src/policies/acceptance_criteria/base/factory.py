"""Factory for acceptance criteria. Construction is ``ACCEPTANCE.create``."""

import copy
import sys
from contextlib import suppress
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
        if name == "bmc":
            consumed.update(
                params={
                    "initial_temp": inst.initial_temp,
                    "alpha": inst.alpha,
                    "seed": inst.seed,
                    "start_temp_control": inst.start_temp_control,
                },
                instance_T=inst.T,
                instance_alpha=inst.alpha,
            )
        recorder = sys.modules.get("logic.src.pipeline.simulations.actions.base")
        if recorder is not None:
            with suppress(Exception):
                recorder._record_live_params("consumer_acceptance", consumed)
        return inst

    @classmethod
    def build(cls, spec: Any) -> IAcceptanceCriterion:
        """Return a criterion for ``spec``, building a fresh one from a config (P-11).

        Args:
            spec: A criterion prototype (copied for each solve), an ``AcceptanceConfig``-like object
                with ``method``/``params``, a ``{"method", "params"}`` mapping, or None
                (only-improving).

        Returns:
            The acceptance criterion.
        """
        if spec is None:
            return cls.create("oi")
        if hasattr(spec, "accept"):
            return copy.deepcopy(spec)
        if isinstance(spec, dict):
            method, params = spec.get("method", "oi"), spec.get("params")
        else:
            method, params = spec.method, getattr(spec, "params", None)
        name = _ALIASES.get(method, method)
        assert isinstance(name, str)
        return cls.create(name, config=params)
