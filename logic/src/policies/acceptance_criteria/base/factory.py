"""Factory for acceptance criteria. Construction is ``ACCEPTANCE.create``."""

import sys
from typing import Any, Dict, Optional, Union

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion
from logic.src.utils.registry import ACCEPTANCE

from .registry import AcceptanceCriterionRegistry


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
