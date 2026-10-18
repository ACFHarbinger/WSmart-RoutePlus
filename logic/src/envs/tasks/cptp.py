"""Capacitated Profitable Tour Problem (CPTP) environment, generator, and task specification.

In CPTP, the vehicle has finite capacity per trip. Visiting the depot unloads the vehicle
and resets trip capacity. Selecting the depot while already at the depot terminates the day.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch

from logic.src.envs.base.resources import LoadResource
from logic.src.envs.generators.cptp import CPTPGenerator
from logic.src.envs.tasks.common.released_specs import CPTP_SPEC
from logic.src.envs.tasks.core.ptp import evaluate_profitable_tour
from logic.src.envs.tasks.ptp import PTP, PTPEnv
from logic.src.envs.tasks.spec import PROBLEMS


class CPTPEnv(PTPEnv):
    """Capacitated Profitable Tour Problem Environment.

    Attributes:
        NAME: Environment identifier "cptp".
        explicit_stop: True. Selecting the depot while already at the depot ends the day.
    """

    NAME: str = "cptp"
    name: str = "cptp"
    explicit_stop: bool = True
    mask_empty_bins: bool = False

    def __init__(
        self,
        generator: Optional[CPTPGenerator] = None,
        generator_params: Optional[dict] = None,
        waste_weight: float = 1.0,
        cost_weight: float = 1.0,
        revenue_kg: Optional[float] = None,
        cost_km: Optional[float] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        generator_params = generator_params or kwargs
        if generator is None:
            generator = CPTPGenerator(**generator_params, device=device)

        super().__init__(
            generator=generator,
            generator_params=generator_params,
            waste_weight=waste_weight,
            cost_weight=cost_weight,
            revenue_kg=revenue_kg,
            cost_km=cost_km,
            device=device,
            **kwargs,
        )
        # Register LoadResource on the routing base
        self.resources = (LoadResource(),)


class CPTP(PTP):
    """Capacitated Profitable Tour Problem (CPTP) task definition."""

    NAME: str = "cptp"

    @staticmethod
    def evaluate_tour(
        dataset: Dict[str, Any],
        pi: torch.Tensor,
        dist_matrix: Optional[torch.Tensor] = None,
        cw_dict: Optional[Dict[str, float]] = None,
        capacity: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Evaluate a completed CPTP tour sequence with vectorized capacity checks."""
        return evaluate_profitable_tour(
            dataset,
            pi,
            dist_matrix=dist_matrix,
            cw_dict=cw_dict,
            capacity=capacity,
            **kwargs,
        )


PROBLEMS.register(CPTP_SPEC)

__all__ = ["CPTPGenerator", "CPTPEnv", "CPTP", "CPTP_SPEC"]
